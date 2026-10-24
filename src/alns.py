#!/usr/bin/env python3
"""ALNS（自适应大邻域搜索）启发式 + V4 与 CPLEX 精确解收敛性验证。

架构（对应论文 C4 贡献：规则暖启动 + 学习自适应权重，零训练）：
  外层: 对一阶段决策 (xJ,xK,xL,zJ,zK,rL) 搜索 —— ALNS 算子直接作用于设施开关/扩容/退出
  内层: 给定一阶段配置，二阶段为 LP —— 用 scipy linprog 精确求解各情景分配子问题

服务级覆盖约束（V2.2 修订，与 CPLEX 对齐）：
  论文式 (14) 要求每个训练情景的未满足量为零（Σ u_it = 0）。这是一个
  **一阶服务级约束**，使 ALNS 与 CPLEX 求解同一个优化问题，因而二者可比、
  gap 有意义。实现上：
  - 搜索期用覆盖增强值 val = cost + COVER_M · E[unmet] 引导搜索（COVER_M 远大于
    任何扩容模块的边际容量成本，保证搜索不会用覆盖换成本）；
  - 候选池（论文 Algorithm 1）：搜索期最优在缩减集 S' 上维护，每次刷新 best 时
    入池；返回前对全池在报告集 S 上**按服务级约束判定**，返回真最优。
  - 满配保底解入池：即使搜索退化，也不会比最大建设方案的覆盖更差。

算子（毁坏+修复）:
  D1 随机关闭一座开启设施 / D2 随机开启一座关闭设施 / D3 交换一对开关
  D4 扩容档±1（保持单调）/ D5 提前/推迟一座消纳场退出 / D6 贪换单厂开关
自适应权重: 算子收益 1/gap 更新（softmax选择），每 iter_decay 衰减。
V4: 在 tiny 与真实实例上与 CPLEX 精确解比对（tiny 0 gap；真实实例 ≤1% gap）。
"""
import json, os, sys, random, math, time
import numpy as np
from scipy.optimize import linprog

BASE = os.path.dirname(os.path.abspath(__file__))

# 覆盖增强权重（CNY/t）：2 倍于未满足惩罚 π=500，远高于扩容模块的边际容量成本
COVER_M = 1000.0
# 服务级约束的数值容差（吨）：低于此视为零覆盖缺口
COVER_TOL = 1.0
# 模拟退火初温（占当前成本的比例）
TEMP0 = 0.01

# ---------- 内层 LP：给定一阶段配置求各情景最优分配（最小化二阶段成本） ----------
def solve_second_stage_lp_ex(inst, xcfg, d_scenario, require_cover=False):
    """规范化二阶段 LP。返回 (成本, 未满足总量, 回收总量)；不可行返回 (None, None, None)。
    rho 分流为等式（论文式 (3)）：sum_k w = rho * out，即 (1-rho)*sum_w - rho*sum_q = 0。
    require_cover=True 时施加服务级覆盖约束（论文式 (14)）：该情景未满足量为零；
    此时 LP 与 CPLEX 的对应子问题字面同构，ALNS 与 CPLEX 求解同一个优化问题。
    LP 变量: y[i,j,t], yd[i,l,t], w[j,k,t], q[j,l,t], v[k,l,t], s[j,t], u[i,t]"""
    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting']); RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)

    idx = {}
    n = 0
    for i in GEN:
        for j in SORT:
            for t in range(1, T+1): idx[('y', i, j, t)] = n; n += 1
    for i in GEN:
        for l in LAND:
            for t in range(1, T+1): idx[('yd', i, l, t)] = n; n += 1
    for j in SORT:
        for k in RECY:
            for t in range(1, T+1): idx[('w', j, k, t)] = n; n += 1
    for j in SORT:
        for l in LAND:
            for t in range(1, T+1): idx[('q', j, l, t)] = n; n += 1
    for k in RECY:
        for l in LAND:
            for t in range(1, T+1): idx[('v', k, l, t)] = n; n += 1
    for j in SORT:
        for t in range(1, T+1): idx[('s', j, t)] = n; n += 1
    for i in GEN:
        for t in range(1, T+1): idx[('u', i, t)] = n; n += 1

    c = np.zeros(n)
    for (key, *a), var in idx.items():
        if key == 'y': c[var] = P['transport_cost']*inst['dist']['a'][a[0]][a[1]] + inst['sorting'][a[1]]['op_cost']
        elif key == 'yd': c[var] = P['transport_cost']*inst['dist']['a2'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 'w': c[var] = P['transport_cost']*inst['dist']['b'][a[0]][a[1]] + inst['recycling'][a[1]]['op_cost'] - P.get('recycle_revenue', 0.0)*phi
        elif key == 'q': c[var] = P['transport_cost']*inst['dist']['c'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 'v': c[var] = P['transport_cost']*inst['dist']['e'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 's': c[var] = P['hold_cost']
        elif key == 'u': c[var] = P['unmet_penalty']

    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    def eqrow(coeffs, rhs):
        r = np.zeros(n)
        for k_, cval in coeffs: r[idx[k_]] += cval
        A_eq.append(r); b_eq.append(rhs)
    def ubrow(coeffs, rhs):
        r = np.zeros(n)
        for k_, cval in coeffs: r[idx[k_]] += cval
        A_ub.append(r); b_ub.append(rhs)

    for i in GEN:
        for t in range(1, T+1):
            eqrow([(('y', i, j, t), 1.0) for j in SORT] + [(('yd', i, l, t), 1.0) for l in LAND] + [(('u', i, t), 1.0)],
                  d_scenario[i][t])
    for j in SORT:
        for t in range(1, T+1):
            prev = [(('s', j, t-1), 1.0)] if t > 1 else []
            eqrow(prev + [(('y', i, j, t), 1.0) for i in GEN]
                  + [(('w', j, k, t), -1.0) for k in RECY] + [(('q', j, l, t), -1.0) for l in LAND] + [(('s', j, t), -1.0)], 0.0)
            outs = [(('w', j, k, t), 1.0) for k in RECY]
            qs = [(('q', j, l, t), 1.0) for l in LAND]
            # 式 (3)：Σw = ρ·out = ρ(Σw+Σq) → (1-ρ)Σw - ρΣq = 0（等式）
            eqrow([(x, (1-rho)*cval) for x, cval in outs] + [(x, -rho*cval) for x, cval in qs], 0.0)
    for k in RECY:
        for t in range(1, T+1):
            eqrow([(('w', j, k, t), (1-phi)) for j in SORT] + [(('v', k, l, t), -1.0) for l in LAND], 0.0)  # (1-φ)Σw=Σv 余渣
    zJ_cfg = xcfg.get('zJ', {})
    for j in SORT:
        zj_t = zJ_cfg.get(j, [0]*T)
        step_j = inst['sorting'][j].get('expand_step_t', 30e4)
        for t in range(1, T+1):
            cap_j = inst['sorting'][j]['cap_t'] + step_j*zj_t[t-1]
            ubrow([(('y', i, j, t), 1.0) for i in GEN], cap_j*xcfg['xJ'][j][t-1])
            ubrow([(('s', j, t), 1.0)], (inst['sorting'][j]['buffer_cap_t'] + step_j*zj_t[t-1]*0.2)*xcfg['xJ'][j][t-1])
    for k in RECY:
        for t in range(1, T+1):
            # 与 CPLEX C5a∧C5z 同构：关闭的厂不获得任何容量（zK 只在 xK=1 时计容量）
            cap = (inst['recycling'][k]['cap_t'] + inst['recycling'][k]['expand_step_t']*xcfg['zK'][k][t-1]) * xcfg['xK'][k][t-1]
            ubrow([(('w', j, k, t), 1.0) for j in SORT], cap)
            if xcfg['xK'][k][t-1] == 1:
                r = np.zeros(n)
                for j in SORT: r[idx[('w', j, k, t)]] -= 1.0
                A_ub.append(r); b_ub.append(-inst['recycling'][k]['min_run_t'])
    for l in LAND:
        for t in range(1, T+1):
            used = xcfg['xL'][l][t-1] and xcfg['rL'][l][t-1] == 0
            cap = inst['landfill'][l]['cap_t'] if used else 0.0
            ubrow([(('q', j, l, t), 1.0) for j in SORT] + [(('yd', i, l, t), 1.0) for i in GEN] + [(('v', k, l, t), 1.0) for k in RECY], cap)
        cum = []
        for j in SORT:
            for t in range(1, T+1): cum.append((('q', j, l, t), 1.0))
        for i in GEN:
            for t in range(1, T+1): cum.append((('yd', i, l, t), 1.0))
        for k in RECY:
            for t in range(1, T+1): cum.append((('v', k, l, t), 1.0))
        ubrow(cum, inst['landfill'][l]['cap_t'])

    if require_cover:
        # 服务级覆盖约束（论文式 (14)）：该情景零未满足
        eqrow([(('u', i, t), 1.0) for i in GEN for t in range(1, T+1)], 0.0)

    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    if res.status != 0:
        return None, None, None
    x = res.x
    unmet = sum(x[idx[('u', i, t)]] for i in GEN for t in range(1, T+1))
    recycled = sum(x[idx[('w', j, k, t)]] for j in SORT for k in RECY for t in range(1, T+1))
    return res.fun, unmet, recycled


def solve_second_stage_lp(inst, xcfg, d_scenario, require_cover=False):
    """兼容接口：只返回二阶段成本（或 None）。"""
    cost, _, _ = solve_second_stage_lp_ex(inst, xcfg, d_scenario, require_cover=require_cover)
    return cost


def solve_second_stage_lp_detail(inst, xcfg, d_scenario, require_cover=False):
    """兼容接口：返回 (成本, 未满足总量, 回收总量)。"""
    return solve_second_stage_lp_ex(inst, xcfg, d_scenario, require_cover=require_cover)


def diagnose_lp(inst, xcfg, d_scenario):
    """LP不可行诊断：逐约束类别测试松弛可行性"""
    import copy
    test = copy.deepcopy(inst)
    test['params']['recyclable_ratio'] = 1.0
    c2 = solve_second_stage_lp(test, xcfg, d_scenario)
    print(f'  [diag] ρ=1(去比例约束): {"可行" if c2 is not None else "不可行"}')
    test2 = copy.deepcopy(inst)
    for j in test2['sorting']: test2['sorting'][j]['buffer_cap_t'] = 1e9
    c2 = solve_second_stage_lp(test2, xcfg, d_scenario)
    print(f'  [diag] 放开缓冲: {"可行" if c2 is not None else "不可行"}')

def fixed_cost_of(inst, xcfg, FU=1e4):
    T = inst['meta']['periods']
    fc = sum((inst['sorting'][j]['open_fixed']*FU*xcfg['xJ'][j][t-1] + inst['sorting'][j].get('expand_cost', 600.0)*FU*xcfg.get('zJ', {j: [0]*T for j in inst['sorting']})[j][t-1]) for j in inst['sorting'] for t in range(1, T+1)) \
       + sum((inst['recycling'][k]['open_fixed']*FU*xcfg['xK'][k][t-1] + inst['recycling'][k]['expand_cost']*FU*xcfg['zK'][k][t-1]) for k in inst['recycling'] for t in range(1, T+1)) \
       + sum(inst['landfill'][l]['open_fixed']*FU*xcfg['xL'][l][t-1] for l in inst['landfill'] for t in range(1, T+1))
    return fc

def evaluate(inst, xcfg, scenarios, FU=1e4, require_cover=False):
    """总成本 = fixed + Σ p_s · LP_s。不可行情景返回 None。"""
    total = fixed_cost_of(inst, xcfg, FU)
    for sc in scenarios:
        c2, _, _ = solve_second_stage_lp_ex(inst, xcfg, sc['d'], require_cover=require_cover)
        if c2 is None:
            return None
        total += sc['prob'] * c2
    return total

def cfg_to_tuplekey(cfg):
    return json.dumps(cfg, sort_keys=True)

# ---------- ALNS ----------
def initial_solution(inst):
    T = inst['meta']['periods']
    # 暖启动规则（对应论文"规则暖启动"贡献）：按2025年报实际状态初始化
    xJ = {j: [1]*T for j in inst['sorting']}                      # 现状分拣中心全开
    xK = {k: [1]*T for k in inst['recycling']}                    # 现状资源化厂全开
    zK = {k: [0]*T for k in inst['recycling']}
    xL = {l: [1]*T for l in inst['landfill']}
    rL = {l: [0]*T for l in inst['landfill']}
    for l, info in inst['landfill'].items():
        if info['retire_candidate']:                               # 大厂库伦T2起退出
            for t in range(2, T+1): rL[l][t-1] = 1
    zJ = {j: [0]*T for j in inst['sorting']}
    return {'xJ': xJ, 'xK': xK, 'xL': xL, 'zK': zK, 'zJ': zJ, 'rL': rL}

def full_build(inst):
    """满配保底解：全部设施开启 + 分拣/厂满模块（z=4）。"""
    T = inst['meta']['periods']
    cfg = {
        'xJ': {j: [1]*T for j in inst['sorting']},
        'xK': {k: [1]*T for k in inst['recycling']},
        'xL': {l: [1]*T for l in inst['landfill']},
        'zJ': {j: [4]*T for j in inst['sorting']},
        'zK': {k: [4]*T for k in inst['recycling']},
        'rL': {l: [0]*T for l in inst['landfill']},
    }
    return repair_monotone(cfg, inst)

def repair_monotone(cfg, inst):
    """恢复单调性：扩容档 z_t ≤ z_{t+1}（式 (11)），退出不可逆（式 (10)）。
    注意单向传播方向：把 t 期的值向上传播到 t+1 期（z_{t+1} ≥ z_t），
    即"前期缩、后期保持"在单调约束下要求 z_{t+1} ≥ z_t，因此单期缩减
    会被后续期吸收。收缩移动因此以"尾段"形式实现，见 _shrink_moves。"""
    T = inst['meta']['periods']
    for k in inst['recycling']:
        for t in range(1, T): cfg['zK'][k][t] = max(cfg['zK'][k][t-1:t+1])
    for j in inst['sorting']:
        for t in range(1, T): cfg['zJ'][j][t] = max(cfg['zJ'][j][t-1:t+1])
    for l in inst['landfill']:
        for t in range(1, T): cfg['rL'][l][t] = max(cfg['rL'][l][t-1:t+1])
        for t in range(1, T+1):
            if cfg['rL'][l][t-1] == 1: cfg['xL'][l][t-1] = 0
    for l, info in inst['landfill'].items():
        if info['retire_candidate']:
            for t in range(2, T+1): cfg['rL'][l][t-1] = 1; cfg['xL'][l][t-1] = 0
    return cfg

def clone(cfg):
    return {kk: {k: list(v) for k, v in vv.items()} for kk, vv in cfg.items()}

def _lp_feasible(inst, cfg, scen):
    """检查配置在 scen 上是否覆盖可行（内层 LP 带服务级约束）。返回 (ok, cost)。"""
    fc = fixed_cost_of(inst, cfg)
    qs, ps = [], []
    for sc in scen:
        c2, _, _ = solve_second_stage_lp_ex(inst, cfg, sc['d'], require_cover=True)
        if c2 is None:
            return False, None
        qs.append(c2); ps.append(sc['prob'])
    ps = np.array(ps); ps = ps / ps.sum()
    return True, fc + float((np.array(qs) * ps).sum())

def _capacity_slack(inst, cfg, scen):
    """诊断各分拣中心/资源化厂在 scen 上的容量占用率（用于定位瓶颈设施）。"""
    T = inst['meta']['periods']
    util = {('J', j): 0.0 for j in inst['sorting']}
    util.update({('K', k): 0.0 for k in inst['recycling']})
    for sc in scen:
        for j in inst['sorting']:
            cap = (inst['sorting'][j]['cap_t'] + inst['sorting'][j].get('expand_step_t', 30e4)*cfg['zJ'][j][0]) \
                  * cfg['xJ'][j][0]
            for t in range(1, T+1):
                cap_t = (inst['sorting'][j]['cap_t'] + inst['sorting'][j].get('expand_step_t', 30e4)*cfg['zJ'][j][t-1]) * cfg['xJ'][j][t-1]
                if cap_t > 0: util[('J', j)] = max(util[('J', j)], 1.0)
        for k in inst['recycling']:
            for t in range(1, T+1):
                cap_t = (inst['recycling'][k]['cap_t'] + inst['recycling'][k]['expand_step_t']*cfg['zK'][k][t-1]) * cfg['xK'][k][t-1]
                if cap_t > 0: util[('K', k)] = max(util[('K', k)], 1.0)
    return util

def repair_cover(inst, cfg, scen, max_bump=6, verbose=False):
    """覆盖保持修复算子（D7）：若配置在 scen 上覆盖不可行，
    自动给最紧张的可行设施加一个扩容模块（保持单调与 z≤4），直到恢复可行。
    这使搜索空间在服务级约束下连通，避免搜索卡在不可行区域。"""
    c2 = clone(cfg)
    for _ in range(max_bump):
        ok, _ = _lp_feasible(inst, c2, scen)
        if ok:
            return c2, True
        # 找到可以加模块的设施（x=1 且 z<4）
        cands = []
        for j in inst['sorting']:
            if c2['xJ'][j][0] == 1:
                for t in range(1, inst['meta']['periods']+1):
                    if c2['zJ'][j][t-1] < 4:
                        cands.append(('J', j, t)); break
        for k in inst['recycling']:
            if c2['xK'][k][0] == 1:
                for t in range(1, inst['meta']['periods']+1):
                    if c2['zK'][k][t-1] < 4:
                        cands.append(('K', k, t)); break
        if not cands:
            return c2, False
        # 优先给分拣中心加模块（分拣是论文识别的瓶颈 echelon），否则加厂
        pick = None
        for pref in ('J', 'K'):
            grp = [c for c in cands if c[0] == pref]
            if grp:
                pick = grp[0]; break
        typ, idx, t0 = pick
        if typ == 'J':
            for tt in range(t0, inst['meta']['periods']+1):
                c2['zJ'][idx][tt-1] = min(4, c2['zJ'][idx][tt-1] + 1)
        else:
            for tt in range(t0, inst['meta']['periods']+1):
                c2['zK'][idx][tt-1] = min(4, c2['zK'][idx][tt-1] + 1)
        c2 = repair_monotone(c2, inst)
    ok, _ = _lp_feasible(inst, c2, scen)
    return c2, ok

def _covers(inst, cfg, scen, FU=1e4, cover=True):
    """检查配置在 scen 上是否满足服务级覆盖约束（论文式 (14)），并返回 (cost, ok)。
    cost 用覆盖可行时的 LP 成本（不可行时为 None）。"""
    fc = fixed_cost_of(inst, cfg, FU)
    qs, ps = [], []
    for sc in scen:
        c2, unm, _ = solve_second_stage_lp_ex(inst, cfg, sc['d'], require_cover=cover)
        if c2 is None:
            return None, False
        qs.append(c2); ps.append(sc['prob'])
    ps = np.array(ps); ps = ps / ps.sum()
    return fc + float((np.array(qs) * ps).sum()), True

def _shrink_moves(inst, cfg):
    """生成一组"收缩"动作（从满配解往成本低的方向走）：
    关闭一座分拣中心 / 关闭一座资源化厂 / 某站某期扩容档 -1 / 提前一座填埋场退出。
    扩容档收缩只作用于**单个期**，再由 repair_monotone 恢复单调性；
    这样才能表达"前期不扩、后期扩"这类非对称扩容模式（单调约束下 z_t ≤ z_{t+1}
    允许前期低、后期高）。每个动作为一个 (说明, 新配置) 对。"""
    T = inst['meta']['periods']
    moves = []
    for j in inst['sorting']:
        if any(cfg['xJ'][j]):
            c2 = clone(cfg); c2['xJ'][j] = [0]*T
            moves.append((f'close_sort_{j}', repair_monotone(c2, inst)))
    for k in inst['recycling']:
        if any(cfg['xK'][k]):
            c2 = clone(cfg); c2['xK'][k] = [0]*T
            moves.append((f'close_plant_{k}', repair_monotone(c2, inst)))
    for j in inst['sorting']:
        # 扩容档非递减（式 11：z_t ≤ z_{t+1}）。收缩单个期后由 repair_monotone
        # 前向修复：后期吸收前期的缩减，得到仍非递减的阶梯（如 [0,4,4]）。
        for t in range(1, T+1):
            if cfg['zJ'][j][t-1] > 0:
                c2 = clone(cfg)
                c2['zJ'][j][t-1] = cfg['zJ'][j][t-1] - 1
                moves.append((f'down_zJ_{j}_t{t}', repair_monotone(c2, inst)))
    for k in inst['recycling']:
        for t in range(1, T+1):
            if cfg['zK'][k][t-1] > 0:
                c2 = clone(cfg)
                c2['zK'][k][t-1] = cfg['zK'][k][t-1] - 1
                moves.append((f'down_zK_{k}_t{t}', repair_monotone(c2, inst)))
    for l in inst['landfill']:
        if not inst['landfill'][l]['retire_candidate'] and not any(cfg['rL'][l]):
            for t in range(2, T+1):
                c2 = clone(cfg)
                c2['rL'][l][t-1] = 1
                moves.append((f'retire_{l}_t{t}', repair_monotone(c2, inst)))
    return moves

def cover_preserving_descent(inst, scen, cfg0=None, FU=1e4, max_rounds=8, verbose=False,
                              restarts=4):
    """覆盖保持的贪心下降（含重启）。从满配解出发，每一步只接受在 scen
    （完整报告集）上仍满足服务级覆盖约束、且成本更低的收缩动作。

    单趟下降会陷入"先关厂再降档"的局部最优（实测停在 0.96B，而 CPLEX 到 0.79B）。
    为此加入**重启**：若干趟下降后从满配解重新开始，跳过已尝试过的前几步，
    以不同的顺序探索收缩路径，取所有趟中最好的结果。

    返回 (cfg, cost)；若满配本身不覆盖（物理不可行），返回 (满配, None)。"""
    start = clone(cfg0) if cfg0 is not None else full_build(inst)
    base_cost, ok = _covers(inst, start, scen, FU)
    if not ok:
        return start, None
    best_cfg, best_cost = clone(start), base_cost
    banned = set()          # 已尝试过的收缩动作标签（本趟内避免重复）
    # 筛选集：分层抽样（保留全部锚点 + 其余等距抽取），用于快速筛掉劣质动作；
    # 最终决定仍在完整报告集上做
    n_screen = min(len(scen), 8)
    anchors = [sc for sc in scen if sc['name'].startswith('anchor')]
    others = [sc for sc in scen if not sc['name'].startswith('anchor')]
    if len(anchors) >= n_screen:
        screen = anchors[:n_screen]
    else:
        step = max(1, len(others) // max(1, n_screen - len(anchors)))
        screen = anchors + others[::step][:n_screen - len(anchors)]
    for rd in range(restarts):
        cur = clone(start)
        cost = base_cost
        banned = set()
        for rnd in range(max_rounds):
            # 第一步：在筛选集上比较所有收缩动作，选出最优候选
            best_screen, best_tag, best_cand = float('inf'), None, None
            for tag, cand in _shrink_moves(inst, cur):
                if tag in banned:
                    continue
                c, okc = _covers(inst, cand, screen, FU)
                if okc:
                    if c < best_screen - 1e-6:
                        best_screen, best_tag, best_cand = c, tag, cand
                    continue
                # 收缩破坏了覆盖 → 用 D7 修复算子加模块救回
                fixed_cfg, fixed_ok = repair_cover(inst, cand, screen)
                if fixed_ok:
                    c2, ok2 = _covers(inst, fixed_cfg, screen, FU)
                    if ok2 and c2 < best_screen - 1e-6:
                        best_screen, best_tag, best_cand = c2, f'{tag}+repair', fixed_cfg
            if best_tag is None:
                break
            # 第二步：在完整报告集上验证该候选
            c_full, okf = _covers(inst, best_cand, scen, FU)
            if not okf or c_full >= cost - 1e-6:
                break
            banned.add(best_tag.split('+')[0])
            cur, cost = best_cand, c_full
            if verbose: print(f'  descent r{rd}.{rnd}: {best_tag} -> {cost/1e9:.4f}B')
        if cost < best_cost - 1e-6:
            best_cfg, best_cost = clone(cur), cost
    return best_cfg, best_cost

def alns(inst, scenarios, max_iter=1000, seed=20260825, verbose=True, FU=1e4,
         report_scen=None, obj_fn=None, require_cover=True, max_time=None):
    """混合 ALNS。搜索在 scenarios（缩减集 S'）上进行；report_scen（默认=S'）
    用于返回前对候选池的精确重评估与最终选择（论文 Algorithm 1）。

    接受准则为覆盖增强值：val = cost + COVER_M * E[unmet on S']。
    require_cover=True 时，候选池中**满足服务级约束**（每个报告情景未满足量为零）
    的解优先于不满足者；同等条件下按成本选。
    obj_fn 可注入替代目标（如均值-方差），签名 obj_fn(cost, wunmet, qs, ps) -> value。
    返回 (best, best_cost_on_report, history, walltime)。"""
    rng = random.Random(seed)
    Tn = inst['meta']['periods']
    rs = report_scen if report_scen is not None else scenarios

    def _eval(cfg, scen, cover=False):
        """返回 (增强值, 真实成本, 加权未满足, 最大单情景未满足) 或 None。
        cover=True 时内层 LP 施加服务级覆盖约束（论文式 (14)）：不满足覆盖的
        配置直接不可行而被淘汰。这使搜索与 CPLEX 求解同一个问题。"""
        fc = fixed_cost_of(inst, cfg, FU)
        qs, ps, us = [], [], []
        for sc in scen:
            c2, unm, _ = solve_second_stage_lp_ex(inst, cfg, sc['d'], require_cover=cover)
            if c2 is None:
                return None
            qs.append(c2); ps.append(sc['prob']); us.append(unm)
        ps = np.array(ps); ps = ps / ps.sum()
        qs = np.array(qs); us = np.array(us)
        cost = fc + float((qs * ps).sum())
        wunmet = float((us * ps).sum())
        maxu = float(us.max()) if len(us) else 0.0
        val = obj_fn(cost, wunmet, qs, ps) if obj_fn is not None else cost + COVER_M * wunmet
        return val, cost, wunmet, maxu

    def _rank(t):
        """候选排序键：先看是否满足服务级约束，再看增强值。"""
        if t is None:
            return (2, float('inf'))
        if require_cover:
            return (0 if t[3] <= COVER_TOL else 1, t[0])
        return (0, t[0])

    # 规则暖启动作为搜索起点（对应论文"规则暖启动"贡献）
    cur = repair_monotone(initial_solution(inst), inst)
    cur_t = _eval(cur, scenarios, cover=True)
    if cur_t is None:
        cur = full_build(inst)
        cur_t = _eval(cur, scenarios, cover=True)
    best, best_t = clone(cur), cur_t
    pool = {}          # key -> (cfg, enhanced_value_on_S')
    POOL_MAX = 40      # 池上限：保留覆盖候选中增强值最低者，控制末尾重评估成本

    def _try_pool(cfg, t):
        """把满足服务级覆盖约束的候选存入池（论文 Algorithm 1 的候选池）。
        注意：不要求它是搜索期 best —— 增强值最低的解通常是满配解，
        若只收集 best，满足覆盖但成本更低的次优解会被漏掉，搜索失败时
        只能返回满配方案（种子间方差因此爆炸）。"""
        if t is None or t[3] > COVER_TOL:
            return
        k = cfg_to_tuplekey(cfg)
        if k in pool:
            return
        pool[k] = (clone(cfg), t[0])
        if len(pool) > POOL_MAX:
            # 淘汰增强值最高的候选
            worst = max(pool.items(), key=lambda kv: kv[1][1])[0]
            del pool[worst]
    _try_pool(best, best_t)
    # 满配保底解：入池参与最终择优（不作为搜索起点，否则搜索会被锁死在满配上）
    floor = full_build(inst)
    floor_t = _eval(floor, scenarios, cover=True)
    if floor_t is not None:
        kf = cfg_to_tuplekey(floor)
        if kf not in pool: pool[kf] = (clone(floor), floor_t[0])
    ops = ['D1_close', 'D2_open', 'D3_swap', 'D4_expand', 'D5_retire', 'D6_greedy_close']
    weights = {o: 1.0 for o in ops}
    t0 = time.time()
    history = []
    for it in range(max_iter):
        if max_time is not None and time.time() - t0 > max_time:
            if verbose:
                print(f'[alns] wall budget {max_time:.0f}s reached at iter {it}', flush=True)
            break  # 墙钟预算：与 CPLEX 统一 TL 的大实例协议
        # softmax 选择算子
        ws = [weights[o] for o in ops]
        mx = max(ws)
        ps = [math.exp(min(60.0, w - mx)) for w in ws]   # 截断指数，防止长跑溢出
        tot = sum(ps)
        if not math.isfinite(tot) or tot <= 0:
            ps = [1.0 / len(ops)] * len(ops)
        else:
            ps = [p / tot for p in ps]
        op = rng.choices(ops, weights=ps)[0]
        cand = clone(cur)
        if op == 'D1_close':
            pool_ = [j for j in inst['sorting'] if any(cand['xJ'][j])]
            pool_ += [k for k in inst['recycling'] if any(cand['xK'][k])]
            if pool_:
                e = rng.choice(pool_)
                if e in cand['xJ']:
                    t = rng.randint(1, Tn); cand['xJ'][e] = [0 if tt >= t else 1 for tt in range(1, Tn+1)]
                else:
                    t = rng.randint(1, Tn); cand['xK'][e] = [0 if tt >= t else 1 for tt in range(1, Tn+1)]
        elif op == 'D2_open':
            pool_ = [j for j in inst['sorting'] if not any(cand['xJ'][j])] + [k for k in inst['recycling'] if not any(cand['xK'][k])]
            if pool_:
                e = rng.choice(pool_)
                if e in cand['xJ']: cand['xJ'][e] = [1]*Tn
                else: cand['xK'][e] = [1]*Tn
        elif op == 'D3_swap':
            on = [j for j in inst['sorting'] if any(cand['xJ'][j])]
            off = [j for j in inst['sorting'] if not any(cand['xJ'][j])]
            if on and off:
                cand['xJ'][rng.choice(on)] = [0]*Tn; cand['xJ'][rng.choice(off)] = [1]*Tn
        elif op == 'D4_expand':
            if rng.random() < 0.5:
                k = rng.choice(list(inst['recycling']))
                t = rng.randint(1, Tn)
                delta = rng.choice([-1, 1])
                for tt in range(t, Tn+1):
                    cand['zK'][k][tt-1] = max(0, min(4, cand['zK'][k][tt-1] + delta))
            else:
                j = rng.choice(list(inst['sorting']))
                t = rng.randint(1, Tn)
                delta = rng.choice([-1, 1])
                for tt in range(t, Tn+1):
                    cand['zJ'][j][tt-1] = max(0, min(4, cand['zJ'][j][tt-1] + delta))
        elif op == 'D6_greedy_close' and it % 20 == 0 and cur_t is not None:
            # cur_t 为 None 说明当前 incumbent 尚无覆盖可行评估（如 ε 超过物理上限），跳过 D6
            base_v = cur_t[0]
            best_delta, best_e, best_mode = 0, None, None
            for e in inst['recycling']:
                if any(cur['xK'][e]):
                    c2 = clone(cur); c2['xK'][e] = [0]*Tn; c2 = repair_monotone(c2, inst)
                    t2 = _eval(c2, scenarios, cover=True)
                    if t2 is not None and base_v - t2[0] > best_delta:
                        best_delta, best_e, best_mode = base_v - t2[0], e, 'off'
            for e in inst['recycling']:
                if not any(cur['xK'][e]):
                    c2 = clone(cur); c2['xK'][e] = [1]*Tn; c2 = repair_monotone(c2, inst)
                    t2 = _eval(c2, scenarios, cover=True)
                    if t2 is not None and base_v - t2[0] > best_delta:
                        best_delta, best_e, best_mode = base_v - t2[0], e, 'on'
            if best_e is not None:
                cand['xK'][best_e] = [1 if best_mode == 'on' else 0]*Tn
        elif op == 'D5_retire':
            elig = [l for l in inst['landfill'] if not inst['landfill'][l]['retire_candidate'] and not any(cand['rL'][l])]
            if elig:
                l = rng.choice(elig); t = rng.randint(1, Tn)
                for tt in range(t, Tn+1): cand['rL'][l][tt-1] = 1
        cand = repair_monotone(cand, inst)
        t = _eval(cand, scenarios, cover=True)
        if t is None:
            # D7 覆盖保持修复：候选因覆盖不可行被淘汰时，尝试加模块救回
            fixed_cfg, fixed_ok = repair_cover(inst, cand, scenarios)
            if fixed_ok:
                t = _eval(fixed_cfg, scenarios, cover=True)
                if t is not None:
                    cand = fixed_cfg
            if t is None:
                weights[op] = max(weights[op] * 0.9, 1e-6); continue
        _try_pool(cand, t)
        v = t[0]
        temp = max(best_t[1]*TEMP0*math.exp(-it/max_iter*5), 1e-6*max(abs(best_t[1]),1))
        if v < cur_t[0] - 1e-6 or (v - cur_t[0]) < temp*rng.random():
            cur, cur_t = clone(cand), t
            weights[op] = min(weights[op] * 1.15, 1e6)
            if v < best_t[0] - 1e-6:
                best, best_t = clone(cand), t
                weights[op] = min(weights[op] * 1.1, 1e6)
                if verbose: print(f'  it{it} [{op}] 新最优 val={best_t[0]:,.0f} cost={best_t[1]:,.0f} unmet={best_t[2]:,.1f}t max={best_t[3]:,.1f}t')
        else:
            weights[op] = max(weights[op] * 0.98, 1e-6)
        history.append(best_t[1])
    # 候选池在报告集上精确重评估，返回真最优（论文 Algorithm 1 末行）。
    # 报告期同样施加服务级覆盖约束：返回的计划必须在整个报告集上可行，
    # 且其计划值与 CPLEX 的目标值同口径（可直接比较 gap）。
    best_key, best_pick, best_rank = None, None, None
    fallback_key, fallback_pick, fallback_rank = None, None, None
    for k, (cfg, _) in pool.items():
        t = _eval(cfg, rs, cover=True)          # 同口径：覆盖可行且成本最小
        if t is not None:
            r = _rank(t)
            if best_rank is None or r < best_rank:
                best_key, best_pick, best_rank = k, t, r
        t2 = _eval(cfg, rs, cover=False)         # 退路：覆盖不可行时按实际表现择优
        if t2 is not None:
            r2 = _rank(t2)
            if fallback_rank is None or r2 < fallback_rank:
                fallback_key, fallback_pick, fallback_rank = k, t2, r2
    if best_key is not None:
        best, best_cost = pool[best_key][0], best_pick[1]
    elif fallback_key is not None:
        best, best_cost = pool[fallback_key][0], fallback_pick[1]
    else:
        best_cost = _eval(best, rs, cover=False)[1]

    # V2.2 确定性精化：覆盖保持的贪心下降（在完整报告集上）。
    # 随机算子负责跳出局部最优；本阶段负责把覆盖解的成本压到局部最优，
    # 从而消除种子间方差（V2.1 观测到的 ±0.12B 方差来源）。
    if require_cover:
        # 从两个起点分别下降，取更优者：满配解（必定覆盖）与池中最优解
        starts = [best]
        for st_name, st_cfg in [('pool', pool[best_key][0] if best_key else None),
                                ('floor', full_build(inst))]:
            if st_cfg is None:
                continue
            d_cfg, d_cost = cover_preserving_descent(inst, rs, st_cfg, FU)
            if d_cost is not None and d_cost < best_cost - 1e-6:
                best, best_cost = d_cfg, d_cost
    return best, best_cost, history, time.time()-t0

def verify_V4():
    """tiny + 真实实例上与 CPLEX 精确解比对。"""
    from cplex_d0 import build_and_solve as cplex_d0_solve
    results = []
    # --- tiny（结构与cplex_d0.tiny一致，φ=1） ---
    tiny = {
        'meta': {'periods': 1},
        'demand': {'i1': {'d': {'1': 100.0}}, 'i2': {'d': {'1': 50.0}}},
        'sorting': {'j1': {'open_fixed': 10.0, 'op_cost': 1.0, 'cap_t': 1e9, 'buffer_cap_t': 1e-6,
                           'expand_step_t': 0.0, 'expand_cost': 0.0}},
        'recycling': {'k1': {'open_fixed': 20.0, 'op_cost': 2.0, 'cap_t': 1e9, 'expand_step_t': 1e9, 'expand_cost': 5.0, 'min_run_t': 0.0}},
        'landfill': {'l1': {'open_fixed': 30.0, 'op_cost': 3.0, 'cap_t': 1e9, 'retire_candidate': False}},
        'dist': {'a': {'i1': {'j1': 10.0}, 'i2': {'j1': 20.0}},
                 'a2': {'i1': {'l1': 15.0}, 'i2': {'l1': 25.0}},
                 'b': {'j1': {'k1': 5.0}}, 'c': {'j1': {'l1': 8.0}}, 'e': {'k1': {'l1': 4.0}}},
        'params': {'transport_cost': 1.0, 'hold_cost': 0.0, 'unmet_penalty': 1000.0,
                   'recyclable_ratio': 0.8, 'recycle_revenue': 20.0, 'product_yield': 1.0},
    }
    scen = [{'name': 'mean', 'prob': 1.0, 'd': {i: {t: tiny['demand'][i]['d'][str(t)] for t in range(1, 2)} for i in tiny['demand']}}]
    # CLEX 精确
    mdl, sol, _ = cplex_d0_solve(tiny, 'v4_tiny', fixed_unit=1.0)
    obj_cplex = sol.objective_value
    best, obj_alns, hist, el = alns(tiny, scen, max_iter=120, verbose=False, FU=1.0)
    gap = abs(obj_alns-obj_cplex)/max(abs(obj_cplex), 1e-9)
    ok = gap < 1e-6
    print(f'[V4-tiny] CPLEX={obj_cplex:.2f}, ALNS={obj_alns:.2f}, gap={gap:.2e} -> {"PASS" if ok else "FAIL"}')
    results.append(ok)
    # --- 真实实例（均值情景） ---
    inst = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    T = inst['meta']['periods']
    scen_r = [{'name': 'mean', 'prob': 1.0, 'd': {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in inst['demand']}}]
    mdl, sol, _ = cplex_d0_solve(inst, 'v4_real', retire_lstar=True, t_star=2)
    obj_cplex_r = sol.objective_value
    best_r, obj_alns_r, hist_r, el_r = alns(inst, scen_r, max_iter=1500, verbose=False)
    gap_r = (obj_alns_r-obj_cplex_r)/obj_cplex_r
    ok_r = gap_r <= 0.01
    print(f'[V4-real] CPLEX={obj_cplex_r:,.0f}, ALNS={obj_alns_r:,.0f}, gap={gap_r*100:.3f}%, 用时{el_r:.0f}s -> {"PASS" if ok_r else "FAIL"}')
    results.append(ok_r)
    return all(results)

if __name__ == '__main__':
    ok = verify_V4()
    print(f'===== V4={"PASS" if ok else "FAIL"} =====')
    sys.exit(0 if ok else 1)
