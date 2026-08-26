#!/usr/bin/env python3
"""ALNS（自适应大邻域搜索）启发式 + V4 与 CPLEX 精确解收敛性验证。

架构（对应论文 C4 贡献：规则暖启动 + 学习自适应权重，零训练）：
  外层: 对一阶段决策 (xJ,xK,xL,zK,rL) 搜索 —— ALNS 算子直接作用于设施开关/扩容/退出
  内层: 给定一阶段配置，二阶段为 LP —— 用 CPLEX 快速求解各情景分配子问题（精确回收子问题）
       小实例用 scipy linprog 求解 LP 以提速（V4 用）。

算子（毁坏+修复）:
  D1 随机关闭一座开启设施 / D2 随机开启一座关闭设施 / D3 交换一对开关
  D4 扩容档±1（保持单调）/ D5 提前/推迟一座消纳场退出
自适应权重: 算子收益 1/gap 更新（softmax选择），每 iter_decay 衰减。
V4: 在 tiny 与真实实例上与 CPLEX 精确解比对（小实例0 gap；真实实例 ≤1% gap）。
"""
import json, os, sys, random, math, time
import numpy as np
from scipy.optimize import linprog

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- 内层 LP：给定一阶段配置求各情景最优分配（最小化二阶段成本） ----------
def solve_second_stage_lp(inst, xcfg, d_scenario):
    """xcfg: {xJ:{j:[t]}, xK:{k:[t]}, xL:{l:[t]}, zK:{k:[t]}, rL:{l:[t]}}
    返回 (二阶段成本, None或失败)。LP变量: y[i,j,t], yd[i,l,t], w[j,k,t], q[j,l,t], v[k,l,t], s[j,t], u[i,t]"""
    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting']); RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)

    # 变量索引
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

    # 目标
    c = np.zeros(n)
    for (key, *a), var in idx.items():
        if key == 'y': c[var] = P['transport_cost']*inst['dist']['a'][a[0]][a[1]] + inst['sorting'][a[1]]['op_cost']
        elif key == 'yd': c[var] = P['transport_cost']*inst['dist']['a2'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 'w': c[var] = P['transport_cost']*inst['dist']['b'][a[0]][a[1]] + inst['recycling'][a[1]]['op_cost'] - P.get('recycle_revenue', 0.0)*phi
        elif key == 'q': c[var] = P['transport_cost']*inst['dist']['c'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 'v': c[var] = P['transport_cost']*inst['dist']['e'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']
        elif key == 's': c[var] = P['hold_cost']
        elif key == 'u': c[var] = P['unmet_penalty']

    # 约束构造 A_ub·z ≤ b_ub, A_eq·z = b_eq
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
            # (1) Σw ≤ ρ·out = ρ(Σw+Σq)  →  (1-ρ)Σw - ρΣq ≤ 0
            ubrow([(x, (1-rho)*cval) for x, cval in outs] + [(x, -rho*cval) for x, cval in qs], 0.0)
            # (2) Σq ≥ (1-ρ)·out  →  (1-ρ)Σw - ρΣq ≤ 0 同一式？不——(2)等价 (1-ρ)Σw+(1-ρ)Σq-Σq ≤ 0 → (1-ρ)Σw - ρΣq ≤ 0
            # 注：(1)与(2)数学等价（因 out=Σw+Σq 恰好由C2平衡式定义），保留一条即可；为忠实规范两条都列时二者相同。
            ubrow([(x, (1-rho)*cval) for x, cval in outs] + [(x, -rho*cval) for x, cval in qs], 0.0)
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
            cap = inst['recycling'][k]['cap_t']*xcfg['xK'][k][t-1] + inst['recycling'][k]['expand_step_t']*xcfg['zK'][k][t-1]
            ubrow([(('w', j, k, t), 1.0) for j in SORT], cap)
            if xcfg['xK'][k][t-1] == 1:
                lbrow_coeffs = [(('w', j, k, t), -1.0) for j in SORT]
                r = np.zeros(n)
                for k_, cval in lbrow_coeffs: r[idx[k_]] += cval
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

    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    if res.status != 0:
        return None
    return res.fun

def diagnose_lp(inst, xcfg, d_scenario):
    """LP不可行诊断：逐约束类别测试松弛可行性"""
    import copy
    T = inst['meta']['periods']
    test = copy.deepcopy(inst)
    # 测试1: 去掉ρ比例约束
    test['params']['recyclable_ratio'] = 1.0
    c2 = solve_second_stage_lp(test, xcfg, d_scenario)
    print(f'  [diag] ρ=1(去比例约束): {"可行" if c2 is not None else "不可行"}')
    test2 = copy.deepcopy(inst)
    for j in test2['sorting']: test2['sorting'][j]['buffer_cap_t'] = 1e9
    c2 = solve_second_stage_lp(test2, xcfg, d_scenario)
    print(f'  [diag] 放开缓冲: {"可行" if c2 is not None else "不可行"}')

def solve_second_stage_lp_detail(inst, xcfg, d_scenario):
    """同 solve_second_stage_lp，额外返回 (cost, unmet_total, recycled_total)。"""
    import numpy as _np
    from scipy.optimize import linprog as _lp
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
    c = _np.zeros(n)
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
        r = _np.zeros(n)
        for k_, cval in coeffs: r[idx[k_]] += cval
        A_eq.append(r); b_eq.append(rhs)
    def ubrow(coeffs, rhs):
        r = _np.zeros(n)
        for k_, cval in coeffs: r[idx[k_]] += cval
        A_ub.append(r); b_ub.append(rhs)
    for i in GEN:
        for t in range(1, T+1):
            eqrow([(('y', i, j, t), 1.0) for j in SORT] + [(('yd', i, l, t), 1.0) for l in LAND] + [(('u', i, t), 1.0)], d_scenario[i][t])
    for j in SORT:
        for t in range(1, T+1):
            prev = [(('s', j, t-1), 1.0)] if t > 1 else []
            eqrow(prev + [(('y', i, j, t), 1.0) for i in GEN]
                  + [(('w', j, k, t), -1.0) for k in RECY] + [(('q', j, l, t), -1.0) for l in LAND] + [(('s', j, t), -1.0)], 0.0)
            outs = [(('w', j, k, t), 1.0) for k in RECY]
            qs = [(('q', j, l, t), 1.0) for l in LAND]
            ubrow([(x, (1-rho)*cval) for x, cval in outs] + [(x, -rho*cval) for x, cval in qs], 0.0)
            ubrow([(x, (1-rho)*cval) for x, cval in outs] + [(x, -rho*cval) for x, cval in qs], 0.0)
    for k in RECY:
        for t in range(1, T+1):
            eqrow([(('w', j, k, t), (1-phi)) for j in SORT] + [(('v', k, l, t), -1.0) for l in LAND], 0.0)
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
            cap = inst['recycling'][k]['cap_t']*xcfg['xK'][k][t-1] + inst['recycling'][k]['expand_step_t']*xcfg['zK'][k][t-1]
            ubrow([(('w', j, k, t), 1.0) for j in SORT], cap)
            if xcfg['xK'][k][t-1] == 1:
                r = _np.zeros(n)
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
    res = _lp(c, A_ub=_np.array(A_ub) if A_ub else None, b_ub=_np.array(b_ub) if b_ub else None,
              A_eq=_np.array(A_eq), b_eq=_np.array(b_eq), bounds=(0, None), method='highs')
    if res.status != 0:
        return None, None, None
    x = res.x
    unmet = sum(x[idx[('u', i, t)]] for i in GEN for t in range(1, T+1))
    recycled = sum(x[idx[('w', j, k, t)]] for j in SORT for k in RECY for t in range(1, T+1))
    return res.fun, unmet, recycled

def fixed_cost_of(inst, xcfg, FU=1e4):
    T = inst['meta']['periods']
    fc = sum((inst['sorting'][j]['open_fixed']*FU*xcfg['xJ'][j][t-1] + inst['sorting'][j].get('expand_cost', 600.0)*FU*xcfg.get('zJ', {j: [0]*T for j in inst['sorting']})[j][t-1]) for j in inst['sorting'] for t in range(1, T+1)) \
       + sum((inst['recycling'][k]['open_fixed']*FU*xcfg['xK'][k][t-1] + inst['recycling'][k]['expand_cost']*FU*xcfg['zK'][k][t-1]) for k in inst['recycling'] for t in range(1, T+1)) \
       + sum(inst['landfill'][l]['open_fixed']*FU*xcfg['xL'][l][t-1] for l in inst['landfill'] for t in range(1, T+1))
    return fc

def evaluate(inst, xcfg, scenarios, FU=1e4):
    """总成本 = fixed + Σ p_s · LP_s。不可行情景返回 None（惩罚由LP内部u实现，不可行指LP本身失败）。"""
    total = fixed_cost_of(inst, xcfg, FU)
    for sc in scenarios:
        c2 = solve_second_stage_lp(inst, xcfg, sc['d'])
        if c2 is None:
            return None
        total += sc['prob'] * c2
    return total

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

def repair_monotone(cfg, inst):
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

def alns(inst, scenarios, max_iter=200, seed=20260825, verbose=True, FU=1e4):
    rng = random.Random(seed)
    cur = repair_monotone(initial_solution(inst), inst)
    cur_cost = evaluate(inst, cur, scenarios, FU)
    best, best_cost = clone(cur), cur_cost
    ops = ['D1_close', 'D2_open', 'D3_swap', 'D4_expand', 'D5_retire', 'D6_greedy_close']
    weights = {o: 1.0 for o in ops}
    Tn = inst['meta']['periods']
    t0 = time.time()
    history = []
    for it in range(max_iter):
        # softmax 选择算子
        ws = [weights[o] for o in ops]
        mx = max(ws)
        ps = [math.exp(w-mx) for w in ws]
        ps = [p/sum(ps) for p in ps]
        op = rng.choices(ops, weights=ps)[0]
        cand = clone(cur)
        if op == 'D1_close':
            pool = [j for j in inst['sorting'] if any(cand['xJ'][j])]
            pool += [k for k in inst['recycling'] if any(cand['xK'][k])]
            if pool:
                e = rng.choice(pool)
                if e in cand['xJ']:
                    t = rng.randint(1, Tn); cand['xJ'][e] = [0 if tt >= t else 1 for tt in range(1, Tn+1)]
                else:
                    t = rng.randint(1, Tn); cand['xK'][e] = [0 if tt >= t else 1 for tt in range(1, Tn+1)]
        elif op == 'D2_open':
            pool = [j for j in inst['sorting'] if not any(cand['xJ'][j])] + [k for k in inst['recycling'] if not any(cand['xK'][k])]
            if pool:
                e = rng.choice(pool)
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
        elif op == 'D6_greedy_close' and it % 20 == 0:
            base_cost = evaluate(inst, cur, scenarios, FU)
            best_delta, best_e, best_mode = 0, None, None
            for e in inst['recycling']:
                if any(cur['xK'][e]):
                    c2 = clone(cur); c2['xK'][e] = [0]*Tn; c2 = repair_monotone(c2, inst)
                    cc = evaluate(inst, c2, scenarios, FU)
                    if cc is not None and base_cost - cc > best_delta:
                        best_delta, best_e, best_mode = base_cost - cc, e, 'off'
            for e in inst['recycling']:
                if not any(cur['xK'][e]):
                    c2 = clone(cur); c2['xK'][e] = [1]*Tn; c2 = repair_monotone(c2, inst)
                    cc = evaluate(inst, c2, scenarios, FU)
                    if cc is not None and base_cost - cc > best_delta:
                        best_delta, best_e, best_mode = base_cost - cc, e, 'on'
            if best_e is not None:
                cand['xK'][best_e] = [1 if best_mode == 'on' else 0]*Tn
        elif op == 'D5_retire':
            elig = [l for l in inst['landfill'] if not inst['landfill'][l]['retire_candidate'] and not any(cand['rL'][l])]
            if elig:
                l = rng.choice(elig); t = rng.randint(1, Tn)
                for tt in range(t, Tn+1): cand['rL'][l][tt-1] = 1
        cand = repair_monotone(cand, inst)
        c = evaluate(inst, cand, scenarios, FU)
        if c is None:
            weights[op] *= 0.9; continue
        temp = max(cur_cost*0.01*math.exp(-it/max_iter*5), 1e-6*max(abs(cur_cost),1))
        if c < cur_cost - 1e-6 or (c-cur_cost) < temp*rng.random():
            cur, cur_cost = clone(cand), c
            weights[op] *= 1.15
            if c < best_cost - 1e-6:
                best, best_cost = clone(cand), c
                weights[op] *= 1.1
                if verbose: print(f'  it{it} [{op}] 新最优 {best_cost:,.0f}')
        else:
            weights[op] *= 0.98
        history.append(best_cost)
    return best, best_cost, history, time.time()-t0

def verify_V4():
    """tiny + 真实实例上与 CPLEX 精确解比对。"""
    from cplex_d0 import build_and_solve as cplex_d0_solve
    from cplex_d0 import tiny_instance as cplex_tiny  # 注意：tiny 是 cplex 版结构，这里构造等价 ALNS tiny
    results = []
    # --- tiny（结构与cplex_d0.tiny一致，φ=1） ---
    tiny = {
        'meta': {'periods': 1},
        'demand': {'i1': {'d': {'1': 100.0}}, 'i2': {'d': {'1': 50.0}}},
        'sorting': {'j1': {'open_fixed': 10.0, 'op_cost': 1.0, 'cap_t': 1e9, 'buffer_cap_t': 1e-6}},
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
