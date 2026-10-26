# -*- coding: utf-8 -*-
"""GRI：贪心回收评估的岛模型纯元启发（V2.4）。

设计要点：
  1. 搜索循环内零 LP：候选评估 = fixed_cost + Σ p_s·贪心二阶段成本(认证上界)
     + COVER_M·Σ p_s·未满足量（软惩罚，保持搜索流动性）
  2. 岛模型并行：N 个独立进程岛，各岛独立自适应算子权重 + 评估扰动（多样性），
     每 MIG_EVERY 代迁移全局最优 incumbent
  3. 算子 D1-D7（继承自诊断版 ALNS 的领域结构，全部重写为贪心评估）：
     D1 关随机设施 / D2 开随机设施 / D3 换开关对 / D4 扩容±1 /
     D5 退出时点±1 / D6 贪关最优单设施 / D7 覆盖修复（加模块/开设施）
  4. 配置有效性：z 单调、r 单调不可逆、xL+rL≤1、退出候选强制
  5. 候选池：覆盖可行（训练集零未满足）配置按贪心值收集 top-K；
     精确评估在算法外（验证/报告层）进行——本模块不含任何 LP/求解器调用

返回：({'pool': [cfg,...], 'greedy_val': [...], 'cover': [bool,...], 'iters': n, ...})
"""
import json, os, sys, time, random, math
from multiprocessing import Process, Manager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_recourse import greedy_recourse, greedy_recourse2

COVER_M = 1000.0
BIG_INF = 1e12          # 未满足软惩罚（CNY/t），与诊断版一致
MIG_EVERY = 200           # 迁移间隔（迭代）
POOL_MAX = 24             # 岛内池上限


# ---------- 配置工具（自包含，不依赖 alns.py） ----------

def clone(cfg):
    return {k: {f: list(v) for f, v in d.items()} for k, d in cfg.items()}


def initial_full(inst):
    T = inst['meta']['periods']
    return {'xJ': {j: [1] * T for j in inst['sorting']},
            'xK': {k: [1] * T for k in inst['recycling']},
            'xL': {l: [1] * T for l in inst['landfill']},
            'zJ': {j: [0] * T for j in inst['sorting']},
            'zK': {k: [0] * T for k in inst['recycling']},
            'rL': {l: [0] * T for l in inst['landfill']}}


def initial_current(inst):
    cfg = initial_full(inst)
    T = inst['meta']['periods']
    for l, info in inst['landfill'].items():
        if info.get('retire_candidate'):
            for t in range(2, T + 1):
                cfg['rL'][l][t - 1] = 1
                cfg['xL'][l][t - 1] = 0
    return cfg


def initial_random(inst, rng, k_frac=0.6):
    T = inst['meta']['periods']
    cfg = initial_full(inst)
    for key, facs in [('xJ', inst['sorting']), ('xK', inst['recycling'])]:
        ks = list(facs)
        rng.shuffle(ks)
        keep = set(ks[:max(1, int(len(ks) * k_frac))])
        for f in facs:
            if f not in keep:
                cfg[key][f] = [0] * T
    return repair_monotone(cfg, inst)


def repair_monotone(cfg, inst):
    T = inst['meta']['periods']
    for key in ('zJ', 'zK'):
        for f in inst['sorting'] if key == 'zJ' else inst['recycling']:
            for t in range(T):
                cfg[key][f][t] = max(0, min(4, int(cfg[key][f][t])))
            for t in range(1, T):
                cfg[key][f][t] = max(cfg[key][f][t - 1:t + 1])
    for l in inst['landfill']:
        for t in range(1, T):
            cfg['rL'][l][t] = max(cfg['rL'][l][t - 1:t + 1])
        for t in range(1, T + 1):
            if cfg['rL'][l][t - 1] == 1:
                cfg['xL'][l][t - 1] = 0
    for l, info in inst['landfill'].items():
        if info.get('retire_candidate'):
            for t in range(2, T + 1):
                cfg['rL'][l][t - 1] = 1
                cfg['xL'][l][t - 1] = 0
    return cfg


def fixed_cost_of(inst, cfg, FU=1e4):
    T = inst['meta']['periods']
    fc = 0.0
    for j in inst['sorting']:
        for t in range(1, T + 1):
            fc += (inst['sorting'][j]['open_fixed'] * FU * cfg['xJ'][j][t - 1]
                   + inst['sorting'][j].get('expand_cost', 600.0) * FU * cfg['zJ'][j][t - 1])
    for k in inst['recycling']:
        for t in range(1, T + 1):
            fc += (inst['recycling'][k]['open_fixed'] * FU * cfg['xK'][k][t - 1]
                   + inst['recycling'][k]['expand_cost'] * FU * cfg['zK'][k][t - 1])
    for l in inst['landfill']:
        for t in range(1, T + 1):
            fc += inst['landfill'][l]['open_fixed'] * FU * cfg['xL'][l][t - 1]
    return fc


def gri_value(inst, cfg, scenarios, rng=None, jitter=0.0):
    """贪心评估：返回 (value, unmet_total, recycled)。结构性失败给 BIG 惩罚。"""
    fc = fixed_cost_of(inst, cfg)
    tot, unm_t, rec_t = 0.0, 0.0, 0.0
    for sc in scenarios:
        c2, unm, recy = greedy_recourse2(inst, cfg, sc['d'], require_cover=False,
                                         rng=rng, jitter=jitter)
        if c2 is None:
            return fc + 1e12, 1e9, 0.0      # 结构性失败：大惩罚但有限（可逃逸）
        tot += sc['prob'] * c2
        unm_t += sc['prob'] * unm
        rec_t += sc['prob'] * recy
    return fc + tot + COVER_M * unm_t, unm_t, rec_t


def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True)


class SubsetEval:
    """锚点子集精确评估器（可 pickle，供子进程使用）。
    在 k 个情景上求精确 LP，成本随情景数线性缩放；用于搜索期排序。
    """
    def __init__(self, inst, subset, require_cover=False):
        self.inst = inst
        self.subset = subset
        self.require_cover = require_cover

    def __call__(self, cfg):
        from alns import fixed_cost_of, solve_second_stage_lp_ex
        fc = fixed_cost_of(self.inst, cfg)
        tot = 0.0
        for sc in self.subset:
            c2, _, _ = solve_second_stage_lp_ex(self.inst, cfg, sc['d'],
                                               require_cover=self.require_cover)
            if c2 is None:
                return None
            tot += sc['prob'] * c2
        return fc + tot


def make_subset_eval(inst, train_scen, k=3, require_cover=False):
    """构造锚点子集精确评估器（锚点优先，不足则取前 k 个）。"""
    anchors = [sc for sc in train_scen if str(sc.get('name', '')).startswith('anchor')]
    subset = anchors if len(anchors) >= k else train_scen[:k]
    return SubsetEval(inst, subset, require_cover=require_cover)


# ---------- 单岛搜索 ----------

def _island(inst, scenarios, seed, wall_budget, init_kind, jitter,
            shared, island_id, max_iter=2_000_000, eval_fn=None):
    rng = random.Random(seed)
    T = inst['meta']['periods']
    deadline = time.time() + wall_budget

    if init_kind == 'full':
        cur = initial_current(inst)
    elif init_kind == 'rand':
        cur = initial_random(inst, rng)
    elif init_kind == 'warm' and 'last' in _WARM_REGISTRY:
        cur = json.loads(_WARM_REGISTRY['last'])
    else:
        cur = initial_full(inst)
        for j in inst['sorting']:
            cur['zJ'][j] = [4] * T
        for k in inst['recycling']:
            cur['zK'][k] = [4] * T
        cur = repair_monotone(cur, inst)
    def _ev(c):
        if eval_fn is not None:
            v = eval_fn(c)
            if v is None:
                return (BIG_INF, 1e9, 0.0)   # 有限大值：保持退火可计算、可逃逸
            return (v, 0.0, 0.0)
        return gri_value(inst, c, scenarios, rng, jitter)

    cur_v = _ev(cur)
    best, best_v = clone(cur), cur_v[0]
    pool, pool_keys = {}, set()
    pool[cfg_key(cur)] = (clone(cur), cur_v[0])

    ops = ['D1_close', 'D2_open', 'D3_swap', 'D4_expand', 'D5_retire',
           'D6_greedy', 'D7_cover', 'D8_time', 'D9_realloc']
    w = {o: 1.0 for o in ops}
    TEMP0, it_done, mig_at = 0.01, 0, MIG_EVERY

    def try_pool(cfg, v):
        # 池只按代理值收集 top-K；可行性由外层全集精确 LP 判定
        # （贪心的覆盖判定偏保守，若在此过滤会把可行解一并挡掉）
        k = cfg_key(cfg)
        if k in pool_keys:
            return
        pool[k] = (clone(cfg), v[0])
        pool_keys.add(k)
        if len(pool) > POOL_MAX:
            worst = max(pool.items(), key=lambda kv: kv[1][1])[0]
            del pool[worst]

    while it_done < max_iter and time.time() < deadline:
        it_done += 1
        # 迁移
        if it_done >= mig_at:
            mig_at += MIG_EVERY
            gb = shared.get('best')
            if gb is not None and gb[1] < best_v - 1e-6:
                cand = json.loads(gb[0])
                cv = _ev(cand)
                if cv[0] < best_v:
                    best, best_v = clone(cand), cv[0]
        # softmax 选算子
        ws = [w[o] for o in ops]
        mx = max(ws)
        exps = [math.exp((x - mx) / max(0.1, mx * 0.2)) for x in ws]
        op = rng.choices(ops, weights=exps)[0]
        cand = clone(cur)
        if op == 'D1_close':
            pool_open = [f for f in inst['sorting'] if any(cand['xJ'][f])] + \
                        [f for f in inst['recycling'] if any(cand['xK'][f])]
            if pool_open:
                f = rng.choice(pool_open)
                key = 'xJ' if f in cand['xJ'] else 'xK'
                cand[key][f] = [0] * T
        elif op == 'D2_open':
            closed = [f for f in inst['sorting'] if not any(cand['xJ'][f])] + \
                     [f for f in inst['recycling'] if not any(cand['xK'][f])]
            if closed:
                f = rng.choice(closed)
                key = 'xJ' if f in cand['xJ'] else 'xK'
                cand[key][f] = [1] * T
        elif op == 'D3_swap':
            on = [f for f in inst['sorting'] if any(cand['xJ'][f])]
            off = [f for f in inst['sorting'] if not any(cand['xJ'][f])]
            if on and off:
                cand['xJ'][rng.choice(on)] = [0] * T
                cand['xJ'][rng.choice(off)] = [1] * T
            else:
                on = [f for f in inst['recycling'] if any(cand['xK'][f])]
                off = [f for f in inst['recycling'] if not any(cand['xK'][f])]
                if on and off:
                    cand['xK'][rng.choice(on)] = [0] * T
                    cand['xK'][rng.choice(off)] = [1] * T
        elif op == 'D4_expand':
            if rng.random() < 0.5:
                f = rng.choice(list(inst['sorting']))
                t = rng.randint(1, T)
                d = rng.choice([-1, 1])
                mode = rng.choice(['tail', 'head', 'point'])
                for tt in range(1, T + 1):
                    if (mode == 'tail' and tt >= t) or (mode == 'head' and tt <= t) or (mode == 'point' and tt == t):
                        cand['zJ'][f][tt - 1] = max(0, min(4, cand['zJ'][f][tt - 1] + d))
            else:
                f = rng.choice(list(inst['recycling']))
                t = rng.randint(1, T)
                d = rng.choice([-1, 1])
                mode = rng.choice(['tail', 'head', 'point'])
                for tt in range(1, T + 1):
                    if (mode == 'tail' and tt >= t) or (mode == 'head' and tt <= t) or (mode == 'point' and tt == t):
                        cand['zK'][f][tt - 1] = max(0, min(4, cand['zK'][f][tt - 1] + d))
        elif op == 'D5_retire':
            active = [l for l in inst['landfill']
                      if not inst['landfill'][l].get('retire_candidate')
                      and any(cand['rL'][l])]
            if active and rng.random() < 0.5:
                l = rng.choice(active)
                if rng.random() < 0.5:       # 全部撤销退役并重新开放
                    cand['rL'][l] = [0] * T
                    cand['xL'][l] = [1] * T
                else:                         # 退役推迟一期
                    t0 = next(t for t in range(1, T + 1) if cand['rL'][l][t - 1])
                    cand['rL'][l][t0 - 1] = 0
                    cand['xL'][l][t0 - 1] = 1
            else:
                elig = [l for l in inst['landfill']
                        if not inst['landfill'][l].get('retire_candidate')
                        and not any(cand['rL'][l])]
                if elig:
                    l = rng.choice(elig)
                    t = rng.randint(1, T)
                    for tt in range(t, T + 1):
                        cand['rL'][l][tt - 1] = 1
        elif op == 'D6_greedy' and it_done % 10 == 0:
            base_v = cur_v[0]
            best_delta, best_f, best_key = 0, None, None
            for f in inst['recycling']:
                if any(cur['xK'][f]):
                    c2 = clone(cur); c2['xK'][f] = [0] * T
                    v2 = _ev(c2)
                    if v2[0] < base_v - 1e-6 and base_v - v2[0] > best_delta:
                        best_delta, best_f, best_key = base_v - v2[0], f, 'xK'
            if best_f:
                cand[best_key][best_f] = [0] * T
        elif op == 'D9_realloc':
            # 同期跨设施容量转移：减一处、增一处，保持覆盖同时消除过建
            zkey, facs = (('zJ', list(inst['sorting'])) if rng.random() < 0.5
                          else ('zK', list(inst['recycling'])))
            donors = [f for f in facs if max(cand[zkey][f]) > 0]
            receivers = [f for f in facs if min(cand[zkey][f]) < 4]
            if donors and receivers:
                f = rng.choice(donors); g = rng.choice(receivers)
                if f != g:
                    t = rng.randint(1, T)
                    for tt in range(t, T + 1):
                        if cand[zkey][f][tt - 1] > 0 and cand[zkey][g][tt - 1] < 4:
                            cand[zkey][f][tt - 1] -= 1
                            cand[zkey][g][tt - 1] += 1
        elif op == 'D8_time':
            # 时间维开闭：x_{f,t} 单期翻转，覆盖 CPLEX 可表达的迟到/间歇开设施配置
            fam = rng.choice(['xJ', 'xK', 'xL'])
            if fam == 'xJ':
                f = rng.choice(list(inst['sorting']))
            elif fam == 'xK':
                f = rng.choice(list(inst['recycling']))
            else:
                f = rng.choice(list(inst['landfill']))
            t = rng.randint(1, T)
            cand[fam][f][t - 1] = 1 - cand[fam][f][t - 1]
        elif op == 'D7_cover':
            if cur_v[1] > 1e-6:      # 有未满足：修复（加模块/开厂）
                cands = []
                for j in inst['sorting']:
                    if any(cand['xJ'][j]):
                        for t in range(1, T + 1):
                            if cand['zJ'][j][t - 1] < 4:
                                cands.append(('zJ', j, t)); break
                for k in inst['recycling']:
                    if any(cand['xK'][k]):
                        for t in range(1, T + 1):
                            if cand['zK'][k][t - 1] < 4:
                                cands.append(('zK', k, t)); break
                closed_k = [k for k in inst['recycling'] if not any(cand['xK'][k])]
                if cands:
                    key, f, t0 = rng.choice(cands)
                    for tt in range(t0, T + 1):
                        cand[key][f][tt - 1] = min(4, cand[key][f][tt - 1] + 1)
                elif closed_k:
                    cand['xK'][rng.choice(closed_k)] = [1] * T
        cand = repair_monotone(cand, inst)
        v = _ev(cand)
        # 接受（模拟退火 lite）
        Tt = max(TEMP0 * cur_v[0] * 0.05, 1e-9)   # 恒温退火（预算主导）
        acc = v[0] < cur_v[0] or rng.random() < math.exp(-(v[0] - cur_v[0]) / Tt)
        if acc:
            cur, cur_v = cand, v
            if v[0] < best_v - 1e-6:
                best, best_v = clone(cand), v[0]
                w[op] = min(w[op] * 1.05, 50.0)
                try_pool(cand, v)
                with shared['lock']:
                    if shared['best'] is None or shared['best'][1] > best_v:
                        shared['best'] = (cfg_key(best), best_v)
            else:
                w[op] = min(w[op] * 1.02, 50.0)
        else:
            w[op] = max(w[op] * 0.99, 0.02)
    return best, best_v, pool, it_done


def _island_wrap(inst, scenarios, seed, wall_budget, init_kind, jitter,
                 shared, island_id, out_q, eval_fn=None):
    try:
        best, best_v, pool, it_done = _island(inst, scenarios, seed, wall_budget,
                                              init_kind, jitter, shared, island_id,
                                              eval_fn=eval_fn)
        out_q.put((island_id, best, best_v,
                   [(cfg, v) for cfg, v in pool.values()], it_done))
    except Exception as e:
        out_q.put((island_id, None, None, [], 0))


# ---------- 主入口 ----------

def gri_run(inst, train_scen, n_islands=6, wall_budget=120, seed=0,
            methods_jitter=(0.0, 0.02, 0.05), eval_fn=None):
    """一次 GRI 运行（= 一组岛）。返回 pool 列表 [(cfg, greedy_v)] 按贪心值升序。"""
    from multiprocessing import Queue
    mgr = Manager()
    shared = mgr.dict()
    shared['best'] = None
    shared['lock'] = mgr.Lock()
    out_q = Queue()
    # 覆盖约束下只有满配必定可行；从不可行初值出发会被困住（评估恒为 BIG_INF）
    inits = ['max'] * n_islands
    procs = []
    for i in range(n_islands):
        jit = methods_jitter[i % len(methods_jitter)]
        p = Process(target=_island_wrap,
                    args=(inst, train_scen, seed * 1000 + i, wall_budget,
                          inits[i % len(inits)], jit, shared, i, out_q, eval_fn))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    all_pool = {}
    while not out_q.empty():
        _, best, best_v, pool, it_done = out_q.get()
        for cfg, v in pool + ([(best, best_v)] if best is not None else []):
            k = cfg_key(cfg)
            if k not in all_pool or v < all_pool[k]:
                all_pool[k] = v
    ranked = sorted(all_pool.items(), key=lambda kv: kv[1])
    return [(json.loads(k), v) for k, v in ranked]


if __name__ == '__main__':
    # 冒烟：real 实例 SAA(10) 一次运行
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=60)
    ap.add_argument('--islands', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from cplex_saa import gen_scenarios
    from run_experiments import reduce_scenarios
    inst = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'instance_v1.json'), encoding='utf-8'))
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)
    t0 = time.time()
    pool = gri_run(inst, train, n_islands=a.islands, wall_budget=a.budget, seed=a.seed)
    dt = time.time() - t0
    print(f'GRI 冒烟: {len(pool)} 池成员, {dt:.0f}s')
    for cfg, v in pool[:5]:
        _, unm, _ = gri_value(inst, cfg, train)
        print(f'  greedy_v={v/1e6:.1f}M  unmet={unm/1e4:.2f}万t')


# ---------- 终端精确抛光（matheuristic 第二阶段，预算有界） ----------

def exact_polish(inst, cfg0, scenarios, exact_eval, budget_s=120, top_moves=2000,
                 screen_k=25, greedy_eval=None):
    """有界精确抛光（快速版）：邻域全量贪心排序 → 前 screen_k 精确确认。
    贪心 ~1-4ms/候选，精确 ~35-3400ms/候选，故只对极少数候选调用精确评估。
    greedy_eval(cfg) -> float|None；默认用 gri_value。
    """
    import time as _t
    t0 = _t.time()
    cache = {}
    n_exact = 0
    if greedy_eval is None:
        greedy_eval = lambda c: gri_value(inst, c, scenarios)[0]

    def ex(c):
        nonlocal n_exact
        k = cfg_key(c)
        if k not in cache:
            cache[k] = exact_eval(c)
            n_exact += 1
        return cache[k]

    cur_cfg = clone(cfg0)
    cur_val = ex(cur_cfg)
    best_cfg, best_val = clone(cur_cfg), cur_val
    if cur_val is None:
        return cur_cfg, None, n_exact
    T = inst['meta']['periods']

    for _round in range(12):
        if _t.time() - t0 > budget_s or n_exact >= top_moves:
            break
        cand = {}

        def add(c):
            c = repair_monotone(c, inst)
            k = cfg_key(c)
            if k not in cache and k not in cand:
                cand[k] = c

        # ---- 单步邻域 ----
        for fam, facs, zkey in (('xJ', inst['sorting'], 'zJ'),
                                ('xK', inst['recycling'], 'zK')):
            for f in facs:
                for t in range(1, T + 1):
                    c = clone(cur_cfg); c[fam][f][t - 1] = 1 - c[fam][f][t - 1]; add(c)
                    if cur_cfg[fam][f][t - 1] == 0 and cur_cfg[zkey][f][t - 1] < 4:
                        for dz in (1, 2):
                            c = clone(cur_cfg); c[fam][f][t - 1] = 1
                            for tt in range(t, T + 1):
                                c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + dz)
                            add(c)
                for t in range(1, T):
                    c = clone(cur_cfg); c[fam][f][t - 1], c[fam][f][t] = c[fam][f][t], c[fam][f][t - 1]; add(c)
        for l in inst['landfill']:
            for t in range(1, T + 1):
                c = clone(cur_cfg); c['xL'][l][t - 1] = 1 - c['xL'][l][t - 1]; add(c)
            if not inst['landfill'][l].get('retire_candidate'):
                if any(cur_cfg['rL'][l]):
                    t_ret = next(t for t in range(1, T + 1) if cur_cfg['rL'][l][t - 1])
                    c = clone(cur_cfg); c['rL'][l][t_ret - 1] = 0; c['xL'][l][t_ret - 1] = 1; add(c)
                    c = clone(cur_cfg); c['rL'][l] = [0] * T; c['xL'][l] = [1] * T; add(c)
                else:
                    for t in range(1, T + 1):
                        c = clone(cur_cfg)
                        for tt in range(t, T + 1):
                            c['rL'][l][tt - 1] = 1
                        add(c)
        for zkey, facs in (('zJ', inst['sorting']), ('zK', inst['recycling'])):
            for f in facs:
                for t in range(1, T + 1):
                    for mode in ('tail', 'head', 'point'):
                        if cur_cfg[zkey][f][t - 1] > 0:
                            c = clone(cur_cfg)
                            for tt in range(1, T + 1):
                                if (mode == 'tail' and tt >= t) or (mode == 'head' and tt <= t) or (mode == 'point' and tt == t):
                                    c[zkey][f][tt - 1] = max(0, c[zkey][f][tt - 1] - 1)
                            add(c)
                        if cur_cfg[zkey][f][t - 1] < 4:
                            c = clone(cur_cfg)
                            for tt in range(1, T + 1):
                                if (mode == 'tail' and tt >= t) or (mode == 'head' and tt <= t) or (mode == 'point' and tt == t):
                                    c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + 1)
                            add(c)
        if not cand:
            break
        # ---- 贪心预筛 → 精确确认 top-K ----
        scored = []
        for c in cand.values():
            gv = greedy_eval(c)
            if gv is not None:
                scored.append((gv, c))
            if _t.time() - t0 > budget_s:
                break
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        improved = False
        for gv, c in scored[:screen_k]:
            if _t.time() - t0 > budget_s or n_exact >= top_moves:
                break
            v = ex(c)
            if v is not None and v < best_val - 1e-6:
                best_cfg, best_val = c, v
                improved = True
        cur_cfg, cur_val = best_cfg, best_val
        if not improved:
            break
    return best_cfg, best_val, n_exact


def exact_pair_polish(inst, cfg0, scenarios, exact_eval, budget_s=120, max_rounds=4,
                      screen_k=20, n_sample=900, greedy_eval=None, seed=0):
    """二阶联合抛光（快速版）：随机采样联合移动 → 贪心排序 → 前 screen_k 精确确认。
    采样上限 n_sample 控制单轮成本，避免 |X|x|Z| 的组合爆炸。
    """
    import time as _t
    import random as _r
    t_start = _t.time()
    rng = _r.Random(seed)
    cache = {}
    n_exact = 0
    if greedy_eval is None:
        greedy_eval = lambda c: gri_value(inst, c, scenarios)[0]

    def ex(c):
        nonlocal n_exact
        k = cfg_key(c)
        if k not in cache:
            cache[k] = exact_eval(c)
            n_exact += 1
        return cache[k]

    cur = clone(cfg0)
    cur_v = ex(cur)
    if cur_v is None:
        return cur, None, n_exact
    T = inst['meta']['periods']

    def base_moves(cfg):
        moves = []
        for fam, facs in (('xJ', inst['sorting']), ('xK', inst['recycling']),
                          ('xL', inst['landfill'])):
            for f in facs:
                moves.append(('xall', fam, f))
                for t in range(1, T):
                    moves.append(('xshift', fam, f, t, +1))
                    moves.append(('xshift', fam, f, t, -1))
                for a in range(1, T + 1):
                    for b in range(a, T + 1):
                        moves.append(('xwindow', fam, f, a, b))
        for l in inst['landfill']:
            if inst['landfill'][l].get('retire_candidate'):
                continue
            if any(cfg['rL'][l]):
                moves.append(('unretire', l, 'one'))
                moves.append(('unretire', l, 'all'))
            else:
                for t in range(1, T + 1):
                    moves.append(('retire', l, t))
        return moves

    def z_moves(cfg):
        moves = []
        for zkey, facs in (('zJ', inst['sorting']), ('zK', inst['recycling'])):
            for f in facs:
                for t in range(1, T + 1):
                    if cfg[zkey][f][t - 1] > 0:
                        moves += [('zdec', zkey, f, t), ('zhead', zkey, f, t),
                                  ('zpoint', zkey, f, t)]
                    if cfg[zkey][f][t - 1] < 4:
                        moves += [('zinc', zkey, f, t), ('zinchead', zkey, f, t),
                                  ('zincpoint', zkey, f, t)]
        moves.append(('znone', None, None, None))
        return moves

    def apply(c, mv):
        kind = mv[0]
        if kind == 'xall':
            _, fam, f = mv
            c[fam][f] = [1 - max(c[fam][f])] * T
        elif kind == 'xshift':
            _, fam, f, t, d = mv
            if d == +1 and c[fam][f][t - 1] == 1 and c[fam][f][t] == 0:
                c[fam][f][t - 1] = 0; c[fam][f][t] = 1
            elif d == -1 and c[fam][f][t - 1] == 0 and c[fam][f][t] == 1:
                c[fam][f][t - 1] = 1; c[fam][f][t] = 0
        elif kind == 'xwindow':
            _, fam, f, a, b = mv
            for t in range(1, T + 1):
                c[fam][f][t - 1] = 1 if a <= t <= b else 0
            if fam == 'xL':
                c['rL'][f] = [0] * T
        elif kind == 'unretire':
            _, l, mode = mv
            if mode == 'all':
                c['rL'][l] = [0] * T; c['xL'][l] = [1] * T
            else:
                t0 = next((t for t in range(1, T + 1) if c['rL'][l][t - 1]), None)
                if t0 is not None:
                    c['rL'][l][t0 - 1] = 0; c['xL'][l][t0 - 1] = 1
        elif kind == 'retire':
            _, l, t = mv
            for tt in range(t, T + 1):
                c['rL'][l][tt - 1] = 1
        elif kind in ('zdec', 'zhead', 'zpoint'):
            _, zkey, f, t = mv
            if kind == 'zdec':
                rng_ = range(t, T + 1)
            elif kind == 'zhead':
                rng_ = range(1, t + 1)
            else:
                rng_ = [t]
            for tt in rng_:
                c[zkey][f][tt - 1] = max(0, c[zkey][f][tt - 1] - 1)
        elif kind in ('zinc', 'zinchead', 'zincpoint'):
            _, zkey, f, t = mv
            if kind == 'zinc':
                rng_ = range(t, T + 1)
            elif kind == 'zinchead':
                rng_ = range(1, t + 1)
            else:
                rng_ = [t]
            for tt in rng_:
                c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + 1)
        return repair_monotone(c, inst)

    for _ in range(max_rounds):
        if _t.time() - t_start > budget_s:
            break
        X = base_moves(cur)
        Z = z_moves(cur)
        pairs = []
        if len(X) * len(Z) <= n_sample:
            pairs = [(x, z) for x in X for z in Z]
        else:
            for _k in range(n_sample):
                pairs.append((rng.choice(X), rng.choice(Z)))
        scored = []
        for x, z in pairs:
            if _t.time() - t_start > budget_s:
                break
            c = apply(apply(clone(cur), x), z)
            if cfg_key(c) == cfg_key(cur):
                continue
            gv = greedy_eval(c)
            if gv is not None:
                scored.append((gv, c))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        improved = False
        for gv, c in scored[:screen_k]:
            if _t.time() - t_start > budget_s:
                break
            v = ex(c)
            if v is not None and v < cur_v - 1e-6:
                cur, cur_v = c, v
                improved = True
        if not improved:
            break
    return cur, cur_v, n_exact


def config_violations(inst, cfg):
    """一阶定义域/耦合/单调性/退役约束审计。返回违规列表（应为空）。"""
    T = inst['meta']['periods']
    out = []
    for zkey, xkey, facs in (('zJ', 'xJ', inst['sorting']),
                              ('zK', 'xK', inst['recycling'])):
        for f in facs:
            for t in range(T):
                z = cfg[zkey][f][t]
                if z < 0 or z > 4 or int(z) != z:
                    out.append((zkey, f, t + 1, 'domain', z))
                if z > 4 * cfg[xkey][f][t]:
                    out.append((zkey, f, t + 1, 'z>4x', z))
                if t < T - 1 and z > cfg[zkey][f][t + 1]:
                    out.append((zkey, f, t + 1, 'nonmonotone'))
    for l in inst['landfill']:
        for t in range(T):
            if cfg['xL'][l][t] + cfg['rL'][l][t] > 1:
                out.append(('xL+rL', l, t + 1))
            if t < T - 1 and cfg['rL'][l][t] > cfg['rL'][l][t + 1]:
                out.append(('rL monotone', l, t + 1))
        if inst['landfill'][l].get('retire_candidate'):
            for t in range(1, T):
                if cfg['rL'][l][t] != 1:
                    out.append(('forced retirement', l, t + 1))
    return out


def repair_feasibility(inst, cfg, exact_eval, max_bump=40):
    """把全集不可行的配置修到覆盖可行：优先抬分拣扩容，其次开分拣中心，
    再其次抬厂扩容。每次改动都用精确 LP 复检（调用次数受 max_bump 限制）。
    返回 (cfg, val) 或 (None, None)。
    """
    cur = clone(cfg)
    T = inst['meta']['periods']
    v = exact_eval(cur)
    if v is not None:
        return cur, v
    for _ in range(max_bump):
        moved = False
        # 1) 抬分拣扩容（从第 1 期起，保持单调）
        for j in inst['sorting']:
            if cur['xJ'][j][0] != 1:
                continue
            if cur['zJ'][j][0] < 4:
                cur['zJ'][j] = [min(4, z + 1) for z in cur['zJ'][j]]
                cur = repair_monotone(cur, inst)
                moved = True
                break
        if not moved:
            # 2) 开一个关闭的分拣中心
            closed = [j for j in inst['sorting'] if not any(cur['xJ'][j])]
            if closed:
                j = closed[0]
                cur['xJ'][j] = [1] * T
                moved = True
        if not moved:
            # 3) 抬厂扩容
            for k in inst['recycling']:
                if cur['xK'][k][0] == 1 and cur['zK'][k][0] < 4:
                    cur['zK'][k] = [min(4, z + 1) for z in cur['zK'][k]]
                    cur = repair_monotone(cur, inst)
                    moved = True
                    break
        if not moved:
            # 4) 开一个关闭的厂
            closed = [k for k in inst['recycling'] if not any(cur['xK'][k])]
            if closed:
                k = closed[0]
                cur['xK'][k] = [1] * T
                moved = True
        if not moved:
            break
        v = exact_eval(cur)
        if v is not None:
            return cur, v
    return None, None


def gri_solve(inst, train_scen, exact_eval, budget_search=180, budget_polish=90,
              n_islands=6, seed=0, subset_k=3, verbose=False):
    """GRI 统一入口：自适应代理搜索 → 池精确评估 + 可行性修复 → 有界抛光。
    exact_eval(cfg) -> float|None，须为全集精确评估（含覆盖约束）。
    """
    import time as _t
    t0 = _t.time()
    stats = {}
    # 代理选择：子集 LP 若单次成本可接受则用之（更忠实），否则退回贪心
    probe = initial_full(inst)
    for j in inst['sorting']:
        probe['zJ'][j] = [4] * inst['meta']['periods']
    for k in inst['recycling']:
        probe['zK'][k] = [4] * inst['meta']['periods']
    probe = repair_monotone(probe, inst)
    sub = make_subset_eval(inst, train_scen, subset_k, require_cover=True)
    t1 = _t.time()
    sub(probe)
    sub_cost = _t.time() - t1
    stats['subset_eval_ms'] = sub_cost * 1000
    if sub_cost * 400 < budget_search:      # 至少能跑 ~400 次
        eval_fn, stats['surrogate'] = sub, 'subset-lp'
    else:
        eval_fn, stats['surrogate'] = None, 'greedy'

    pool = gri_run_pool(inst, train_scen, n_islands=n_islands,
                        wall_budget=max(20.0, budget_search / 2.0), seed=seed,
                        eval_fn=eval_fn, rounds=2)
    stats['n_pool'] = len(pool)

    # 池评估 + 可行性修复
    cands = []
    for cfg, gv in pool[:24]:
        v = exact_eval(cfg)
        if v is None:
            v = None
            rcfg, rv = repair_feasibility(inst, cfg, exact_eval)
            if rv is not None:
                cands.append((rv, rcfg))
        else:
            cands.append((v, cfg))
    # 满配保底
    v0 = exact_eval(probe)
    if v0 is not None:
        cands.append((v0, probe))
    if not cands:
        stats['time_s'] = _t.time() - t0
        stats['status'] = 'no-feasible'
        return None, None, stats
    cands.sort(key=lambda x: x[0])
    best_v, best_c = cands[0]
    stats['v_after_search'] = best_v
    stats['t_search'] = _t.time() - t0

    # 有界抛光：子集代理排序 + 全集精确确认
    screen = eval_fn if stats['surrogate'] == 'subset-lp' else None
    best_c, v2, n1 = exact_polish(inst, best_c, train_scen, exact_eval,
                                  budget_s=budget_polish * 0.6, screen_k=20,
                                  greedy_eval=screen)
    if v2 is not None and v2 < best_v:
        best_v = v2
    best_c, v3, n2 = exact_pair_polish(inst, best_c, train_scen, exact_eval,
                                        budget_s=budget_polish * 0.4, screen_k=15,
                                        greedy_eval=screen)
    if v3 is not None and v3 < best_v:
        best_v = v3
    stats.update({'v_final': best_v, 'polish_exact_evals': n1 + n2,
                  'time_s': _t.time() - t0, 'status': 'ok'})
    return best_c, best_v, stats


def _island_worker(args):
    """模块级 worker（可 pickle）：跑一个岛，返回 (best, best_v, pool_list)。"""
    inst, scenarios, seed, wall_budget, init_kind, jitter, eval_fn = args
    class _L:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    shared = {'best': None, 'lock': _L()}
    try:
        best, best_v, pool, iters = _island(inst, scenarios, seed, wall_budget,
                                            init_kind, jitter, shared, 0,
                                            eval_fn=eval_fn)
        return (best, best_v, list(pool.values()), iters)
    except Exception:
        return (None, None, [], 0)


def gri_run_pool(inst, train_scen, n_islands=6, wall_budget=120, seed=0,
                 eval_fn=None, rounds=2):
    """并行岛搜索（Pool.map，无 Manager）。rounds=2 时第二轮从首轮最优热启动。"""
    from multiprocessing import Pool
    all_pool = {}
    warm = None
    for rd in range(rounds):
        jobs = []
        for i in range(n_islands):
            if rd == 0:
                jobs.append((inst, train_scen, seed * 1000 + i, wall_budget,
                             'max', 0.02 * (i % 3), eval_fn))
            else:
                jobs.append((inst, train_scen, seed * 1000 + 100 + i,
                             wall_budget, 'warm', 0.02 * (i % 3), eval_fn))
        with Pool(processes=min(n_islands, os.cpu_count() or 4)) as p:
            for best, best_v, pool, iters in p.map(_island_worker, jobs):
                for cfg, v in pool:
                    k = cfg_key(cfg)
                    if k not in all_pool or v < all_pool[k]:
                        all_pool[k] = v
                if best is not None:
                    k = cfg_key(best)
                    if k not in all_pool or best_v < all_pool[k]:
                        all_pool[k] = best_v
        if not all_pool:
            break
        warm = min(all_pool.items(), key=lambda kv: kv[1])[0]
        # 把首轮最优注入第二轮：作为 'warm' 初值的来源
        _WARM_REGISTRY[cfg_key(train_scen[0]['d']) if False else 'last'] = warm
    ranked = sorted(all_pool.items(), key=lambda kv: kv[1])
    return [(json.loads(k), v) for k, v in ranked]


_WARM_REGISTRY = {}


def _apply_level(inst, cfg_hi, cfg_lo, lam):
    """在两配置之间做结构插值：λ=1 取 cfg_hi（满配），λ=0 取 cfg_lo（最省）。
    设施开闭取整，模块按比例取整并保持单调。"""
    T = inst['meta']['periods']
    out = {'xJ': {}, 'xK': {}, 'xL': {}, 'zJ': {}, 'zK': {}, 'rL': {}}
    for key in ('xJ', 'xK', 'xL', 'rL'):
        for f in cfg_hi[key]:
            hi, lo = cfg_hi[key][f], cfg_lo[key][f]
            out[key][f] = [hi[t] if lam >= 0.5 else lo[t] for t in range(T)]
    for key in ('zJ', 'zK'):
        for f in cfg_hi[key]:
            seq = []
            for t in range(T):
                hi = cfg_hi[key][f][t]
                lo = cfg_lo[key][f][t]
                v = lo + (hi - lo) * lam
                seq.append(int(round(max(0, min(4, v)))))
            out[key][f] = seq
    return repair_monotone(out, inst)


def capacity_level_search(inst, cfg_full, exact_eval, budget_s=60, iters=12):
    """容量水平二分：满配 → 逐级压缩总容量，找到仍覆盖可行的最低水平。
    针对"满配冗余大、单点减容必破约束"的结构：先定总水平，再局部重分配。
    返回 (best_cfg, best_val, n_eval)。
    """
    import time as _t
    t0 = _t.time()
    T = inst['meta']['periods']
    base = initial_full(inst)          # 全开、零扩容 = 最省结构
    hi_cfg, hi_val = clone(cfg_full), exact_eval(cfg_full)
    n = 1
    if hi_val is None:
        return None, None, n
    lo_cfg = clone(base)
    best_cfg, best_val = hi_cfg, hi_val
    # 二分：找最小的可行 λ
    a, b = 0.0, 1.0
    for _ in range(iters):
        if _t.time() - t0 > budget_s:
            break
        mid = (a + b) / 2.0
        c = _apply_level(inst, hi_cfg, lo_cfg, mid)
        v = exact_eval(c)
        n += 1
        if v is None:
            a = mid                      # 不可行 → 提高水平
        else:
            b = mid                      # 可行 → 继续下压
            if v < best_val:
                best_cfg, best_val = c, v
    return best_cfg, best_val, n


def redistribute_polish(inst, cfg0, exact_eval, budget_s=60, max_move=40, screen=None):
    """在容量水平确定后做"转移式"重分配：把冗余设施的模块挪给紧张的设施。
    每次移动都用精确评估确认；不可行即回滚。返回 (cfg, val, n_eval)。
    """
    import time as _t
    t0 = _t.time()
    T = inst['meta']['periods']
    cur, cur_v = clone(cfg0), exact_eval(cfg0)
    n = 1
    if cur_v is None:
        return None, None, n
    moved = 0
    improved = True
    while improved and moved < max_move and _t.time() - t0 < budget_s:
        improved = False
        donors = [(f, t) for f in inst['sorting'] for t in range(T) if cur['zJ'][f][t] > 0]
        recv = [(f, t) for f in inst['sorting'] for t in range(T) if cur['zJ'][f][t] < 4]
        for (fd, td) in donors:
            if improved or _t.time() - t0 > budget_s:
                break
            for (fr, tr_) in recv:
                if fd == fr and td == tr_:
                    continue
                c = clone(cur)
                c['zJ'][fd][td] -= 1
                c['zJ'][fr][tr_] = min(4, c['zJ'][fr][tr_] + 1)
                c = repair_monotone(c, inst)
                v = exact_eval(c)
                n += 1
                moved += 1
                if v is not None and v < cur_v - 1e-6:
                    cur, cur_v = c, v
                    improved = True
                    break
                if _t.time() - t0 > budget_s:
                    break
    return cur, cur_v, n
