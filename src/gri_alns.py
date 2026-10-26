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

COVER_M = 1000.0          # 未满足软惩罚（CNY/t），与诊断版一致
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


# ---------- 单岛搜索 ----------

def _island(inst, scenarios, seed, wall_budget, init_kind, jitter,
            shared, island_id, max_iter=2_000_000):
    rng = random.Random(seed)
    T = inst['meta']['periods']
    deadline = time.time() + wall_budget

    if init_kind == 'full':
        cur = initial_current(inst)
    elif init_kind == 'rand':
        cur = initial_random(inst, rng)
    else:
        cur = initial_full(inst)
        for j in inst['sorting']:
            cur['zJ'][j] = [4] * T
        for k in inst['recycling']:
            cur['zK'][k] = [4] * T
        cur = repair_monotone(cur, inst)
    cur_v = gri_value(inst, cur, scenarios, rng, jitter)
    best, best_v = clone(cur), cur_v[0]
    pool, pool_keys = {}, set()
    if cur_v[1] <= 1e-6:
        pool[cfg_key(cur)] = (clone(cur), cur_v[0])

    ops = ['D1_close', 'D2_open', 'D3_swap', 'D4_expand', 'D5_retire',
           'D6_greedy', 'D7_cover', 'D8_time', 'D9_realloc']
    w = {o: 1.0 for o in ops}
    TEMP0, it_done, mig_at = 0.01, 0, MIG_EVERY

    def try_pool(cfg, v):
        if v[1] > 1e-6:
            return
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
                cv = gri_value(inst, cand, scenarios, rng, jitter)
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
                    v2 = gri_value(inst, c2, scenarios, rng, jitter)
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
        v = gri_value(inst, cand, scenarios, rng, jitter)
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
                 shared, island_id, out_q):
    try:
        best, best_v, pool, it_done = _island(inst, scenarios, seed, wall_budget,
                                              init_kind, jitter, shared, island_id)
        out_q.put((island_id, best, best_v,
                   [(cfg, v) for cfg, v in pool.values()], it_done))
    except Exception as e:
        out_q.put((island_id, None, None, [], 0))


# ---------- 主入口 ----------

def gri_run(inst, train_scen, n_islands=6, wall_budget=120, seed=0,
            methods_jitter=(0.0, 0.02, 0.05)):
    """一次 GRI 运行（= 一组岛）。返回 pool 列表 [(cfg, greedy_v)] 按贪心值升序。"""
    from multiprocessing import Queue
    mgr = Manager()
    shared = mgr.dict()
    shared['best'] = None
    shared['lock'] = mgr.Lock()
    out_q = Queue()
    inits = ['full', 'rand', 'max'] + ['full'] * max(0, n_islands - 3)
    procs = []
    for i in range(n_islands):
        jit = methods_jitter[i % len(methods_jitter)]
        p = Process(target=_island_wrap,
                    args=(inst, train_scen, seed * 1000 + i, wall_budget,
                          inits[i % len(inits)], jit, shared, i, out_q))
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

def exact_polish(inst, cfg0, scenarios, exact_eval, budget_s=120, top_moves=10000,
                 screen_k=100000):
    """有界精确抛光 + ILS：贪心筛邻域、精确 best-improvement、近优 kick。"""
    import time as _t
    t0 = _t.time()
    cache = {}
    n_eval = 0

    def ex(c):
        nonlocal n_eval
        k = cfg_key(c)
        if k not in cache:
            cache[k] = exact_eval(c)
            n_eval += 1
        return cache[k]

    cur_cfg = clone(cfg0)
    cur_val = ex(cur_cfg)
    best_cfg, best_val = clone(cur_cfg), cur_val
    rng = random.Random(0)
    stall = 0
    if cur_val is None:
        return cur_cfg, None, n_eval

    while _t.time() - t0 < budget_s and n_eval < top_moves:
        T = inst['meta']['periods']
        cand = {}

        def add(c):
            c = repair_monotone(c, inst)
            k = cfg_key(c)
            if k not in cache and k not in cand:
                cand[k] = c

        # 单期开闭（J/K/L），以及开设施+模块的联合动作
        for fam, facs, zkey in (('xJ', inst['sorting'], 'zJ'),
                                ('xK', inst['recycling'], 'zK')):
            for f in facs:
                for t in range(1, T + 1):
                    c = clone(cur_cfg)
                    c[fam][f][t - 1] = 1 - c[fam][f][t - 1]
                    add(c)
                    if cur_cfg[fam][f][t - 1] == 0 and cur_cfg[zkey][f][t - 1] < 4:
                        for dz in (1, 2):
                            c = clone(cur_cfg)
                            c[fam][f][t - 1] = 1
                            for tt in range(t, T + 1):
                                c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + dz)
                            add(c)
        for l in inst['landfill']:
            for t in range(1, T + 1):
                c = clone(cur_cfg)
                c['xL'][l][t - 1] = 1 - c['xL'][l][t - 1]
                add(c)
        # 扩容尾部±1
        for zkey, facs in (('zJ', inst['sorting']), ('zK', inst['recycling'])):
            for f in facs:
                for t in range(1, T + 1):
                    if cur_cfg[zkey][f][t - 1] > 0:
                        c = clone(cur_cfg)
                        for tt in range(t, T + 1):
                            c[zkey][f][tt - 1] = cur_cfg[zkey][f][tt - 1] - 1
                        add(c)
                        c = clone(cur_cfg)
                        c[zkey][f][t - 1] -= 1
                        add(c)
                        c = clone(cur_cfg)
                        for tt in range(1, t + 1):
                            c[zkey][f][tt - 1] -= 1
                        add(c)
                    if cur_cfg[zkey][f][t - 1] < 4:
                        c = clone(cur_cfg)
                        for tt in range(t, T + 1):
                            c[zkey][f][tt - 1] = cur_cfg[zkey][f][tt - 1] + 1
                        add(c)
        # 退役时点
        for l in inst['landfill']:
            if not inst['landfill'][l].get('retire_candidate') and not any(cur_cfg['rL'][l]):
                for t in range(1, T + 1):
                    c = clone(cur_cfg)
                    for tt in range(t, T + 1):
                        c['rL'][l][tt - 1] = 1
                    add(c)
        # 退役推迟/撤销（跨解邻域，不违反单一解内的不可逆约束）
        for l in inst['landfill']:
            if not inst['landfill'][l].get('retire_candidate') and any(cur_cfg['rL'][l]):
                t_ret = next(t for t in range(1, T + 1) if cur_cfg['rL'][l][t - 1])
                c = clone(cur_cfg)
                c['rL'][l][t_ret - 1] = 0
                c['xL'][l][t_ret - 1] = 1
                add(c)
                c = clone(cur_cfg)
                c['rL'][l] = [0] * T
                c['xL'][l] = [1] * T
                add(c)
        # 跨设施容量转移（减容与增容必须协同才能保持覆盖）
        for zkey, facs in (('zJ', list(inst['sorting'])), ('zK', list(inst['recycling']))):
            for f in facs:
                for g in facs:
                    if f == g:
                        continue
                    for t in range(1, T + 1):
                        if cur_cfg[zkey][f][t - 1] > 0 and cur_cfg[zkey][g][t - 1] < 4:
                            c = clone(cur_cfg)
                            for tt in range(t, T + 1):
                                if c[zkey][f][tt - 1] > 0 and c[zkey][g][tt - 1] < 4:
                                    c[zkey][f][tt - 1] -= 1
                                    c[zkey][g][tt - 1] += 1
                            add(c)
        # 跨层级扩容替代（分拣模块与再生厂模块联合增减）
        for t in range(1, T + 1):
            for k in inst['recycling']:
                if cur_cfg['zK'][k][t - 1] > 0:
                    for j in inst['sorting']:
                        if cur_cfg['zJ'][j][t - 1] < 4:
                            c = clone(cur_cfg)
                            for tt in range(t, T + 1):
                                if c['zK'][k][tt - 1] > 0 and c['zJ'][j][tt - 1] < 4:
                                    c['zK'][k][tt - 1] -= 1
                                    c['zJ'][j][tt - 1] += 1
                            add(c)
            for j in inst['sorting']:
                if cur_cfg['zJ'][j][t - 1] > 0:
                    for k in inst['recycling']:
                        if cur_cfg['zK'][k][t - 1] < 4:
                            c = clone(cur_cfg)
                            for tt in range(t, T + 1):
                                if c['zJ'][j][tt - 1] > 0 and c['zK'][k][tt - 1] < 4:
                                    c['zJ'][j][tt - 1] -= 1
                                    c['zK'][k][tt - 1] += 1
                            add(c)
        # 关闭中的分拣中心：开启并承接一个转移模块（三项联合动作）
        for j in inst['sorting']:
            for t in range(1, T + 1):
                if cur_cfg['xJ'][j][t - 1] == 0 and cur_cfg['zJ'][j][t - 1] < 4:
                    for k in inst['recycling']:
                        if cur_cfg['zK'][k][t - 1] > 0:
                            c = clone(cur_cfg)
                            c['xJ'][j][t - 1] = 1
                            for tt in range(t, T + 1):
                                if c['zK'][k][tt - 1] > 0 and c['zJ'][j][tt - 1] < 4:
                                    c['zK'][k][tt - 1] -= 1
                                    c['zJ'][j][tt - 1] += 1
                            add(c)
                    for j2 in inst['sorting']:
                        if j2 != j and cur_cfg['zJ'][j2][t - 1] > 0:
                            c = clone(cur_cfg)
                            c['xJ'][j][t - 1] = 1
                            for tt in range(t, T + 1):
                                if c['zJ'][j2][tt - 1] > 0 and c['zJ'][j][tt - 1] < 4:
                                    c['zJ'][j2][tt - 1] -= 1
                                    c['zJ'][j][tt - 1] += 1
                            add(c)
        # 同期设施开关交换（J/K 各自同类交换）
        for fam, facs in (('xJ', list(inst['sorting'])), ('xK', list(inst['recycling']))):
            for f in facs:
                for g in facs:
                    if f == g:
                        continue
                    for t in range(1, T + 1):
                        if cur_cfg[fam][f][t - 1] == 1 and cur_cfg[fam][g][t - 1] == 0:
                            c = clone(cur_cfg)
                            c[fam][f][t - 1] = 0
                            c[fam][g][t - 1] = 1
                            add(c)
        if not cand:
            break
        ranked = sorted(((gri_value(inst, c, scenarios)[0], c) for c in cand.values()),
                        key=lambda x: x[0])
        improved = False
        round_best_cfg, round_best_val = cur_cfg, cur_val
        evaluated = []
        for pos, (gv, c) in enumerate(ranked[:screen_k * 3]):
            if _t.time() - t0 > budget_s or n_eval >= top_moves:
                break
            v = ex(c)
            if v is not None:
                evaluated.append((v, c))
                if v < round_best_val - 1e-6:
                    round_best_cfg, round_best_val, improved = c, v, True
            if improved and pos + 1 >= screen_k:
                break
        if improved:
            cur_cfg, cur_val = round_best_cfg, round_best_val
            if cur_val < best_val - 1e-6:
                best_cfg, best_val = clone(cur_cfg), cur_val
                stall = 0
            else:
                stall += 1
        else:
            # ILS kick：跳到已精确评估的次优点，全局最优始终保留
            stall += 1
            near = sorted(((v, c) for v, c in evaluated if v <= best_val * 1.08), key=lambda x: x[0])
            if not near:
                break
            cur_val, cur_cfg = rng.choice(near[:min(6, len(near))])
        if stall >= 20:
            break
    return best_cfg, best_val, n_eval


def exact_pair_polish(inst, cfg0, scenarios, exact_eval, budget_s=180, max_rounds=4):
    """二阶精确抛光：联合执行「开闭/退役动作 + 减容动作」并精确评估。
    用于穿越单动作不可行或变差的壁垒。返回 (best_cfg, best_val, evals)。
    """
    import time as _t
    t_start = _t.time()
    cur = clone(cfg0)
    cur_v = exact_eval(cur)
    n_eval = 1
    if cur_v is None:
        return cur, None, n_eval
    T = inst['meta']['periods']

    def apply(c, mv):
        kind = mv[0]
        if kind == 'x':
            _, fam, f, t = mv
            c[fam][f][t - 1] = 1 - c[fam][f][t - 1]
        elif kind == 'xall':
            _, fam, f = mv
            v = 1 - max(c[fam][f])
            c[fam][f] = [v] * T
        elif kind == 'xtail':
            _, fam, f, t, v = mv
            for tt in range(t, T + 1):
                c[fam][f][tt - 1] = v
        elif kind == 'xshift':
            _, fam, f, t, d = mv
            if d == +1:
                if c[fam][f][t - 1] == 1 and c[fam][f][t] == 0:
                    c[fam][f][t - 1] = 0
                    c[fam][f][t] = 1
            else:
                if c[fam][f][t - 1] == 0 and c[fam][f][t] == 1:
                    c[fam][f][t - 1] = 1
                    c[fam][f][t] = 0
        elif kind == 'xwindow':
            _, fam, f, a, b = mv
            for t in range(1, T + 1):
                c[fam][f][t - 1] = 1 if a <= t <= b else 0
            if fam == 'xL':
                c['rL'][f] = [0] * T
        elif kind == 'zdec':
            _, zkey, f, t = mv
            for tt in range(t, T + 1):
                c[zkey][f][tt - 1] = max(0, c[zkey][f][tt - 1] - 1)
        elif kind == 'zhead':
            _, zkey, f, t = mv
            for tt in range(1, t + 1):
                c[zkey][f][tt - 1] = max(0, c[zkey][f][tt - 1] - 1)
        elif kind == 'zpoint':
            _, zkey, f, t = mv
            c[zkey][f][t - 1] = max(0, c[zkey][f][t - 1] - 1)
        elif kind == 'znone':
            pass
        elif kind == 'zinc':
            _, zkey, f, t = mv
            for tt in range(t, T + 1):
                c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + 1)
        elif kind == 'zinchead':
            _, zkey, f, t = mv
            for tt in range(1, t + 1):
                c[zkey][f][tt - 1] = min(4, c[zkey][f][tt - 1] + 1)
        elif kind == 'zincpoint':
            _, zkey, f, t = mv
            c[zkey][f][t - 1] = min(4, c[zkey][f][t - 1] + 1)
        elif kind == 'retire':
            _, l, t = mv
            for tt in range(t, T + 1):
                c['rL'][l][tt - 1] = 1
        elif kind == 'unretire':
            _, l, mode = mv
            if mode == 'all':
                c['rL'][l] = [0] * T
                c['xL'][l] = [1] * T
            else:
                t0 = next((t for t in range(1, T + 1) if c['rL'][l][t - 1]), None)
                if t0 is not None:
                    c['rL'][l][t0 - 1] = 0
                    c['xL'][l][t0 - 1] = 1
        return repair_monotone(c, inst)

    for _ in range(max_rounds):
        if _t.time() - t_start > budget_s:
            break
        x_moves = []
        for fam, facs in (('xJ', inst['sorting']), ('xK', inst['recycling']),
                          ('xL', inst['landfill'])):
            for f in facs:
                for t in range(1, T + 1):
                    x_moves.append(('x', fam, f, t))
        for fam, facs in (('xJ', inst['sorting']), ('xK', inst['recycling']),
                          ('xL', inst['landfill'])):
            for f in facs:
                x_moves.append(('xall', fam, f))
                for t in range(1, T + 1):
                    x_moves.append(('xtail', fam, f, t, 0))
                    x_moves.append(('xtail', fam, f, t, 1))
                    for t in range(1, T):
                        x_moves.append(('xshift', fam, f, t, +1))
                        x_moves.append(('xshift', fam, f, t, -1))
                    for a in range(1, T + 1):
                        for b in range(a, T + 1):
                            x_moves.append(('xwindow', fam, f, a, b))
        for l in inst['landfill']:
            if inst['landfill'][l].get('retire_candidate'):
                continue
            if any(cur['rL'][l]):
                x_moves.append(('unretire', l, 'one'))
                x_moves.append(('unretire', l, 'all'))
            else:
                for t in range(1, T + 1):
                    x_moves.append(('retire', l, t))
        z_moves = [('znone', None, None, None)]
        for zkey, facs in (('zJ', inst['sorting']), ('zK', inst['recycling'])):
            for f in facs:
                for t in range(1, T + 1):
                    if cur[zkey][f][t - 1] > 0:
                        z_moves.append(('zdec', zkey, f, t))
                        z_moves.append(('zhead', zkey, f, t))
                        z_moves.append(('zpoint', zkey, f, t))
                        if cur[zkey][f][t - 1] < 4:
                            z_moves.append(('zinc', zkey, f, t))
                            z_moves.append(('zinchead', zkey, f, t))
                            z_moves.append(('zincpoint', zkey, f, t))
        round_cfg, round_val = cur, cur_v
        seen = set()
        for mx in x_moves:
            if _t.time() - t_start > budget_s:
                break
            for mz in z_moves:
                if _t.time() - t_start > budget_s:
                    break
                c = apply(apply(clone(cur), mx), mz)
                k = cfg_key(c)
                if k in seen or k == cfg_key(cur):
                    continue
                seen.add(k)
                v = exact_eval(c)
                n_eval += 1
                if v is not None and v < round_val - 1e-6:
                    round_cfg, round_val = c, v
        if round_val >= cur_v - 1e-6:
            break
        cur, cur_v = round_cfg, round_val
    return cur, cur_v, n_eval


def config_violations(inst, cfg):
    """一阶定义域/耦合/单调性/退役约束的显式审计。返回违规列表（应为空）。"""
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
