#!/usr/bin/env python3
"""E3-E6 v2: 主实验批处理（ALNS 混合算法路线）。
求解策略（与论文 C4 贡献一致）:
  - 两阶段方法（SAA/DRO/ROB）用 ALNS 搜索（内层=精确 LP 回收子问题），
    搜索用缩减情景集（k=12），最终解用全训练情景集精确评价 plan_obj；
  - DET 用 CPLEX 单块求解（均值情景，秒级）；
  - CPLEX 单块求解 SAA(30) 于真实实例做精确性对照（报告 gap）。
样本外: 200 独立情景共享（配对设计）。
"""
import json, os, sys, time, copy, random
import numpy as np
import pandas as pd
import math

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = f'{BASE}/experiments'
os.makedirs(EXP, exist_ok=True)

from cplex_saa import build_and_solve_saa, gen_scenarios
from cplex_dro import solve_dro, worst_case_scenarios
from alns import alns, solve_second_stage_lp, fixed_cost_of

def make_batch_instance(base_inst, seed):
    rng = np.random.default_rng(seed)
    inst = copy.deepcopy(base_inst)
    T = inst['meta']['periods']
    GEN = list(inst['demand'])
    conc = np.array([sum(inst['demand'][i]['d'].values()) for i in GEN])
    conc = conc / conc.sum() * 60
    w = rng.dirichlet(conc)
    year_tot = {t: sum(inst['demand'][i]['d'][str(t)] for i in GEN) for t in range(1, T+1)}
    idx = {i: n for n, i in enumerate(GEN)}
    for i in GEN:
        for t in range(1, T+1):
            inst['demand'][i]['d'][str(t)] = float(w[idx[i]] * year_tot[t])
    for arc in ['a', 'a2', 'b', 'c', 'e']:
        for u in inst['dist'][arc]:
            for v2 in inst['dist'][arc][u]:
                inst['dist'][arc][u][v2] *= float(np.exp(rng.normal(0, 0.15)))
    for j in inst['sorting']:
        inst['sorting'][j]['op_cost'] *= float(rng.uniform(0.85, 1.15))
        inst['sorting'][j]['open_fixed'] *= float(rng.uniform(0.9, 1.1))
    for k in inst['recycling']:
        inst['recycling'][k]['op_cost'] *= float(rng.uniform(0.85, 1.15))
        inst['recycling'][k]['open_fixed'] *= float(rng.uniform(0.9, 1.1))
    for l in inst['landfill']:
        inst['landfill'][l]['open_fixed'] *= float(rng.uniform(0.9, 1.1))
    inst['params']['recycle_revenue'] *= float(rng.uniform(0.8, 1.2))
    inst['meta']['batch_seed'] = seed
    return inst

def reduce_scenarios(inst, scenarios, k=12, seed=7):
    """简单场景缩减：保留3锚点 + 均匀抽取 k-3 个采样情景，权重归一。"""
    rng = random.Random(seed)
    anchors = [sc for sc in scenarios if sc['name'].startswith('anchor')]
    others = [sc for sc in scenarios if not sc['name'].startswith('anchor')]
    keep = anchors + rng.sample(others, min(k - len(anchors), len(others)))
    tot = sum(sc['prob'] for sc in keep)
    for sc in keep:
        sc['prob'] = sc['prob'] / tot
    return keep

def eval_config(inst, cfg, scenarios, FU=1e4):
    fc = fixed_cost_of(inst, cfg, FU)
    tot = fc
    for sc in scenarios:
        c2 = solve_second_stage_lp(inst, cfg, sc['d'])
        if c2 is None:
            return None
        tot += sc['prob'] * c2
    return tot

def extract_xconfig_cpx(inst, V):
    T = inst['meta']['periods']
    gv = lambda v: v.solution_value
    return {
        'xJ': {inst['sorting'][j]['name']: [round(gv(V['xJ'][j, t])) for t in range(1, T+1)] for j in inst['sorting']},
        'xK': {inst['recycling'][k]['name']: [round(gv(V['xK'][k, t])) for t in range(1, T+1)] for k in inst['recycling']},
        'xL': {inst['landfill'][l]['name']: [round(gv(V['xL'][l, t])) for t in range(1, T+1)] for l in inst['landfill']},
        'zK': {inst['recycling'][k]['name']: [round(gv(V['zK'][k, t])) for t in range(1, T+1)] for k in inst['recycling']},
        'rL': {inst['landfill'][l]['name']: [round(gv(V['rL'][l, t])) for t in range(1, T+1)] for l in inst['landfill']},
    }

def cfg_to_tuplekey(cfg):
    return json.dumps(cfg, sort_keys=True)

from alns import solve_second_stage_lp_detail

def oos_eval(inst, cfg, test_scenarios, FU=1e4):
    """返回 (costs, probs, unmets, recycs)——加权口径 + 决策指标。"""
    fc = fixed_cost_of(inst, cfg, FU)
    costs, probs, unmets, recycs = [], [], [], []
    for sc in test_scenarios:
        c2, unmet, recyc = solve_second_stage_lp_detail(inst, cfg, sc['d'])
        costs.append(fc + (c2 if c2 is not None else np.nan))
        probs.append(sc['prob'])
        unmets.append(unmet if unmet is not None else np.nan)
        recycs.append(recyc if recyc is not None else np.nan)
    return np.array(costs), np.array(probs), np.array(unmets), np.array(recycs)

def weighted_stats(costs, probs, plan_obj, unmets=None, recycs=None, budget=None):
    m = ~np.isnan(costs)
    c, p = costs[m], probs[m]
    p = p / p.sum()
    wmean = float((c*p).sum())
    wvar = float((p*(c-wmean)**2).sum())
    order = np.argsort(c)
    cum = np.cumsum(p[order])
    p95 = float(c[order][np.searchsorted(cum, 0.95)])
    disc = float(p[c > plan_obj*1.001].sum())
    disc_mag = float((np.maximum(c-plan_obj, 0)*p).sum())
    out = {'oos_mean': wmean, 'oos_std': math.sqrt(wvar), 'oos_cv': math.sqrt(wvar)/wmean,
           'oos_p50': float(c[order][np.searchsorted(cum, 0.50)]), 'oos_p95': p95,
           'disappointment_rate': disc, 'disappointment_mean': disc_mag}
    if budget is not None:  # 公共预算基准（消除会计循环）
        out['budget_exceed_rate'] = float(p[c > budget*1.001].sum())
    if unmets is not None:
        um = unmets[~np.isnan(unmets)]
        up = p[:len(um)] if len(um) == len(p) else p
        out['service_fail_rate'] = float(up[um > 1.0].sum())
        out['unmet_mean_t'] = float((um*up).sum())
    if recycs is not None:
        rm = recycs[~np.isnan(recycs)]
        rp = p[:len(rm)] if len(rm) == len(p) else p
        out['recycled_mean_t'] = float((rm*rp).sum())
    return out

def run_method_alns(inst, method, train_scen, alns_iters=250, alns_seed=42):
    """返回 (plan_obj_on_full_train, cfg, walltime, alns_gap_vs_saa?)"""
    t0 = time.time()
    if method.startswith('DRO'):
        eps = float(method.split('(')[1].rstrip(')'))
        wc = worst_case_scenarios(inst, train_scen, eps)
        lam = min(eps / 0.3, 1.0)
        full = []
        for sc, w in zip(train_scen, wc):
            full.append({'name': sc['name'], 'prob': sc['prob']*(1-lam), 'd': sc['d']})
            full.append({'name': w['name'], 'prob': sc['prob']*lam, 'd': w['d']})
    elif method == 'ROB':
        T = inst['meta']['periods']
        full = [{'name': 'wc', 'prob': 1.0,
                 'd': {i: {t: train_scen[0]['d'][i][t]*1.2 for t in range(1, T+1)} for i in inst['demand']}}]
    else:  # SAA
        full = train_scen
    # 情景缩减搜索
    search_scen = reduce_scenarios(inst, full, k=12, seed=7) if len(full) > 12 else full
    best, best_cost_search, _, _ = alns(inst, search_scen, max_iter=alns_iters, seed=alns_seed, verbose=False)
    plan_obj = eval_config(inst, best, full)
    return plan_obj, best, time.time()-t0

def run_method_det(inst):
    t0 = time.time()
    T = inst['meta']['periods']
    mean_sc = [{'name': 'mean', 'prob': 1.0,
                'd': {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in inst['demand']}}]
    mdl, sol, V = build_and_solve_saa(inst, mean_sc, 'det', retire_lstar=True, t_star=2)
    xcfg = extract_xconfig_cpx(inst, V)
    # 转为 cfg 结构
    cfg = {'xJ': {}, 'xK': {}, 'xL': {}, 'zK': {}, 'rL': {}}
    for name, seq in xcfg['xJ'].items():
        for j, info in inst['sorting'].items():
            if info['name'] == name: cfg['xJ'][j] = seq
    for name, seq in xcfg['xK'].items():
        for k, info in inst['recycling'].items():
            if info['name'] == name: cfg['xK'][k] = seq
    for name, seq in xcfg['xL'].items():
        for l, info in inst['landfill'].items():
            if info['name'] == name: cfg['xL'][l] = seq
    for name, seq in xcfg['zK'].items():
        for k, info in inst['recycling'].items():
            if info['name'] == name: cfg['zK'][k] = seq
    for name, seq in xcfg['rL'].items():
        for l, info in inst['landfill'].items():
            if info['name'] == name: cfg['rL'][l] = seq
    return sol.objective_value, cfg, time.time()-t0

# ================= E3 =================
def run_main_experiment(n_batches=10, methods=None, n_test=200):
    methods = methods or ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)', 'ROB']
    base = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    rows = []
    for seed in [0] + list(range(1, n_batches+1)):
        inst = base if seed == 0 else make_batch_instance(base, seed)
        iname = 'real' if seed == 0 else f'batch{seed}'
        train = gen_scenarios(inst, 30, seed=1000+seed)
        test = gen_scenarios(inst, n_test, seed=20260901, include_anchor=True)
        # 先解 DET 取公共预算基准
        obj_det, cfg_det, wt_det = run_method_det(inst)
        budget = obj_det
        for m in methods:
            try:
                if m == 'DET':
                    obj, cfg, wt = obj_det, cfg_det, wt_det
                else:
                    obj, cfg, wt = run_method_alns(inst, m, train)
                if obj is None:
                    print(f'[{iname}] {m}: 不可行，跳过'); continue
                costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
                st = weighted_stats(costs, probs, obj, unmets, recycs, budget=budget)
                row = {'instance': iname, 'method': m, 'plan_obj': obj, 'time_s': wt, **st}
                rows.append(row)
                json.dump(cfg, open(f'{EXP}/config_{iname}_{m}.json', 'w'), ensure_ascii=False)
                print(f"[{iname}] {m}: plan={obj:,.0f} oos={row['oos_mean']:,.0f} bER={row.get('budget_exceed_rate', float('nan')):.2f} sfr={row.get('service_fail_rate', float('nan')):.2f} t={wt:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(f'{EXP}/results_main.csv', index=False)
            except Exception as e:
                print(f'[{iname}] {m} ERR: {e}', flush=True)
    return rows

# ================= E4 灵敏度 =================
def run_sensitivity(real_only=True):
    base = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    train = gen_scenarios(base, 30, seed=1000)
    rows = []
    grid = []
    for tc in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for rev in [0, 10, 20, 30, 40]:
            grid.append(('transport_cost', tc, 'recycle_revenue', rev))
    for tc, rev in [(1.0, 20.0)] + [(p[1], p[3]) for p in grid if (p[1], p[3]) != (1.0, 20.0)]:
        inst = copy.deepcopy(base)
        inst['params']['transport_cost'] = tc
        inst['params']['recycle_revenue'] = rev
        for m in ['DET', 'SAA', 'DRO(0.10)']:
            if m == 'DET':
                obj, cfg, wt = run_method_det(inst)
            else:
                obj, cfg, wt = run_method_alns(inst, m, train, alns_iters=150)
            costs, probs, unmets, recycs = oos_eval(inst, cfg, gen_scenarios(inst, 100, seed=20260901))
            st = weighted_stats(costs, probs, obj)
            rows.append({'tc': tc, 'rev': rev, 'method': m, 'oos_mean': st['oos_mean'],
                         'disc_rate': st['disappointment_rate']})
        print(f'[sens tc={tc} rev={rev}] done', flush=True)
        pd.DataFrame(rows).to_csv(f'{EXP}/results_sensitivity.csv', index=False)
    return rows

# ================= E5 回测 =================
def run_backtest():
    """用 2023-24 校准（训练），2025 官方总量作真实检验。"""
    base = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    # 训练情景：锚定 2023/2024 模式（把三期需求都替换为当年模式）
    GEN = list(inst['demand'])
    scen = []
    for yr in [1, 2]:
        d = {i: {t: inst['demand'][i]['d'][str(yr)] for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'anchor_hist{yr}', 'prob': 0.5, 'd': d})
    for s in range(8):
        import random as _r
        rng = _r.Random(500+s)
        d = {i: {t: scen[0]['d'][i][t]*rng.uniform(0.85, 1.15) for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'hist_s{s}', 'prob': 0.5/8, 'd': d})
    # 测试 = 2025 真实（各期同2025模式）
    test = [{'name': 'actual_2025', 'prob': 1.0,
             'd': {i: {t: inst['demand'][i]['d']['3'] for t in range(1, 4)} for i in GEN}}]
    rows = []
    for m in ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)', 'ROB']:
        if m == 'DET':
            obj, cfg, wt = run_method_det(inst)
        else:
            obj, cfg, wt = run_method_alns(inst, m, scen)
        costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
        actual = costs[0]
        rows.append({'method': m, 'plan_obj': obj, 'actual_2025': actual, 'regret': actual - obj,
                     'disappointed': bool(actual > obj*1.001)})
        print(f"[backtest] {m}: plan={obj:,.0f} actual2025={actual:,.0f} regret={actual-obj:,.0f}", flush=True)
    pd.DataFrame(rows).to_csv(f'{EXP}/results_backtest.csv', index=False)
    return rows

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='main', choices=['main', 'sens', 'backtest', 'all'])
    args = ap.parse_args()
    if args.stage in ('main', 'all'):
        run_main_experiment()
    if args.stage in ('sens', 'all'):
        run_sensitivity()
    if args.stage in ('backtest', 'all'):
        run_backtest()
