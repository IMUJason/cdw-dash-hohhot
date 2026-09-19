#!/usr/bin/env python3
"""R1 补充实验：M3 紧容量实例族 / M6 惩罚灵敏度 / M7 三折回测 / 计时多种子。"""
import json, os, sys, time, copy, random
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = f'{BASE}/experiments'

from cplex_saa import gen_scenarios
from run_experiments import (make_batch_instance, run_method_det, run_method_alns,
                             oos_eval, weighted_stats)

base = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))

# ---------- M3: 紧容量实例族（能力×0.7/0.8）——构型分化 ----------
rows = []
for cap_scale in [0.7, 0.8]:
    for seed in [1, 2, 3]:
        inst = make_batch_instance(base, seed)
        for j in inst['sorting']:
            inst['sorting'][j]['cap_t'] *= cap_scale
        for k in inst['recycling']:
            inst['recycling'][k]['cap_t'] *= cap_scale
        iname = f'tight{int(cap_scale*100)}_s{seed}'
        train = gen_scenarios(inst, 30, seed=1000+seed)
        test = gen_scenarios(inst, 200, seed=20260901)
        obj_det, cfg_det, wt_det = run_method_det(inst)
        budget = obj_det
        cfgs = {'DET': (obj_det, cfg_det, wt_det)}
        for m in ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)', 'ROB']:
            obj, cfg, wt = run_method_alns(inst, m, train)
            cfgs[m] = (obj, cfg, wt)
        # 构型签名
        sigs = {}
        for m, (obj, cfg, wt) in cfgs.items():
            sig = json.dumps(cfg, sort_keys=True)
            sigs[m] = sig
        uniq = len(set(sigs.values()))
        for m, (obj, cfg, wt) in cfgs.items():
            costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
            st = weighted_stats(costs, probs, obj, unmets, recycs, budget=budget)
            rows.append({'instance': iname, 'method': m, 'plan_obj': obj, **st,
                         'config_sig': hash(sigs[m]) % 10**8, 'uniq_configs': uniq})
            print(f'[{iname}] {m}: oos={st["oos_mean"]:,.0f} bER={st.get("budget_exceed_rate", float("nan")):.2f} sfr={st.get("service_fail_rate", float("nan")):.2f} uniq={uniq}', flush=True)
        pd.DataFrame(rows).to_csv(f'{EXP}/results_tight.csv', index=False)

# ---------- M6: 惩罚灵敏度 ----------
rows2 = []
for pen in [100.0, 250.0, 500.0, 1000.0]:
    inst = copy.deepcopy(base)
    inst['params']['unmet_penalty'] = pen
    train = gen_scenarios(inst, 30, seed=1000)
    test = gen_scenarios(inst, 200, seed=20260901)
    for m in ['DET', 'SAA', 'DRO(0.10)']:
        if m == 'DET':
            obj, cfg, wt = run_method_det(inst)
        else:
            obj, cfg, wt = run_method_alns(inst, m, train, alns_iters=150)
        costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
        st = weighted_stats(costs, probs, obj, unmets, recycs, budget=None)
        # 无惩罚口径成本（把 u 的惩罚项剥离）：oos_nopen = oos_mean - (pen-10)*E[u] 近似（u的运营按填埋计）
        E_u = st.get('unmet_mean_t', 0.0)
        rows2.append({'penalty': pen, 'method': m, 'oos_mean': st['oos_mean'],
                      'oos_nopen_approx': st['oos_mean'] - (pen - 10.0)*E_u,
                      'unmet_mean_t': E_u, 'service_fail': st.get('service_fail_rate', float('nan'))})
        print(f'[pen={pen}] {m}: oos={st["oos_mean"]:,.0f} unmet={E_u:,.0f}t', flush=True)
    pd.DataFrame(rows2).to_csv(f'{EXP}/results_penalty.csv', index=False)

# ---------- M7: 三折留一年回测 ----------
rows3 = []
for hold in [1, 2, 3]:  # 1=2023, 2=2024, 3=2025
    inst = copy.deepcopy(base)
    GEN = list(inst['demand'])
    scen = []
    for yr in [1, 2, 3]:
        if yr == hold: continue
        d = {i: {t: inst['demand'][i]['d'][str(yr)] for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'anchor{yr}', 'prob': 0.5, 'd': d})
    for s in range(8):
        rng = random.Random(600+s)
        base_d = scen[0]['d']
        d = {i: {t: base_d[i][t]*rng.uniform(0.85, 1.15) for t in range(1, 4)} for i in GEN}
        scen.append({'name': f's{s}', 'prob': 0.5/8, 'd': d})
    test = [{'name': f'actual{hold}', 'prob': 1.0,
             'd': {i: {t: inst['demand'][i]['d'][str(hold)] for t in range(1, 4)} for i in GEN}}]
    for m in ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)', 'ROB']:
        if m == 'DET':
            obj, cfg, wt = run_method_det(inst)
        else:
            obj, cfg, wt = run_method_alns(inst, m, scen)
        costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
        actual = costs[0]
        rows3.append({'holdout_year': hold, 'method': m, 'plan_obj': obj, 'actual': actual,
                      'regret': actual - obj, 'disappointed': bool(actual > obj*1.001)})
        print(f'[loo-{hold}] {m}: plan={obj:,.0f} actual={actual:,.0f}', flush=True)
    pd.DataFrame(rows3).to_csv(f'{EXP}/results_loo_backtest.csv', index=False)

# ---------- 计时多种子（真实实例，ALNS 3 种子） ----------
rows4 = []
for alns_seed in [42, 43, 44]:
    train = gen_scenarios(base, 30, seed=1000)
    for m in ['SAA', 'DRO(0.10)']:
        t0 = time.time()
        obj, cfg, wt = run_method_alns(base, m, train, alns_seed=alns_seed)
        rows4.append({'alns_seed': alns_seed, 'method': m, 'plan_obj': obj, 'time_s': wt})
        print(f'[timing seed{alns_seed}] {m}: {wt:.1f}s obj={obj:,.0f}', flush=True)
pd.DataFrame(rows4).to_csv(f'{EXP}/results_timing.csv', index=False)
print('R1 补充实验全部完成')
