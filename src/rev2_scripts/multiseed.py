"""多种子实验：10 种子 × {SAA, DASH(0.05/0.10/0.20)} 在 real 实例上，报告 mean±sd"""
import os
import json, sys, os, time
import numpy as np, pandas as pd
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import run_method_alns, oos_eval, weighted_stats, EXP

base = json.load(open('instance_v1.json', encoding='utf-8'))
train = gen_scenarios(base, 30, seed=1000)
test  = gen_scenarios(base, 200, seed=20260901, include_anchor=True)

rows = []
for seed in range(1, 11):  # 10 seeds
    for m in ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']:
        obj, cfg, wt = run_method_alns(base, m, train, alns_seed=seed)
        costs, probs, unmets, recycs = oos_eval(base, cfg, test)
        st = weighted_stats(costs, probs, obj, unmets, recycs)
        rows.append({'seed':seed, 'method':m, 'plan_obj':obj, 'oos_mean':st['oos_mean'],
                     'oos_p95':st['oos_p95'], 'disappointment':st['disappointment_rate'],
                     'service_fail':st.get('service_fail_rate',0), 'recycled':st['recycled_mean_t']})
    print(f'seed {seed}: ' + '  '.join(f'{r["method"]}={r["oos_mean"]/1e9:.3f}' for r in rows[-4:]), flush=True)
    pd.DataFrame(rows).to_csv(f'{EXP}/results_seeds10.csv', index=False)

df = pd.DataFrame(rows)
print('\n=== 多种子汇总（mean±sd, n=10）===')
for m in ['SAA','DRO(0.05)','DRO(0.10)','DRO(0.16)']:
    g = df[df.method==m]
    print(f'{m:12s}: oos={g.oos_mean.mean()/1e9:.3f}±{g.oos_mean.std()/1e9:.3f}  '
          f'disc={g.disappointment.mean():.3f}±{g.disappointment.std():.3f}  '
          f'sfr={g.service_fail.mean():.3f}  recycled={g.recycled.mean()/1e6:.2f}Mt')
