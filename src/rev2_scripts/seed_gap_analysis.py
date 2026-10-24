# -*- coding: utf-8 -*-
"""seeds10 修后分析：每个种子的 plan_obj vs 同训练集 CPLEX 精确最优。
（为论文 §5.4 两段提供可复现数字：有利/不利种子 gap 与失望率梯度。）
"""
import json, sys, os
import numpy as np, pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa

base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
train = gen_scenarios(base, 30, seed=1000)

mdl, sol, V = build_and_solve_saa(base, train, 'seedgap_exact', retire_lstar=True, t_star=2,
                                  require_cover=True, time_limit=1800, log=False)
opt = sol.objective_value
gap = mdl.solve_details.mip_relative_gap
print(f'CPLEX exact optimum (30-scen train, cover): {opt:.2f}  gap={gap:.2e}')

df = pd.read_csv(f'{D}/experiments/results_seeds10.csv')
d = df[df.method == 'SAA'].copy()
d['gap_pct'] = (d.plan_obj - opt) / opt * 100
print('\nSAA per-seed gap vs exact optimum (%):')
print(d[['seed', 'plan_obj', 'gap_pct']].round(2).to_string(index=False))
print(f'\nfavourable {d.gap_pct.min():.1f}%  unfavourable {d.gap_pct.max():.1f}%')

print('\nDisappointment by method (mean±sd, n=10):')
g = df.groupby('method')['disappointment'].agg(['mean', 'std']).round(3)
print(g)
g2 = df.groupby('method')['service_fail'].agg(['mean', 'std']).round(3)
print('\nService-fail by method:')
print(g2)
out = {'exact_opt_30scen': opt, 'saa_gap_min': float(d.gap_pct.min()),
       'saa_gap_max': float(d.gap_pct.max())}
json.dump(out, open(f'{D}/experiments/seedgap_summary.json', 'w'), indent=1)
print('\nwrote experiments/seedgap_summary.json')
