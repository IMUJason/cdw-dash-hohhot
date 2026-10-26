# -*- coding: utf-8 -*-
"""从 results_main_v22.csv 生成论文所需全部派生数字（终端清空后版本）。
输出：控制台对照表 + experiments/paper_numbers_v24.json
"""
import json, os
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{D}/experiments'
df = pd.read_csv(f'{EXP}/results_main_v22.csv')
METHODS = ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']

real = df[df.instance == 'real'].set_index('method')
rep = df[df.instance != 'real']

out = {}
print('=== 真实实例 ===')
print(f"{'方法':12s} {'plan(B)':>9s} {'oos(B)':>9s} {'P95(B)':>9s} {'disc':>7s} {'sfr':>7s} {'recyc(Mt)':>9s}")
for m in METHODS:
    if m not in real.index:
        print(f'{m:12s} 缺失'); continue
    r = real.loc[m]
    print(f"{m:12s} {r.plan_obj/1e9:9.3f} {r.oos_mean/1e9:9.3f} {r.oos_p95/1e9:9.3f} "
          f"{r.disc:7.3f} {r.sfr:7.3f} {r.recycled/1e6:9.2f}")
    out[f'real_{m}'] = {k: (None if pd.isna(r[k]) else float(r[k]))
                        for k in ['plan_obj', 'oos_mean', 'oos_p95', 'disc', 'sfr', 'recycled']}

det = real.loc['DET']
print('\n=== 相对 DET 的变化 ===')
for m in METHODS[1:]:
    if m not in real.index:
        continue
    r = real.loc[m]
    d_cost = (r.oos_mean / det.oos_mean - 1) * 100
    d_tail = (r.oos_p95 / det.oos_p95 - 1) * 100
    prem = (r.plan_obj / det.plan_obj - 1) * 100
    print(f'{m:12s} 期望成本 {d_cost:+6.2f}%  P95 {d_tail:+6.2f}%  计划溢价 {prem:+6.2f}%')
    out[f'delta_{m}'] = {'oos': d_cost, 'p95': d_tail, 'premium': prem}

print('\n=== 副本汇总（n=10, 相对 DET）===')
for m in METHODS[1:]:
    g = rep[rep.method == m]
    gd = rep[rep.method == 'DET']
    if len(g) == 0:
        continue
    ratio = (g.oos_mean.values / gd.oos_mean.values - 1) * 100
    print(f'{m:12s} 均值 {ratio.mean():+6.2f}%  sd {ratio.std():6.2f}  '
          f'n_ok {g.ok.sum()}/{len(g)}  sfr_max {g.sfr.max():.3f}')
    out[f'rep_{m}'] = {'mean': float(ratio.mean()), 'sd': float(ratio.std()),
                       'n_ok': int(g.ok.sum()), 'sfr_max': float(g.sfr.max())}

sfr_det = rep[rep.method == 'DET'].sfr
print(f"\nDET 服务失败率（副本）: 均值 {sfr_det.mean():.3f} 范围 "
      f"{sfr_det.min():.3f}–{sfr_det.max():.3f}")
out['det_sfr_replicas'] = {'mean': float(sfr_det.mean()),
                           'min': float(sfr_det.min()), 'max': float(sfr_det.max())}

json.dump(out, open(f'{EXP}/paper_numbers_v24.json', 'w'), indent=1)
print(f'\n写出 {EXP}/paper_numbers_v24.json')
