"""V2.2 主表的统计检验：replica 层 Wilcoxon（Holm 校正）+ Cliff's delta。"""
import os
import json, sys, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)

df = pd.read_csv(f'{D}/experiments/results_main_v22.csv')
rep = df[df.instance != 'real']
real = df[df.instance == 'real'].set_index('method')
METHODS = ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']


def cliff(rep_x, rep_y):
    gt = sum(1 for a in rep_x for b in rep_y if a > b)
    lt = sum(1 for a in rep_x for b in rep_y if a < b)
    return (gt - lt) / (len(rep_x) * len(rep_y))


det_oos = rep[rep.method == 'DET'].oos_mean.values
print('=== replica 层（n=10）Wilcoxon vs DET，Holm 校正 ===')
raw = []
for m in METHODS:
    x = rep[rep.method == m].oos_mean.values
    stat, p = wilcoxon(x, det_oos)
    raw.append((m, p, stat, cliff(x, det_oos), x.mean(), x.std()))
order = sorted(range(len(raw)), key=lambda i: raw[i][1])
adj = [0.0] * len(raw)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, (len(raw) - rank) * raw[i][1])
    adj[i] = min(run, 1.0)
for (m, p, stat, d, mu, sd), pa in zip(raw, adj):
    print(f'{m:10s} oos={mu/1e9:.4f}±{sd/1e9:.4f}B  p={p:.4f}  Holm p={pa:.4f}  '
          f'Cliff d={d:+.2f}  {"显著" if pa < 0.05 else "不显著"}')

print('\n=== real 实例 ===')
det = real.loc['DET']
for m in METHODS:
    r = real.loc[m]
    print(f'{m:10s} plan={r.plan_obj/1e9:.4f}B  oos={r.oos_mean/1e9:.4f}B '
          f'({(r.oos_mean-det.oos_mean)/det.oos_mean*100:+.2f}%)  p95={r.oos_p95/1e9:.4f}B '
          f'({(r.oos_p95-det.oos_p95)/det.oos_p95*100:+.2f}%)  disc={r.disc:.3f}  sfr={r.sfr:.4f}  '
          f'recyc={r.recycled/1e6:.3f}Mt  unmet={r.unmet_mean:,.0f}t')

print('\n=== P95 降幅（real）===')
for m in METHODS:
    r = real.loc[m]
    print(f'  {m:10s} {(1-r.oos_p95/det.oos_p95)*100:5.1f}%')

print('\n=== 计划成本溢价（vs SAA, real）===')
saa = real.loc['SAA']
for m in METHODS:
    r = real.loc[m]
    print(f'  {m:10s} {(r.plan_obj-saa.plan_obj)/saa.plan_obj*100:+5.1f}%  '
          f'(实现成本 {(r.oos_mean-saa.oos_mean)/saa.oos_mean*100:+.1f}%)')

print('\n=== 失望率梯度（replica mean±sd）===')
for m in ['DET'] + METHODS:
    g = rep[rep.method == m]
    print(f'  {m:10s} {g.disc.mean():.3f}±{g.disc.std():.3f}')

print('\n=== 回收量（real）===')
for m in ['DET'] + METHODS:
    r = real.loc[m]
    print(f'  {m:10s} {r.recycled/1e6:.3f}Mt')

print('\n=== 求解时间（real）===')
for m in ['DET'] + METHODS:
    r = real.loc[m]
    print(f'  {m:10s} {r.time_s:.0f}s')
