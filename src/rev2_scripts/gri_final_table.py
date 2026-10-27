# -*- coding: utf-8 -*-
"""GRI 小规模验证终表（V2.4）。
来源：
  results_gri_small_verify.csv                基础 11 实例 x 5 种子（seed 0-4）
  results_gri_small_verify_inst{0,1,4,5,6,9}.csv  补充种子（seed 5-14）
  results_gri_refine_inst*.csv                精修
  results_gri_small_cplex_tight.csv           参照最优（mip_gap=1e-9）
口径：
  attains optimum := 与证明最优相对误差 < 1e-9
  certified gap   := 精确算术差（%）
"""
import glob, os
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{D}/experiments'

base = pd.read_csv(f'{EXP}/results_gri_small_verify.csv')
base['source'] = 'base'

extra_frames = []
for i in [0, 1, 4, 5, 6, 9]:
    fn = f'{EXP}/results_gri_small_verify_inst{i}.csv'
    if not os.path.exists(fn):
        continue
    d = pd.read_csv(fn)
    d = d[d.seed >= 5]
    d['source'] = 'extra'
    extra_frames.append(d)
extra = (pd.concat(extra_frames, ignore_index=True) if extra_frames
         else pd.DataFrame(columns=list(base.columns)))

runs = pd.concat([base, extra], ignore_index=True)
runs = runs.drop_duplicates(subset=['instance', 'seed'], keep='first')

refine_files = sorted(glob.glob(f'{EXP}/results_gri_refine_inst*.csv'))
refine = (pd.concat([pd.read_csv(f) for f in refine_files], ignore_index=True)
          if refine_files else pd.DataFrame(columns=['instance', 'gap_pct', 'exact_hit']))

ref = pd.read_csv(f'{EXP}/results_gri_small_cplex_tight.csv').set_index('seed')

rows = []
for i in sorted(runs.instance.unique()):
    g = runs[runs.instance == i]
    best_gap = float(g.gap_pct.min())
    hit_seeds = [str(int(s)) for s in g[g.exact_hit].seed]
    r = refine[refine.instance == i]
    if len(r):
        rg = float(r.gap_pct.iloc[0])
        if rg < best_gap:
            best_gap = rg
        if bool(r.exact_hit.iloc[0]):
            hit_seeds.append('refine')
    rows.append({
        'instance': 'real' if i == 0 else f'batch{i}',
        'n_seeds': len(g),
        'attains_opt': bool(len(hit_seeds) > 0),
        'hit_seeds': ','.join(hit_seeds),
        'best_gap_pct': round(best_gap, 5),
        'mean_gap_pct': round(float(g.gap_pct.mean()), 4),
        'worst_gap_pct': round(float(g.gap_pct.max()), 4),
        'gri_time_s': round(float(g.time_s.mean()), 0),
        'cplex_time_s': round(float(ref.loc[i, 'time_s']), 0),
    })

df = pd.DataFrame(rows)
df.to_csv(f'{EXP}/results_gri_final_table.csv', index=False)
print(df.to_string(index=False))
n_hit = int(df.attains_opt.sum())
print('')
print(f'达到精确最优的实例: {n_hit}/{len(df)}')
print(f'全部实例 best certified gap <= {df.best_gap_pct.max():.4f}%')
print(f'参照: CPLEX mip_gap=1e-9 全部证明最优 '
      f'({df.cplex_time_s.min():.0f}-{df.cplex_time_s.max():.0f}s)')
print(f'GRI 单次运行 {df.gri_time_s.min():.0f}-{df.gri_time_s.max():.0f}s')
