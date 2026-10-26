# -*- coding: utf-8 -*-
"""GRI 小规模验证终表：合并基础 5 种子 + 补充种子 + 精修。
输出 experiments/results_gri_final_table.csv 与控制台汇总。
口径：exact_hit = 与证明最优相对误差 < 1e-9；certified gap = 精确算术差。
"""
import glob, os
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{D}/experiments'

base = pd.read_csv(f'{EXP}/results_gri_small_verify.csv')
extra_files = glob.glob(f'{EXP}/results_gri_small_verify_inst*_extra.csv')
extra = pd.concat([pd.read_csv(f) for f in extra_files], ignore_index=True) if extra_files else pd.DataFrame()
refine_files = glob.glob(f'{EXP}/results_gri_refine_inst*.csv')
refine = pd.concat([pd.read_csv(f) for f in refine_files], ignore_index=True) if refine_files else pd.DataFrame()

runs = pd.concat([base, extra], ignore_index=True)
rows = []
for i in sorted(runs.instance.unique()):
    g = runs[runs.instance == i]
    r = refine[refine.instance == i]
    best_gap = g.gap_pct.min()
    hit_seeds = sorted(g[g.exact_hit].seed.tolist())
    if len(r):
        best_gap = min(best_gap, r.gap_pct.iloc[0])
        if bool(r.exact_hit.iloc[0]):
            hit_seeds.append('refine')
    rows.append({'instance': 'real' if i == 0 else f'batch{i}',
                 'n_seeds': len(g),
                 'exact_hits': len([s for s in hit_seeds if s != 'refine']),
                 'hit_seeds': ','.join(map(str, hit_seeds)),
                 'best_gap_pct': round(best_gap, 5),
                 'mean_gap_pct': round(g.gap_pct.mean(), 4),
                 'time_per_run_s': round(g.time_s.mean(), 0)})
df = pd.DataFrame(rows)
df.to_csv(f'{EXP}/results_gri_final_table.csv', index=False)
print(df.to_string(index=False))
n_hit = (df.exact_hits > 0).sum()
print(f'\n精确命中实例: {n_hit}/{len(df)}；全部实例 best gap ≤ {df.best_gap_pct.max():.4f}%')
print(f'CPLEX 基线: 见 results_gri_small_cplex.csv（单实例 22–63s）')
