"""V2.2 成本×收益网格的最优方法统计：每格中哪个方法最优。"""
import pandas as pd
import numpy as np
D = '/Users/cjx-ms/Documents/我的坚果云/科研/论文/2026 cdw 0825/hohhot case data/model'
df = pd.read_csv(f'{D}/experiments/results_grid_v22.csv')
ok = df[df.ok]
piv = ok.pivot_table(index=['tc', 'rev'], columns='method', values='oos_mean')
print('=== 每格最优方法（按实现成本）===')
best = piv.idxmin(axis=1)
print(best.value_counts().to_string())
print()
print('DET 最优的格:', list(best[best == 'DET'].index))
print()
print('=== 实现成本范围 ===')
for m in ['DET', 'SAA', 'DRO(0.10)']:
    g = ok[ok.method == m]
    print(f'{m:10s} {g.oos_mean.min()/1e9:.3f}--{g.oos_mean.max()/1e9:.3f}B  '
          f'disc {g.disc.min():.3f}--{g.disc.max():.3f}  sfr {g.sfr.mean():.3f}')
print()
print('=== SAA vs DASH(0.10) 逐格对比 ===')
comp = piv.dropna(subset=['SAA', 'DRO(0.10)'])
win = (comp['DRO(0.10)'] < comp['SAA']).sum()
print(f'DASH(0.10) 实现成本低于 SAA 的格数: {win}/{len(comp)}')
disc_piv = ok.pivot_table(index=['tc', 'rev'], columns='method', values='disc')
dwin = (disc_piv['DRO(0.10)'] < disc_piv['SAA']).sum()
print(f'DASH(0.10) 失望率低于 SAA 的格数: {dwin}/{len(disc_piv)}')
print(f'DASH(0.10) 失望率范围 {disc_piv["DRO(0.10)"].min():.3f}--{disc_piv["DRO(0.10)"].max():.3f}, '
      f'SAA {disc_piv["SAA"].min():.3f}--{disc_piv["SAA"].max():.3f}')
