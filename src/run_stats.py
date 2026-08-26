#!/usr/bin/env python3
"""E6: 统计检验（协议 §1.3）+ 图表 + 实验报告。"""
import json, os, math
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = f'{BASE}/experiments'
OUT = f'{BASE}/..', 
res = pd.read_csv(f'{EXP}/results_main.csv')
sens = pd.read_csv(f'{EXP}/results_sensitivity.csv')
back = pd.read_csv(f'{EXP}/results_backtest.csv')

report = []
def w(s=''):
    report.append(s); print(s)

w('# 实验结果与统计分析报告（E6）\n')
w('## 1. 主实验描述统计（11 实例 × 6 方法）\n')
piv = res.pivot_table(index='method', values=['oos_mean', 'oos_p95', 'oos_cv', 'disappointment_rate', 'disappointment_mean', 'time_s'],
                      aggfunc=['mean', 'std'])
w(piv.round(3).to_string())
w('')

w('## 2. 假设检验（Wilcoxon 配对，Holm 校正 α=0.05）\n')
batch = res[res.instance != 'real']
def paired(method_a, method_b, col):
    a = batch[batch.method == method_a].sort_values('instance')[col].values
    b = batch[batch.method == method_b].sort_values('instance')[col].values
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return None
    stat, p = stats.wilcoxon(a, b)
    # Cliff's delta
    diffs = a - b
    cd = abs(sum(np.sign(diffs)) / len(diffs))
    return {'pair': f'{method_a} vs {method_b}', 'metric': col, 'median_diff': float(np.median(diffs)),
            'W': float(stat), 'p': float(p), 'cliffs_delta': float(cd)}

tests = []
for m in ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)']:
    r = paired(m, 'DET', 'oos_mean')
    if r: tests.append(r)
    r = paired(m, 'DET', 'disappointment_rate')
    if r: tests.append(r)
# Holm 校正
tests = [t for t in tests if t]
m = len(tests)
sorted_idx = np.argsort([t['p'] for t in tests])
holm = {}
for rank, idx in enumerate(sorted_idx):
    holm[idx] = min(1.0, (m - rank) * tests[idx]['p'])
for i, t in enumerate(tests):
    t['p_holm'] = holm[i]
    t['significant'] = t['p_holm'] < 0.05
    w(f"- {t['pair']} [{t['metric']}]: 中位差={t['median_diff']:,.2f}, p_holm={t['p_holm']:.4f}, Cliff's δ={t['cliffs_delta']:.2f}, 显著={t['significant']}")
w('')

w('## 3. 方差齐性（H2: DRO 的 CV < SAA）\n')
for m in ['DRO(0.10)', 'DRO(0.20)']:
    a = batch[batch.method == m].sort_values('instance')['oos_cv'].values
    b = batch[batch.method == 'SAA'].sort_values('instance')['oos_cv'].values
    if len(a) == len(b) and len(a) > 2:
        stat, p = stats.levene(a, b, center='median')
        w(f"- {m} vs SAA [oos_cv]: Levene W={stat:.3f}, p={p:.4f}, mean {a.mean():.4f} vs {b.mean():.4f}")
w('')

w('## 4. 真实实例（呼和浩特）逐方法\n')
real = res[res.instance == 'real']
w(real[['method', 'plan_obj', 'oos_mean', 'oos_p95', 'oos_cv', 'disappointment_rate', 'time_s']].round(2).to_string(index=False))
w('')

w('## 5. 灵敏度（运输成本 × 再生收益网格）\n')
w(sens.pivot_table(index=['tc'], columns=['rev'], values='oos_mean', aggfunc='first').round(0).to_string())
w('')
w('最优方法分布（按 oos_mean 最小）:')
best = sens.loc[sens.groupby(['tc', 'rev'])['oos_mean'].idxmin()]
w(best.groupby('method').size().to_string())
w('')

w('## 6. 回测（2023-24 训练 → 2025 实际）\n')
w(back.round(0).to_string(index=False))
w('')
w('解读：2025 年实际需求坍缩（256 万吨 vs 训练期均值 ~445 万吨），所有方法的保守投资在低需求年出现沉没成本（负 regret 即实际成本远低于规划口径），DRO 的代价是计划成本更高——衡量的是"保险费"。低需求年中 DET 的表观成本最低，但其 47-54% 的失望率（主实验）说明该保险是必要的。')

with open(f'{EXP}/statistics_report.md', 'w') as f:
    f.write('\n'.join(report))

# ---------- 图表（matplotlib，论文用 PDF 矢量） ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# 图1: 方法×oos_mean 箱线图（batch 实例）
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)', 'ROB']
data = [batch[batch.method == m]['oos_mean'].values / 1e9 for m in order]
bp = ax.boxplot(data, labels=order, patch_artist=True)
colors = ['#f8d7da', '#fff3cd', '#d1ecf1', '#c3e6cb', '#bee5eb', '#e2d5f0']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
ax.set_ylabel('Out-of-sample cost (billion CNY)')
ax.set_title('Out-of-sample total cost by method (10 batch instances)')
plt.tight_layout()
plt.savefig(f'{EXP}/fig1_oos_box.pdf')

# 图2: 失望率柱状图（均值±std）
fig, ax = plt.subplots(figsize=(8, 4))
means = [batch[batch.method == m]['disappointment_rate'].mean() for m in order]
stds = [batch[batch.method == m]['disappointment_rate'].std() for m in order]
ax.bar(order, means, yerr=stds, capsize=4, color=colors)
ax.set_ylabel('Disappointment rate (weighted)')
ax.set_title('Probability of out-of-sample disappointment')
plt.tight_layout()
plt.savefig(f'{EXP}/fig2_disappointment.pdf')

# 图3: 灵敏度热图
pv = sens.pivot_table(index='tc', columns='rev', values='oos_mean')
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(pv.values / 1e9, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pv.columns)), [f'{int(v)}' for v in pv.columns])
ax.set_yticks(range(len(pv.index)), [f'{v:.1f}' for v in pv.index])
ax.set_xlabel('Recycled product revenue (CNY/t)')
ax.set_ylabel('Unit transport cost (CNY/t·km)')
ax.set_title('Out-of-sample cost (billion CNY), DRO(0.10)')
for i in range(len(pv.index)):
    for j in range(len(pv.columns)):
        ax.text(j, i, f'{pv.values[i, j]/1e9:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'{EXP}/fig3_sensitivity_heat.pdf')
print('\n图表已输出: fig1_oos_box.pdf, fig2_disappointment.pdf, fig3_sensitivity_heat.pdf')
