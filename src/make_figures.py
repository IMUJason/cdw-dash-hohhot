#!/usr/bin/env python3
"""JCP 论文图表（出版级，色盲安全，矢量 PDF）。"""
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = f'{BASE}/../hohhot case data/model'
if not os.path.isdir(MODEL):
    MODEL = BASE  # repo layout: experiments/ sits next to this file
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'DejaVu Sans', 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
# Okabe-Ito 色盲安全色
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
     'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'black': '#000000'}
METHODS = ['DET', 'SAA', 'DASH(0.05)', 'DASH(0.10)', 'DASH(0.16)']
MC = {'DET': C['red'], 'SAA': C['sky'], 'DASH(0.05)': C['green'], 'DASH(0.10)': C['blue'],
      'DASH(0.16)': C['purple']}

# ============ 图1：网络地图（真实坐标） ============
def fig1_network():
    """真实边界+英文标注版（高德行政区API）。"""
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    names = {'150102':'Xincheng','150103':'Huimin','150104':'Yuquan','150105':'Saihan',
             '150121':'Tumed Left Banner','150122':'Togtoh','150123':'Horinger','150124':'Qingshuihe','150125':'Wuchuan'}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_facecolor('#f7fbff')
    patches = []
    for ad in names:
        segs = json.load(open(f'/tmp/bnd_{ad}.json'))
        for seg in segs: patches.append(MplPolygon(np.array(seg), closed=True))
    ax.add_collection(PatchCollection(patches, facecolor='white', edgecolor='#4a4a4a', linewidth=0.7, zorder=1))
    for ad, en in names.items():
        arr = np.vstack(json.load(open(f'/tmp/bnd_{ad}.json')))
        ax.annotate(en, (arr[:,0].mean(), arr[:,1].mean()), ha='center', va='center', fontsize=7, color='#666666', style='italic')
    import csv as _csv
    nodes = {'J':{}, 'K':{}, 'L':{}}
    for r in _csv.DictReader(open(f'{BASE}/../hohhot case data/cdw_facilities_geocoded.csv', encoding='utf-8-sig')):
        if r['lng']:
            t = 'J' if '分拣' in r['type'] else ('K' if '资源化' in r['type'] else 'L')
            nodes[t][r['facility']] = (float(r['lng']), float(r['lat']))
    gen = {}
    for r in _csv.DictReader(open(f'{BASE}/../hohhot case data/nodes_generation_points.csv', encoding='utf-8-sig')):
        gen[r['节点ID']] = (float(r['lng']), float(r['lat']))
    for g,(x1,y1) in gen.items():
        js = sorted(nodes['J'].items(), key=lambda kv:(kv[1][0]-x1)**2+(kv[1][1]-y1)**2)[:2]
        for j,(x2,y2) in js: ax.plot([x1,x2],[y1,y2], color='#c8d7e3', lw=0.8, zorder=2)
    for j,(x1,y1) in nodes['J'].items():
        for k,(x2,y2) in nodes['K'].items():
            if (x2-x1)**2+(y2-y1)**2 < 0.35: ax.plot([x1,x2],[y1,y2], color='#c8d7e3', lw=0.8, zorder=2)
    for k,(x1,y1) in nodes['K'].items():
        ls = sorted(nodes['L'].items(), key=lambda kv:(kv[1][0]-x1)**2+(kv[1][1]-y1)**2)[:2]
        for l,(x2,y2) in ls: ax.plot([x1,x2],[y1,y2], color='#c8d7e3', lw=0.8, zorder=2)
    for cat,marker,size,color,label in [('G','s',32,'#000000','Generation districts (9)'),
                                         ('J','^',40,'#009E73','Sorting centers (8)'),
                                         ('K','D',48,'#0072B2','Recycling plants (4)'),
                                         ('L','v',42,'#D55E00','Landfills (6)')]:
        pts = gen.values() if cat=='G' else nodes[cat].values()
        xs=[v[0] for v in pts]; ys=[v[1] for v in pts]
        ax.scatter(xs, ys, s=size, c=color, marker=marker, zorder=4, label=label, edgecolors='white', linewidths=0.7)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=7.5)
    ax.set_title('CDW reverse logistics network of Hohhot: administrative boundaries,\nfour echelons, and facility retirement (real coordinates)')
    plt.savefig(f'{FIG}/fig1_network.pdf'); plt.close()
    print('fig1 done')

# ============ 图2：真实实例样本外成本分布（小提琴） ============
def fig2_oos_violin():
    import sys
    sys.path.insert(0, MODEL)
    os.chdir(MODEL)
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval
    inst = json.load(open(f'{MODEL}/instance_v1.json', encoding='utf-8'))
    test = gen_scenarios(inst, 200, seed=20260901)
    data, labels, cols = [], [], []
    for m in METHODS:
        fname = m.replace('DASH', 'DRO')
        cfg = json.load(open(f'{MODEL}/experiments/config_real_{fname}.json'))
        costs, probs, um, rc = oos_eval(inst, cfg, test)
        data.append(costs/1e9); labels.append(m.replace('DASH', 'DASH\n')); cols.append(MC[m])
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    vp = ax.violinplot(data, showmedians=True, widths=0.75)
    for pc, col in zip(vp['bodies'], cols):
        pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor(col)
    for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[part].set_color('0.3'); vp[part].set_linewidth(0.8)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel('Realized total cost (billion CNY)')
    ax.set_title('Out-of-sample cost distributions, real Hohhot instance (200 shared scenarios)')
    plt.savefig(f'{FIG}/fig2_oos_violin.pdf'); plt.close()
    print('fig2 done')

# ============ 图3：政策拨盘 ε（成本-失望-P95-CO2） ============
def fig3_dial():
    main = pd.read_csv(f'{MODEL}/experiments/results_main_v22.csv')
    real = main[main.instance == 'real'].set_index('method')
    eps_labels = ['SAA\n(ε=0)', '0.05', '0.10', '0.16']
    eps_keys = ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    x = np.arange(4)
    plan = [real.loc[k, 'plan_obj']/1e9 for k in eps_keys]
    oos = [real.loc[k, 'oos_mean']/1e9 for k in eps_keys]
    p95 = [real.loc[k, 'oos_p95']/1e9 for k in eps_keys]
    ax.plot(x, plan, 'o-', color=C['red'], ms=4, lw=1.2, label='Planned cost')
    ax.plot(x, oos, 's-', color=C['blue'], ms=4, lw=1.2, label='Realized mean')
    ax.plot(x, p95, '^--', color=C['purple'], ms=4, lw=1.2, label='Realized P95')
    ax.set_xticks(x); ax.set_xticklabels(eps_labels); ax.set_xlabel('Hedging level ε')
    ax.set_ylabel('Billion CNY'); ax.legend(frameon=False, loc='upper left')
    plt.savefig(f'{FIG}/fig3_dial.pdf'); plt.close()
    print('fig3 done')

# ============ 图4：LOO 回测后悔 ============
def fig4_loo():
    loo = pd.read_csv(f'{MODEL}/experiments/results_loo_v22.csv')
    loo['method'] = loo['method'].str.replace('DRO', 'DASH')
    piv = loo.pivot(index='method', columns='holdout_year', values='regret')
    order = ['DET', 'SAA', 'DASH(0.05)', 'DASH(0.10)', 'DASH(0.16)']
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(order)); w = 0.26
    yr_colors = {1: C['sky'], 2: C['red'], 3: C['green']}
    yr_names = {1: '2023 held out', 2: '2024 peak held out', 3: '2025 collapse held out'}
    for i, yr in enumerate([1, 2, 3]):
        vals = piv[yr]/1e9
        ax.bar(x + (i-1)*w, vals, w, color=yr_colors[yr], label=yr_names[yr])
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([m.replace('DASH', 'DASH\n') for m in order])
    ax.set_ylabel('Regret: realized − planned (billion CNY)')
    ax.set_title('Leave-one-year-out backtests: peak-year protection vs collapse-year premium')
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower left')
    plt.savefig(f'{FIG}/fig4_loo.pdf'); plt.close()
    print('fig4 done')

# ============ 图5：灵敏度（双面板：tc×rev 热图 + 惩罚条件曲线） ============
def fig5_sensitivity():
    sens = pd.read_csv(f'{MODEL}/experiments/results_grid_v22.csv')
    pen = pd.read_csv(f'{MODEL}/experiments/results_penalty_v22.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw={'width_ratios': [1.05, 1], 'wspace': 0.55})
    # (a) DET minus best alternative (SAA, DASH(0.10)): positive everywhere = DET never cheapest
    det = sens[sens.method == 'DET'].set_index(['tc', 'rev']).oos_mean
    alt = sens[sens.method != 'DET'].groupby(['tc', 'rev']).oos_mean.min()
    diff = (det - alt).unstack() / 1e6   # M CNY
    ax = axes[0]
    im = ax.imshow(diff.values/1e3, origin='lower', aspect='auto', cmap='OrRd')
    ax.set_xticks(range(len(diff.columns)), [int(v) for v in diff.columns])
    ax.set_yticks(range(len(diff.index)), [f'{v:.1f}' for v in diff.index])
    ax.set_xlabel('Recycled product revenue (CNY/t)')
    ax.set_ylabel('Unit transport cost (CNY/t$\\cdot$km)')
    ax.set_title('(a) DET minus best alternative (billion CNY)', fontsize=8.5)
    for i in range(len(diff.index)):
        for j in range(len(diff.columns)):
            ax.text(j, i, f'{diff.values[i, j]/1e3:.2f}', ha='center', va='center', fontsize=6.5)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    # (b) penalty condition
    ax = axes[1]
    for m, col in [('DET', C['red']), ('SAA', C['sky']), ('DRO(0.10)', C['blue'])]:
        sub = pen[pen.method == m]
        lbl = m.replace('DRO', 'DASH')
        ax.plot(sub.penalty, sub.oos_mean/1e9, 'o-', ms=4, lw=1.3, color=col, label=lbl)
    ax.set_xscale('log')
    ax.set_xlabel('Unserved-waste penalty $\\pi$ (CNY/t)')
    ax.set_ylabel('Realized cost (billion CNY)')
    ax.legend(frameon=False, fontsize=7.5)
    ax.axvline(270, color='0.4', ls=':', lw=1.0)
    ax.text(272, ax.get_ylim()[1]*0.97, r'$\pi^{*}\approx$270 CNY/t', va='top', fontsize=7, color='0.3')
    ax.set_title('(b) The penalty condition', fontsize=8.5)
    plt.savefig(f'{FIG}/fig5_sensitivity.pdf'); plt.close()
    print('fig5 done')

if __name__ == '__main__':
    import sys as _s
    if '--skip-network' not in _s.argv:
        fig1_network()
    fig2_oos_violin()
    fig3_dial()
    fig4_loo()
    fig5_sensitivity()
    print('全部图表完成 → paper/figures/')
