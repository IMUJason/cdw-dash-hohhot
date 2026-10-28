# -*- coding: utf-8 -*-
"""Generate the V2.5 graphical abstract from verified paper numbers."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'cm',
                     'figure.dpi': 300, 'savefig.bbox': 'tight'})
NAVY, BLUE, TEAL, GOLD, RED, GREY = '#16324F', '#2F75B5', '#2A9D8F', '#E9A23B', '#B84A4A', '#52606D'

fig = plt.figure(figsize=(11.2, 6.0), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
fig.text(0.045, 0.94, 'Robust CDW network design under demand drift',
         fontsize=19, fontweight='bold', color=NAVY, va='top')
fig.text(0.045, 0.895, 'A disposal-capacity certificate, calibrated on three official years',
         fontsize=10.5, color=GREY, va='top')

# Left: network flow
fig.text(0.055, 0.80, 'THE NETWORK', fontsize=9, fontweight='bold', color=BLUE)
boxes = [
    (0.055, 0.57, 0.16, 0.13, 'Districts', '9 generation\nzones', '#EAF2F8'),
    (0.275, 0.57, 0.16, 0.13, 'Sorting', 'modular capacity\n+ buffers', '#E8F5F1'),
    (0.495, 0.57, 0.16, 0.13, 'Recycling', '4 plants\nopen / expand', '#EAF2F8'),
    (0.715, 0.57, 0.16, 0.13, 'Landfills', 'retire when\nexhausted', '#FCEDE8'),
]
for x, y, w, h, title, sub, fc in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.018',
                                linewidth=1.1, edgecolor=BLUE, facecolor=fc))
    fig.text(x + w/2, y + h*0.66, title, ha='center', va='center', fontsize=10,
             fontweight='bold', color=NAVY)
    fig.text(x + w/2, y + h*0.29, sub, ha='center', va='center', fontsize=7.8, color=GREY)
for x0, x1 in [(0.215, 0.275), (0.435, 0.495), (0.655, 0.715)]:
    ax.add_patch(FancyArrowPatch((x0, 0.635), (x1, 0.635), arrowstyle='-|>',
                                 mutation_scale=13, linewidth=1.3, color=BLUE))
ax.add_patch(FancyArrowPatch((0.135, 0.562), (0.795, 0.562), arrowstyle='-|>',
                             mutation_scale=12, linewidth=1.1, color=GOLD,
                             connectionstyle='arc3,rad=0.18'))
fig.text(0.465, 0.505, 'direct haul for inert spoil', fontsize=8, color=GOLD, ha='center')

# Certificate strip
ax.add_patch(FancyBboxPatch((0.055, 0.285), 0.82, 0.105, boxstyle='round,pad=0.012,rounding_size=0.012',
                            linewidth=0, facecolor=NAVY))
fig.text(0.075, 0.355, 'DISPOSAL-CAPACITY CERTIFICATE', fontsize=8.5, fontweight='bold', color='white')
fig.text(0.075, 0.315, r'$F_{\min}(D)=D-(1-\kappa)\min(A,D)\;\leq\;R$',
         fontsize=14, color='white')
fig.text(0.53, 0.315, r'$\varepsilon^*=D_{\max}/D_0-1$',
         fontsize=12, color='#BFE3FF')
fig.text(0.53, 0.285, r'$A=\min(S,K/\rho)$', fontsize=8.5, color='#BFE3FF')

# KPI cards
cards = [
    (0.055, 0.105, '49% → 0', 'service failure\nprobability mass', RED),
    (0.275, 0.105, '−11.7%', 'expected cost\non the real network', TEAL),
    (0.495, 0.105, '−49.6%', 'realized P95\ncost tail', BLUE),
    (0.715, 0.105, '16%', 'physical hedging ceiling\n(ε* = 0.1647)', GOLD),
]
for x, y, big, small, col in cards:
    fig.text(x, y + 0.075, big, fontsize=20, fontweight='bold', color=col, va='center')
    fig.text(x, y + 0.012, small, fontsize=8.2, color=GREY, va='bottom', linespacing=1.25)

fig.text(0.055, 0.025, 'Service is a constraint, not a price  ·  Solver optima throughout (MIP gap $\\leq 10^{-4}$)  ·  Threshold $\\approx$ 270 CNY/t',
         fontsize=8.5, color=GREY)
fig.savefig(os.path.join(OUT, 'graphical_abstract.pdf'))
fig.savefig(os.path.join(OUT, 'graphical_abstract.png'), dpi=300)
print('wrote', os.path.join(OUT, 'graphical_abstract.pdf'))
