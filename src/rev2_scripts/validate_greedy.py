# -*- coding: utf-8 -*-
"""贪心分配器验证（V2.4 门槛建议验）：真实网 + 10 情景训练集。
  A. CPLEX 最优配置：贪心必须成功且 UB ≥ LP 值（逐情景）
  B. 随机配置 N=50：可行性语义对齐 + UB≥LP 不变量 + 偏差分布
"""
import json, sys, os, random
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, extract_xconfig_cpx, _cfg_from_xconfig
from alns import solve_second_stage_lp_ex, repair_monotone, clone, initial_solution
from greedy_recourse import greedy_recourse

inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)

# ---- A. CPLEX 最优配置 ----
mdl, sol, V = build_and_solve_saa(inst, train, 'gv_opt', retire_lstar=True, t_star=2,
                                  require_cover=True, time_limit=900, log=False)
opt = sol.objective_value
cfg = _cfg_from_xconfig(inst, extract_xconfig_cpx(inst, V))
print(f'CPLEX exact optimum (cover): {opt:.2f}')

lp_vals, gd_vals = [], []
for sc in train:
    lpv, _, _ = solve_second_stage_lp_ex(inst, cfg, sc['d'], require_cover=True)
    gdv, unm, _ = greedy_recourse(inst, cfg, sc['d'], require_cover=True)
    lp_vals.append(lpv); gd_vals.append(gdv)
lp_tot = sum(p * v for p, v in zip([s['prob'] for s in train], lp_vals))
gd_tot = sum(p * v for p, v in zip([s['prob'] for s in train], gd_vals))
print(f'A. 最优配置: LP={lp_tot:.1f}  greedyUB={gd_tot:.1f}  UB-LP={gd_tot-lp_tot:.1f} '
      f'({(gd_tot/lp_tot-1)*100:.2f}%)  覆盖情景 {sum(1 for g in gd_vals if g is not None)}/10')
viol = [i for i, (g, l) in enumerate(zip(gd_vals, lp_vals))
        if g is not None and l is not None and g < l - 1e-6]
print(f'   UB<LP 违例: {len(viol)} （必须为 0）')

# ---- B. 随机配置 ----
rng = random.Random(7)
def rand_cfg():
    c = clone(initial_solution(inst))
    for _ in range(rng.randint(3, 12)):
        r = rng.random()
        if r < 0.3:
            j = rng.choice(list(inst['sorting'])); c['xJ'][j] = [rng.randint(0, 1)] * 3
        elif r < 0.5:
            k = rng.choice(list(inst['recycling'])); c['xK'][k] = [rng.randint(0, 1)] * 3
        elif r < 0.65:
            l = rng.choice(list(inst['landfill'])); c['xL'][l] = [rng.randint(0, 1)] * 3
        elif r < 0.85:
            j = rng.choice(list(inst['sorting']) + list(inst['recycling']))
            key = 'zJ' if j in inst['sorting'] else 'zK'
            c[key][j] = [rng.randint(0, 4)] * 3
        else:
            l = rng.choice(list(inst['landfill'])); c['rL'][l] = [rng.randint(0, 1)] * 3
    return repair_monotone(c, inst)

n_both, n_gnone_lpok, n_both_none, n_gok_lpnone = 0, 0, 0, 0
ub_gaps = []
viol_b = 0
for it in range(50):
    c = rand_cfg()
    sc = train[rng.randrange(len(train))]
    lpv, _, _ = solve_second_stage_lp_ex(inst, c, sc['d'], require_cover=False)
    gdv, _, _ = greedy_recourse(inst, c, sc['d'], require_cover=False)
    if lpv is None and gdv is None:
        n_both_none += 1
    elif lpv is not None and gdv is None:
        n_gnone_lpok += 1          # 贪心过紧（可接受但计数）
    elif lpv is None and gdv is not None:
        n_gok_lpnone += 1          # 严重：贪心声称可行而 LP 不可行 → 需查明
    else:
        n_both += 1
        if gdv < lpv - 1e-6:
            viol_b += 1
        else:
            ub_gaps.append((gdv - lpv) / lpv * 100)
print(f'B. 50 随机配置×随机情景: 双可行={n_both} 贪心过紧={n_gnone_lpok} '
      f'双不可行={n_both_none} 贪心虚报可行={n_gok_lpnone}')
if ub_gaps:
    g = np.array(ub_gaps)
    print(f'   UB−LP 偏差: 均值={g.mean():.2f}%  中位={np.median(g):.2f}%  '
          f'P90={np.percentile(g, 90):.2f}%  max={g.max():.2f}%  违例={viol_b}（必须 0）')
