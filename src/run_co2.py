#!/usr/bin/env python3
"""Bounded cradle-to-gate carbon accounting (§6.3 of the paper).

Computes, for each method configuration on the real instance:
  - transport work (tonne-km) and CO2 (factor 0.10 kg CO2/t.km, range 0.08-0.12)
  - plant intake (recycled / yield) for the processing term (2.2-22 kg CO2e/t,
    process-LCA anchor: Chen et al. 2025, Waste Management, doi:10.1016/j.wasman.2025.115107)
  - substitution credit (0-10 kg CO2e/t saleable output,
    Hasheminezhad et al. 2024, Sci. Total Environ., doi:10.1016/j.scitotenv.2024.175310)
Output: results/results_co2.csv
"""
import json, os, glob
import numpy as np
from scipy.optimize import linprog

BASE = os.path.dirname(os.path.abspath(__file__))
EMIS = 0.10

def solve_tkm(inst, xcfg, d_scenario):
    """Min-cost recourse LP returning (cost, tonne-km, unmet, recycled)."""
    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting']); RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)
    idx = {}; n = 0
    for i in GEN:
        for j in SORT:
            for t in range(1, T+1): idx[('y', i, j, t)] = n; n += 1
    for i in GEN:
        for l in LAND:
            for t in range(1, T+1): idx[('yd', i, l, t)] = n; n += 1
    for j in SORT:
        for k in RECY:
            for t in range(1, T+1): idx[('w', j, k, t)] = n; n += 1
    for j in SORT:
        for l in LAND:
            for t in range(1, T+1): idx[('q', j, l, t)] = n; n += 1
    for k in RECY:
        for l in LAND:
            for t in range(1, T+1): idx[('v', k, l, t)] = n; n += 1
    for j in SORT:
        for t in range(1, T+1): idx[('s', j, t)] = n; n += 1
    for i in GEN:
        for t in range(1, T+1): idx[('u', i, t)] = n; n += 1
    c = np.zeros(n)
    arc_d = {}
    for (key, *a), var in idx.items():
        if key == 'y': c[var] = P['transport_cost']*inst['dist']['a'][a[0]][a[1]] + inst['sorting'][a[1]]['op_cost']; arc_d[var] = inst['dist']['a'][a[0]][a[1]]
        elif key == 'yd': c[var] = P['transport_cost']*inst['dist']['a2'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']; arc_d[var] = inst['dist']['a2'][a[0]][a[1]]
        elif key == 'w': c[var] = P['transport_cost']*inst['dist']['b'][a[0]][a[1]] + inst['recycling'][a[1]]['op_cost'] - P.get('recycle_revenue', 0.0)*phi; arc_d[var] = inst['dist']['b'][a[0]][a[1]]
        elif key == 'q': c[var] = P['transport_cost']*inst['dist']['c'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']; arc_d[var] = inst['dist']['c'][a[0]][a[1]]
        elif key == 'v': c[var] = P['transport_cost']*inst['dist']['e'][a[0]][a[1]] + inst['landfill'][a[1]]['op_cost']; arc_d[var] = inst['dist']['e'][a[0]][a[1]]
        elif key == 's': c[var] = P['hold_cost']
        elif key == 'u': c[var] = P['unmet_penalty']
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    def eqrow(cf, rhs):
        r = np.zeros(n)
        for k_, cv in cf: r[idx[k_]] += cv
        A_eq.append(r); b_eq.append(rhs)
    def ubrow(cf, rhs):
        r = np.zeros(n)
        for k_, cv in cf: r[idx[k_]] += cv
        A_ub.append(r); b_ub.append(rhs)
    for i in GEN:
        for t in range(1, T+1):
            eqrow([(('y', i, j, t), 1.0) for j in SORT] + [(('yd', i, l, t), 1.0) for l in LAND] + [(('u', i, t), 1.0)], d_scenario[i][t])
    for j in SORT:
        for t in range(1, T+1):
            prev = [(('s', j, t-1), 1.0)] if t > 1 else []
            eqrow(prev + [(('y', i, j, t), 1.0) for i in GEN] + [(('w', j, k, t), -1.0) for k in RECY] + [(('q', j, l, t), -1.0) for l in LAND] + [(('s', j, t), -1.0)], 0.0)
            outs = [(('w', j, k, t), 1.0) for k in RECY]; qs = [(('q', j, l, t), 1.0) for l in LAND]
            ubrow([(x, (1-rho)*cv) for x, cv in outs] + [(x, -rho*cv) for x, cv in qs], 0.0)
            ubrow([(x, (1-rho)*cv) for x, cv in outs] + [(x, -rho*cv) for x, cv in qs], 0.0)
    for k in RECY:
        for t in range(1, T+1):
            eqrow([(('w', j, k, t), (1-phi)) for j in SORT] + [(('v', k, l, t), -1.0) for l in LAND], 0.0)
    zJ_cfg = xcfg.get('zJ', {})
    for j in SORT:
        zj_t = zJ_cfg.get(j, [0]*T)
        for t in range(1, T+1):
            ubrow([(('y', i, j, t), 1.0) for i in GEN], (inst['sorting'][j]['cap_t'] + inst['sorting'][j].get('expand_step_t', 1e5)*zj_t[t-1])*xcfg['xJ'][j][t-1])
            ubrow([(('s', j, t), 1.0)], inst['sorting'][j]['buffer_cap_t'] + inst['sorting'][j].get('expand_step_t', 1e5)*zj_t[t-1]*0.2)
    for k in RECY:
        for t in range(1, T+1):
            ubrow([(('w', j, k, t), 1.0) for j in SORT], inst['recycling'][k]['cap_t']*xcfg['xK'][k][t-1] + inst['recycling'][k]['expand_step_t']*xcfg['zK'][k][t-1])
            if xcfg['xK'][k][t-1] == 1:
                r = np.zeros(n)
                for j in SORT: r[idx[('w', j, k, t)]] -= 1.0
                A_ub.append(r); b_ub.append(-inst['recycling'][k]['min_run_t'])
    for l in LAND:
        for t in range(1, T+1):
            used = xcfg['xL'][l][t-1] and xcfg['rL'][l][t-1] == 0
            cap = inst['landfill'][l]['cap_t'] if used else 0.0
            ubrow([(('q', j, l, t), 1.0) for j in SORT] + [(('yd', i, l, t), 1.0) for i in GEN] + [(('v', k, l, t), 1.0) for k in RECY], cap)
        cum = []
        for j in SORT:
            for t in range(1, T+1): cum.append((('q', j, l, t), 1.0))
        for i in GEN:
            for t in range(1, T+1): cum.append((('yd', i, l, t), 1.0))
        for k in RECY:
            for t in range(1, T+1): cum.append((('v', k, l, t), 1.0))
        ubrow(cum, inst['landfill'][l]['cap_t'])
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    if res.status != 0:
        return None
    x = res.x
    tkm = sum(x[var]*arc_d[var] for var in arc_d)
    unmet = sum(x[idx[('u', i, t)]] for i in GEN for t in range(1, T+1))
    recyc = sum(x[idx[('w', j, k, t)]] for j in SORT for k in RECY for t in range(1, T+1))
    return res.fun, tkm, unmet, recyc

if __name__ == '__main__':
    import sys
    inst = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    for f in sys.argv[1:]:
        cfg = json.load(open(f))
        T = inst['meta']['periods']
        d_mean = {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in inst['demand']}
        out = solve_tkm(inst, cfg, d_mean)
        if out is None:
            print(f, 'infeasible'); continue
        cost, tkm, unmet, recyc = out
        phi = inst['params']['product_yield']
        print(f"{f}: tkm={tkm/1e6:.1f}M  co2_lo={tkm*0.08/1e3:.1f}kt  co2_mid={tkm*0.10/1e3:.1f}kt  co2_hi={tkm*0.12/1e3:.1f}kt  intake={recyc/phi/1e6:.2f}Mt  recycled={recyc/1e6:.2f}Mt  unmet={unmet/1e6:.2f}Mt")
