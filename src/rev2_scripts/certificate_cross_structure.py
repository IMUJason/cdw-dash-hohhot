# -*- coding: utf-8 -*-
"""跨结构证书盲验：改变 D0/S/R 比例而非仅同城参数扰动。
每种结构先由公式给出 ε*，再用 CPLEX 在 ε*±0.01 检查可行性。
"""
import json, os, sys, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, dash_scenarios
from gri_alns import initial_full, repair_monotone
from eps_formula_v2 import predict

BASE = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))


def variant(name, demand_scale=1.0, sort_scale=1.0, landfill_scale=1.0):
    inst = json.loads(json.dumps(BASE))
    for i in inst['demand']:
        for t in ('1', '2', '3'):
            inst['demand'][i]['d'][t] *= demand_scale
    for j in inst['sorting']:
        inst['sorting'][j]['cap_t'] *= sort_scale
    for l in inst['landfill']:
        inst['landfill'][l]['cap_t'] *= landfill_scale
    return name, inst


def check(name, inst):
    r = predict(inst)
    eps_star = r['eps_pred']
    cfg = initial_full(inst)
    for j in inst['sorting']:
        cfg['zJ'][j] = [4] * 3
    for k in inst['recycling']:
        cfg['zK'][k] = [4] * 3
    cfg = repair_monotone(cfg, inst)
    rows = []
    for eps in [max(0.0, eps_star - 0.01), eps_star + 0.01]:
        train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
        full = dash_scenarios(inst, train, eps)
        t0 = time.time()
        mdl, sol, V = build_and_solve_saa(inst, full, f'xs_{name}_{eps:.3f}',
                                          retire_lstar=True, t_star=2, log=False,
                                          require_cover=True, time_limit=600)
        for key in ['xJ', 'xK', 'xL', 'zJ', 'zK', 'rL']:
            for f in cfg[key]:
                for t in range(1, 4):
                    mdl.add_constraint(V[key][f, t] == cfg[key][f][t - 1])
        sol2 = mdl.solve()
        rows.append({'structure': name, 'eps_pred': eps_star, 'eps': eps,
                     'cplex_feasible': sol2 is not None,
                     'time_s': round(time.time() - t0, 1)})
        print(f'{name:18s} pred={eps_star:.4f} eps={eps:.3f} '
              f'{"OK" if sol2 else "INFEAS"}', flush=True)
    return rows


if __name__ == '__main__':
    specs = [variant('baseline'),
             variant('sorting_plus20', sort_scale=1.2),
             variant('demand_minus20', demand_scale=0.8),
             variant('landfill_minus10', landfill_scale=0.9)]
    rows = [r for name, inst in specs for r in check(name, inst)]
    pd.DataFrame(rows).to_csv(f'{D}/experiments/results_certificate_cross_structure.csv',
                              index=False)
    print(f'wrote {D}/experiments/results_certificate_cross_structure.csv')
