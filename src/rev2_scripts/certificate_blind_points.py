# -*- coding: utf-8 -*-
"""证书盲测点：网格之外的 rho/phi 组合，先预测边界再用 CPLEX 验证。"""
import json, os, sys, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, dash_scenarios
from gri_alns import initial_full, repair_monotone
from eps_formula_v2 import predict

BASE = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))


def main():
    rows = []
    for rho, phi in [(0.75, 0.80), (0.75, 0.90), (0.85, 0.80), (0.85, 0.90)]:
        inst = json.loads(json.dumps(BASE))
        inst['params']['recyclable_ratio'] = rho
        inst['params']['product_yield'] = phi
        r = predict(inst, rho, phi)
        eps_star = r['eps_pred']
        cfg = initial_full(inst)
        for j in inst['sorting']:
            cfg['zJ'][j] = [4] * 3
        for k in inst['recycling']:
            cfg['zK'][k] = [4] * 3
        cfg = repair_monotone(cfg, inst)
        for eps in [max(0, eps_star - 0.01), eps_star + 0.01]:
            train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
            full = dash_scenarios(inst, train, eps)
            mdl, sol, V = build_and_solve_saa(inst, full, f'blind_{rho}_{phi}_{eps:.3f}',
                                              retire_lstar=True, t_star=2, log=False,
                                              require_cover=True, time_limit=600)
            for key in ['xJ', 'xK', 'xL', 'zJ', 'zK', 'rL']:
                for f in cfg[key]:
                    for t in range(1, 4):
                        mdl.add_constraint(V[key][f, t] == cfg[key][f][t - 1])
            sol2 = mdl.solve()
            rows.append({'rho': rho, 'phi': phi, 'eps_pred': eps_star,
                         'eps': eps, 'cplex_feasible': sol2 is not None})
            print(f'rho={rho} phi={phi} pred={eps_star:.4f} eps={eps:.3f} '
                  f'{"OK" if sol2 else "INFEAS"}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/experiments/results_certificate_blind_points.csv',
                              index=False)


if __name__ == '__main__':
    main()
