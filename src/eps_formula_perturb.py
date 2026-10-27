# -*- coding: utf-8 -*-
"""ε* 闭式证书的容量扰动验证：R_l 总量 ×0.8/1.0/1.2，比较预测边界与 CPLEX。
只测真实网，满配固定一阶解，覆盖约束。输出 experiments/eps_formula_perturb.csv
"""
import json, os, sys, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, dash_scenarios
from gri_alns import initial_full, repair_monotone
from eps_formula_v2 import predict

base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))

if __name__ == '__main__':
    rows = []
    for scale in (0.8, 1.0, 1.2):
        inst = json.loads(json.dumps(base))
        for l in inst['landfill']:
            inst['landfill'][l]['cap_t'] *= scale
        r = predict(inst)
        pred = r['eps_pred']
        for eps in [round(max(0, pred - 0.01), 3), round(pred + 0.01, 3)]:
            train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
            full = dash_scenarios(inst, train, eps)
            cfg = initial_full(inst)
            for j in inst['sorting']:
                cfg['zJ'][j] = [4] * 3
            for k in inst['recycling']:
                cfg['zK'][k] = [4] * 3
            cfg = repair_monotone(cfg, inst)
            t0 = time.time()
            mdl, sol, V = build_and_solve_saa(inst, full, f'pert_{scale}_{eps}',
                                              retire_lstar=True, t_star=2, log=False,
                                              require_cover=True, time_limit=600)
            for key in ['xJ', 'xK', 'xL', 'zJ', 'zK', 'rL']:
                for f in cfg[key]:
                    for t in range(1, 4):
                        mdl.add_constraint(V[key][f, t] == cfg[key][f][t - 1])
            sol2 = mdl.solve()
            rows.append({'R_scale': scale, 'eps_pred': pred, 'eps': eps,
                         'cplex_feasible': sol2 is not None,
                         'time_s': round(time.time() - t0, 1)})
            print(f'R×{scale}: pred={pred:.4f}  eps={eps:.3f}  '
                  f'CPLEX={"OK" if sol2 else "INFEAS"}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/experiments/eps_formula_perturb.csv', index=False)
