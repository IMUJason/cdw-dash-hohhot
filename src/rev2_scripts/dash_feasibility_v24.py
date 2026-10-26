# -*- coding: utf-8 -*-
"""前置验证 2：DASH 档位 ε∈{0.12,0.14,0.16,0.18} 在 11 个实例上的可行性。
终端清空修复后 ε*=0.185 失效，最高档需要重新确定。
输出 experiments/results_dash_feasibility_v24.csv
"""
import json, sys, os, time
import pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{D}/experiments'
EPS_LIST = [0.12, 0.14, 0.16, 0.18]


def job(seed):
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios, make_batch_instance, dash_scenarios
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if seed == 0 else make_batch_instance(base, seed)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed), k=10, seed=7)
    rows = []
    for eps in EPS_LIST:
        full = dash_scenarios(inst, train, eps)
        t0 = time.time()
        try:
            mdl, sol, V = build_and_solve_saa(inst, full, f'eps_{seed}_{eps}',
                                              retire_lstar=True, t_star=2, log=False,
                                              require_cover=True, time_limit=600)
            ok = sol is not None
            obj = sol.objective_value if ok else None
        except Exception:
            ok, obj = False, None
        rows.append({'instance': 'real' if seed == 0 else f'batch{seed}',
                     'eps': eps, 'feasible': ok, 'obj': obj,
                     'time_s': round(time.time() - t0, 1)})
    return rows


if __name__ == '__main__':
    sys.path.insert(0, D); os.chdir(D)
    with Pool(processes=6) as p:
        allrows = [r for rows in p.map(job, range(11)) for r in rows]
    df = pd.DataFrame(allrows)
    df.to_csv(f'{EXP}/results_dash_feasibility_v24.csv', index=False)
    piv = df.pivot_table(index='instance', columns='eps', values='feasible',
                         aggfunc='first')
    print('各实例在各 ε 下是否可行：')
    print(piv.to_string())
    print('\n各 ε 的可行实例数：')
    print(df.groupby('eps')['feasible'].sum().to_string())
