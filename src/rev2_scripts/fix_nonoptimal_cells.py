# -*- coding: utf-8 -*-
"""对未达最优的 5 个格子做隔离重解（单进程、无并发、TL=3600）。
逐个串行执行，输出 experiments/results_main_exactfix.csv
"""
import json, sys, os, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [('batch2', 2, 'SAA'), ('batch3', 3, 'SAA'), ('batch4', 4, 'SAA'),
           ('batch5', 5, 'SAA'), ('batch5', 5, 'DRO(0.16)')]


def main():
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import make_batch_instance, dash_scenarios
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    rows = []
    for iname, seed, method in TARGETS:
        inst = make_batch_instance(base, seed)
        train = gen_scenarios(inst, 30, seed=1000 + seed)
        full = train if method == 'SAA' else dash_scenarios(inst, train, 0.16)
        t0 = time.time()
        mdl, sol, V = build_and_solve_saa(inst, full, f'exactfix_{iname}_{method}',
                                          retire_lstar=True, t_star=2, log=False,
                                          require_cover=True, time_limit=3600)
        obj = sol.objective_value if sol is not None else None
        gap = None
        try:
            gap = mdl.solve_details.mip_relative_gap if sol is not None else None
        except Exception:
            pass
        elapsed = time.time() - t0
        rows.append({'instance': iname, 'seed': seed, 'method': method,
                     'obj': obj, 'gap': gap, 'time_s': round(elapsed, 1),
                     'optimal': bool(gap is not None and gap <= 1e-5)})
        pd.DataFrame(rows).to_csv(f'{D}/experiments/results_main_exactfix.csv', index=False)
        print(f"{iname:8s} {method:12s} obj={obj} gap={gap} "
              f"optimal={rows[-1]['optimal']} t={elapsed:.0f}s", flush=True)


if __name__ == '__main__':
    main()
