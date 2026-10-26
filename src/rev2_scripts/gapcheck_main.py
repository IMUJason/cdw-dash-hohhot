# -*- coding: utf-8 -*-
"""核验主表中慢格子是否真的达到最优（gap<=1e-4）。
对 time_s>600 的行重解并记录 mip gap。输出 experiments/results_main_gapcheck.csv
"""
import json, sys, os, time
import pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{D}/experiments'


def job(arg):
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import make_batch_instance, run_method_det, reduce_scenarios
    iname, seed, method = arg
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if seed == 0 else make_batch_instance(base, seed)
    train = gen_scenarios(inst, 30, seed=1000 + seed)
    t0 = time.time()
    if method == 'DET':
        # DET 走均值轨迹，单独处理
        from run_experiments import run_method_det
        obj, cfg, wt = run_method_det(inst, time_limit=1200)
        return {'instance': iname, 'method': method, 'obj': obj, 'gap': None,
                'time_s': time.time() - t0, 'note': 'DET(均值轨迹)'}
    mdl, sol, V = build_and_solve_saa(inst, train, f'gc_{iname}_{method}',
                                      retire_lstar=True, t_star=2, log=False,
                                      require_cover=True, time_limit=1200)
    obj = sol.objective_value if sol is not None else None
    try:
        gap = mdl.solve_details.mip_relative_gap if sol is not None else None
    except Exception:
        gap = None
    return {'instance': iname, 'method': method, 'obj': obj, 'gap': gap,
            'time_s': time.time() - t0, 'note': ''}


if __name__ == '__main__':
    sys.path.insert(0, D); os.chdir(D)
    df = pd.read_csv(f'{EXP}/results_main_v22.csv')
    slow = df[df.time_s > 600][['instance', 'seed', 'method']].values.tolist()
    print(f'{len(slow)} slow rows to verify', flush=True)
    with Pool(processes=3) as p:
        rows = p.map(job, [(a, b, c) for a, b, c in slow])
    out = pd.DataFrame(rows)
    out.to_csv(f'{EXP}/results_main_gapcheck.csv', index=False)
    print(out.to_string(index=False))
    bad = out[(out.gap.notna()) & (out.gap > 1e-4)]
    print(f'\n未达最优的格子: {len(bad)}')
