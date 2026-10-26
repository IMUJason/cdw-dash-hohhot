# -*- coding: utf-8 -*-
"""根因测试 B：CPLEX 在 m=4/6 的失败是"规模"还是"近对称"？
对比 clone 实例（近对称，扰动 ±8–25%）与 hetero 实例（异构，容量跨 2 倍、
成本 ±45%、距离 σ=0.32），同规模同预算。
输出 experiments/results_symmetry_test.csv
"""
import json, sys, os, time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(fn, tag, TL=600):
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios
    inst = json.load(open(fn, encoding='utf-8'))
    seed = inst['meta'].get('gen_seed', 1001)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100),
                             k=10, seed=7)
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, train, f'sym_{tag}', retire_lstar=True,
                                      t_star=2, log=False, require_cover=True,
                                      time_limit=TL)
    obj = sol.objective_value if sol is not None else None
    gap = None
    try:
        gap = mdl.solve_details.mip_relative_gap if sol is not None else None
    except Exception:
        pass
    print(f'{tag:16s} disc={(len(inst["sorting"])+len(inst["recycling"])+len(inst["landfill"]))*3*2:4d} '
          f'TL={TL:.0f}s  obj={"None" if obj is None else round(obj)}  '
          f'gap={gap}  t={time.time()-t0:.0f}s', flush=True)
    return {'case': tag, 'obj': obj, 'gap': gap, 'time_s': time.time() - t0}


if __name__ == '__main__':
    TL = float(os.environ.get('TL', '600'))
    rows = []
    which = os.environ.get('WHICH', 'm4')
    rows.append(run(f'{D}/experiments/synth_instances/city_{which}_s1001.json',
                    f'clone_{which}', TL))
    rows.append(run(f'{D}/experiments/synth_instances/hetero_{which}_s1001.json',
                    f'hetero_{which}', TL))
    import pandas as pd
    pd.DataFrame(rows).to_csv(
        f'{D}/experiments/results_symmetry_test_{which}.csv', index=False)
