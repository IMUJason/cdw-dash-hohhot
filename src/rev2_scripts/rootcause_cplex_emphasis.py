# -*- coding: utf-8 -*-
"""根因测试 A：CPLEX 的"无解"是否是求解器强调模式问题？
对比 default / feasibility-emphasis / 提供满配 MIP start 三种设置，m=4 与 m=6。
"""
import json, sys, os, time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_case(case, mode, TL=600):
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import initial_full, repair_monotone

    fn = f'{D}/experiments/synth_instances/city_{case}_s1001.json'
    inst = json.load(open(fn, encoding='utf-8'))
    T = inst['meta']['periods']
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)

    mdl, sol, V = build_and_solve_saa(inst, train, f'{case}_{mode}', retire_lstar=True,
                                      t_star=2, log=False, require_cover=True,
                                      time_limit=TL)
    if mode == 'feas':
        mdl.parameters.emphasis.mip = 1          # 1 = feasibility
        sol = mdl.solve()
    elif mode == 'heur':
        mdl.parameters.emphasis.mip = 4          # 4 = hidden feasible solutions
        mdl.parameters.mip.strategy.heuristiceffort = 4
        sol = mdl.solve()
    obj = sol.objective_value if sol is not None else None
    best_bound = None
    try:
        best_bound = float(sol.objective_value) if sol is not None else None
    except Exception:
        pass
    print(f'{case:4s} {mode:5s}: obj={obj}  '
          f'status={mdl.solve_details.status}  '
          f'gap={getattr(mdl.solve_details, "mip_relative_gap", None)}', flush=True)
    return obj


if __name__ == '__main__':
    case = os.environ.get('CASE', 'm4')
    for mode in ['default', 'feas', 'heur']:
        run_case(case, mode, TL=float(os.environ.get('TL', '600')))
