# -*- coding: utf-8 -*-
"""规模阶梯基准：CPLEX(TL) vs GRI(固定预算)，同一实例同一模型。
ONLY=<index> 指定实例。输出 experiments/results_ladder_<idx>.csv
实例序：0=real, 1=m2_s1001, 2=m4_s1001, 3=m6_s1001
"""
import json, sys, os, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ('real', 'instance_v1.json'),
    ('m2', 'experiments/synth_instances/city_m2_s1001.json'),
    ('m4', 'experiments/synth_instances/city_m4_s1001.json'),
    ('m6', 'experiments/synth_instances/city_m6_s1001.json'),
]


def main():
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import gri_solve, config_violations

    idx = int(os.environ.get('ONLY', '0'))
    tag, fn = CASES[idx]
    inst = json.load(open(fn, encoding='utf-8'))
    seed = inst['meta'].get('gen_seed', 0)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100),
                             k=10, seed=7)
    TL = float(os.environ.get('TL', '900'))
    BS = float(os.environ.get('BS', '160'))
    BP = float(os.environ.get('BP', '60'))

    row = {'case': tag, 'n_discrete':
           (len(inst['sorting']) + len(inst['recycling']) + len(inst['landfill'])) * 3 * 2}
    # CPLEX
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, train, f'ladder_{tag}', retire_lstar=True,
                                      t_star=2, log=False, require_cover=True,
                                      time_limit=TL)
    row['cplex_time'] = time.time() - t0
    row['cplex_obj'] = sol.objective_value if sol else None
    try:
        row['cplex_gap'] = mdl.solve_details.mip_relative_gap if sol else None
    except Exception:
        row['cplex_gap'] = None
    row['cplex_proved'] = bool(sol is not None and (row['cplex_gap'] or 1) < 1e-6)

    def ex(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)

    cfg, val, st = gri_solve(inst, train, ex, budget_search=BS, budget_polish=BP,
                             n_islands=6, seed=0)
    row.update({'gri_time': st.get('time_s'), 'gri_obj': val,
                'gri_surrogate': st.get('surrogate'),
                'gri_polish_evals': st.get('polish_exact_evals')})
    if val is not None and row['cplex_obj']:
        row['gri_vs_cplex_pct'] = (val / row['cplex_obj'] - 1) * 100
    else:
        row['gri_vs_cplex_pct'] = None
    pd.DataFrame([row]).to_csv(f'{D}/experiments/results_ladder_{idx}.csv', index=False)
    print(f"[{tag}] disc={row['n_discrete']} | CPLEX {row['cplex_time']:.0f}s "
          f"proved={row['cplex_proved']} obj={row['cplex_obj']} | "
          f"GRI {row['gri_time']:.0f}s obj={val} | vs_cplex={row['gri_vs_cplex_pct']}",
          flush=True)


if __name__ == '__main__':
    main()
