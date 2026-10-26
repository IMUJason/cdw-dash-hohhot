# -*- coding: utf-8 -*-
"""设施轴 GRI vs CPLEX（V2.4 新模型，m=1..6）。
CPLEX 若在 1800s 内给出 mip gap，则报精确/认证 gap；否则报
"CPLEX 未证明最优，GRI 在同等墙钟预算下给出可行计划，相对 CPLEX 当前最好界 X%"。
输出 experiments/results_gri_facility.csv
"""
import json, sys, os, time, glob
import pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, make_batch_instance, extract_xconfig_cpx, _cfg_from_xconfig
import alns as A
from gri_alns import gri_run, exact_polish, exact_pair_polish, config_violations

EXP = f'{D}/experiments'
TL = float(os.environ.get('TL', '1800'))
GRI_BUDGET = float(os.environ.get('GRI_BUDGET', '600'))
POLISH = float(os.environ.get('POLISH', '300'))
NISO = int(os.environ.get('NISO', '6'))


def job(arg):
    fn, tag = arg
    inst = json.load(open(fn, encoding='utf-8'))
    m = inst['meta'].get('replication', 1)
    seed = inst['meta'].get('gen_seed', 0)
    ndisc = (len(inst['sorting']) + len(inst['recycling']) + len(inst['landfill'])) * 3 * 2
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100), k=10, seed=7)
    row = {'instance': tag, 'm': m, 'n_discrete': ndisc,
           'n_scen': 10}
    # --- CPLEX ---
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, train, f'fac_{tag}', retire_lstar=True, t_star=2,
                                      log=False, require_cover=True, time_limit=TL)
    row['cplex_time'] = time.time() - t0
    row['cplex_obj'] = sol.objective_value if sol else None
    gap = None
    if sol is not None:
        try:
            gap = mdl.solve_details.mip_relative_gap
        except Exception:
            gap = None
    row['cplex_gap'] = gap
    row['cplex_proved'] = bool(sol is not None and gap is not None and gap < 1e-6)
    # CPLEX 的当前最好界（未证明时用于算 relative gap）
    cplex_bound = None
    if sol is not None:
        cplex_bound = float(sol.objective_value)
    # --- GRI ---
    def ex(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)
    t0 = time.time()
    pool = gri_run(inst, train, n_islands=NISO, wall_budget=GRI_BUDGET, seed=0)
    best_v, best_c = None, None
    for cfg, gv in pool:
        v = ex(cfg)
        if v is not None and (best_v is None or v < best_v):
            best_v, best_c = v, cfg
    if best_c is not None:
        best_c, v2, _ = exact_polish(inst, best_c, train, ex, budget_s=POLISH)
        if v2 is not None and (best_v is None or v2 < best_v):
            best_v = v2
        best_c, v3, _ = exact_pair_polish(inst, best_c, train, ex, budget_s=POLISH)
        if v3 is not None and (best_v is None or v3 < best_v):
            best_v = v3
    row['gri_time'] = time.time() - t0
    row['gri_obj'] = best_v
    row['gri_feasible'] = best_v is not None
    if best_v is not None and cplex_bound:
        row['gri_vs_cplex_pct'] = (best_v / cplex_bound - 1) * 100
    else:
        row['gri_vs_cplex_pct'] = None
    return row


if __name__ == '__main__':
    tasks = [(f'{D}/instance_v1.json', 'real')]
    for fn in sorted(glob.glob(f'{EXP}/synth_instances/city_m*.json')):
        tasks.append((fn, os.path.basename(fn).replace('.json', '')))
    only = os.environ.get('ONLY')
    if only is not None:
        r = job(tasks[int(only)])
        pd.DataFrame([r]).to_csv(f'{EXP}/results_gri_facility_inst{only}.csv', index=False)
        print(f"done {r['instance']} m={r['m']} cplex={r['cplex_time']:.0f}s "
              f"proved={r['cplex_proved']} gri={r['gri_time']:.0f}s "
              f"gap={r['gri_vs_cplex_pct']}", flush=True)
    else:
        print(f'{len(tasks)} instances, TL={TL}s, GRI budget={GRI_BUDGET}s', flush=True)
        rows = []
        for k, t in enumerate(tasks):
            r = job(t)
            rows.append(r)
            pd.DataFrame(rows).to_csv(f'{EXP}/results_gri_facility.csv', index=False)
            print(f"[{time.strftime('%H:%M')}] {r['instance']} m={r['m']} "
                  f"disc={r['n_discrete']} cplex={r['cplex_time']:.0f}s "
                  f"proved={r['cplex_proved']} gri={r['gri_time']:.0f}s "
                  f"gap_vs_cplex={r['gri_vs_cplex_pct']}", flush=True)
