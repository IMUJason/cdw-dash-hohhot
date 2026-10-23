"""V2.2 规模对比：3 规模 × 10 实例 × 2 方法 × 2 求解器，统一 1800s 上限。
CPLEX 带服务级覆盖约束（与 ALNS 同解一个问题），ALNS 同样带约束。"""
import json, sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import (reduce_scenarios, make_batch_instance, run_method_alns,
                             oos_eval, weighted_stats, extract_xconfig_cpx, dash_scenarios)
import alns as A
EXP = f'{D}/experiments'
TIME_LIMIT = 1800
SIZES = {'small': 10, 'medium': 20, 'large': 30}

def job(arg):
    size_name, n_scen, inst_id, method = arg
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if inst_id == 1 else make_batch_instance(base, inst_id)
    iname = f'{size_name}_{inst_id}'
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000+inst_id), k=n_scen, seed=7)
    full = train if method == 'SAA' else dash_scenarios(inst, train, 0.10)
    # --- CPLEX（带覆盖约束）---
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, full, f'cp_{iname}_{method}', retire_lstar=True,
                                       t_star=2, log=False, require_cover=True, time_limit=TIME_LIMIT)
    t_cplex = time.time() - t0
    cplex_obj = sol.objective_value if sol else None
    cplex_gap = None
    if sol:
        try: cplex_gap = mdl.solve_details.mip_relative_gap
        except Exception: pass
    # --- ALNS（同样带覆盖约束）---
    t0 = time.time()
    alns_obj, alns_cfg, _ = run_method_alns(inst, method, train, alns_iters=1000, alns_seed=42)
    t_alns = time.time() - t0
    # ALNS 在训练集上的覆盖可行性
    c2, unm, _ = A.solve_second_stage_lp_ex(inst, alns_cfg, train[0]['d'], require_cover=True)
    alns_cover = (c2 is not None)
    return {'size': size_name, 'instance': iname, 'method': method, 'n_scen': n_scen,
            'cplex_obj': cplex_obj, 'cplex_time': t_cplex, 'cplex_gap': cplex_gap,
            'cplex_solved': bool(sol is not None and cplex_gap is not None and cplex_gap < 0.01),
            'alns_obj': alns_obj, 'alns_time': t_alns, 'alns_cover': alns_cover,
            'alns_gap_vs_opt': (abs(alns_obj-cplex_obj)/cplex_obj*100) if (cplex_obj and cplex_gap is not None and cplex_gap < 0.01) else None}

if __name__ == '__main__':
    tasks = [(sn, ns, i, m) for sn, ns in SIZES.items() for i in range(1, 11)
             for m in ['SAA', 'DRO(0.10)']]
    with Pool(processes=10) as p:
        rows = p.map(job, tasks)
    df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_scale_v22.csv', index=False)
    print('\n=== 规模对比 (TL=1800s) ===')
    for size in ['small', 'medium', 'large']:
        for m in ['SAA', 'DRO(0.10)']:
            g = df[(df['size']==size)&(df['method']==m)]
            solved = g.cplex_solved.sum(); n = len(g)
            gaps = g.alns_gap_vs_opt.dropna()
            gt = f'{gaps.mean():.1f}%' if len(gaps) else '---'
            print(f'{size:7s} {m:10s} CPLEX {solved}/{n} 平均{g.cplex_time.mean():6.0f}s | '
                  f'ALNS 平均{g.alns_time.mean():5.0f}s 覆盖可行 {int(g.alns_cover.sum())}/{n} gap={gt}', flush=True)
