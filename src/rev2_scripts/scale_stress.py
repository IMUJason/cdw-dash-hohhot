"""压力规模实验：找到 CPLEX 的真实规模边界（覆盖约束下），并评估 ALNS 的性价比。
规模 = 训练情景数（DASH 抬升后翻倍）。统一 1800s 上限。
"""
import json, sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
EXP = f'{D}/experiments'
TIME_LIMIT = 1800
# 更激进的情景规模，覆盖约束下 CPLEX 的边界
SIZES = {'xsmall': 10, 'small': 20, 'medium': 40, 'large': 60}


def job(arg):
    size_name, n_scen, inst_id, method = arg
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import (reduce_scenarios, make_batch_instance, run_method_alns,
                                 dash_scenarios)
    import alns as A
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if inst_id == 1 else make_batch_instance(base, inst_id)
    iname = f'{size_name}_{inst_id}'
    # 训练集按目标规模生成（先建 60 情景再缩减）
    train = reduce_scenarios(inst, gen_scenarios(inst, max(n_scen, 30), seed=1000+inst_id),
                             k=n_scen, seed=7)
    full = train if method == 'SAA' else dash_scenarios(inst, train, 0.10)
    # --- CPLEX ---
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, full, f'cp_{iname}_{method}', retire_lstar=True,
                                       t_star=2, log=False, require_cover=True,
                                       time_limit=TIME_LIMIT)
    t_cplex = time.time() - t0
    cplex_obj = sol.objective_value if sol else None
    cplex_gap = None
    if sol:
        try: cplex_gap = mdl.solve_details.mip_relative_gap
        except Exception: pass
    # --- ALNS ---
    t0 = time.time()
    alns_obj, alns_cfg, _ = run_method_alns(inst, method, train, alns_iters=1000, alns_seed=42)
    t_alns = time.time() - t0
    ok_cover = True
    for sc in train:
        c2, _, _ = A.solve_second_stage_lp_ex(inst, alns_cfg, sc['d'], require_cover=True)
        if c2 is None:
            ok_cover = False; break
    return {'size': size_name, 'n_scen': n_scen, 'instance': iname, 'method': method,
            'cplex_obj': cplex_obj, 'cplex_time': t_cplex, 'cplex_gap': cplex_gap,
            'cplex_solved': bool(sol is not None and cplex_gap is not None and cplex_gap < 0.01),
            'cplex_status': str(mdl.solve_details.status),
            'alns_obj': alns_obj, 'alns_time': t_alns, 'alns_cover': ok_cover,
            'alns_gap_vs_opt': (abs(alns_obj-cplex_obj)/cplex_obj*100) if (cplex_obj and cplex_gap is not None and cplex_gap < 0.01) else None}


if __name__ == '__main__':
    tasks = [(sn, ns, i, m) for sn, ns in SIZES.items() for i in range(1, 6)
             for m in ['SAA', 'DRO(0.10)']]
    with Pool(processes=10) as p:
        rows = p.map(job, tasks)
    df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_scale_stress.csv', index=False)
    print('\n=== 压力规模 (TL=1800s, 5 实例/格) ===')
    for size in ['xsmall', 'small', 'medium', 'large']:
        for m in ['SAA', 'DRO(0.10)']:
            g = df[(df['size'] == size) & (df['method'] == m)]
            solved = int(g.cplex_solved.sum()); n = len(g)
            gaps = g.alns_gap_vs_opt.dropna()
            gs = f'{gaps.mean():.1f}%' if len(gaps) else '---'
            print(f'{size:7s} n={g.n_scen.iloc[0]:3d} {m:10s} CPLEX {solved}/{n} '
                  f'平均{g.cplex_time.mean():6.0f}s | ALNS 平均{g.alns_time.mean():6.0f}s '
                  f'覆盖{int(g.alns_cover.sum())}/{n} gap={gs}', flush=True)
