# -*- coding: utf-8 -*-
"""设施轴规模实验（V2.3，rev2）：合成城市 m∈{2,3,4,6}（+真实网 m=1 对照）。
情景数固定 10（隔离设施维）。CPLEX 与 ALNS 同享 1800s 墙钟预算（统一 TL 协议）；
imap_unordered 增量落盘（experiments/results_scale_facility.csv），防长任务丢失。

运行前提：rev2_scripts/gen_synthetic_city.py 已生成实例。
"""
import json, sys, os, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, run_method_alns, dash_scenarios
import alns as A

EXP = f'{D}/experiments'
TL = 1800
N_SCEN = 10
OUT = f'{EXP}/results_scale_facility.csv'

def n_discrete(inst):
    return (len(inst['sorting']) + len(inst['recycling']) + len(inst['landfill'])) * 3 * 2

def job(arg):
    fn, method = arg
    inst = json.load(open(fn, encoding='utf-8'))
    m = inst['meta'].get('replication', 1); seed = inst['meta'].get('gen_seed', 0)
    iname = os.path.basename(fn).replace('.json', '')
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100), k=N_SCEN, seed=7)
    full = train if method == 'SAA' else dash_scenarios(inst, train, 0.10)
    row = {'instance': iname, 'm': m, 'n_discrete': n_discrete(inst), 'method': method,
           'n_scen': N_SCEN}
    # --- CPLEX（覆盖约束，TL） ---
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, full, f'fac_{iname}_{method}', retire_lstar=True,
                                      t_star=2, log=False, require_cover=True, time_limit=TL)
    row['cplex_time'] = time.time() - t0
    row['cplex_obj'] = sol.objective_value if sol else None
    gap = None
    if sol:
        try: gap = mdl.solve_details.mip_relative_gap
        except Exception: pass
    row['cplex_gap'] = gap
    row['cplex_solved'] = bool(sol is not None and gap is not None and gap < 0.01)
    # --- ALNS（同约束、同 1800s 墙钟预算） ---
    t0 = time.time()
    alns_obj, cfg, _ = run_method_alns(inst, method, train, alns_iters=1000, alns_seed=42,
                                       alns_max_time=TL)
    row['alns_time'] = time.time() - t0
    row['alns_obj'] = alns_obj
    c2, unm, _ = A.solve_second_stage_lp_ex(inst, cfg, train[0]['d'], require_cover=True)
    row['alns_cover'] = bool(c2 is not None)
    if row['cplex_solved'] and alns_obj is not None:
        row['alns_gap_vs_opt'] = abs(alns_obj - row['cplex_obj']) / row['cplex_obj'] * 100
    else:
        row['alns_gap_vs_opt'] = None
    return row

if __name__ == '__main__':
    tasks = []
    for method in ('SAA', 'DRO(0.10)'):
        tasks.append((f'{D}/instance_v1.json', method))
    for fn in sorted(glob.glob(f'{EXP}/synth_instances/city_m*.json')):
        for method in ('SAA', 'DRO(0.10)'):
            tasks.append((fn, method))
    print(f'{len(tasks)} jobs (incremental save to {OUT})', flush=True)
    done = 0
    with Pool(processes=8) as p, open(OUT, 'w') as f:
        f.write('instance,m,n_discrete,method,n_scen,cplex_time,cplex_obj,cplex_gap,'
                'cplex_solved,alns_time,alns_obj,alns_cover,alns_gap_vs_opt\n')
        for row in p.imap_unordered(job, tasks):
            f.write(','.join(str(row[k]) for k in
                    ['instance','m','n_discrete','method','n_scen','cplex_time','cplex_obj',
                     'cplex_gap','cplex_solved','alns_time','alns_obj','alns_cover',
                     'alns_gap_vs_opt']) + '\n')
            f.flush()
            done += 1
            print(f"[{time.strftime('%H:%M')}] {done}/{len(tasks)} {row['instance']} "
                  f"{row['method']}: cpx_solved={row['cplex_solved']} "
                  f"cpx={row['cplex_time']:.0f}s alns={row['alns_time']:.0f}s "
                  f"gap={row['alns_gap_vs_opt']}", flush=True)
    print('\n=== 设施轴规模（TL=1800s 双侧, 10 情景）===')
    df = pd.read_csv(OUT)
    for m in sorted(df.m.unique()):
        for meth in ('SAA', 'DRO(0.10)'):
            g = df[(df.m == m) & (df.method == meth)]
            solved = g.cplex_solved.sum()
            gaps = g.alns_gap_vs_opt.dropna()
            alns_cov = g.alns_cover.sum()
            gt = f'{gaps.mean():.1f}%' if len(gaps) else '---'
            print(f'm={m} ({int(g.n_discrete.iloc[0])} 离散var) {meth:10s} '
                  f'CPLEX {solved}/{len(g)} 均{g.cplex_time.mean():6.0f}s | '
                  f'ALNS 均{g.alns_time.mean():6.0f}s 覆盖{alns_cov}/{len(g)} gap={gt}', flush=True)
