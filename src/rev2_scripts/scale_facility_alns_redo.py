# -*- coding: utf-8 -*-
"""设施轴 ALNS 侧重跑（V2.3 rev3）：统一硬预算 1800s（alns() 已修好预算贯穿
搜索/池重评估/下降三阶段）。CPLEX 列复用既有 results_scale_facility.csv；
仅缺失的 2 行 m=6 DASH 重算 CPLEX。输出覆盖写回同文件（m=1/2 行原样保留）。
"""
import json, sys, os, time, glob
import pandas as pd
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
PREV = f'{EXP}/results_scale_facility.partial.save'

def alns_job(arg):
    fn, method, cpx = arg   # cpx = 复用的 CPLEX 行（dict）或 None
    inst = json.load(open(fn, encoding='utf-8'))
    seed = inst['meta'].get('gen_seed', 0)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100), k=N_SCEN, seed=7)
    if cpx is None:   # 缺失的 CPLEX 行：重算
        full = train if method == 'SAA' else dash_scenarios(inst, train, 0.10)
        t0 = time.time()
        mdl, sol, V = build_and_solve_saa(inst, full, f'redo_{os.path.basename(fn)}_{method}',
                                          retire_lstar=True, t_star=2, log=False,
                                          require_cover=True, time_limit=TL)
        cpx_time = time.time() - t0
        cpx_obj = sol.objective_value if sol else None
        try: gap = mdl.solve_details.mip_relative_gap
        except Exception: gap = None
        cpx_solved = bool(sol is not None and gap is not None and gap < 0.01)
    else:
        cpx_time, cpx_obj, gap, cpx_solved = (cpx['cplex_time'], cpx['cplex_obj'],
                                              cpx['cplex_gap'], cpx['cplex_solved'])
    t0 = time.time()
    alns_obj, cfg, _ = run_method_alns(inst, method, train, alns_iters=1000, alns_seed=42,
                                       alns_max_time=TL)
    alns_time = time.time() - t0
    c2, unm, _ = A.solve_second_stage_lp_ex(inst, cfg, train[0]['d'], require_cover=True)
    gap_alns = (abs(alns_obj - cpx_obj) / cpx_obj * 100) if (cpx_solved and alns_obj is not None) else None
    return {'instance': os.path.basename(fn).replace('.json', ''),
            'm': inst['meta'].get('replication', 1),
            'n_discrete': (len(inst['sorting'])+len(inst['recycling'])+len(inst['landfill']))*3*2,
            'method': method, 'n_scen': N_SCEN,
            'cplex_time': cpx_time, 'cplex_obj': cpx_obj, 'cplex_gap': gap,
            'cplex_solved': cpx_solved, 'alns_time': alns_time, 'alns_obj': alns_obj,
            'alns_cover': bool(c2 is not None), 'alns_gap_vs_opt': gap_alns}

if __name__ == '__main__':
    old = pd.read_csv(OUT)
    os.replace(OUT, PREV)
    have = {(r.instance, r.method): r for r in old.itertuples()}
    tasks = []
    for fn in sorted(glob.glob(f'{EXP}/synth_instances/city_m*.json')):
        iname = os.path.basename(fn).replace('.json', '')
        for method in ('SAA', 'DRO(0.10)'):
            cpx = have.get((iname, method))
            cpx = None if cpx is None else cpx._asdict()
            if cpx is not None and str(cpx.get('cplex_solved')) == 'False':
                cpx = None  # 之前超时未解出的行也重算（换 TL 内运气更好的一轮不采：仍如实记录）
            tasks.append((fn, method, cpx))
    print(f'{len(tasks)} ALNS-redo jobs (hard {TL}s budget)', flush=True)
    keep = old[old.instance == 'instance_v1'].copy()  # m=1 行原样保留
    rows = list(keep.to_dict('records'))
    with Pool(processes=8) as p:
        for i, row in enumerate(p.imap_unordered(alns_job, tasks), 1):
            rows.append(row)
            print(f"[{time.strftime('%H:%M')}] {i}/{len(tasks)} {row['instance']} {row['method']}: "
                  f"cpx={row['cplex_time']:.0f}s solved={row['cplex_solved']} "
                  f"alns={row['alns_time']:.0f}s cover={row['alns_cover']} "
                  f"gap={None if row['alns_gap_vs_opt'] is None else round(row['alns_gap_vs_opt'],1)}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('=== 汇总 ===')
    df = pd.read_csv(OUT)
    for m in sorted(df.m.unique()):
        for meth in ('SAA', 'DRO(0.10)'):
            g = df[(df.m == m) & (df.method == meth)]
            gaps = g.alns_gap_vs_opt.dropna()
            print(f'm={m} {meth:10s} CPLEX {g.cplex_solved.sum()}/{len(g)} 均{g.cplex_time.mean():6.0f}s '
                  f'| ALNS 均{g.alns_time.mean():6.0f}s gap均={gaps.mean() if len(gaps) else float("nan"):.1f}%', flush=True)
