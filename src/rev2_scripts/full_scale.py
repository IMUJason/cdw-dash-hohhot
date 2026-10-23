"""完整规模实验：3 规模 × 10 实例 × 2 方法 × 2 求解器，统一 1800s 终止协议"""
import json, sys, os, time, copy
import numpy as np, pandas as pd
D='/Users/cjx-ms/Documents/我的坚果云/科研/论文/2026 cdw 0825/hohhot case data/model'
sys.path.insert(0,D); os.chdir(D)
from docplex.mp.model import Model
from cplex_saa import gen_scenarios, build_and_solve_saa
from cplex_dro import solve_dro
from run_experiments import reduce_scenarios, make_batch_instance, run_method_alns, oos_eval, weighted_stats, extract_xconfig_cpx
from alns import evaluate

base = json.load(open('instance_v1.json', encoding='utf-8'))
TIME_LIMIT = 1800  # 30 min unified protocol

# 定义规模
SIZES = {
    'small':  {'n_scen': 10},   # 10→20 lifted for DASH
    'medium': {'n_scen': 20},   # 20→40 lifted
    'large':  {'n_scen': 30},   # 30→60 lifted (full problem)
}

results = []

for size_name, cfg in SIZES.items():
    n_scen = cfg['n_scen']
    for inst_id in range(1, 11):  # 10 instances per size
        inst = make_batch_instance(base, inst_id) if inst_id > 1 else base
        iname = f'{size_name}_{inst_id}'
        
        train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000+inst_id), k=n_scen, seed=7)
        
        for method in ['SAA', 'DASH(0.10)']:
            # === CPLEX ===
            t0 = time.time()
            if method == 'SAA':
                mdl, sol, V = build_and_solve_saa(inst, train, f'{method}_{iname}', log=False)
            else:
                mdl, sol, V = solve_dro(inst, train, 0.10, log=False)
            t_cplex = time.time() - t0
            
            cplex_obj = sol.objective_value if sol else None
            cplex_gap = None
            if sol and hasattr(mdl, 'solve_details'):
                try: cplex_gap = mdl.solve_details.mip_relative_gap
                except: pass
            
            # === ALNS ===
            t0 = time.time()
            if method == 'SAA':
                alns_obj, alns_cfg, _ = run_method_alns(inst, 'SAA', train, alns_iters=1000, alns_seed=42)
            else:
                alns_obj, alns_cfg, _ = run_method_alns(inst, 'DRO(0.10)', train, alns_iters=1000, alns_seed=42)
            t_alns = time.time() - t0
            
            # Gap: vs CPLEX optimum if available
            if cplex_obj and cplex_gap is not None and cplex_gap < 0.01:
                gap_vs_opt = abs(alns_obj - cplex_obj) / cplex_obj * 100
                benchmark = 'CPLEX_optimal'
            else:
                gap_vs_opt = None
                benchmark = 'CPLEX_timeout'
            
            results.append({
                'size': size_name, 'instance': iname, 'method': method,
                'n_scen': n_scen,
                'cplex_obj': cplex_obj, 'cplex_time': t_cplex, 'cplex_gap': cplex_gap,
                'alns_obj': alns_obj, 'alns_time': t_alns,
                'alns_gap_vs_opt': gap_vs_opt, 'benchmark': benchmark,
                'cplex_solved': cplex_gap is not None and cplex_gap < 0.01,
            })
            
            status = 'OPT' if results[-1]['cplex_solved'] else f'TIMEOUT(gap={cplex_gap:.3f})' if cplex_gap else 'FAIL'
            print(f'[{iname}] {method}: CPLEX {status} {t_cplex:.0f}s | ALNS {t_alns:.0f}s obj={alns_obj/1e9:.3f}B', flush=True)
            pd.DataFrame(results).to_csv(f'{D}/experiments/results_scale_full.csv', index=False)

print('\n=== 汇总 ===')
df = pd.DataFrame(results)
for size in ['small', 'medium', 'large']:
    for m in ['SAA', 'DASH(0.10)']:
        g = df[(df['size']==size) & (df['method']==m)]
        n_solved = g['cplex_solved'].sum()
        n_total = len(g)
        avg_gap = g['alns_gap_vs_opt'].mean() if g['alns_gap_vs_opt'].notna().any() else None
        avg_t_cp = g['cplex_time'].mean()
        avg_t_al = g['alns_time'].mean()
        print(f'{size:6s} {m:12s}: CPLEX {n_solved}/{n_total} solved, avg_time={avg_t_cp:.0f}s | ALNS avg_time={avg_t_al:.0f}s, gap_vs_opt={avg_gap:.2f}%' if avg_gap else f'{size:6s} {m:12s}: CPLEX {n_solved}/{n_total} solved, avg_time={avg_t_cp:.0f}s | ALNS avg_time={avg_t_al:.0f}s')
