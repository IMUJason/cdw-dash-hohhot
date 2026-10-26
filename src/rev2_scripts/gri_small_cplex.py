# -*- coding: utf-8 -*-
"""小规模 CPLEX 基线（V2.4 小规模验证，不是全量重跑）。
real + batch1..10；SAA(10)；覆盖约束；终端清空；TL=900s。
输出 results_gri_small_cplex.csv + 最优配置 config_gri_small_<i>.json。
"""
import json, sys, os, time
import pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import reduce_scenarios, make_batch_instance, extract_xconfig_cpx, _cfg_from_xconfig

EXP = f'{D}/experiments'

def job(seed):
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if seed == 0 else make_batch_instance(base, seed)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed), k=10, seed=7)
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, train, f'gri_small_cpx_{seed}', retire_lstar=True,
                                      t_star=2, log=False, require_cover=True, time_limit=900)
    dt = time.time() - t0
    if sol is None:
        return {'seed': seed, 'ok': False, 'time_s': dt}
    gap = mdl.solve_details.mip_relative_gap
    cfg = _cfg_from_xconfig(inst, extract_xconfig_cpx(inst, V))
    json.dump(cfg, open(f'{EXP}/config_gri_small_{seed}.json', 'w'), ensure_ascii=False)
    return {'seed': seed, 'ok': True, 'opt': sol.objective_value,
            'mip_gap': gap, 'time_s': dt}

if __name__ == '__main__':
    with Pool(processes=4) as p:
        rows = []
        for r in p.imap_unordered(job, range(11)):
            rows.append(r)
            print(f"seed={r['seed']} ok={r['ok']} "
                  f"opt={r.get('opt', float('nan'))/1e6:.1f}M t={r['time_s']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f'{EXP}/results_gri_small_cplex.csv', index=False)
    print('done')
