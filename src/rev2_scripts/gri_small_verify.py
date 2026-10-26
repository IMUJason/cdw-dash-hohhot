# -*- coding: utf-8 -*-
"""GRI 小规模验证（real + 10 个实例；K 种子/实例；精确 gap）。
CPLEX 基线来自 results_gri_small_cplex.csv（同实例同训练集同模型）。
每个种子：零 LP 岛搜索 → 全池精确评估 → 有界精确抛光。
输出 results_gri_small_verify.csv（逐种子）与 config_gri_small_best_<i>.json。
"""
import json, sys, os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios, make_batch_instance
import alns as A
from gri_alns import gri_run, exact_polish, exact_pair_polish, config_violations

EXP = f'{D}/experiments'
NSEEDS = int(os.environ.get('NSEEDS', '5'))
BUDGET = float(os.environ.get('BUDGET', '180'))
POLISH = float(os.environ.get('POLISH', '180'))


def job(seed):
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if seed == 0 else make_batch_instance(base, seed)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed), k=10, seed=7)
    cpx = pd.read_csv(f'{EXP}/results_gri_small_cplex.csv')
    opt = float(cpx[cpx.seed == seed].opt.iloc[0])
    def valid_eval(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)
    rows = []
    best_cfg, best_val = None, None
    SEED0 = int(os.environ.get('SEED0', '0'))
    TAG = os.environ.get('TAG', '')
    for s in range(SEED0, SEED0 + NSEEDS):
        t0 = time.time()
        pool = gri_run(inst, train, n_islands=6, wall_budget=BUDGET, seed=s)
        val, cfg_best = None, None
        n_feas = 0
        for cfg, gv in pool:
            v = valid_eval(cfg)
            if v is not None:
                n_feas += 1
                if val is None or v < val:
                    val, cfg_best = v, cfg
        if cfg_best is not None:
            def ex(c):
                return valid_eval(c)
            cfg_best, pv, _ = exact_polish(inst, cfg_best, train, ex, budget_s=POLISH)
            if pv is not None and pv < val:
                val = pv
            cfg_best, pv2, _ = exact_pair_polish(inst, cfg_best, train, ex,
                                                  budget_s=float(os.environ.get('PAIR', '180')))
            if pv2 is not None and pv2 < val:
                val = pv2
        gap = (val / opt - 1) * 100 if val is not None else float('nan')
        hit = bool(val is not None and abs(val - opt) / opt < 1e-6)
        dt = time.time() - t0
        row = {'instance': seed, 'seed': s, 'opt': opt, 'gri_obj': val,
               'gap_pct': gap, 'exact_hit': hit, 'pool_n': len(pool),
               'pool_lp_feas': n_feas, 'time_s': dt}
        rows.append(row)
        json.dump(cfg_best, open(f'{EXP}/config_gri_small_{seed}_s{s}.json', 'w'),
                  ensure_ascii=False)
        if val is not None and (best_val is None or val < best_val):
            best_val, best_cfg = val, cfg_best
        print(f'inst={seed} seed={s}: gap={gap:.3f}% hit={hit} t={dt:.0f}s', flush=True)
    if best_cfg is not None:
        json.dump(best_cfg, open(f'{EXP}/config_gri_small_best_{seed}.json', 'w'),
                  ensure_ascii=False)
    return rows


if __name__ == '__main__':
    only = os.environ.get('ONLY')
    if only is not None:
        rows = job(int(only))
        pd.DataFrame(rows).to_csv(f'{EXP}/results_gri_small_verify_inst{only}.csv', index=False)
    else:
        # 默认逐实例串行（批量并行由 gri_small_driver.py 通过独立进程实现）
        all_rows = []
        for i in range(11):
            rows = job(i)
            all_rows.extend(rows)
            pd.DataFrame(rows).to_csv(f'{EXP}/results_gri_small_verify_inst{i}{TAG}.csv', index=False)
            pd.DataFrame(all_rows).to_csv(f'{EXP}/results_gri_small_verify.csv', index=False)
        df = pd.DataFrame(all_rows)
        print('\n=== 小规模验证汇总 ===')
        for i, g in df.groupby('instance'):
            print(f'inst={i}: best={g.gap_pct.min():.3f}% mean={g.gap_pct.mean():.3f}% '
                  f'hits={g.exact_hit.sum()}/{len(g)} time_mean={g.time_s.mean():.0f}s')
