# -*- coding: utf-8 -*-
"""未命中实例的二次精修：best-of-5 配置 + 满配保底，延长一阶/二阶精确抛光。
ONLY=<instance>。结果写 results_gri_refine_inst<i>.csv。
"""
import json, sys, os, glob, time
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios, make_batch_instance
import alns as A
from gri_alns import (exact_polish, exact_pair_polish, initial_full,
                      repair_monotone, config_violations)

EXP = f'{D}/experiments'
ONLY = int(os.environ.get('ONLY', '0'))


def main():
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if ONLY == 0 else make_batch_instance(base, ONLY)
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + ONLY), k=10, seed=7)
    opt = float(pd.read_csv(f'{EXP}/results_gri_small_cplex.csv').query('seed==@ONLY').opt.iloc[0])

    def ex(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)

    candidates = []
    for fn in (glob.glob(f'{EXP}/config_gri_small_{ONLY}_s*.json')
               + glob.glob(f'{EXP}/config_gri_refine_{ONLY}.json')):
        c = json.load(open(fn))
        v = ex(c)
        if v is not None:
            candidates.append((v, c, os.path.basename(fn)))
    # 满配保底：即使某轮搜索没有 LP 可行池，也有合法起点
    c = initial_full(inst)
    for j in inst['sorting']:
        c['zJ'][j] = [4] * 3
    for k in inst['recycling']:
        c['zK'][k] = [4] * 3
    c = repair_monotone(c, inst)
    v = ex(c)
    if v is not None:
        candidates.append((v, c, 'max-build'))
    if not candidates:
        print('no feasible starting point', flush=True)
        return
    candidates.sort(key=lambda x: x[0])
    print(f'inst={ONLY} starts={len(candidates)} best={candidates[0][0]/1e6:.3f}M', flush=True)
    best_v, best_c = candidates[0][0], candidates[0][1]
    t0 = time.time()
    # 从最好的两个不同起点分别精修，避免单点轨迹依赖
    for rank, (v0, c0, nm) in enumerate(candidates[:2]):
        c, v, _ = exact_polish(inst, c0, train, ex, budget_s=180)
        c, v, _ = exact_pair_polish(inst, c, train, ex, budget_s=300, max_rounds=8)
        print(f'  {nm}: {v/1e6:.4f}M gap={(v/opt-1)*100:.5f}%', flush=True)
        if v < best_v:
            best_v, best_c = v, c
        if abs(best_v - opt) / opt < 1e-6:
            break
    hit = abs(best_v - opt) / opt < 1e-6
    json.dump(best_c, open(f'{EXP}/config_gri_refine_{ONLY}.json', 'w'), ensure_ascii=False)
    pd.DataFrame([{'instance': ONLY, 'opt': opt, 'gri_obj': best_v,
                   'gap_pct': (best_v/opt-1)*100, 'exact_hit': hit,
                   'time_s': time.time()-t0}]).to_csv(
        f'{EXP}/results_gri_refine_inst{ONLY}.csv', index=False)
    print(f'REFINE inst={ONLY}: gap={(best_v/opt-1)*100:.5f}% hit={hit} '
          f't={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
