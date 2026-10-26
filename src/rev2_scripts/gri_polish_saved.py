# -*- coding: utf-8 -*-
"""在已保存的 GRI 配置上测试增强抛光（不重新搜索）。"""
import json, sys, os, glob, time
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios
import alns as A
from gri_alns import exact_polish, exact_pair_polish

inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
opt = 853904000.0
import pandas as pd
cpx = pd.read_csv(f'{D}/experiments/results_gri_small_cplex.csv')
opt = float(cpx[cpx.seed == 0].opt.iloc[0])

def ex(c):
    return A.evaluate(inst, c, train, require_cover=True)

best = None
for fn in sorted(glob.glob(f'{D}/experiments/config_gri_small_0_s*.json')):
    cfg = json.load(open(fn))
    v = ex(cfg)
    if v is not None:
        print(f'{os.path.basename(fn)}: {v/1e6:.2f}M gap={(v/opt-1)*100:.2f}%', flush=True)
        if best is None or v < best[0]:
            best = (v, cfg, fn)
if os.path.exists(f'{D}/experiments/config_gri_polished_test.json'):
    c2 = json.load(open(f'{D}/experiments/config_gri_polished_test.json'))
    v2 = ex(c2)
    if v2 is not None and (best is None or v2 < best[0]):
        best = (v2, c2, f'{D}/experiments/config_gri_polished_test.json')
if best:
    print(f'polishing {os.path.basename(best[2])} ...', flush=True)
    t0 = time.time()
    cfg, val, nev = exact_polish(inst, best[1], train, ex, budget_s=120)
    cfg, val, nev2 = exact_pair_polish(inst, cfg, train, ex, budget_s=180)
    nev += nev2
    print(f'after: {val/1e6:.3f}M gap={(val/opt-1)*100:.4f}% '
          f'evals={nev} t={time.time()-t0:.0f}s', flush=True)
    json.dump(cfg, open(f'{D}/experiments/config_gri_polished_test.json', 'w'), ensure_ascii=False)
