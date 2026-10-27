# -*- coding: utf-8 -*-
"""GRI 性能剖析：定位时间去向。"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios, make_batch_instance
from greedy_recourse import greedy_recourse2
import alns as A
from gri_alns import (gri_value, initial_current, exact_polish,
                      exact_pair_polish, config_violations)

base = json.load(open('instance_v1.json', encoding='utf-8'))

def bench(inst, label, n_rep=20):
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
    cfg = initial_current(inst)
    # greedy per scenario
    t0 = time.time()
    for _ in range(n_rep):
        greedy_recourse2(inst, cfg, train[0]['d'], require_cover=False)
    tg = (time.time() - t0) / n_rep * 1000
    # greedy full 10 scenarios (one candidate evaluation)
    t0 = time.time()
    for _ in range(max(3, n_rep // 4)):
        gri_value(inst, cfg, train)
    tc = (time.time() - t0) / max(3, n_rep // 4) * 1000
    # one exact LP eval (10 scenarios)
    t0 = time.time()
    for _ in range(3):
        A.evaluate(inst, cfg, train, require_cover=False)
    te = (time.time() - t0) / 3 * 1000
    print(f'{label:9s} |J|={len(inst["sorting"]):3d} |K|={len(inst["recycling"]):3d} '
          f'|L|={len(inst["landfill"]):3d} |I|={len(inst["demand"]):3d} | '
          f'greedy/scen={tg:6.1f}ms  candidate(10sc)={tc:7.1f}ms  exact(10sc)={te:8.1f}ms  '
          f'ratio exact/cand={te/max(tc,1e-9):5.1f}x', flush=True)
    return tc, te

for m, label in [(1, 'real'), (2, 'm2'), (4, 'm4'), (6, 'm6')]:
    inst = base if m == 1 else make_batch_instance(
        json.load(open('experiments/synth_instances/city_m%d_s1001.json' % m, encoding='utf-8')), 1) \
        if os.path.exists('experiments/synth_instances/city_m%d_s1001.json' % m) else None
    if inst is None:
        print(f'{label}: instance missing', flush=True)
        continue
    bench(inst, label)
