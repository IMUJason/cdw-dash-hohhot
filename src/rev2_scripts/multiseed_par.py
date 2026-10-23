"""多种子并行验证：10 种子 × 4 方法（ALNS 覆盖修复后的 §3.4 验收）。
用 multiprocessing Pool 并行（32 核），每个任务独立进程。
输出 experiments/results_seeds10.csv（V2.1 修复版）。"""
import json, sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)

BASE = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
METHODS = ['SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)']
ITERS = int(os.environ.get('ALNS_ITERS', '1000'))
NSEEDS = int(os.environ.get('NSEEDS', '10'))
EXP = f'{D}/experiments'


def job(arg):
    seed, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import run_method_alns, oos_eval, weighted_stats
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    train = gen_scenarios(base, 30, seed=1000)
    test = gen_scenarios(base, 200, seed=20260901, include_anchor=True)
    obj, cfg, wt = run_method_alns(base, m, train, alns_iters=ITERS, alns_seed=seed)
    costs, probs, unmets, recycs = oos_eval(base, cfg, test)
    st = weighted_stats(costs, probs, obj, unmets, recycs)
    return {'seed': seed, 'method': m, 'plan_obj': obj, 'oos_mean': st['oos_mean'],
            'oos_p95': st['oos_p95'], 'disappointment': st['disappointment_rate'],
            'service_fail': st.get('service_fail_rate', 0),
            'recycled': st.get('recycled_mean_t', 0), 'time_s': wt}


if __name__ == '__main__':
    t0 = time.time()
    tasks = [(s, m) for s in range(1, NSEEDS + 1) for m in METHODS]
    with Pool(processes=min(len(tasks), 16)) as p:
        rows = p.map(job, tasks)
    df = pd.DataFrame(rows).sort_values(['seed', 'method'])
    df.to_csv(f'{EXP}/results_seeds10.csv', index=False)
    print(f'\n=== 多种子汇总（mean±sd, n={NSEEDS}, iters={ITERS}）=== [{time.time()-t0:.0f}s]')
    for m in METHODS:
        g = df[df.method == m]
        print(f'{m:12s}: oos={g.oos_mean.mean()/1e9:.3f}±{g.oos_mean.std()/1e9:.3f}B  '
              f'plan={g.plan_obj.mean()/1e9:.3f}±{g.plan_obj.std()/1e9:.3f}B  '
              f'disc={g.disappointment.mean():.3f}±{g.disappointment.std():.3f}  '
              f'sfr={g.service_fail.mean():.4f} (max {g.service_fail.max():.4f})  '
              f'recycled={g.recycled.mean()/1e6:.2f}Mt  t={g.time_s.mean():.0f}s')
