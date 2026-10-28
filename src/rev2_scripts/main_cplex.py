"""主实验（V2.2）：CPLEX 精确解，10 replica + real 实例，200 共享测试情景。

方法：DET（均值规划，不承诺覆盖）/ SAA / DASH(0.05, 0.10, 0.16)
DASH(0.16) 为当前标定下的最高可行档（0.18 全实例不可行）。

输出：
  experiments/results_main_v22.csv       主表
  experiments/config_<instance>_<method>.json   一阶段配置
"""
import json, sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
EXP = f'{D}/experiments'
METHODS = ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']


def job(arg):
    iname, seed, m, tl = arg
    from cplex_saa import gen_scenarios
    from run_experiments import (make_batch_instance, run_method_det, run_method_exact,
                                 oos_eval, weighted_stats)
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = base if seed == 0 else make_batch_instance(base, seed)
    train = gen_scenarios(inst, 30, seed=1000 + seed)
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    try:
        if m == 'DET':
            obj, cfg, wt = run_method_det(inst, time_limit=tl)
        else:
            obj, cfg, wt = run_method_exact(inst, m, train, time_limit=tl, require_cover=True)
    except Exception as e:
        return {'instance': iname, 'seed': seed, 'method': m, 'ok': False, 'err': str(e)}
    if obj is None:
        return {'instance': iname, 'seed': seed, 'method': m, 'ok': False,
                'err': 'infeasible_or_timeout', 'time_s': wt}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    # DET 的公共预算基准
    if m == 'DET':
        budget = obj
    else:
        budget = None
    json.dump(cfg, open(f'{EXP}/config_{iname}_{m}.json', 'w'), ensure_ascii=False)
    return {'instance': iname, 'seed': seed, 'method': m, 'ok': True, 'time_s': wt,
            'plan_obj': obj, 'oos_mean': st['oos_mean'], 'oos_p95': st['oos_p95'],
            'oos_std': st['oos_std'], 'disc': st['disappointment_rate'],
            'disc_mag': st['disappointment_mean'], 'sfr': st.get('service_fail_rate', 0),
            'unmet_mean': st.get('unmet_mean_t', 0), 'recycled': st.get('recycled_mean_t', 0)}


if __name__ == '__main__':
    tl = int(os.environ.get('TL', '900'))
    tasks = []
    for seed in [0] + list(range(1, 11)):
        for m in METHODS:
            tasks.append((('real' if seed == 0 else f'batch{seed}'), seed, m, tl))
    with Pool(processes=11) as p:
        rows = p.map(job, tasks)
    df = pd.DataFrame(rows)
    df.to_csv(f'{EXP}/results_main_v22.csv', index=False)
    print(f'\n=== 主表 V2.2 (CPLEX 精确, TL={tl}s) ===')
    for m in METHODS:
        g = df[(df.method == m)]
        ok = g[g.ok]
        if not len(ok):
            print(f'{m:10s} 求解 0/{len(g)}'); continue
        print(f'{m:10s} {len(ok):2d}/{len(g)}  t={ok.time_s.mean():6.0f}s  '
              f'plan={ok.plan_obj.mean()/1e9:.4f}±{ok.plan_obj.std()/1e9:.4f}B  '
              f'oos={ok.oos_mean.mean()/1e9:.4f}±{ok.oos_mean.std()/1e9:.4f}B  '
              f'p95={ok.oos_p95.mean()/1e9:.4f}B  disc={ok.disc.mean():.3f}±{ok.disc.std():.3f}  '
              f'sfr={ok.sfr.mean():.4f}', flush=True)
    print('\n--- real 实例 ---')
    for m in METHODS:
        g = df[(df.instance == 'real') & (df.method == m)]
        if len(g) and g.ok.iloc[0]:
            r = g.iloc[0]
            print(f'{m:10s} plan={r.plan_obj/1e9:.4f}B oos={r.oos_mean/1e9:.4f}B p95={r.oos_p95/1e9:.4f}B '
                  f'disc={r.disc:.3f} sfr={r.sfr:.4f} unmet={r.unmet_mean:,.0f}t '
                  f'recyc={r.recycled/1e6:.3f}Mt t={r.time_s:.0f}s')
        else:
            print(f'{m:10s} 不可行/超时')
