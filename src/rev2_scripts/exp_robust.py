"""V2.2 补充实验：惩罚扫描 / 回测 / 留一年回测（全部 CPLEX 精确解 + 服务级约束）。
并行执行，输出：
  experiments/results_penalty_v22.csv
  experiments/results_backtest_v22.csv
  experiments/results_loo_v22.csv
"""
import json, sys, os, time, copy, random
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
EXP = f'{D}/experiments'
METHODS = ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.18)']
TL = int(os.environ.get('TL', '900'))


def _solve(inst, m, train, tl):
    from run_experiments import run_method_det, run_method_exact
    if m == 'DET':
        return run_method_det(inst, time_limit=tl)
    return run_method_exact(inst, m, train, time_limit=tl, require_cover=True)


# ---------- 惩罚扫描 ----------
def job_penalty(arg):
    pen, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval, weighted_stats
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    inst['params']['unmet_penalty'] = pen
    train = gen_scenarios(inst, 30, seed=1000)
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    obj, cfg, wt = _solve(inst, m, train, TL)
    if obj is None:
        return {'penalty': pen, 'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    E_u = st.get('unmet_mean_t', 0.0)
    return {'penalty': pen, 'method': m, 'ok': True, 'time_s': wt, 'plan_obj': obj,
            'oos_mean': st['oos_mean'], 'oos_p95': st['oos_p95'],
            'disappointment': st['disappointment_rate'],
            'sfr': st.get('service_fail_rate', 0), 'unmet_mean_t': E_u,
            'oos_nopen_approx': st['oos_mean'] - (pen - 10.0) * E_u,
            'recycled': st.get('recycled_mean_t', 0)}


# ---------- 回测（2023-24 校准，2025 检验） ----------
def job_backtest(arg):
    m, = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    GEN = list(inst['demand'])
    scen = []
    for yr in [1, 2]:
        d = {i: {t: inst['demand'][i]['d'][str(yr)] for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'anchor_hist{yr}', 'prob': 0.5, 'd': d})
    for s in range(8):
        rng = random.Random(500 + s)
        d = {i: {t: scen[0]['d'][i][t]*rng.uniform(0.85, 1.15) for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'hist_s{s}', 'prob': 0.5/8, 'd': d})
    test = [{'name': 'actual_2025', 'prob': 1.0,
             'd': {i: {t: inst['demand'][i]['d']['3'] for t in range(1, 4)} for i in GEN}}]
    obj, cfg, wt = _solve(inst, m, scen, TL)
    if obj is None:
        return {'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    actual = costs[0]
    return {'method': m, 'ok': True, 'time_s': wt, 'plan_obj': obj, 'actual_2025': actual,
            'regret': actual - obj, 'disappointed': bool(actual > obj*1.001)}


# ---------- 留一年回测 ----------
def job_loo(arg):
    hold, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    GEN = list(inst['demand'])
    scen = []
    for yr in [1, 2, 3]:
        if yr == hold:
            continue
        d = {i: {t: inst['demand'][i]['d'][str(yr)] for t in range(1, 4)} for i in GEN}
        scen.append({'name': f'anchor{yr}', 'prob': 0.5, 'd': d})
    for s in range(8):
        rng = random.Random(600 + s)
        base_d = scen[0]['d']
        d = {i: {t: base_d[i][t]*rng.uniform(0.85, 1.15) for t in range(1, 4)} for i in GEN}
        scen.append({'name': f's{s}', 'prob': 0.5/8, 'd': d})
    test = [{'name': f'actual{hold}', 'prob': 1.0,
             'd': {i: {t: inst['demand'][i]['d'][str(hold)] for t in range(1, 4)} for i in GEN}}]
    obj, cfg, wt = _solve(inst, m, scen, TL)
    if obj is None:
        return {'holdout_year': hold, 'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    actual = costs[0]
    return {'holdout_year': hold, 'method': m, 'ok': True, 'time_s': wt, 'plan_obj': obj,
            'actual': actual, 'regret': actual - obj,
            'disappointed': bool(actual > obj*1.001)}


if __name__ == '__main__':
    which = os.environ.get('WHICH', 'all')
    if which in ('penalty', 'all'):
        tasks = [(pen, m) for pen in [100.0, 250.0, 500.0, 1000.0]
                 for m in ['DET', 'SAA', 'DRO(0.10)']]
        with Pool(processes=12) as p:
            rows = p.map(job_penalty, tasks)
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_penalty_v22.csv', index=False)
        print('=== 惩罚扫描 ===')
        print(df.to_string(index=False))
    if which in ('backtest', 'all'):
        with Pool(processes=6) as p:
            rows = p.map(job_backtest, [(m,) for m in METHODS])
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_backtest_v22.csv', index=False)
        print('\n=== 回测（2023-24 校准 → 2025 检验）===')
        print(df.to_string(index=False))
    if which in ('loo', 'all'):
        tasks = [(hold, m) for hold in [1, 2, 3] for m in METHODS]
        with Pool(processes=12) as p:
            rows = p.map(job_loo, tasks)
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_loo_v22.csv', index=False)
        print('\n=== 留一年回测 ===')
        print(df.to_string(index=False))
