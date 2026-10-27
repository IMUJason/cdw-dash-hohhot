"""V2.2 网格与消融实验（全部 CPLEX 精确解 + 服务级覆盖约束）：
  A. ρ/φ 网格（9 格 × 2 方法）
  B. 成本×收益网格（5×5 × 3 方法）
  C. 消融：MV(λ) / DASH / 空白lift 对比
输出 experiments/results_r2_*_v22.csv
"""
import json, sys, os, time, copy
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
EXP = f'{D}/experiments'
TL = int(os.environ.get('TL', '900'))


def _solve(inst, m, train, tl):
    from run_experiments import run_method_det, run_method_exact
    if m == 'DET':
        return run_method_det(inst, time_limit=tl)
    return run_method_exact(inst, m, train, time_limit=tl, require_cover=True)


def job_grid(arg):
    """成本×收益网格：3 方法 × 25 格"""
    tc, rev, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval, weighted_stats
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    inst['params']['transport_cost'] = tc
    inst['params']['recycle_revenue'] = rev
    train = gen_scenarios(inst, 30, seed=1000)
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    obj, cfg, wt = _solve(inst, m, train, TL)
    if obj is None:
        return {'tc': tc, 'rev': rev, 'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    return {'tc': tc, 'rev': rev, 'method': m, 'ok': True, 'time_s': wt,
            'plan_obj': obj, 'oos_mean': st['oos_mean'], 'oos_p95': st['oos_p95'],
            'disc': st['disappointment_rate'], 'sfr': st.get('service_fail_rate', 0),
            'recycled': st.get('recycled_mean_t', 0)}


def job_rhophi(arg):
    """ρ/φ 网格：2 方法 × 9 格"""
    rho, phi, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval, weighted_stats
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    inst['params']['recyclable_ratio'] = rho
    inst['params']['product_yield'] = phi
    P = inst['params']
    lhs = P.get('recycle_revenue', 0.0) * P.get('product_yield', 1.0)
    rhs = min(P['transport_cost'] * inst['dist']['b'][j][k] + inst['recycling'][k]['op_cost']
              for j in inst['dist']['b'] for k in inst['dist']['b'][j])
    train = gen_scenarios(inst, 30, seed=1000)
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    obj, cfg, wt = _solve(inst, m, train, TL)
    if obj is None:
        return {'rho': rho, 'phi': phi, 'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    return {'rho': rho, 'phi': phi, 'method': m, 'ok': True, 'time_s': wt,
            'assump1_lhs': lhs, 'assump1_rhs': rhs, 'assump1_ok': lhs <= rhs + 1e-9,
            'plan_obj': obj, 'oos_mean': st['oos_mean'], 'oos_p95': st['oos_p95'],
            'disc': st['disappointment_rate'], 'sfr': st.get('service_fail_rate', 0),
            'recycled': st.get('recycled_mean_t', 0)}


def job_ablation(arg):
    """消融：MV(λ) 三档 + 空白lift(ROB) + SAA + DASH 三档"""
    kind, val, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import oos_eval, weighted_stats, run_method_alns
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    train = gen_scenarios(base, 30, seed=1000)
    test = gen_scenarios(base, 200, seed=20260901, include_anchor=True)
    if kind == 'MV':
        import alns as A
        from run_experiments import run_method_alns as rma
        def mv_obj(cost, wunmet, qs, ps):
            mean = float((qs*ps).sum()); sd = float(np.sqrt((ps*(qs-mean)**2).sum()))
            return cost + val*sd + A.COVER_M*wunmet
        obj, cfg, wt = rma(base, 'SAA', train, alns_iters=1000, alns_seed=42, obj_fn=mv_obj)
        if obj is None:
            return {'method': f'MV({val})', 'ok': False}
    elif kind == 'BLANKET':
        # 空白lift：单情景 +20%（无经验中心），ROB
        t_start = time.time()
        T = base['meta']['periods']
        full = [{'name': 'wc', 'prob': 1.0,
                 'd': {i: {t: train[0]['d'][i][t]*1.2 for t in range(1, T+1)} for i in base['demand']}}]
        from cplex_saa import build_and_solve_saa
        from run_experiments import extract_xconfig_cpx, _cfg_from_xconfig
        mdl, sol, V = build_and_solve_saa(base, full, 'blanket', retire_lstar=True, t_star=2,
                                           log=False, require_cover=True, time_limit=TL)
        if not sol:
            return {'method': 'ROB', 'ok': False}
        obj = sol.objective_value
        cfg = _cfg_from_xconfig(base, extract_xconfig_cpx(base, V))
        wt = time.time() - t_start
    else:
        obj, cfg, wt = _solve(base, m, train, TL)
        if obj is None:
            return {'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(base, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    return {'method': (f'MV({val})' if kind == 'MV' else ('ROB' if kind == 'BLANKET' else m)),
            'ok': True, 'time_s': wt, 'plan_obj': obj, 'oos_mean': st['oos_mean'],
            'oos_p95': st['oos_p95'], 'disc': st['disappointment_rate'],
            'sfr': st.get('service_fail_rate', 0), 'recycled': st.get('recycled_mean_t', 0)}


if __name__ == '__main__':
    which = os.environ.get('WHICH', 'all')
    if which in ('grid', 'all'):
        tasks = [(tc, rev, m) for tc in [0.6, 0.8, 1.0, 1.2, 1.5] for rev in [0, 10, 20, 30, 40]
                 for m in ['DET', 'SAA', 'DRO(0.10)']]
        with Pool(processes=10) as p:
            rows = p.map(job_grid, tasks)
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_grid_v22.csv', index=False)
        print('=== 成本×收益网格 ===')
        ok = df[df.ok]
        for m in ['DET', 'SAA', 'DRO(0.10)']:
            g = ok[ok.method == m]
            if len(g):
                wins = sum(1 for _, r in g.iterrows()
                           if r.oos_mean <= ok[ok.method != m].groupby(['tc', 'rev']).oos_mean.min().get((r.tc, r.rev), 1e18))
                print(f'  {m:10s} oos {g.oos_mean.min()/1e9:.3f}--{g.oos_mean.max()/1e9:.3f}B  解出 {len(g)}/{len(tasks)//3}')
    if which in ('rhophi', 'all'):
        tasks = [(rho, phi, m) for rho in [0.70, 0.80, 0.90] for phi in [0.75, 0.85, 0.95]
                 for m in ['SAA', 'DRO(0.10)']]
        with Pool(processes=9) as p:
            rows = p.map(job_rhophi, tasks)
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_rhophi_v22.csv', index=False)
        print('\n=== ρ/φ 网格 ===')
        print(df.to_string(index=False))
    if which in ('ablation', 'all'):
        tasks = [('MV', 0.25, None), ('MV', 0.5, None), ('MV', 1.0, None),
                 ('SAA', None, 'SAA'), ('DASH', None, 'DRO(0.05)'),
                 ('DASH', None, 'DRO(0.10)'), ('DASH', None, 'DRO(0.16)'),
                 ('BLANKET', None, None)]
        with Pool(processes=8) as p:
            rows = p.map(job_ablation, tasks)
        df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_ablation_v22.csv', index=False)
        print('\n=== 消融 ===')
        print(df.to_string(index=False))
