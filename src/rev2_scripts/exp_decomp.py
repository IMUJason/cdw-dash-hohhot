"""V2.2 需求分解：纯投资份额 / 纯人口份额 / 去掉 2023 锚点 / 基线混合份额。
比较 DASH(0.10) 相对 DET 的实现成本降幅。"""
import json, sys, os, time, copy
import numpy as np, pandas as pd
from multiprocessing import Pool
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
EXP = f'{D}/experiments'
TL = int(os.environ.get('TL', '900'))


def job(arg):
    variant, m = arg
    from cplex_saa import gen_scenarios
    from run_experiments import run_method_det, run_method_exact, oos_eval, weighted_stats
    base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    inst = copy.deepcopy(base)
    GEN = list(inst['demand'])
    if variant == 'investment':
        # 纯投资份额：各区份额按投资份额，年度总量不变
        tot = {t: sum(inst['demand'][i]['d'][str(t)] for i in GEN) for t in range(1, 4)}
        # 实例中的份额本身即混合；此处用同一 support 但去掉人口项无法直接分离，
        # 因此改为：所有区等份额（纯人口近似：人口份额≈等权）
        for t in range(1, 4):
            share = tot[t] / len(GEN)
            for i in GEN:
                inst['demand'][i]['d'][str(t)] = share
    elif variant == 'equal':
        tot = {t: sum(inst['demand'][i]['d'][str(t)] for i in GEN) for t in range(1, 4)}
        for t in range(1, 4):
            for i in GEN:
                inst['demand'][i]['d'][str(t)] = tot[t] / len(GEN)
    elif variant == 'drop2023':
        # 去掉 2023 锚点（统计口径变更年）
        train = gen_scenarios(inst, 30, seed=1000)
        train = [sc for sc in train if not sc['name'].startswith('anchor_2023')]
        tot = sum(sc['prob'] for sc in train)
        for sc in train:
            sc['prob'] /= tot
    train = gen_scenarios(inst, 30, seed=1000)
    if variant == 'drop2023':
        train = [sc for sc in train if not sc['name'].startswith('anchor_2023')]
        tot = sum(sc['prob'] for sc in train)
        for sc in train:
            sc['prob'] /= tot
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    if m == 'DET':
        obj, cfg, wt = run_method_det(inst, time_limit=TL)
    else:
        obj, cfg, wt = run_method_exact(inst, m, train, time_limit=TL, require_cover=True)
    if obj is None:
        return {'variant': variant, 'method': m, 'ok': False}
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, obj, unm, rec)
    return {'variant': variant, 'method': m, 'ok': True, 'time_s': wt,
            'plan_obj': obj, 'oos_mean': st['oos_mean'], 'oos_p95': st['oos_p95'],
            'disc': st['disappointment_rate'], 'sfr': st.get('service_fail_rate', 0),
            'recycled': st.get('recycled_mean_t', 0)}


if __name__ == '__main__':
    tasks = [(v, m) for v in ['baseline', 'equal', 'drop2023'] for m in ['DET', 'DRO(0.10)']]
    with Pool(processes=6) as p:
        rows = p.map(job, tasks)
    df = pd.DataFrame(rows); df.to_csv(f'{EXP}/results_decomp_v22.csv', index=False)
    print('=== 需求分解 ===')
    for v in ['baseline', 'equal', 'drop2023']:
        g = df[df.variant == v].set_index('method')
        if 'DET' in g.index and 'DRO(0.10)' in g.index and g.loc['DET'].ok and g.loc['DRO(0.10)'].ok:
            d, h = g.loc['DET'], g.loc['DRO(0.10)']
            print(f'{v:10s}: DET oos={d.oos_mean/1e9:.4f}B  DASH(0.10) oos={h.oos_mean/1e9:.4f}B  '
                  f'降幅 {(1-h.oos_mean/d.oos_mean)*100:5.1f}%  sfr DET={d.sfr:.3f} DASH={h.sfr:.3f}')
