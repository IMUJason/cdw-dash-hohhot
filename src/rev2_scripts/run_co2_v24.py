# -*- coding: utf-8 -*-
"""V2.4 碳核算：200 共享测试情景，当前配置、当前模型。
同时输出：
  raw_tkm_M       最优二阶段流的运输量（百万 tonne-km）
  direct_tkm_M    反事实：把未服务量按当期可用填埋场最近直运弧补运后的运输量
  co2_raw_kt      raw_tkm 对应千吨 CO2（强度 0.10 kg/t-km）
  co2_direct_kt   direct_tkm 对应千吨 CO2
  unmet_Mt        未服务量（百万吨）
  recycled_Mt     再生厂投入量（百万吨）
反事实只用于运输/碳核算，不改变原文的服务失败定义。
"""
import json, os, sys, inspect
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
import run_co2

# 仅在本脚本内存中扩展返回流变量，不修改 run_co2.py
_src = inspect.getsource(run_co2.solve_tkm)
_src = _src.replace('return res.fun, tkm, unmet, recyc',
                    'return res.fun, tkm, unmet, recyc, x, idx')
_ns = {}
exec(_src, run_co2.__dict__, _ns)
solve_detail = _ns['solve_tkm']

METHODS = ['DET', 'SAA', 'DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']
INTENSITY = 0.10  # kg CO2 / tonne-km


def main():
    inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    test = gen_scenarios(inst, 200, seed=20260901, include_anchor=True)
    rows = []
    for m in METHODS:
        cfg = json.load(open(f'{D}/experiments/config_real_{m}.json', encoding='utf-8'))
        raw = direct = unmet_w = rec_w = 0.0
        n = 0
        for sc in test:
            cost, tkm, unmet, recyc, x, idx = solve_detail(inst, cfg, sc['d'])
            if cost is None:
                continue
            p = sc['prob']
            raw += p * tkm
            unmet_w += p * unmet
            rec_w += p * recyc
            add = 0.0
            for i in inst['demand']:
                for t in range(1, inst['meta']['periods'] + 1):
                    u = x[idx[('u', i, t)]]
                    if u <= 1e-9:
                        continue
                    live = [l for l in inst['landfill']
                            if cfg['xL'][l][t - 1] == 1 and cfg['rL'][l][t - 1] == 0]
                    if not live:
                        raise RuntimeError(f'{m}: no live landfill for counterfactual at t={t}')
                    add += u * min(inst['dist']['a2'][i][l] for l in live)
            direct += p * (tkm + add)
            n += 1
        raw_M = raw / 1e6
        direct_M = direct / 1e6
        rows.append({'method': m, 'n_scenarios': n,
                     'raw_tkm_M': raw_M,
                     'direct_tkm_M': direct_M,
                     'co2_raw_kt': raw_M * INTENSITY,
                     'co2_direct_kt': direct_M * INTENSITY,
                     'unmet_Mt': unmet_w / 1e6,
                     'recycled_Mt': rec_w / 1e6,
                     'intensity_kg_per_tkm': INTENSITY})
        print(f'{m:10s} raw={raw_M:.3f}M direct={direct_M:.3f}M '
              f'co2_direct={direct_M*INTENSITY:.3f}kt unmet={unmet_w/1e6:.3f}Mt',
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/experiments/results_co2_v24.csv', index=False)
    det = out[out.method == 'DET'].iloc[0]
    for m in METHODS[1:]:
        r = out[out.method == m].iloc[0]
        print(f'{m}: direct transport vs DET '
              f'{(r.direct_tkm_M/det.direct_tkm_M-1)*100:+.2f}%  '
              f'CO2 diff={r.co2_direct_kt-det.co2_direct_kt:+.3f}kt')
    print(f"wrote {D}/experiments/results_co2_v24.csv")


if __name__ == '__main__':
    main()
