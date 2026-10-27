# -*- coding: utf-8 -*-
"""点质量 Wasserstein 参照（V2.5）。
对同一配置 x*，计算 DASH 混合目标与 lifted support 上最差点质量目标。
点质量分布属于相应 Wasserstein 球，因此其目标值是精确最坏情形的下界，
不是 CCG 精确最坏情形解。输出 results_worstcase_ref.csv。
"""
import json, os, sys
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import dash_scenarios
from gri_alns import fixed_cost_of
import alns as A


def main():
    inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
    train = gen_scenarios(inst, 30, seed=1000)
    rows = []
    for eps, method in [(0.05, 'DRO(0.05)'), (0.10, 'DRO(0.10)'), (0.16, 'DRO(0.16)')]:
        cfg = json.load(open(f'{D}/experiments/config_real_{method}.json', encoding='utf-8'))
        fc = fixed_cost_of(inst, cfg)
        full = dash_scenarios(inst, train, eps)
        mix = fc
        worst = (-1.0, None)
        n_feas = 0
        for sc in full:
            c2, unm, rec = A.solve_second_stage_lp_ex(inst, cfg, sc['d'], require_cover=True)
            if c2 is None:
                continue
            n_feas += 1
            mix += sc['prob'] * c2
            if sc['name'].endswith('_wc') and fc + c2 > worst[0]:
                worst = (fc + c2, sc['name'])
        point_obj, point_sc = worst
        rows.append({'method': method, 'eps': eps, 'fixed_cost': fc,
                     'dash_mix_obj': mix, 'pointmass_obj': point_obj,
                     'pointmass_minus_dash': point_obj - mix,
                     'point_scenario': point_sc, 'n_lifted_feasible': n_feas,
                     'n_scenarios': len(full),
                     'reference': 'point-mass lower bound on exact Wasserstein worst case'})
        print(f'{method}: mix={mix/1e6:.3f}M point={point_obj/1e6:.3f}M '
              f'diff={(point_obj-mix)/1e6:+.3f}M scenario={point_sc} '
              f'feasible={n_feas}/{len(full)}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/experiments/results_worstcase_ref.csv', index=False)
    print(f'wrote {D}/experiments/results_worstcase_ref.csv')


if __name__ == '__main__':
    main()
