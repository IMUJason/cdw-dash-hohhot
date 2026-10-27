# -*- coding: utf-8 -*-
"""ε* 闭式证书（V2.5）：全期需求/满配能力/库容的可行性下界。
F_min(D) = D - (1-kappa) * min(S, D)
DASH: eps_star = (R + (1-kappa) S) / D0 - 1  (D0>S 时)
验证：real + 10 副本的 0.16/0.18 可行性；rho/phi 9 格的 DASH(0.10) 可行性。
"""
import json, os, sys
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios, make_batch_instance
from gri_alns import initial_full, repair_monotone

base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))


def full_capacity(inst):
    cfg = initial_full(inst)
    T = inst['meta']['periods']
    for j in inst['sorting']:
        cfg['zJ'][j] = [4] * T
    for k in inst['recycling']:
        cfg['zK'][k] = [4] * T
    cfg = repair_monotone(cfg, inst)
    S = sum((inst['sorting'][j]['cap_t'] + inst['sorting'][j].get('expand_step_t', 30e4) * 4)
            * cfg['xJ'][j][t - 1]
            for j in inst['sorting'] for t in range(1, T + 1))
    R = sum(inst['landfill'][l]['cap_t'] for l in inst['landfill'])
    return S, R


def predict(inst, rho=None, phi=None):
    P = inst['params']
    rho = P['recyclable_ratio'] if rho is None else rho
    phi = P.get('product_yield', 1.0) if phi is None else phi
    kappa = (1 - rho) + rho * (1 - phi)
    S, R = full_capacity(inst)
    # 2024 峰值锚点（全期总量）
    D0 = 3 * sum(inst['demand'][i]['d']['2'] for i in inst['demand'])
    if D0 > S:
        eps = (R + (1 - kappa) * S) / D0 - 1
    else:
        eps = R / (kappa * D0) - 1
    return {'D0': D0, 'S': S, 'R': R, 'kappa': kappa, 'eps_pred': eps}


if __name__ == '__main__':
    rows = []
    for seed in [0] + list(range(1, 11)):
        inst = base if seed == 0 else make_batch_instance(base, seed)
        r = predict(inst)
        r['instance'] = 'real' if seed == 0 else f'batch{seed}'
        rows.append(r)
    df = pd.DataFrame(rows)
    print('=== 11 实例闭式预测 ===')
    print(df[['instance', 'D0', 'S', 'R', 'kappa', 'eps_pred']].to_string(index=False))
    df.to_csv(f'{D}/experiments/eps_formula_v2_instances.csv', index=False)

    print('\n=== rho/phi 网格：预测 eps* 与 DASH(0.10) 可行性 ===')
    grid = []
    for rho in (0.70, 0.80, 0.90):
        for phi in (0.75, 0.85, 0.95):
            r = predict(base, rho, phi)
            r.update({'rho': rho, 'phi': phi, 'dash010_feasible_pred': r['eps_pred'] >= 0.10})
            grid.append(r)
    gd = pd.DataFrame(grid)
    print(gd[['rho', 'phi', 'kappa', 'eps_pred', 'dash010_feasible_pred']].to_string(index=False))
    gd.to_csv(f'{D}/experiments/eps_formula_v2_rhophi.csv', index=False)
