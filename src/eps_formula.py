# -*- coding: utf-8 -*-
"""ε* 闭式候选量验证：满配下用最小总库容余量公式推上限。
R_total = 各填埋场可用量（当期+累计取小）之和。
峰值情景需求 D_peak = max_t Σ_i d_it。压力后 D_peak*(1+ε)。
满配最省库容方案所需库容 = D_peak*(1+ε)*min(1, κ*S_cap + ... )?
实际上满配全分拣时库容=κ*D_peak*(1+ε)（若分拣能力够）。
若分拣能力不足，需部分直运，库容=κ*S_cap + (D-S_cap)。
故所需最小库容 = D - (1-κ)*min(S_cap, D)。
可行 iff D - (1-κ)*min(S_cap, D) <= R_total
⇒ (1-κ)*min(S_cap, D) >= D - R_total
若 S_cap >= D: 需要 (1-κ)D >= D-R_total → κD <= R_total → D <= R_total/κ
若 S_cap < D: 需要 (1-κ)S_cap >= D-R_total → D <= R_total + (1-κ)S_cap
⇒ ε* = [R_total + (1-κ)*S_cap] / D_peak - 1 （当 S_cap < D_peak 时）
或 ε* = R_total/(κ D_peak) - 1 （当 S_cap >= D_peak 时）
"""
import json, os, sys, pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import reduce_scenarios, make_batch_instance
from gri_alns import initial_full, repair_monotone

base = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))


def predict(inst, peak_scen):
    cfg = initial_full(inst)
    T = inst['meta']['periods']
    for j in inst['sorting']:
        cfg['zJ'][j] = [4] * T
    for k in inst['recycling']:
        cfg['zK'][k] = [4] * T
    cfg = repair_monotone(cfg, inst)
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)
    kappa = (1 - rho) + rho * (1 - phi)
    D_peak = max(sum(peak_scen['d'][i][t] for i in inst['demand']) for t in range(1, T + 1))
    S_cap = sum(inst['sorting'][j]['cap_t'] + inst['sorting'][j].get('expand_step_t', 30e4) * 4
                for j in inst['sorting'] if cfg['xJ'][j][0] == 1)
    # 可用填埋量：当期+累计取小之和（保守：取当期可用量）
    R_total = sum(inst['landfill'][l]['cap_t'] for l in inst['landfill']
                  if cfg['xL'][l][0] == 1 and cfg['rL'][l][0] == 0)
    if S_cap >= D_peak:
        eps_star = R_total / (kappa * D_peak) - 1
    else:
        eps_star = (R_total + (1 - kappa) * S_cap) / D_peak - 1
    return {'D_peak': D_peak, 'S_cap': S_cap, 'R_total': R_total,
            'kappa': kappa, 'eps_pred': eps_star}


if __name__ == '__main__':
    rows = []
    for seed in [0] + list(range(1, 11)):
        inst = base if seed == 0 else make_batch_instance(base, seed)
        train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed), k=10, seed=7)
        peak = max(train, key=lambda sc: max(sum(sc['d'][i][t] for i in inst['demand'])
                                            for t in range(1, 4)))
        r = predict(inst, peak)
        r['instance'] = 'real' if seed == 0 else f'batch{seed}'
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df[['instance', 'D_peak', 'S_cap', 'R_total', 'kappa', 'eps_pred']].to_string(index=False))
    df.to_csv(f'{D}/experiments/eps_formula_prediction.csv', index=False)
