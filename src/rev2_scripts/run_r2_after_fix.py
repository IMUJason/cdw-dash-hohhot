#!/usr/bin/env python3
"""R2 补充实验：rho/phi 敏感性网格 + 鲁棒化方案消融（承诺口径）。

ALNS 修复后需重跑本脚本，输出：
  experiments/results_r2_rhophi.csv      (9 格 × 2 方法)
  experiments/results_r2_baselines.csv   (SAA / MV×3 / DASH×3 / ROB)

用法（在 model 目录下）：
  /opt/anaconda3/envs/env310/bin/python rev2_scripts/run_r2_after_fix.py
口径约定：所有方法的 committed_cost 用其自身目标值（DASH 用应力分布期望，
SAA 用经验期望，ROB 用单情景值，MV 用均值+λ·sd），与主表 Plan 列一致。
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
os.chdir(D)

from cplex_saa import gen_scenarios                      # noqa: E402
import alns as A                                         # noqa: E402
from run_experiments import (run_method_alns, oos_eval,  # noqa: E402
                             weighted_stats, EXP)

BASE = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))


def assumption1(inst):
    P = inst['params']
    lhs = P.get('recycle_revenue', 0.0) * P.get('product_yield', 1.0)
    rhs = min(P['transport_cost'] * inst['dist']['b'][j][k] + inst['recycling'][k]['op_cost']
              for j in inst['dist']['b'] for k in inst['dist']['b'][j])
    return lhs, rhs, lhs <= rhs + 1e-9


def record(tag, native, cfg, inst, test, rows):
    costs, probs, unm, rec = oos_eval(inst, cfg, test)
    st = weighted_stats(costs, probs, native, unm, rec)
    rows.append({'method': tag, 'committed_cost': native, **st})


def run_mv(inst, train, lam, iters=1000, seed=42):
    """均值-方差基线：min F + E[Q] + lam*sd(Q)，通过 alns 的 obj_fn 注入
    （接受准则保留覆盖增强项，与主方法同一搜索协议，隔离目标差异）。"""
    def mv_obj(cost, wunmet, qs, ps):
        mean = float((qs * ps).sum())
        sd = float(np.sqrt((ps * (qs - mean) ** 2).sum()))
        return cost + lam * sd + A.COVER_M * wunmet

    _, cfg, _ = run_method_alns(inst, 'SAA', train,
                                alns_iters=iters, alns_seed=seed, obj_fn=mv_obj)
    # committed cost = MV 自身目标值（在全训练集上重算）
    fc = A.fixed_cost_of(inst, cfg)
    qs, ps = [], []
    for sc in train:
        c2 = A.solve_second_stage_lp(inst, cfg, sc['d'])
        if c2 is None:
            return None, cfg
        qs.append(c2); ps.append(sc['prob'])
    qs = np.array(qs); ps = np.array(ps); ps = ps / ps.sum()
    mean = float((qs * ps).sum()); sd = float(np.sqrt((ps * (qs - mean) ** 2).sum()))
    return fc + mean + lam * sd, cfg


def main():
    train = gen_scenarios(BASE, 30, seed=1000)
    test = gen_scenarios(BASE, 200, seed=20260901, include_anchor=True)

    # ---------- A. rho / phi 网格 ----------
    rows = []
    for rho in [0.70, 0.80, 0.90]:
        for phi in [0.75, 0.85, 0.95]:
            inst = copy.deepcopy(BASE)
            inst['params']['recyclable_ratio'] = rho
            inst['params']['product_yield'] = phi
            lhs, rhs, ok = assumption1(inst)
            for m in ['SAA', 'DRO(0.10)']:
                native, cfg, wt = run_method_alns(inst, m, train)
                costs, probs, unm, rec = oos_eval(inst, cfg, test)
                st = weighted_stats(costs, probs, native, unm, rec)
                rows.append({'rho': rho, 'phi': phi, 'method': m, 'time_s': wt,
                             'assump1_lhs': lhs, 'assump1_rhs': rhs, 'assump1_ok': ok,
                             'committed_cost': native, **st})
                print(f"[A rho={rho} phi={phi}] {m:9s} "
                      f"commit={native/1e9:.3f}B oos={st['oos_mean']/1e9:.3f}B "
                      f"disc={st['disappointment_rate']:.3f} "
                      f"sfr={st.get('service_fail_rate', 0):.3f} "
                      f"A1={'ok' if ok else 'FAIL'}", flush=True)
                pd.DataFrame(rows).to_csv(f'{EXP}/results_r2_rhophi.csv', index=False)

    # ---------- B. 鲁棒化方案消融 ----------
    rows = []
    native, cfg, _ = run_method_alns(BASE, 'SAA', train)
    record('SAA', native, cfg, BASE, test, rows)
    print(f"[B] SAA commit={native/1e9:.3f}B", flush=True)
    for lam in [0.25, 0.50, 1.00]:
        native, cfg = run_mv(BASE, train, lam)
        record(f'MV(lambda={lam})', native, cfg, BASE, test, rows)
        print(f"[B] MV({lam}) commit={native/1e9:.3f}B", flush=True)
    for m in ['DRO(0.05)', 'DRO(0.10)', 'DRO(0.16)']:
        native, cfg, _ = run_method_alns(BASE, m, train)
        record(m, native, cfg, BASE, test, rows)
        print(f"[B] {m} commit={native/1e9:.3f}B", flush=True)
    native, cfg, _ = run_method_alns(BASE, 'ROB', train)
    record('ROB', native, cfg, BASE, test, rows)
    print(f"[B] ROB commit={native/1e9:.3f}B", flush=True)
    pd.DataFrame(rows).to_csv(f'{EXP}/results_r2_baselines.csv', index=False)

    print('\n完成 → experiments/results_r2_*.csv')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'总耗时 {time.time()-t0:.0f}s')
