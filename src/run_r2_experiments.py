#!/usr/bin/env python3
"""R2 补充实验（审稿人要求的两组）：
A. rho/phi 敏感性 —— 扫 recyclable_ratio x product_yield，检验 lean(SAA) vs robust(DASH 0.10)
   的结论是否稳健，并逐格校验 Assumption 1(i)（净弧成本非负）。
B. 保守度-收益光谱（DRO 基线消融）—— SAA / 均值-方差(MV) / DASH(eps) / ROB 在同一实例上的
   "计划成本 vs 样本外风险"权衡，回答"DASH 每单位保守度换来多少收益"。

设置与主实验严格一致：train=30 情景 seed=1000；test=200 情景 seed=20260901(include_anchor)；
ALNS 250 迭代 seed=42。SAA 行可逐位复现 results_main.csv（已校验偏差 0.0000%）。
"""
import json, os, sys, copy, time, math
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)
from cplex_saa import gen_scenarios
import alns as alns_mod
from run_experiments import run_method_alns, oos_eval, weighted_stats, EXP

BASE_INST = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
ITERS = 250


# ---------- 统一口径：planned_cost = F + Σ p_s Q(ξ_s)（规划者按经验模型的预期成本） ----------
def soft_stats(inst, cfg, scenarios):
    """返回 (planned_cost, sd) —— 按给定情景的加权均值与标准差。"""
    fc = alns_mod.fixed_cost_of(inst, cfg)
    qs, ps = [], []
    for sc in scenarios:
        c2 = alns_mod.solve_second_stage_lp(inst, cfg, sc['d'])
        if c2 is None:
            return None, None
        qs.append(c2); ps.append(sc['prob'])
    qs = np.array(qs); ps = np.array(ps); ps = ps / ps.sum()
    mean = float((qs * ps).sum())
    sd = float(np.sqrt((ps * (qs - mean) ** 2).sum()))
    return fc + mean, sd


def run_mv(inst, train, lam, iters=ITERS, seed=42):
    """均值-方差基线：min F + E[Q] + lam*sd(Q)。通过临时替换 alns.evaluate 实现。"""
    orig = alns_mod.evaluate

    def mv_eval(inst_, cfg, scen, FU=1e4):
        fc = alns_mod.fixed_cost_of(inst_, cfg, FU)
        qs, ps = [], []
        for sc in scen:
            c2 = alns_mod.solve_second_stage_lp(inst_, cfg, sc['d'])
            if c2 is None:
                return None
            qs.append(c2); ps.append(sc['prob'])
        qs = np.array(qs); ps = np.array(ps); ps = ps / ps.sum()
        mean = float((qs * ps).sum()); sd = float(np.sqrt((ps * (qs - mean) ** 2).sum()))
        return fc + mean + lam * sd

    alns_mod.evaluate = mv_eval
    try:
        t0 = time.time()
        planned, cfg, _ = run_method_alns(inst, 'SAA', train, alns_iters=iters, alns_seed=seed)
        wt = time.time() - t0
    finally:
        alns_mod.evaluate = orig
    _, sd = soft_stats(inst, cfg, train)
    return planned, cfg, wt, planned + lam * sd


def assumption1_check(inst):
    """Assumption 1(i): pi_rev*phi <= min_{j,k}(c_tr*d_jk + oK_k)（净弧成本非负）。"""
    P = inst['params']
    lhs = P.get('recycle_revenue', 0.0) * P.get('product_yield', 1.0)
    rhs = min(P['transport_cost'] * inst['dist']['b'][j][k] + inst['recycling'][k]['op_cost']
              for j in inst['dist']['b'] for k in inst['dist']['b'][j])
    return lhs, rhs, (lhs <= rhs + 1e-9)


# ==================== A. rho / phi 敏感性 ====================
def stage_A():
    rows = []
    train = gen_scenarios(BASE_INST, 30, seed=1000)          # 与主实验同种子（rho/phi 不影响情景）
    test = gen_scenarios(BASE_INST, 200, seed=20260901, include_anchor=True)
    for rho in [0.70, 0.80, 0.90]:
        for phi in [0.75, 0.85, 0.95]:
            inst = copy.deepcopy(BASE_INST)
            inst['params']['recyclable_ratio'] = rho
            inst['params']['product_yield'] = phi
            lhs, rhs, ok = assumption1_check(inst)
            for m in ['SAA', 'DRO(0.10)']:
                _native, cfg, wt = run_method_alns(inst, m, train, alns_iters=ITERS)
                planned, _sd = soft_stats(inst, cfg, train)
                costs, probs, unmets, recycs = oos_eval(inst, cfg, test)
                st = weighted_stats(costs, probs, planned, unmets, recycs)
                rows.append({
                    'rho': rho, 'phi': phi, 'method': m, 'time_s': wt,
                    'assump1_lhs': lhs, 'assump1_rhs': rhs, 'assump1_ok': ok,
                    'planned_cost': planned, **st})
                print(f'[A rho={rho} phi={phi}] {m:10s} plan={planned/1e9:.3f}B '
                      f'oos={st["oos_mean"]/1e9:.3f}B p95={st["oos_p95"]/1e9:.3f}B '
                      f'disc={st["disappointment_rate"]:.3f} recycled={st["recycled_mean_t"]/1e6:.2f}Mt '
                      f'A1={"ok" if ok else "FAIL"} {wt:.0f}s', flush=True)
                pd.DataFrame(rows).to_csv(f'{EXP}/results_r2_rhophi.csv', index=False)
    return rows


# ==================== B. 保守度-收益光谱 ====================
def stage_B():
    rows = []
    train = gen_scenarios(BASE_INST, 30, seed=1000)
    test = gen_scenarios(BASE_INST, 200, seed=20260901, include_anchor=True)

    def record(tag, planned, cfg, wt, native=None):
        _p2, sd = soft_stats(BASE_INST, cfg, train)
        costs, probs, unmets, recycs = oos_eval(BASE_INST, cfg, test)
        st = weighted_stats(costs, probs, planned, unmets, recycs)
        rows.append({'method': tag, 'native_obj': native, 'planned_cost': planned,
                     'train_sd': sd, **st})
        print(f'[B] {tag:12s} plan={planned/1e9:.3f}B oos={st["oos_mean"]/1e9:.3f}B '
              f'p95={st["oos_p95"]/1e9:.3f}B disc={st["disappointment_rate"]:.3f} '
              f'recycled={st["recycled_mean_t"]/1e6:.2f}Mt {wt:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(f'{EXP}/results_r2_baselines.csv', index=False)

    # 1) SAA（lean 参照）——同时作为与论文表 1 的一致性自检
    native, cfg, wt = run_method_alns(BASE_INST, 'SAA', train, alns_iters=ITERS)
    planned, _ = soft_stats(BASE_INST, cfg, train)
    assert abs(planned - 1071547247.85) / 1071547247.85 < 1e-6, f'SAA 复现失败: {planned}'
    print('[B] SAA 与论文表 1 逐位一致（自检通过）', flush=True)
    record('SAA', planned, cfg, wt, native=native)

    # 2) 均值-方差（矩基鲁棒化的可实现形式）
    for lam in [0.25, 0.50, 1.00]:
        planned, cfg, wt, mvobj = run_mv(BASE_INST, train, lam)
        record(f'MV(lambda={lam})', planned, cfg, wt, native=mvobj)

    # 3) DASH 各档
    for m in ['DRO(0.05)', 'DRO(0.10)', 'DRO(0.20)']:
        native, cfg, wt = run_method_alns(BASE_INST, m, train, alns_iters=ITERS)
        planned, _ = soft_stats(BASE_INST, cfg, train)
        record(m, planned, cfg, wt, native=native)

    # 4) ROB（固定抬升单情景）
    native, cfg, wt = run_method_alns(BASE_INST, 'ROB', train, alns_iters=ITERS)
    planned, _ = soft_stats(BASE_INST, cfg, train)
    record('ROB', planned, cfg, wt, native=native)
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if which in ('A', 'all'):
        print('=== Stage A: rho/phi 敏感性 ===', flush=True)
        stage_A()
    if which in ('B', 'all'):
        print('=== Stage B: 保守度-收益光谱 ===', flush=True)
        stage_B()
    print(f'\n总耗时 {time.time()-t0:.0f}s → {EXP}/results_r2_*.csv')
