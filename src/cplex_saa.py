#!/usr/bin/env python3
"""两阶段 SAA（样本平均近似）实现 + V3 退化一致性验证。
对应 model_spec_v1.md §5 + 本文件 §SAA 扩展：
  一阶段(here-and-now): xJ/xK/xL/zK/rL（与情景无关）
  二阶段(wait-and-see): y/yd/w/q/v/s/u（随情景 s，概率 p_s）
  min fixed + Σ_s p_s·(transport+ops+hold+penalty−revenue)_s

场景生成（诚实小样本原则，不可声称经验分布）:
  需求支撑集 = v3 基准 × [0.7, 1.3] 区间（±30%），各区县独立均匀采样 + 3 个官方年份锚点情景。
V3 验证: S=1（均值情景）时 SAA 目标值必须等于 D0 目标值（误差<1e-3）。
"""
import json, os, sys, random
from docplex.mp.model import Model

BASE = os.path.dirname(os.path.abspath(__file__))

def gen_scenarios(inst, n_saa, seed=20260825, include_anchor=True):
    """生成需求情景列表 [{name, prob, d: {i: {t: 值}}}]。
    锚点情景：3 个官方年份总量模式（v3份额×官方总量）；
    采样情景：支撑集 [0.7,1.3]×基准 独立均匀采样。"""
    rng = random.Random(seed)
    GEN = list(inst['demand'])
    T = inst['meta']['periods']
    base = {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in GEN}
    scenarios = []
    if include_anchor:
        # 官方年份锚点（概率各1/9）：2023低/2024峰/2025谷
        for yr, tag in [(1, 'anchor_2023'), (2, 'anchor_2024'), (3, 'anchor_2025')]:
            scenarios.append({'name': tag, 'prob': 1/9,
                              'd': {i: {t: base[i][yr] for t in range(1, T+1)} for i in GEN}})
        n_sample = n_saa - 3
        p_sample = 2/3 / max(n_sample, 1)
    else:
        n_sample = n_saa
        p_sample = 1.0 / n_saa
    for s in range(n_sample):
        d = {i: {t: base[i][t] * rng.uniform(0.7, 1.3) for t in range(1, T+1)} for i in GEN}
        scenarios.append({'name': f's{s+1}', 'prob': p_sample, 'd': d})
    # 归一化概率
    tot = sum(sc['prob'] for sc in scenarios)
    for sc in scenarios:
        sc['prob'] /= tot
    return scenarios

def build_and_solve_saa(inst, scenarios, name='saa', retire_lstar=False, t_star=2, log=False, fixed_unit=1e4):
    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting']); RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)
    NS = len(scenarios)

    mdl = Model(name)
    # 一阶段变量
    xJ = mdl.binary_var_dict([(j, t) for j in SORT for t in range(1, T+1)], name='xJ')
    xK = mdl.binary_var_dict([(k, t) for k in RECY for t in range(1, T+1)], name='xK')
    xL = mdl.binary_var_dict([(l, t) for l in LAND for t in range(1, T+1)], name='xL')
    zK = mdl.integer_var_dict([(k, t) for k in RECY for t in range(1, T+1)], lb=0, ub=4, name='zK')
    zJ = mdl.integer_var_dict([(j, t) for j in SORT for t in range(1, T+1)], lb=0, ub=4, name='zJ')  # 分拣扩容档（V1.2：现实2→16家分拣扩张）
    rL = mdl.binary_var_dict([(l, t) for l in LAND for t in range(1, T+1)], name='rL')
    # 二阶段变量（按情景）
    y = mdl.continuous_var_dict([(i, j, t, sn) for i in GEN for j in SORT for t in range(1, T+1) for sn in range(NS)], lb=0, name='y')
    yd = mdl.continuous_var_dict([(i, l, t, sn) for i in GEN for l in LAND for t in range(1, T+1) for sn in range(NS)], lb=0, name='yd')
    w = mdl.continuous_var_dict([(j, k, t, sn) for j in SORT for k in RECY for t in range(1, T+1) for sn in range(NS)], lb=0, name='w')
    q = mdl.continuous_var_dict([(j, l, t, sn) for j in SORT for l in LAND for t in range(1, T+1) for sn in range(NS)], lb=0, name='q')
    v = mdl.continuous_var_dict([(k, l, t, sn) for k in RECY for l in LAND for t in range(1, T+1) for sn in range(NS)], lb=0, name='v')
    s = mdl.continuous_var_dict([(j, t, sn) for j in SORT for t in range(1, T+1) for sn in range(NS)], lb=0, name='s')
    u = mdl.continuous_var_dict([(i, t, sn) for i in GEN for t in range(1, T+1) for sn in range(NS)], lb=0, name='u')

    FU = fixed_unit
    fixed = mdl.sum((inst['sorting'][j]['open_fixed']*FU*xJ[j, t] + inst['sorting'][j].get('expand_cost', 600.0)*FU*zJ[j, t]) for j in SORT for t in range(1, T+1)) \
          + mdl.sum((inst['recycling'][k]['open_fixed']*FU*xK[k, t] + inst['recycling'][k]['expand_cost']*FU*zK[k, t]) for k in RECY for t in range(1, T+1)) \
          + mdl.sum(inst['landfill'][l]['open_fixed']*FU*xL[l, t] for l in LAND for t in range(1, T+1))
    stage2 = 0
    for sn, sc in enumerate(scenarios):
        ps = sc['prob']
        trans = mdl.sum(P['transport_cost'] * (
              mdl.sum(inst['dist']['a'][i][j]*y[i, j, t, sn] for i in GEN for j in SORT)
            + mdl.sum(inst['dist']['a2'][i][l]*yd[i, l, t, sn] for i in GEN for l in LAND)
            + mdl.sum(inst['dist']['b'][j][k]*w[j, k, t, sn] for j in SORT for k in RECY)
            + mdl.sum(inst['dist']['c'][j][l]*q[j, l, t, sn] for j in SORT for l in LAND)
            + mdl.sum(inst['dist']['e'][k][l]*v[k, l, t, sn] for k in RECY for l in LAND)) for t in range(1, T+1))
        oper = mdl.sum(inst['sorting'][j]['op_cost']*mdl.sum(y[i, j, t, sn] for i in GEN) for j in SORT for t in range(1, T+1)) \
             + mdl.sum(inst['recycling'][k]['op_cost']*mdl.sum(w[j, k, t, sn] for j in SORT) for k in RECY for t in range(1, T+1)) \
             + mdl.sum(inst['landfill'][l]['op_cost']*(mdl.sum(q[j, l, t, sn] for j in SORT)+mdl.sum(v[k, l, t, sn] for k in RECY)+mdl.sum(yd[i, l, t, sn] for i in GEN)) for l in LAND for t in range(1, T+1))
        hold = P['hold_cost']*mdl.sum(s[j, t, sn] for j in SORT for t in range(1, T+1))
        pen = P['unmet_penalty']*mdl.sum(u[i, t, sn] for i in GEN for t in range(1, T+1))
        rev = P.get('recycle_revenue', 0.0)*phi*mdl.sum(w[j, k, t, sn] for j in SORT for k in RECY for t in range(1, T+1))
        stage2 += ps * (trans + oper + hold + pen - rev)
    mdl.minimize(fixed + stage2)

    for sn, sc in enumerate(scenarios):
        for i in GEN:
            for t in range(1, T+1):
                mdl.add_constraint(mdl.sum(y[i, j, t, sn] for j in SORT) + mdl.sum(yd[i, l, t, sn] for l in LAND) + u[i, t, sn] == sc['d'][i][t], ctname=f'C1_{i}_{t}_{sn}')
        for j in SORT:
            for t in range(1, T+1):
                prev = s[j, t-1, sn] if t > 1 else 0
                out_t = mdl.sum(w[j, k, t, sn] for k in RECY) + mdl.sum(q[j, l, t, sn] for l in LAND)
                mdl.add_constraint(prev + mdl.sum(y[i, j, t, sn] for i in GEN) == out_t + s[j, t, sn], ctname=f'C2_{j}_{t}_{sn}')
                mdl.add_constraint(mdl.sum(w[j, k, t, sn] for k in RECY) <= rho*out_t, ctname=f'C2rho1_{j}_{t}_{sn}')
                mdl.add_constraint(mdl.sum(q[j, l, t, sn] for l in LAND) >= (1-rho)*out_t, ctname=f'C2rho2_{j}_{t}_{sn}')
        for k in RECY:
            for t in range(1, T+1):
                mdl.add_constraint((1-phi)*mdl.sum(w[j, k, t, sn] for j in SORT) == mdl.sum(v[k, l, t, sn] for l in LAND), ctname=f'C3_{k}_{t}_{sn}')
        for j in SORT:
            for t in range(1, T+1):
                mdl.add_constraint(zJ[j, t] <= 4*xJ[j, t], ctname=f'C4z_{j}_{t}_{sn}')
                mdl.add_constraint(mdl.sum(y[i, j, t, sn] for i in GEN) <= inst['sorting'][j]['cap_t']*xJ[j, t] + inst['sorting'][j].get('expand_step_t', 30e4)*zJ[j, t], ctname=f'C4_{j}_{t}_{sn}')
        for k in RECY:
            for t in range(1, T+1):
                inflow = mdl.sum(w[j, k, t, sn] for j in SORT)
                mdl.add_constraint(zK[k, t] <= 4*xK[k, t], ctname=f'C5z_{k}_{t}_{sn}')
                mdl.add_constraint(inflow <= inst['recycling'][k]['cap_t']*xK[k, t] + inst['recycling'][k]['expand_step_t']*zK[k, t], ctname=f'C5a_{k}_{t}_{sn}')
                mdl.add_constraint(inflow >= inst['recycling'][k]['min_run_t']*xK[k, t], ctname=f'C5b_{k}_{t}_{sn}')
        for l in LAND:
            for t in range(1, T+1):
                inflow_t = mdl.sum(q[j, l, t, sn] for j in SORT) + mdl.sum(yd[i, l, t, sn] for i in GEN) + mdl.sum(v[k, l, t, sn] for k in RECY)
                mdl.add_constraint(inflow_t <= inst['landfill'][l]['cap_t']*xL[l, t], ctname=f'C6a_{l}_{t}_{sn}')
                mdl.add_constraint(inflow_t + inst['landfill'][l]['cap_t']*rL[l, t] <= inst['landfill'][l]['cap_t'], ctname=f'C6b_{l}_{t}_{sn}')
            cum_inflow = (mdl.sum(q[j, l, t, sn] for j in SORT for t in range(1, T+1))
                        + mdl.sum(yd[i, l, t, sn] for i in GEN for t in range(1, T+1))
                        + mdl.sum(v[k, l, t, sn] for k in RECY for t in range(1, T+1)))
            mdl.add_constraint(cum_inflow <= inst['landfill'][l]['cap_t'], ctname=f'C6c_{l}_{sn}')
        for j in SORT:
            for t in range(1, T+1):
                mdl.add_constraint(s[j, t, sn] <= inst['sorting'][j]['buffer_cap_t']*xJ[j, t] + inst['sorting'][j].get('expand_step_t', 30e4)*0.2*zJ[j, t], ctname=f'C10_{j}_{t}_{sn}')

    for l in LAND:
        for t in range(1, T):
            mdl.add_constraint(rL[l, t] <= rL[l, t+1], ctname=f'C7a_{l}_{t}')
        for t in range(1, T+1):
            mdl.add_constraint(xL[l, t] + rL[l, t] <= 1, ctname=f'C7b_{l}_{t}')
    if retire_lstar:
        for l in LAND:
            if inst['landfill'][l]['retire_candidate']:
                for t in range(t_star, T+1):
                    mdl.add_constraint(rL[l, t] == 1, ctname=f'C8_{l}_{t}')
    for k in RECY:
        for t in range(1, T):
            mdl.add_constraint(zK[k, t] <= zK[k, t+1], ctname=f'C9_{k}_{t}')
    for j in SORT:
        for t in range(1, T):
            mdl.add_constraint(zJ[j, t] <= zJ[j, t+1], ctname=f'C9J_{j}_{t}')

    mdl.set_time_limit(300)  # 5min cap per instance
    sol = mdl.solve(log_output=log)
    return mdl, sol, dict(xJ=xJ, xK=xK, xL=xL, zK=zK, zJ=zJ, rL=rL, y=y, yd=yd, w=w, q=q, v=v, s=s, u=u)

def verify_V3():
    """S=1（均值情景）时 SAA 目标值必须等于 D0 目标值（<1e-3）。"""
    from cplex_d0 import build_and_solve
    inst = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    _, sol_d0, _ = build_and_solve(inst, 'v3_d0', retire_lstar=True, t_star=2)
    obj_d0 = sol_d0.objective_value
    T = inst['meta']['periods']
    mean_sc = [{'name': 'mean', 'prob': 1.0,
                'd': {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in inst['demand']}}]
    _, sol_saa, _ = build_and_solve_saa(inst, mean_sc, 'v3_saa1', retire_lstar=True, t_star=2)
    obj_saa = sol_saa.objective_value
    diff = abs(obj_d0 - obj_saa)
    ok = diff < 1e-3
    print(f'[V3] D0={obj_d0:,.2f}, SAA(S=1)={obj_saa:,.2f}, 差异={diff:.2e} -> {"PASS" if ok else "FAIL"}')
    return ok

if __name__ == '__main__':
    ok3 = verify_V3()
    # SAA 演示：S=30 情景（3锚点+27采样）
    inst = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    scenarios = gen_scenarios(inst, 30)
    mdl, sol, V = build_and_solve_saa(inst, scenarios, 'saa_30', retire_lstar=True, t_star=2, log=True)
    if sol:
        print(f'[SAA-30] 两阶段目标值={sol.objective_value:,.0f}')
        gv = lambda v: v.solution_value
        for k in inst['recycling']:
            print(f"  {inst['recycling'][k]['name'][:12]}: xK={[round(gv(V['xK'][k,t])) for t in range(1,4)]}, zK={[round(gv(V['zK'][k,t])) for t in range(1,4)]}")
    print(f'===== V3={"PASS" if ok3 else "FAIL"} =====')
    sys.exit(0 if ok3 else 1)
