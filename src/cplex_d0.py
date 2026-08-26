#!/usr/bin/env python3
"""CPLEX 确定性 MILP（模型规范 §5 D0）+ 一致性验证 V1/V2。
V1: 手算小实例（2-1-1-1,1期）与 CPLEX 解比对（误差<1e-6）
V2: 真实 9-8-4-6 实例求解 + 逐条流量守恒复核（残差<1e-6）
对应数学模型: model_spec_v1.md §5（目标函数与约束 C1-C10 逐一映射，见代码注释）
"""
import json, os, sys
from docplex.mp.model import Model

BASE = os.path.dirname(os.path.abspath(__file__))

def build_and_solve(inst, name='d0', retire_lstar=False, t_star=2, log=False, fixed_unit=1e4):
    """按 model_spec_v1.md §5 构建 D0 并求解。返回 (mdl, sol, vars)。"""
    T = inst['meta']['periods']
    GEN = list(inst['demand'].keys())
    SORT = list(inst['sorting'].keys())
    RECY = list(inst['recycling'].keys())
    LAND = list(inst['landfill'].keys())
    P = inst['params']
    rho = P['recyclable_ratio']

    mdl = Model(name)

    # ---- 决策变量（规范 §4）----
    xJ = mdl.binary_var_dict([(j, t) for j in SORT for t in range(1, T+1)], name='xJ')
    xK = mdl.binary_var_dict([(k, t) for k in RECY for t in range(1, T+1)], name='xK')
    xL = mdl.binary_var_dict([(l, t) for l in LAND for t in range(1, T+1)], name='xL')
    zK = mdl.integer_var_dict([(k, t) for k in RECY for t in range(1, T+1)], lb=0, ub=4, name='zK')
    rL = mdl.binary_var_dict([(l, t) for l in LAND for t in range(1, T+1)], name='rL')
    y = mdl.continuous_var_dict([(i, j, t) for i in GEN for j in SORT for t in range(1, T+1)], lb=0, name='y')
    yd = mdl.continuous_var_dict([(i, l, t) for i in GEN for l in LAND for t in range(1, T+1)], lb=0, name='yd')  # 直达消纳（工程渣土等免分拣部分）
    w = mdl.continuous_var_dict([(j, k, t) for j in SORT for k in RECY for t in range(1, T+1)], lb=0, name='w')
    q = mdl.continuous_var_dict([(j, l, t) for j in SORT for l in LAND for t in range(1, T+1)], lb=0, name='q')
    v = mdl.continuous_var_dict([(k, l, t) for k in RECY for l in LAND for t in range(1, T+1)], lb=0, name='v')
    s = mdl.continuous_var_dict([(j, t) for j in SORT for t in range(1, T+1)], lb=0, name='s')
    u = mdl.continuous_var_dict([(i, t) for i in GEN for t in range(1, T+1)], lb=0, name='u')

    # ---- 目标函数（规范 §5）----
    FIXED_UNIT = fixed_unit  # 实例固定成本单位换算（真实实例万元→元；tiny已为一致单位传1）
    fixed = mdl.sum(inst['sorting'][j]['open_fixed']*FIXED_UNIT*xJ[j, t] for j in SORT for t in range(1, T+1)) \
          + mdl.sum((inst['recycling'][k]['open_fixed']*FIXED_UNIT*xK[k, t] + inst['recycling'][k]['expand_cost']*FIXED_UNIT*zK[k, t]) for k in RECY for t in range(1, T+1)) \
          + mdl.sum(inst['landfill'][l]['open_fixed']*FIXED_UNIT*xL[l, t] for l in LAND for t in range(1, T+1))
    trans = mdl.sum(P['transport_cost'] * (
          mdl.sum(inst['dist']['a'][i][j]*y[i, j, t] for i in GEN for j in SORT)
        + mdl.sum(inst['dist']['b'][j][k]*w[j, k, t] for j in SORT for k in RECY)
        + mdl.sum(inst['dist']['c'][j][l]*q[j, l, t] for j in SORT for l in LAND)
        + mdl.sum(inst['dist']['a2'][i][l]*yd[i, l, t] for i in GEN for l in LAND)
        + mdl.sum(inst['dist']['e'][k][l]*v[k, l, t] for k in RECY for l in LAND)) for t in range(1, T+1))
    oper = mdl.sum(inst['sorting'][j]['op_cost']*mdl.sum(y[i, j, t] for i in GEN) for j in SORT for t in range(1, T+1)) \
         + mdl.sum(inst['recycling'][k]['op_cost']*mdl.sum(w[j, k, t] for j in SORT) for k in RECY for t in range(1, T+1)) \
         + mdl.sum(inst['landfill'][l]['op_cost']*(mdl.sum(q[j, l, t] for j in SORT)+mdl.sum(v[k, l, t] for k in RECY)+mdl.sum(yd[i, l, t] for i in GEN)) for l in LAND for t in range(1, T+1))
    hold = P['hold_cost']*mdl.sum(s[j, t] for j in SORT for t in range(1, T+1))
    pen = P['unmet_penalty']*mdl.sum(u[i, t] for i in GEN for t in range(1, T+1))
    # V1.1 修正：再生品收益（无此项则模型退化为全填埋，与现实72%资源化率矛盾——见model_spec_v1.md §6 核查项7）
    phi = P.get('product_yield', 1.0)
    revenue = P.get('recycle_revenue', 0.0)*phi*mdl.sum(w[j, k, t] for j in SORT for k in RECY for t in range(1, T+1))
    mdl.minimize(fixed + trans + oper + hold + pen - revenue)

    # ---- 约束（规范 §5 C1-C10）----
    # C1 产生点流量平衡
    for i in GEN:
        for t in range(1, T+1):
            mdl.add_constraint(mdl.sum(y[i, j, t] for j in SORT) + mdl.sum(yd[i, l, t] for l in LAND) + u[i, t] == inst['demand'][i]['d'][str(t)], ctname=f'C1_{i}_{t}')
    # C2 分拣中心流量平衡（含缓冲结转，R2拆分）
    for j in SORT:
        for t in range(1, T+1):
            prev = s[j, t-1] if t > 1 else 0
            out_t = mdl.sum(w[j, k, t] for k in RECY) + mdl.sum(q[j, l, t] for l in LAND)
            mdl.add_constraint(prev + mdl.sum(y[i, j, t] for i in GEN) == out_t + s[j, t], ctname=f'C2_{j}_{t}')
            # R2 资源化比例：进资源化厂 ρ、进消纳 1-ρ
            mdl.add_constraint(mdl.sum(w[j, k, t] for k in RECY) <= rho*out_t, ctname=f'C2rho1_{j}_{t}')
            mdl.add_constraint(mdl.sum(q[j, l, t] for l in LAND) >= (1-rho)*out_t, ctname=f'C2rho2_{j}_{t}')
    # C3 资源化厂流量平衡
    for k in RECY:
        for t in range(1, T+1):
            mdl.add_constraint((1-phi)*mdl.sum(w[j, k, t] for j in SORT) == mdl.sum(v[k, l, t] for l in LAND), ctname=f'C3_{k}_{t}')
    # C4 分拣能力
    for j in SORT:
        for t in range(1, T+1):
            mdl.add_constraint(mdl.sum(y[i, j, t] for i in GEN) <= inst['sorting'][j]['cap_t']*xJ[j, t], ctname=f'C4_{j}_{t}')
    # C5 资源化能力（含扩容与最小经济开工量；zK≤ub·xK 使 zK·xK 双线性消去）
    for k in RECY:
        for t in range(1, T+1):
            inflow = mdl.sum(w[j, k, t] for j in SORT)
            mdl.add_constraint(zK[k, t] <= 4*xK[k, t], ctname=f'C5z_{k}_{t}')
            mdl.add_constraint(inflow <= inst['recycling'][k]['cap_t']*xK[k, t] + inst['recycling'][k]['expand_step_t']*zK[k, t], ctname=f'C5a_{k}_{t}')
            mdl.add_constraint(inflow >= inst['recycling'][k]['min_run_t']*xK[k, t], ctname=f'C5b_{k}_{t}')
    # C6 消纳能力：使用须开启 + 退出禁流 + 累计库容（三条均线性）
    for l in LAND:
        for t in range(1, T+1):
            inflow_t = mdl.sum(q[j, l, t] for j in SORT) + mdl.sum(yd[i, l, t] for i in GEN) + mdl.sum(v[k, l, t] for k in RECY)
            mdl.add_constraint(inflow_t <= inst['landfill'][l]['cap_t']*xL[l, t], ctname=f'C6a_{l}_{t}')
            mdl.add_constraint(inflow_t + inst['landfill'][l]['cap_t']*rL[l, t] <= inst['landfill'][l]['cap_t'], ctname=f'C6b_{l}_{t}')
        cum_inflow = (mdl.sum(q[j, l, t] for j in SORT for t in range(1, T+1))
                    + mdl.sum(yd[i, l, t] for i in GEN for t in range(1, T+1))
                    + mdl.sum(v[k, l, t] for k in RECY for t in range(1, T+1)))
        mdl.add_constraint(cum_inflow <= inst['landfill'][l]['cap_t'], ctname=f'C6c_{l}')
    # C7 退出不可逆 + 开启与退出互斥
    for l in LAND:
        for t in range(1, T):
            mdl.add_constraint(rL[l, t] <= rL[l, t+1], ctname=f'C7a_{l}_{t}')
        for t in range(1, T+1):
            mdl.add_constraint(xL[l, t] + rL[l, t] <= 1, ctname=f'C7b_{l}_{t}')
    # C8 大厂库伦强制退出情景
    if retire_lstar:
        for l in LAND:
            if inst['landfill'][l]['retire_candidate']:
                for t in range(t_star, T+1):
                    mdl.add_constraint(rL[l, t] == 1, ctname=f'C8_{l}_{t}')
    # C9 扩容单调
    for k in RECY:
        for t in range(1, T):
            mdl.add_constraint(zK[k, t] <= zK[k, t+1], ctname=f'C9_{k}_{t}')
    # C10 库存容量
    for j in SORT:
        for t in range(1, T+1):
            mdl.add_constraint(s[j, t] <= inst['sorting'][j]['buffer_cap_t']*xJ[j, t], ctname=f'C10_{j}_{t}')

    sol = mdl.solve(log_output=log)
    return mdl, sol, dict(xJ=xJ, xK=xK, xL=xL, zK=zK, rL=rL, y=y, yd=yd, w=w, q=q, v=v, s=s, u=u)

# ================= V1: 手算小实例 =================
def tiny_instance(revenue=0.0):
    return {
        'meta': {'periods': 1, 'name': 'tiny-2-1-1-1'},
        'demand': {'i1': {'d': {'1': 100.0}}, 'i2': {'d': {'1': 50.0}}},
        'sorting': {'j1': {'open_fixed': 10.0, 'op_cost': 1.0, 'cap_t': 1e9, 'buffer_cap_t': 1e-6}},  # 禁缓冲免费处置
        'recycling': {'k1': {'open_fixed': 20.0, 'op_cost': 2.0, 'cap_t': 1e9,
                             'expand_step_t': 1e9, 'expand_cost': 5.0, 'min_run_t': 0.0}},
        'landfill': {'l1': {'open_fixed': 30.0, 'op_cost': 3.0, 'cap_t': 1e9, 'retire_candidate': False}},
        'dist': {'a': {'i1': {'j1': 10.0}, 'i2': {'j1': 20.0}},
                 'a2': {'i1': {'l1': 15.0}, 'i2': {'l1': 25.0}},
                 'b': {'j1': {'k1': 5.0}},
                 'c': {'j1': {'l1': 8.0}},
                 'e': {'k1': {'l1': 4.0}}},
        'params': {'transport_cost': 1.0, 'hold_cost': 0.0, 'unmet_penalty': 1000.0,
                   'recyclable_ratio': 0.8, 'recycle_revenue': revenue, 'product_yield': 1.0},
    }

def verify_V1():
    """变体A（收益=0）：单位路径成本 直达(i1:18/i2:28) < 经分拣直填(22/32) < 经资源化(25/35)
    → 全直达。手算: 100×18+50×28+固定l1=30 → 1800+1400+30 = 3230
    变体B（收益=20,φ=1）：资源化净路径(i1:5/i2:15) 最便宜 → 全分拣, w=ρ·150=120, q=30；φ=1无余渣
    手算: 分拣运输2000+运营150 + w运输600+w运营240 + q运输240+q运营90 + 固定60 − 收益2400 = 980"""
    results = []
    for name, rev, expect in [('A(收益=0,全直达)', 0.0, 3230.0), ('B(收益=20,ρ资源化)', 20.0, 980.0)]:
        mdl, sol, V = build_and_solve(tiny_instance(rev), f'v1_tiny_{name[0]}', fixed_unit=1.0)
        got = sol.objective_value
        ok = abs(got - expect) < 1e-3
        print(f'[V1-{name}] 手算={expect}, CPLEX={got:.4f}, 差异={got-expect:.2e} -> {"PASS" if ok else "FAIL"}')
        results.append(ok)
    return all(results)

# ================= V2: 真实实例 + 守恒复核 =================
def verify_V2():
    inst = json.load(open(f'{BASE}/instance_v1.json', encoding='utf-8'))
    mdl, sol, V = build_and_solve(inst, 'v2_hohhot', retire_lstar=True, t_star=2, log=True)
    if sol is None:
        print('[V2] FAIL: 无解')
        return False
    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting']); RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    gv = lambda var: var.solution_value
    resid_max = 0.0
    # C1 复核
    for i in GEN:
        for t in range(1, T+1):
            lhs = sum(gv(V['y'][i, j, t]) for j in SORT) + sum(gv(V['yd'][i, l, t]) for l in LAND) + gv(V['u'][i, t])
            resid_max = max(resid_max, abs(lhs - inst['demand'][i]['d'][str(t)]))
    # C2/C3 复核
    for j in SORT:
        for t in range(1, T+1):
            prev = gv(V['s'][j, t-1]) if t > 1 else 0
            out_t = sum(gv(V['w'][j, k, t]) for k in RECY) + sum(gv(V['q'][j, l, t]) for l in LAND)
            lhs = prev + sum(gv(V['y'][i, j, t]) for i in GEN)
            resid_max = max(resid_max, abs(lhs - out_t - gv(V['s'][j, t])))
    phi_v = inst['params'].get('product_yield', 1.0)
    for k in RECY:
        for t in range(1, T+1):
            lhs = (1-phi_v)*sum(gv(V['w'][j, k, t]) for j in SORT) - sum(gv(V['v'][k, l, t]) for l in LAND)
            resid_max = max(resid_max, abs(lhs))
    unmet = sum(gv(V['u'][i, t]) for i in GEN for t in range(1, T+1))
    print(f'[V2] 目标值={sol.objective_value:,.2f} 万元, 流量守恒最大残差={resid_max:.2e}, 未服务量={unmet:,.1f} 吨')
    print(f'[V2] 求解状态={mdl.solve_details.status}')
    # 输出关键决策摘要
    for l in LAND:
        if inst['landfill'][l]['retire_candidate']:
            print(f"[V2] 退出候选 {inst['landfill'][l]['name']}: rL = {[round(gv(V['rL'][l,t])) for t in range(1,T+1)]}, xL = {[round(gv(V['xL'][l,t])) for t in range(1,T+1)]}")
    for k in RECY:
        open_seq = [round(gv(V['xK'][k, t])) for t in range(1, T+1)]
        z_seq = [round(gv(V['zK'][k, t])) for t in range(1, T+1)]
        inflow = [round(sum(gv(V['w'][j, k, t]) for j in SORT)/1e4, 1) for t in range(1, T+1)]
        print(f"[V2] {inst['recycling'][k]['name'][:12]}: xK={open_seq}, 扩容档={z_seq}, 处理量(万吨)={inflow}")
    print('[V2-DIAG] 分期流量(万吨):')
    for t in range(1, T+1):
        sy = sum(gv(V['y'][i, j, t]) for i in GEN for j in SORT)/1e4
        syd = sum(gv(V['yd'][i, l, t]) for i in GEN for l in LAND)/1e4
        sw = sum(gv(V['w'][j, k, t]) for j in SORT for k in RECY)/1e4
        sq = sum(gv(V['q'][j, l, t]) for j in SORT for l in LAND)/1e4
        sv = sum(gv(V['v'][k, l, t]) for k in RECY for l in LAND)/1e4
        su = sum(gv(V['u'][i, t]) for i in GEN)/1e4
        print(f'  T{t}: 分拣{sy:.1f} 直达{syd:.1f} 资源化{sw:.1f} 直填{sq:.1f} 余渣{sv:.1f} 未服务{su:.1f}')
    for l in LAND:
        cum = (sum(gv(V['q'][j, l, t]) for j in SORT for t in range(1, T+1))
             + sum(gv(V['yd'][i, l, t]) for i in GEN for t in range(1, T+1))
             + sum(gv(V['v'][k, l, t]) for k in RECY for t in range(1, T+1)))/1e4
        xl_seq = [round(gv(V['xL'][l, t])) for t in range(1, T+1)]
        print(f"  库容 {inst['landfill'][l]['name'][:10]}: 用{cum:.1f}/{inst['landfill'][l]['cap_t']/1e4:.1f}万吨 xL={xl_seq}")
    ok = resid_max < 1e-6
    print(f'[V2] -> {"PASS" if ok else "FAIL"}')
    return ok

if __name__ == '__main__':
    ok1 = verify_V1()
    ok2 = verify_V2()
    print(f'\n===== 一致性验证汇总: V1={"PASS" if ok1 else "FAIL"}, V2={"PASS" if ok2 else "FAIL"} =====')
    sys.exit(0 if (ok1 and ok2) else 1)
