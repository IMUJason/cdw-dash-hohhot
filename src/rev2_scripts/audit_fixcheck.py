# -*- coding: utf-8 -*-
"""Bug 修复验证 + V1 复检（正确口径：轨迹情景）。"""
import json, sys, os
import numpy as np
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_d0 import build_and_solve as d0_solve
from cplex_saa import build_and_solve_saa, gen_scenarios
from run_experiments import reduce_scenarios, extract_xconfig_cpx
import alns as A

inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
T = inst['meta']['periods']; GEN = list(inst['demand'])

# ---- V1 复检：SAA(1) 用【官方轨迹】作为单情景（与 run_method_det 相同口径） ----
traj = [{'name': 'traj', 'prob': 1.0,
         'd': {i: {t: inst['demand'][i]['d'][str(t)] for t in range(1, T+1)} for i in GEN}}]
_, sol_d0, _ = d0_solve(inst, 'fx_d0', retire_lstar=True, t_star=2)
_, sol_s1, _ = build_and_solve_saa(inst, traj, 'fx_saa1', retire_lstar=True, t_star=2,
                                   require_cover=False, time_limit=600)
rel = abs(sol_d0.objective_value - sol_s1.objective_value) / abs(sol_d0.objective_value)
print(f"[{'PASS' if rel < 1e-6 else 'FAIL'}] V1 D0==SAA(轨迹): "
      f"D0={sol_d0.objective_value:.4f} SAA1={sol_s1.objective_value:.4f} rel={rel:.2e}")

# ---- V6 复检：关闭厂+zK 不再获得容量 ----
train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)
_, sol, V = build_and_solve_saa(inst, train, 'fx_cov', retire_lstar=True, t_star=2,
                                require_cover=True, time_limit=900)
xcfg = extract_xconfig_cpx(inst, V)
from run_experiments import _cfg_from_xconfig
cfg = _cfg_from_xconfig(inst, xcfg)
c_all, _, _ = A.solve_second_stage_lp_ex(inst, cfg, train[0]['d'], require_cover=True)
cfg6 = A.clone(cfg); k0 = list(inst['recycling'])[0]
cfg6['xK'][k0] = [0]*T; cfg6['zK'][k0] = [4]*T
c6, _, _ = A.solve_second_stage_lp_ex(inst, cfg6, train[0]['d'], require_cover=False)
cfg6b = A.clone(cfg6); cfg6b['zK'][k0] = [0]*T
c6b, _, _ = A.solve_second_stage_lp_ex(inst, cfg6b, train[0]['d'], require_cover=False)
if c6 is None or c6b is None:
    print(f"[N/A] V6: LP 不可行 c6={c6} c6b={c6b}（关闭+模块配置本身覆盖不可行，无对照）")
else:
    exploited = abs(c6 - c6b) > 1.0
    print(f"[{'FAIL' if exploited else 'PASS'}] V6 修复: zK=4,xK=0 stage2={c6:.1f} "
          f"vs zK=0 stage2={c6b:.1f} {'仍有容量' if exploited else '关闭厂无容量 ✓'}")
# 全配置 LP 重评估 == CPLEX（确认修复未破坏一致性）
val = A.evaluate(inst, cfg, train, require_cover=True)
rel2 = abs(val - sol.objective_value) / abs(sol.objective_value)
print(f"[{'PASS' if rel2 < 1e-6 else 'FAIL'}] V2 复检 LP==CPLEX: rel={rel2:.2e}")
