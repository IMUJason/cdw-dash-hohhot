# -*- coding: utf-8 -*-
"""一致性审计 V-AUD（2026-10-23）：模型/算法脚本交叉验证。

测试项：
  V1  D0(均值情景) == SAA(单情景)            —— cplex_d0 与 cplex_saa 同构性
  V2  固定一阶段配置：evaluate(LP) == CPLEX 目标 —— LP 与 CPLEX 物理一致性（核心）
  V3  DASH(ε=0) == SAA                        —— 退化检验
  V4  覆盖约束两侧一致：CPLEX(cover) 的解在 LP(cover) 下可行且同值
  V5  ALNS 解有效性审计：z≤4x / z 单调 / r 单调 / xL+rL≤1
  V6  【Bug 复现】关闭厂+zK>0 的配置：LP 是否给容量（CPLEX 语义下应无容量）
"""
import json, sys, os, time
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)

from cplex_d0 import build_and_solve as cplex_d0_solve
from cplex_saa import build_and_solve_saa, gen_scenarios
from run_experiments import (reduce_scenarios, dash_scenarios, extract_xconfig_cpx,
                             _cfg_from_xconfig)
import alns as A

inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
T = inst['meta']['periods']
GEN = list(inst['demand'])
results = []

def log(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

# ---------- 均值情景 ----------
d_mean = {i: {t: float(np.mean([inst['demand'][i]['d'][str(y)] for y in (1,2,3)]))
              for t in range(1, T+1)} for i in GEN}
mean_sc = [{'name': 'mean', 'prob': 1.0, 'd': d_mean}]

# ---------- V1: D0 == SAA(1) ----------
try:
    _, sol_d0, Vd0 = cplex_d0_solve(inst, 'aud_d0', retire_lstar=True, t_star=2)
    obj_d0 = sol_d0.objective_value
    mdl, sol_s1, V1v = build_and_solve_saa(inst, mean_sc, 'aud_saa1', retire_lstar=True,
                                           t_star=2, require_cover=False, time_limit=600)
    obj_s1 = sol_s1.objective_value
    rel = abs(obj_d0 - obj_s1) / abs(obj_d0)
    log('V1 D0==SAA(1)', rel < 1e-6, f'D0={obj_d0:.4f} SAA1={obj_s1:.4f} rel={rel:.2e}')
except Exception as e:
    log('V1 D0==SAA(1)', False, f'EXC {e}')

# ---------- 训练集（10 情景，快速） ----------
train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)

# ---------- V2/V4: CPLEX(cover) 解 → LP 评估同值 ----------
try:
    t0 = time.time()
    mdl, sol, V = build_and_solve_saa(inst, train, 'aud_cov', retire_lstar=True, t_star=2,
                                      require_cover=True, time_limit=900)
    t_cpx = time.time() - t0
    obj_cpx = sol.objective_value
    xcfg = extract_xconfig_cpx(inst, V)
    cfg = _cfg_from_xconfig(inst, xcfg)
    # LP 评估（覆盖约束下）
    t0 = time.time()
    val_lp = A.evaluate(inst, cfg, train, require_cover=True)
    t_lp = time.time() - t0
    rel = abs(val_lp - obj_cpx) / abs(obj_cpx)
    log('V2 LP==CPLEX(cover)', rel < 1e-6,
        f'cpx={obj_cpx:.4f} lp={val_lp:.4f} rel={rel:.2e} (cpx {t_cpx:.0f}s, lp {t_lp:.1f}s)')
    # V4b: 同一配置在无覆盖 LP 下的成本应 ≤ 覆盖成本（松弛）
    val_free = A.evaluate(inst, cfg, train, require_cover=False)
    log('V4 cover-relaxation', val_free <= val_lp + 1e-6,
        f'free={val_free:.4f} <= cover={val_lp:.4f}')
    # 保存解供 V5/V6
    json.dump({'train_names': [sc['name'] for sc in train], 'cfg': cfg},
              open('/tmp/aud_cfg.json', 'w'), ensure_ascii=False)
except Exception as e:
    log('V2 LP==CPLEX(cover)', False, f'EXC {e}')

# ---------- V3: DASH(0) == SAA ----------
try:
    dash0 = dash_scenarios(inst, train, 0.0)
    _, sol_d0s, _ = build_and_solve_saa(inst, dash0, 'aud_dash0', retire_lstar=True,
                                        t_star=2, require_cover=True, time_limit=900)
    _, sol_sa, _ = build_and_solve_saa(inst, train, 'aud_saa', retire_lstar=True,
                                       t_star=2, require_cover=True, time_limit=900)
    rel = abs(sol_d0s.objective_value - sol_sa.objective_value) / abs(sol_sa.objective_value)
    log('V3 DASH(0)==SAA', rel < 1e-9,
        f'dash0={sol_d0s.objective_value:.4f} saa={sol_sa.objective_value:.4f} rel={rel:.2e}')
except Exception as e:
    log('V3 DASH(0)==SAA', False, f'EXC {e}')

# ---------- V5: ALNS 短跑 → 解有效性审计 ----------
try:
    pool_cfg = json.load(open('/tmp/aud_cfg.json'))['cfg']
    base = pool_cfg
    viol = []
    for k_ in inst['recycling']:
        for t in range(T):
            if base['zK'][k_][t] > 4 * base['xK'][k_][t]: viol.append(f'zK>4xK {k_} t{t+1}')
    for j in inst['sorting']:
        for t in range(T):
            if base['zJ'][j][t] > 4 * base['xJ'][j][t]: viol.append(f'zJ>4xJ {j} t{t+1}')
    for key, fac in [('zK', inst['recycling']), ('zJ', inst['sorting']), ('rL', inst['landfill'])]:
        for f in fac:
            for t in range(T - 1):
                if base[key][f][t] > base[key][f][t + 1]:
                    viol.append(f'{key} 非单调 {f} t{t+1}')
    for l in inst['landfill']:
        for t in range(T):
            if base['xL'][l][t] + base['rL'][l][t] > 1: viol.append(f'xL+rL>1 {l} t{t+1}')
    log('V5 CPLEX解有效性', len(viol) == 0, '; '.join(viol[:5]) if viol else '全部满足')
except Exception as e:
    log('V5 CPLEX解有效性', False, f'EXC {e}')

# ---------- V6: 【Bug 复现】关闭厂+zK ----------
try:
    cfg6 = A.clone(pool_cfg)
    k0 = list(inst['recycling'])[0]
    cfg6['xK'][k0] = [0] * T          # 关闭该厂（模拟 D6 close_plant 算子）
    cfg6['zK'][k0] = [4] * T          # 但保留扩容模块（repair_monotone 不会清零）
    c2, unm, recy = A.solve_second_stage_lp_ex(inst, cfg6, train[0]['d'], require_cover=False)
    # 对照：模块也清零
    cfg6b = A.clone(cfg6); cfg6b['zK'][k0] = [0] * T
    c2b, unm_b, _ = A.solve_second_stage_lp_ex(inst, cfg6b, train[0]['d'], require_cover=False)
    fc6 = A.fixed_cost_of(inst, cfg6); fc6b = A.fixed_cost_of(inst, cfg6b)
    step = inst['recycling'][k0]['expand_step_t']
    if c2 is None or c2b is None:
        log('V6 关闭厂+zK>0', None, f'LP不可行（无对照意义）c2={c2} c2b={c2b}')
    else:
        # 若 LP 与 CPLEX 语义一致，两配置容量都应为 0，二阶段成本应几乎相同（仅固定成本差）
        diff = (fc6 + c2) - (fc6b + c2b)
        exploited = abs(c2 - c2b) > 1.0  # 二阶段成本差>1元 即说明容量被计入
        log('V6 关闭厂+zK>0', not exploited,
            f'zK=4,xK=0: stage2={c2:.1f} vs zK=0: stage2={c2b:.1f} '
            f'{"← LP 给关闭厂记了容量（bug 实锤）" if exploited else "← LP 未给容量（无 bug）"}')
except Exception as e:
    log('V6 关闭厂+zK>0', False, f'EXC {e}')

print('\n==== 汇总 ====')
for n, ok, d in results:
    print(f"{'PASS' if ok else 'FAIL' if ok is not None else 'N/A '}  {n}")
