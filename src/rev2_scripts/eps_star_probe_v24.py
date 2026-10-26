# -*- coding: utf-8 -*-
"""V2.4 ε* 重定位探针：终端清空后物理上限下移（0.18 全实例不可行）。
对 real 实例 30 情景训练集，扫描 ε ∈ {0.10..0.18}，记录可行性与目标值。
同时验证 LP 解的 s 值全零（无停放）。
"""
import json, sys, os
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)
from cplex_saa import gen_scenarios, build_and_solve_saa
from run_experiments import dash_scenarios

inst = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
train = gen_scenarios(inst, 30, seed=1000)

print('ε 扫描（real 实例，30 情景训练集，覆盖约束，TL=900s）', flush=True)
for eps in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]:
    full = dash_scenarios(inst, train, eps) if eps > 0 else train
    try:
        mdl, sol, V = build_and_solve_saa(inst, full, f'epsprobe_{eps}', retire_lstar=True,
                                          t_star=2, log=False, require_cover=True, time_limit=900)
        if sol is None:
            print(f'ε={eps:.2f}: INFEASIBLE', flush=True)
        else:
            print(f'ε={eps:.2f}: ok  obj={sol.objective_value/1e6:.1f}M', flush=True)
    except Exception as e:
        print(f'ε={eps:.2f}: EXC {e}', flush=True)
