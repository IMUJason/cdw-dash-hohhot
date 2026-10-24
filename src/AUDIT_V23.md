# V2.3 一致性审计记录（2026-10-23）

## 审计结论

模型/算法脚本交叉验证（`rev2_scripts/audit_crosscheck.py` + `audit_fixcheck.py`）：

| 测试 | 结果 |
|---|---|
| 固定一阶配置：ALNS 内层 LP vs CPLEX（覆盖约束双侧） | rel = 2.1e-15，机器精度一致 |
| DASH(ε=0) 退化 = SAA | 精确相等 |
| D0 = SAA(官方轨迹单情景) | rel = 0 |
| CPLEX 解结构有效性（z≤4x / 单调 / xL+rL≤1） | 全部满足 |
| CSV 复现对拍（small_1 SAA 目标值） | 与 results_scale_v22.csv 完全一致 |

ρ 等式、zJ、覆盖约束、退役约束、目标函数在 cplex_saa / cplex_d0 / alns LP /
eval_config（OOS 评估器）四条路径语义一致。cplex_dro.py 无自有模型（情景变换壳）。

## 发现并修复的 bug

**alns.py solve_second_stage_lp_ex 厂容量未被 xK 门控**：
- 修前 `cap = cap·xK + step·zK`（关闭的厂仍可凭模块获得容量）
- CPLEX 语义为 C5a∧C5z（关闭即零容量）
- 对抗复现实测：xK=0, zK=4 的配置单情景便宜 2520 万元（4.2%）
- ALNS 关厂算子（D6 close）恰会把 xK 置零而保留 zK，搜索空间可达该漏洞
- 修复：`cap = (cap + step·zK)·xK`（与分拣侧同构）；修后三项复验全绿

**连带修复**：
- alns.py D6_greedy_close 对 cur_t=None 无防护 → ε>ε* 时崩溃；加 guard
- multiseed_par.py METHODS 去掉 DRO(0.20)（超物理上限 ε*≈0.185，修后无覆盖可行解），改 0.18

## 波及范围与处置

| 结果 | 是否受 bug 影响 | 处置 |
|---|---|---|
| 主表 Table 2（全 CPLEX 精确解） | 否（配置满足 z≤4x，评估路径已证一致） | 不动 |
| penalty/backtest/loo/rhophi/grid/ablation/decomp | 否（同上） | 不动 |
| scale_v22 / scale_stress（ALNS 列） | 是 | **已重跑**（CPLEX 目标值前后 60/60 完全一致=确定性验证通过） |
| seeds10（种子离散） | 是（含 DRO(0.20) 幽灵容量行） | 重跑中（换 0.18） |
| §5.4 引用的 0.348±0.063 等（修复前旧数据） | 旧 CSV 已被覆盖、不可复现 | 重跑后换成修后数据 |

时间线隐患（一并消除）：results_scale_v22.csv 原生成于 alns.py 最终修改之前 2 小时。

## 新增实验：设施轴规模（V2.3）

NP-hard 落在离散设施维（CFLP 归约），而旧规模实验只动情景数（离散变量恒 108）。
新增 `rev2_scripts/gen_synthetic_city.py`（克隆真实网结构 m 份，单设施规模保持真实，
总需求 ∝ m）+ `scale_facility.py`：m∈{1,2,3,4,6} → 108–648 离散变量，情景固定 10，
SAA 与 DASH(0.10)，CPLEX(cover, TL=1800s) vs ALNS。结果：experiments/results_scale_facility.csv。
两种结果都如实写：若 CPLEX 全解，则"设施维硬度在市政尺度内不绑定"本身是结论。

## 论文同步（main.tex）

- Table 3 六格全部换成修后均值；gap 带 1.6–2.0%
- §5.4 "faster at two of the six cells" → 41/100 配对运行 ALNS 更快、质量带不动
- 压力扫描 "19 to 716 seconds" → "41 to 679"
- §5.4 两段（4.2%/22% 与 0.348 种子统计）待 seeds10 重跑后更新
