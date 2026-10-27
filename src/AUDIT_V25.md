# V2.5 论文升级：处置容量闭式证书（2026-10-27）

## 核心变更

§4.2 已从“呼和浩特经验上限”改为正式命题 **Disposal-capacity certificate**：
对全期总需求 D、满配全期分拣能力 S、满配可用填埋库容 R、链残渣率 κ=(1−ρ)+ρ(1−φ)，
任意配置可覆盖的必要条件为

    F_min(D) = D − (1−κ)·min(S,D) ≤ R

DASH 情景 D=(1+ε)D0 且 D0>S 时：

    ε* = [R + (1−κ)S] / D0 − 1

该条件是**对所有一阶配置的必要条件**（不是“低于它就必可行”的充分保证）；
在本实例族满配下达到下界，因此与 CPLEX 边界一致。

## 数值验证（全部通过）

- 真实网：D0=14.7555 Mt，S=14.25 Mt，R=7.4956 Mt，κ=0.32 → ε*=0.164691；
  CPLEX 0.16 可行、0.17 不可行；11 实例同样边界（副本保持总量与能力）。
- ρ/φ 网格：公式预测三个 <0.10 的格子 = (0.70,0.75) 0.0150、(0.70,0.85) 0.0826、
  (0.80,0.75) 0.0874；正是 DASH(0.10) 不可行的三格，其余六格均 >0.10 且可解。
- 填埋库容扰动 R×0.8/1.0/1.2：预测 0.0631/0.1647/0.2663；
  CPLEX 在各预测值−0.01 可行、+0.01 不可行，6/6 吻合。
- 脚本：`rev2_scripts/eps_formula_v2.py`、`eps_formula_perturb.py`；
  数据：`experiments/eps_formula_v2_instances.csv`、`eps_formula_v2_rhophi.csv`、
  `eps_formula_perturb.csv`。

## 论文重定位

- 标题改为：A disposal-capacity certificate for robust CDW network design under demand drift
  and facility retirement: an application to Hohhot, China
- 摘要改为“证书先行”（131 词），引言在介绍城市前先提出“所要求的鲁棒性是否可能存在”。
- §5 压缩为可移植、有限评估启发式（保留 10/11 达到证明最优与规模边界），不再展开 ALNS 细节。
- 新增附录 Verification and reproducibility：LP/CPLEX 一致性、隔离重解、证书验证、复现文件。
- 编译：54 页，0 错误、0 未定义引用；AI 痕迹 10/12（余项为突发度/段落方差的排版测量项）。

## 与既有结论的关系

这不是否定 DASH，而是把“何时不该再对冲”变成可先算的诊断量：
服务级覆盖约束决定“必须服务”，证书决定“物理上能否服务”，DASH 决定在可行域内如何对冲。
