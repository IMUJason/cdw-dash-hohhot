# V2.4 模型与 GRI 算法审计记录（2026-10-26）

## 1. 模型缺陷与修复

### 1.1 期末缓冲处置漏洞
- 无 `s[j,T]=0` 时，LP 可将材料永久停在分拣中心，规避填埋库容与处置成本。
- 实证：anchor_2024 在某配置下，无缓冲 LP 不可行、含缓冲 LP 可行；手算库容缺口与期末停放一致。
- 修复：`cplex_saa.py`、`cplex_d0.py`、`alns.py`、`run_co2.py` 全部加终端清空。
- `run_co2.py` 另修：缓冲容量乘 xJ、厂扩容容量乘 xK（与主模型一致）。

### 1.2 修复后验证
- `audit_fixcheck.py`：
  - D0 == SAA(官方轨迹)：rel = 0
  - 固定配置 LP == CPLEX：rel = 5.66e-15
  - 关闭厂 zK>0 不再获得容量
- `eps_star_probe_v24.py`：real 上 ε=0.16 可行（1075.6M），0.17 不可行。
  新物理上限 ε*∈(0.16,0.17)，替代旧 0.185。
- 主表 real（终端清空后，CPLEX）：
  - DET plan/oos = 718.9/936.3M，sfr=0.490
  - SAA = 834.8/827.0M，sfr=0
  - DASH(0.05) = 877.7/861.1M，sfr=0
  - DASH(0.10) = 944.9/900.4M，sfr=0
  - DASH(0.18) 全 11 实例不可行

## 2. CPLEX 小规模基线

`rev2_scripts/gri_small_cplex.py`：real + 10 个扰动实例，SAA(10)，覆盖约束，终端清空。
全部 11/11 求得整数最优（MIP gap<0.01%），单实例 22–63s。
结果：`experiments/results_gri_small_cplex.csv`；配置：`config_gri_small_<seed>.json`。

## 3. GRI 算法（替代旧 ALNS）

### 3.1 架构
1. 零 LP 岛搜索：6 独立岛、D1–D9、双策略贪心评估、定期迁移。
2. 一阶精确抛光：全邻域精确评估 + ILS（有界）。
3. 二阶联合抛光：开闭/退役/时间尾段动作 × 减容，含前缀与单期调整。
4. 所有报告值由精确 LP 计算；配置必须通过 `config_violations`。

### 3.2 贪心评估器定位
- `greedy_recourse.py` 成功时构造可行流，成本为认证上界。
- 多期库容协调使单遍构造保守；双策略（premium/maxsort）提高覆盖识别。
- 贪心仅用于搜索与排序，不用于论文最终精确数字。

### 3.3 已验证的精确命中
- 文件：`experiments/config_gri_polished_test.json`
- 固定回 CPLEX 重解目标：853903997.6484964
- 与 `results_gri_small_cplex.csv` 的 real 精确最优完全一致。
- 另一独立端到端种子 gap=1.017%，说明必须多种子报告 best-of-K + 命中数 + 均值。

### 3.4 负 gap 事件（已作废）
- 前缀减容未保界，产生 z=-1/-2；LP 求值器未检查一阶定义域，造成虚假负 gap。
- 修复：`repair_monotone` 统一夹紧 0≤z≤4；新增 `config_violations`；验证脚本全部经有效性门禁。
- 任何负 gap 一律视为 bug，不作为算法优势报告。

## 4. 当前验证门栅

- `rev2_scripts/gri_small_driver.py`：11 实例 × 5 种子。
- 每种子：零 LP 岛搜索 → 全池精确评估 → 一阶抛光 → 二阶联合抛光。
- 报告：逐种子 gap、best-of-K、精确命中数、均值、总耗时。
- 未命中实例补种子；用户门栅通过后才启动全量重跑。
- 旧 ALNS 弃用，不再投入算力；代码仅留作诊断历史。
