#!/usr/bin/env python3
"""实例构建器：从 hohhot case data 的 CSV 生成标准实例 JSON（可追溯）。
产出: instance_v1.json + instance_report.md
数据溯源: cdw_generation_by_district_estimates_v3.csv (需求),
          cdw_treatment_facilities_2023_2025.csv (能力),
          distance_matrix_wide_km.csv (距离),
          model_parameters_literature.csv (成本参数假设)
"""
import csv, json, os

BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'

# ---------- 需求（v3 官方分解，份额×年度官方总量） ----------
YEAR_TOTAL = {1: 398.61, 2: 491.85, 3: 256.03}  # 万吨，官方年报；注：2023为工程渣土口径（已声明）
districts = {}
with open(f'{BASE}/cdw_generation_by_district_estimates_v3.csv', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        share = float(r['合成供给份额'].rstrip('%')) / 100
        districts[r['区县']] = {
            'share': share,
            'd': {t: round(share * YEAR_TOTAL[t] * 1e4, 0) for t in (1, 2, 3)},  # 吨/年
        }

# ---------- 设施（2025年报设计能力，万吨/年 -> 吨/年） ----------
SORTING = [  # (名称, 矩阵节点名, 设计能力万吨/年, 2025实际负荷万吨)
    ('新城区装修垃圾分拣中心', '新城区装修垃圾分拣中心', 24, 8.08),
    ('回民区装修垃圾分拣中心', '回民区装修垃圾分拣中心', 21, 14.39),
    ('玉泉装修垃圾分拣中心', '玉泉装修垃圾分拣中心', 19.5, 7.93),
    ('赛罕装修垃圾分拣中心', '赛罕装修垃圾分拣中心', 30, 22.83),
    ('和林格尔县装修垃圾分拣中心', '和林格尔县装修垃圾分拣中心', 30, 14.55),
    ('托克托县装修垃圾分拣中心', '托克托县装修垃圾分拣中心', 17.5, 0.04),
    ('清水河县装修垃圾分拣中心', '清水河县装修垃圾分拣中心', 10, 1.31),
    ('武川县装修垃圾分拣中心', '武川县装修垃圾分拣中心', 3, 0.02),
]
RECYCLING = [
    ('强鑫资源化利用项目', '强鑫资源化利用项目', 50, 13.43),
    ('亿友资源化利用项目', '亿友资源化利用项目', 150, 97.07),
    ('路雅资源化利用项目', '路雅资源化利用项目', 100, 22.38),
    ('土左旗城发物资再生资源利用有限公司', '土左旗城发物资再生资源利用有限公司', 30, 11.95),
]
LANDFILL = [  # (名称, 矩阵节点名, 剩余库容万m³, 备注)
    ('大里堡建筑垃圾消纳场', '大里堡建筑垃圾消纳场', 413.1, '2024年报剩余库容'),
    ('大厂库伦建筑垃圾消纳场', '大厂库伦建筑垃圾消纳场', 6.8, '退出候选：2024剩余6.8,2025年报消失'),
    ('茂林太建筑垃圾消纳场', '茂林太建筑垃圾消纳场', 58, '2024年报；2025年报34.11万方'),
    ('和林格尔县建筑垃圾消纳场', '和林格尔县建筑垃圾消纳场', 36, '2025年报'),
    ('托克托县建筑垃圾消纳场', '托克托县建筑垃圾消纳场', 19.4, '2025年报'),
    ('武川县建筑垃圾消纳场', '武川县建筑垃圾消纳场', 2.1, '2025年报'),
]
DENSITY = 1.4  # t/m³ 建筑垃圾密度（文献常用1.3-1.5，灵敏度参数）

# ---------- 距离矩阵 ----------
dist = {}
with open(f'{BASE}/distance_matrix_wide_km.csv', encoding='utf-8-sig') as f:
    rows = list(csv.reader(f))
    header = rows[0][1:]
    for row in rows[1:]:
        for j, v in enumerate(row[1:]):
            if v != '':
                dist[(row[0], header[j])] = float(v)

def D(a, b):
    if a == b:
        return 0.0
    v = dist.get((a, b))
    if v is None:
        raise KeyError(f'距离缺失: {a} -> {b}')
    return v

G = {f'G{i+1}_{n}': n for i, n in enumerate(districts)}  # 矩阵中的产生点名
GEN = list(G.keys())
SORT = [s[1] for s in SORTING]
RECY = [r[1] for r in RECYCLING]
LAND = [l[1] for l in LANDFILL]

instance = {
    'meta': {
        'name': 'hohhot-cdw-v1',
        'periods': 3,
        'source': 'hohhot case data CSVs, 2026-08-25',
        'density_t_per_m3': DENSITY,
        'notes': '成本类参数为文献/标定假设，均需灵敏度分析；2023需求为工程渣土口径',
    },
    'demand': {g: {'name': n, 'd': {str(t): districts[n]['d'][t] for t in (1,2,3)}} for g, n in G.items()},
    'sorting': {s[1]: {'name': s[0], 'cap_t': s[2]*1e4, 'load2025_t': s[3]*1e4,
                       'open_fixed': 200.0 if '区' in s[0] else 100.0,  # 万元/年（假设）
                       'op_cost': 15.0, 'buffer_cap_t': s[2]*1e4*0.2,
                       'expand_step_t': 1e5, 'expand_cost': 220.0} for s in SORTING},  # V1.3: 分拣模块化10万吨/档、220万元/档（单位能力成本不变）
    'recycling': {r[1]: {'name': r[0], 'cap_t': r[2]*1e4, 'load2025_t': r[3]*1e4,
                         'open_fixed': 800.0 if r[2] >= 100 else 500.0,  # 万元/年（假设，托克托CAPEX标定）
                         'op_cost': 30.0, 'expand_step_t': 1e5, 'expand_cost': 300.0,
                         'min_run_t': r[2]*1e4*0.1} for r in RECYCLING},
    'landfill': {l[1]: {'name': l[0], 'cap_t': round(l[2]*1e4*DENSITY, 0),  # 库容为规划期总量约束（非年度约束）
                        'note': l[3], 'open_fixed': 100.0, 'op_cost': 10.0,
                        'retire_candidate': '大厂库伦' in l[0]} for l in LANDFILL},
    'dist': {
        'a': {g: {j: D(g, j) for j in SORT} for g in GEN},          # 产生->分拣
        'b': {j: {k: D(j, k) for k in RECY} for j in SORT},          # 分拣->资源化
        'c': {j: {l: D(j, l) for l in LAND} for j in SORT},          # 分拣->消纳
        'a2': {g: {l: D(g, l) for l in LAND} for g in GEN},         # 产生->消纳直达
        'e': {k: {l: D(k, l) for l in LAND} for k in RECY},          # 资源化->消纳
    },
    'params': {
        'transport_cost': 1.0,   # 元/吨·km（基准，灵敏度0.6-1.5）
        'hold_cost': 5.0,        # 元/吨·期（假设）
        'unmet_penalty': 500.0,  # 元/吨（>任何合法路径成本2倍）
        'recyclable_ratio': 0.8, # ρ 分拣后可资源化比例（灵敏度0.7-0.9）
        'product_yield': 0.85,   # φ 资源化厂成品率（余渣比1-φ进消纳；文献0.8-0.95）
        'recycle_revenue': 20.0, # 元/吨 再生品收益（V1.1修正项：无收益则模型退化为全填埋；
                                 # 文献假设：再生骨料价为天然骨料60-80%，灵敏度0-40；
                                 # 现实锚点：2024年报资源化利用率72%）
    },
}

with open(f'{BASE}/src/instance_v1.json', 'w', encoding='utf-8') as f:
    json.dump(instance, f, ensure_ascii=False, indent=1)

# ---------- 实例报告（人读核查用） ----------
lines = ['# 实例核查报告 instance_v1', '']
lines.append(f"- 产生点 {len(GEN)} 个，三年需求合计："
             + ', '.join(f"T{t}={sum(d['d'][str(t)] for d in instance['demand'].values())/1e4:.2f}万吨" for t in (1,2,3)))
lines.append(f"- 分拣 {len(SORT)} 座（能力合计 {sum(s[2] for s in SORTING)} 万吨/年），"
             f"资源化 {len(RECY)} 座（{sum(r[2] for r in RECYCLING)} 万吨/年），消纳 {len(LAND)} 座")
lines.append(f"- 分拣→资源化能力比: {sum(s[2] for s in SORTING)/sum(r[2] for r in RECYCLING):.2f}")
lines.append(f"- 2024需求/资源化能力: {YEAR_TOTAL[2]/sum(r[2] for r in RECYCLING):.2f}（>1 需填埋兜底，符合现实）")
lines.append('')
lines.append('| 产生点 | T1(吨) | T2(吨) | T3(吨) |')
lines.append('|---|---|---|---|')
for g, n in G.items():
    d = districts[n]['d']
    lines.append(f"| {n} | {d[1]:,.0f} | {d[2]:,.0f} | {d[3]:,.0f} |")
lines.append('')
lines.append('| 设施 | 类型 | 能力(吨/年) | 2025负荷(吨) | 利用率 |')
lines.append('|---|---|---|---|---|')
for s in SORTING:
    lines.append(f"| {s[0]} | 分拣 | {s[2]*1e4:,.0f} | {s[3]*1e4:,.0f} | {s[3]/s[2]*100:.1f}% |")
for r in RECYCLING:
    lines.append(f"| {r[0]} | 资源化 | {r[2]*1e4:,.0f} | {r[3]*1e4:,.0f} | {r[3]/r[2]*100:.1f}% |")
for l in LANDFILL:
    lines.append(f"| {l[0]} | 消纳 | {l[2]}万m³余量 | — | — |")
with open(f'{BASE}/src/instance_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines[:8]))
print('\n实例已生成: src/instance_v1.json, src/instance_report.md')
