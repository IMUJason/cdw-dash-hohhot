# -*- coding: utf-8 -*-
"""设施轴合成城市生成器（V2.3 审计后新增实验）。

目的：现有规模实验（scale_v22/scale_stress）只放大情景数；NP-hard 落在
离散设施维（CFLP 归约），该维从未被测试。本生成器沿设施轴放大城市：
克隆真实呼和浩特的设施/需求/距离结构 m 次，加可控扰动，使
  - 单设施规模保持真实（容量/成本/最小开工量取自实网），
  - 总需求 ∝ m，避免"拆细单元"退化为平凡问题，
  - 距离分布保持真实（克隆弧长 × lognormal 扰动，非合成网格）。

离散变量数 = 108·m（xJ,xK,xL,rL 二元 + zJ,zK∈{0..4} 整数）。

输出：experiments/synth_instances/city_m{m}_{seed}.json
"""
import json, os, sys
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D); os.chdir(D)

BASE = json.load(open(f'{D}/instance_v1.json', encoding='utf-8'))
OUT = f'{D}/experiments/synth_instances'
os.makedirs(OUT, exist_ok=True)

def gen_city(m, seed):
    """m = 复制倍数（1=真实网）；seed 控制扰动。"""
    rng = np.random.default_rng(seed)
    inst = {'meta': {'name': f'synth-city-m{m}-s{seed}', 'periods': 3,
                     'density_t_per_m3': 1.4, 'source': 'cloned from hohhot instance_v1',
                     'replication': m, 'gen_seed': int(seed)},
            'params': dict(BASE['params'])}
    # ---- 需求：9 区克隆 m 次，份额扰动后按年总量=m×真实总量 重标定 ----
    GEN0 = list(BASE['demand'])
    w = {}
    for i in GEN0:
        for c in range(m):
            w[(i, c)] = float(rng.uniform(0.85, 1.15))
    inst['demand'] = {}
    for i in GEN0:
        for c in range(m):
            nm = f'{i}_c{c}'
            d = {str(t): BASE['demand'][i]['d'][str(t)] * w[(i, c)] for t in (1, 2, 3)}
            inst['demand'][nm] = {'name': nm, 'd': d}
    for t in (1, 2, 3):
        tot = sum(v['d'][str(t)] for v in inst['demand'].values())
        target = m * sum(BASE['demand'][i]['d'][str(t)] for i in GEN0)
        for v in inst['demand'].values():
            v['d'][str(t)] *= target / tot

    def clone_fac(kind, m):
        out = {}
        for nm0, info0 in BASE[kind].items():
            for c in range(m):
                nm = f'{nm0}_c{c}'
                f = dict(info0)
                f['name'] = nm
                f['cap_t'] = info0['cap_t'] * float(rng.uniform(0.8, 1.25))
                if 'load2025_t' in info0:
                    f['load2025_t'] = info0['load2025_t'] * float(rng.uniform(0.8, 1.2))
                if 'buffer_cap_t' in info0:
                    f['buffer_cap_t'] = info0['buffer_cap_t'] * float(rng.uniform(0.8, 1.25))
                f['open_fixed'] = info0['open_fixed'] * float(rng.uniform(0.9, 1.1))
                f['op_cost'] = info0['op_cost'] * float(rng.uniform(0.85, 1.15))
                if kind == 'landfill':
                    # 每克隆组的 c==0 继承 retire_candidate（保真实退出结构）
                    f['retire_candidate'] = bool(info0.get('retire_candidate', False)) and c == 0
                out[nm] = f
        return out

    inst['sorting'] = clone_fac('sorting', m)
    inst['recycling'] = clone_fac('recycling', m)
    inst['landfill'] = clone_fac('landfill', m)

    # ---- 距离：克隆弧长 × lognormal(0,0.15)，保持真实距离分布 ----
    inst['dist'] = {}
    for arc in ('a', 'a2', 'b', 'c', 'e'):
        src = BASE['dist'][arc]
        us, vs = list(src), list(next(iter(src.values())))
        out = {}
        for u0 in us:
            for uc in range(m):
                u = f'{u0}_c{uc}'
                out[u] = {}
                for v0 in vs:
                    for vc in range(m):
                        v = f'{v0}_c{vc}'
                        out[u][v] = float(src[u0][v0] * np.exp(rng.normal(0, 0.15)))
        inst['dist'][arc] = out
    return inst

def n_discrete(inst):
    T = 3
    return (len(inst['sorting']) + len(inst['recycling']) + len(inst['landfill'])) * T * 2

if __name__ == '__main__':
    SPECS = []  # (m, seed) —— 与论文表 4 对应的规模阶梯
    for m in (1, 2, 3, 4, 6):
        for s in range(1, 4):      # 每档 3 实例（m=1 直接用真实网，不做）
            if m == 1:
                continue
            SPECS.append((m, 1000 + s))
    for m, s in SPECS:
        inst = gen_city(m, s)
        fn = f'{OUT}/city_m{m}_s{s}.json'
        json.dump(inst, open(fn, 'w', encoding='utf-8'), ensure_ascii=False)
        nJ, nK, nL = len(inst['sorting']), len(inst['recycling']), len(inst['landfill'])
        print(f'm={m} seed={s}: |I|={len(inst["demand"])} |J|={nJ} |K|={nK} |L|={nL} '
              f'→ {n_discrete(inst)} discrete vars  → {fn}')
    print('真实网（m=1）离散变量数 = 108（对照）')
