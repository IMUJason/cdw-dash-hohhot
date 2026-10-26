# -*- coding: utf-8 -*-
"""链感知贪心回收分配器（GRI-ALNS 评估核，V2.4 rev6 —— 整体重写）。

O(弧数) 构造二阶段可行流（s≡0，与 V2.4 终端清空模型一致）。性质：
  成功 ⇒ 流可行，弧级成本与 LP 系数逐项相同 ⇒ 成本 ≥ Q(x,ξ)（认证上界）。
  失败 ⇒ None（保守）。

机制：
  A. 前瞻库容预算：直运耗 1.0 t/t 库容、分拣链耗 κ=(1−ρ)+ρ(1−φ) t/t；
     R_avail = room − 未来最小需求（未来满负荷分拣估算 − 未来新开填埋）；
     x_min = (D − R_avail)/(1−κ)，加 min_run/ρ 下界
  B. 比价拆分：按溢价 (c_s − c_d) 升序挑分拣吨位，边界区可部分拆分
  C. 兜底与重试：x_min > S_cap → 全量分拣兜底；残渣/弃料溢出 →
     从最贵源弧退款改直运、缩减分拣量后整期重试一次
  D. 弧级记账：placed_arc 支持精确退款；重试前快照恢复库容状态
"""
import numpy as np


def greedy_recourse(inst, xcfg, d_scenario, require_cover=False, rng=None, jitter=0.0,
                     x_policy='premium'):
    """返回 (stage2_cost_ub, unmet_total, recycled_total)；不可行返回 (None, None, None)。"""
    greedy_recourse.last_fail = None

    def fail(site):
        greedy_recourse.last_fail = site
        return None, None, None

    T = inst['meta']['periods']
    GEN = list(inst['demand']); SORT = list(inst['sorting'])
    RECY = list(inst['recycling']); LAND = list(inst['landfill'])
    P = inst['params']; rho = P['recyclable_ratio']; phi = P.get('product_yield', 1.0)
    tr = P['transport_cost']; rev = P.get('recycle_revenue', 0.0)
    pen = P['unmet_penalty']
    kappa = (1 - rho) + rho * (1 - phi)

    dist_a, dist_a2 = inst['dist']['a'], inst['dist']['a2']
    dist_b, dist_c, dist_e = inst['dist']['b'], inst['dist']['c'], inst['dist']['e']
    opJ = {j: inst['sorting'][j]['op_cost'] for j in SORT}
    opK = {k: inst['recycling'][k]['op_cost'] for k in RECY}
    opL = {l: inst['landfill'][l]['op_cost'] for l in LAND}

    def j_():
        return rng.uniform(-jitter, jitter) if (jitter > 0 and rng is not None) else 0.0

    cost = 0.0
    unmet_tot = 0.0
    recycled_tot = 0.0
    cum_used = {l: 0.0 for l in LAND}

    for t in range(1, T + 1):
        open_K = [k for k in RECY if xcfg['xK'][k][t - 1] == 1]
        live_L = [l for l in LAND if xcfg['xL'][l][t - 1] == 1 and xcfg['rL'][l][t - 1] == 0]
        if not live_L:
            return fail('no-landfill')
        open_J = ([j for j in SORT if xcfg['xJ'][j][t - 1] == 1] if open_K else [])
        capJ = {j: inst['sorting'][j]['cap_t']
                + inst['sorting'][j].get('expand_step_t', 30e4) * xcfg['zJ'][j][t - 1]
                for j in open_J}
        capK = {k: inst['recycling'][k]['cap_t']
                + inst['recycling'][k]['expand_step_t'] * xcfg['zK'][k][t - 1] for k in open_K}
        minrun = {k: inst['recycling'][k].get('min_run_t', 0.0) for k in open_K}
        remL0 = {l: inst['landfill'][l]['cap_t'] for l in live_L}
        roomL0 = {l: inst['landfill'][l]['cap_t'] - cum_used[l] for l in live_L}
        room_tot = sum(min(remL0[l], roomL0[l]) for l in live_L)
        D_tot = sum(float(d_scenario[i][t]) for i in GEN)
        S_cap = sum(capJ.values())

        def plant_unit(j, k):
            return tr * dist_b[j][k] + opK[k] - rev * phi

        # ---- (A) 前瞻库容预算 ----
        D_f = sum(float(d_scenario[i][tt]) for i in GEN for tt in range(t + 1, T + 1))
        if open_K:
            S_f = sum(inst['sorting'][j]['cap_t']
                      + inst['sorting'][j].get('expand_step_t', 30e4) * xcfg['zJ'][j][tt - 1]
                      for tt in range(t + 1, T + 1) for j in SORT
                      if xcfg['xJ'][j][tt - 1] == 1)
            new_room_f = sum(inst['landfill'][l]['cap_t']
                             for tt in range(t + 1, T + 1) for l in LAND
                             if xcfg['xL'][l][tt - 1] == 1 and xcfg['rL'][l][tt - 1] == 0
                             and not (xcfg['xL'][l][t - 1] == 1 and xcfg['rL'][l][t - 1] == 0))
        else:
            S_f, new_room_f = 0.0, 0.0
        min_need_f = max(0.0, D_f - (1 - kappa) * min(S_f, D_f)) if kappa < 1 else D_f
        R_avail = max(0.0, room_tot - max(0.0, min_need_f - new_room_f))
        x_min = max(0.0, (D_tot - R_avail) / (1 - kappa)) if kappa < 1 - 1e-9 else D_tot
        x_min = max(x_min, sum(minrun.values()) / max(rho, 1e-9))
        x_min = min(x_min, D_tot)
        if x_min > S_cap + 1e-6:
            x_min = min(S_cap, D_tot)        # 兜底：全量分拣，真不可行由 u 阶段揭示

        # 即将关闭的填埋场（use-it-or-lose-it：其剩余库容下期作废，优先消耗）
        if t < T:
            closing_L = {l for l in live_L
                         if not (xcfg['xL'][l][t] == 1 and xcfg['rL'][l][t] == 0)}
        else:
            closing_L = set()
        def L_key(base, l):
            return (0 if l in closing_L else 1, base + j_())

        # ---- (B) 比价拆分 ----
        down = {j: rho * min(plant_unit(j, k) + (1 - phi) * min(
            tr * dist_e[k][l] + opL[l] for l in live_L) for k in open_K)
            + (1 - rho) * min(tr * dist_c[j][l] + opL[l] for l in live_L)
            for j in open_J}
        cd = {i: min(tr * dist_a2[i][l] + opL[l] for l in live_L) for i in GEN}
        cs = {i: (min((tr * dist_a[i][j] + opJ[j] + down[j], j) for j in open_J)
                  if open_J else (float('inf'), None)) for i in GEN}
        prem_list = sorted(((cs[i][0] - cd[i], i) for i in GEN if open_J), key=lambda x: x[0])
        neg_prem_dem = sum(float(d_scenario[i][t]) for prem, i in prem_list if prem <= 0)
        if x_policy == 'maxsort':
            x_target = min(S_cap, D_tot)          # 最大分拣：覆盖优先的极端策略
        else:
            x_target = min(S_cap, D_tot, max(x_min, neg_prem_dem))
        route_sort = {}
        filled = 0.0
        for prem, i in prem_list:
            if filled >= x_target - 1e-9:
                break
            take = min(float(d_scenario[i][t]), x_target - filled)
            if take > 1e-9:
                route_sort[i] = take
                filled += take

        # ---- 执行（溢出缩减重试至多 2 次） ----
        unmet_t = 0.0
        for attempt in range(2):
            remJ = dict(capJ); remK = dict(capK)
            remL = dict(remL0); roomL = dict(roomL0)
            inJ = {j: 0.0 for j in open_J}
            placed = []            # (i, j, qty, unit_cost)
            spill = []
            period_cost = 0.0
            cum_add = {l: 0.0 for l in live_L}
            # y 入料
            for prem, i in prem_list:
                if i not in route_sort:
                    continue
                d_rem = route_sort[i]
                for jj in sorted(open_J, key=lambda jj: tr * dist_a[i][jj] + opJ[jj] + down[jj]):
                    if d_rem <= 1e-9:
                        break
                    put = min(d_rem, remJ[jj])
                    if put > 1e-9:
                        uc = tr * dist_a[i][jj] + opJ[jj]
                        period_cost += put * uc
                        placed.append((i, jj, put, uc))
                        remJ[jj] -= put; inJ[jj] += put; d_rem -= put
                if d_rem > 1e-9:
                    spill.append((i, d_rem))
            for i in GEN:
                rest = float(d_scenario[i][t]) - route_sort.get(i, 0.0)
                if rest > 1e-9:
                    spill.append((i, rest))
            # 厂容量耦合：退最贵弧改直运
            if open_K and inJ:
                freeK = sum(max(remK[k] - minrun[k], 0.0) for k in open_K)
                needW = rho * sum(inJ.values())
                if needW > freeK + 1e-6:
                    excess = needW - freeK
                    for a in range(len(placed) - 1, -1, -1):
                        if excess <= 1e-6:
                            break
                        i, jj, q, uc = placed[a]
                        refund = min(q, excess / rho)
                        if refund <= 1e-9:
                            continue
                        period_cost -= refund * uc
                        inJ[jj] -= refund
                        excess -= refund * rho
                        placed[a] = (i, jj, q - refund, uc)
                        spill.append((i, refund))
                    if excess > 1e-6:
                        return fail('plant-capacity')
            # w：min_run + 贪心
            inflowK = {k: 0.0 for k in open_K}
            if open_K and inJ:
                w_tot = {j: rho * inJ[j] for j in open_J}
                for k in open_K:
                    need = minrun[k]
                    for j in sorted(open_J, key=lambda j: plant_unit(j, k) + j_()):
                        if need <= 1e-9:
                            break
                        take = min(need, w_tot[j], remK[k] - inflowK[k])
                        if take > 1e-9:
                            period_cost += take * plant_unit(j, k)
                            w_tot[j] -= take; inflowK[k] += take; need -= take
                    if need > 1e-9:
                        return fail('min-run')
                for j in open_J:
                    rem_w = w_tot[j]
                    for k in sorted(open_K, key=lambda k: plant_unit(j, k) + j_()):
                        if rem_w <= 1e-9:
                            break
                        put = min(rem_w, remK[k] - inflowK[k])
                        if put > 1e-9:
                            period_cost += put * plant_unit(j, k)
                            inflowK[k] += put; rem_w -= put
                    if rem_w > 1e-9:
                        return fail('w-capacity')
            # v：厂残渣
            ovf = 0.0
            for k in open_K:
                v = (1 - phi) * inflowK[k]
                for l in sorted(live_L, key=lambda l: L_key(tr * dist_e[k][l] + opL[l], l)):
                    if v <= 1e-9:
                        break
                    put = min(v, remL[l], roomL[l])
                    if put > 1e-9:
                        period_cost += put * (tr * dist_e[k][l] + opL[l])
                        remL[l] -= put; roomL[l] -= put; cum_add[l] += put; v -= put
                ovf += v
            # q：分拣弃料
            for j in open_J:
                q = (1 - rho) * inJ[j]
                for l in sorted(live_L, key=lambda l: L_key(tr * dist_c[j][l] + opL[l], l)):
                    if q <= 1e-9:
                        break
                    put = min(q, remL[l], roomL[l])
                    if put > 1e-9:
                        period_cost += put * (tr * dist_c[j][l] + opL[l])
                        remL[l] -= put; roomL[l] -= put; cum_add[l] += put; q -= put
                ovf += q
            # 溢出 → 缩减分拣量整期重试一次
            if ovf > 1e-6 and attempt == 0 and inJ:
                need_cut = ovf / max(kappa, 1e-9)
                new_route = dict(route_sort)
                for a in range(len(placed) - 1, -1, -1):
                    if need_cut <= 1e-6:
                        break
                    i, jj, q, uc = placed[a]
                    cut = min(q, need_cut)
                    if cut <= 1e-9:
                        continue
                    new_route[i] = new_route.get(i, 0.0) - cut
                    need_cut -= cut
                route_sort = {i: v for i, v in new_route.items() if v > 1e-9}
                continue
            if ovf > 1e-6:
                return fail('residue-room')
            # 成功：落账
            cost += period_cost
            recycled_tot += sum(inflowK.values())
            for l in live_L:
                cum_used[l] += cum_add[l]
            # yd → u
            for i, q in spill:
                for l in sorted(live_L, key=lambda l: L_key(tr * dist_a2[i][l] + opL[l], l)):
                    if q <= 1e-9:
                        break
                    put = min(q, remL[l], roomL[l])
                    if put > 1e-9:
                        cost += put * (tr * dist_a2[i][l] + opL[l])
                        remL[l] -= put; roomL[l] -= put; cum_used[l] += put; q -= put
                if q > 1e-9:
                    cost += q * pen
                    unmet_t += q
            break

        if require_cover and unmet_t > 1e-6:
            return fail('coverage')
        unmet_tot += unmet_t

    return cost, unmet_tot, recycled_tot


def greedy_recourse2(inst, xcfg, d_scenario, require_cover=False, rng=None, jitter=0.0):
    """双策略取优：premium（比价）+ maxsort（最大分拣）。二者均为认证上界。"""
    best = (None, None, None)
    for pol in ('premium', 'maxsort'):
        r = greedy_recourse(inst, xcfg, d_scenario, require_cover=require_cover,
                            rng=rng, jitter=jitter, x_policy=pol)
        if r[0] is not None:
            if best[0] is None or r[0] < best[0]:
                best = r
        elif pol == 'premium':
            continue
    return best
