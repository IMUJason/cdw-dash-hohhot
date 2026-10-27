# -*- coding: utf-8 -*-
"""GRI 门栅测试（real 实例）：多种子 best-of-K 精确 gap。spawn-safe。"""
import json, sys, os, time


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import gri_run

    inst = json.load(open('instance_v1.json', encoding='utf-8'))
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)
    mdl, sol, V = build_and_solve_saa(inst, train, 'gri_gate', retire_lstar=True, t_star=2,
                                      require_cover=True, time_limit=900, log=False)
    opt = sol.objective_value
    print(f'CPLEX exact: {opt/1e6:.1f}M', flush=True)

    NSEEDS = int(os.environ.get('NSEEDS', 5))
    BUDGET = float(os.environ.get('BUDGET', 60))
    rows = []
    for seed in range(NSEEDS):
        t0 = time.time()
        pool = gri_run(inst, train, n_islands=6, wall_budget=BUDGET, seed=seed)
        best_exact, n_cov, n_hit = None, 0, 0
        best_cfg = None
        for cfg, gv in pool:
            val = A.evaluate(inst, cfg, train, require_cover=True)
            if val is not None:
                n_cov += 1
                if abs(val - opt) / opt < 1e-6:
                    n_hit += 1
                if best_exact is None or val < best_exact:
                    best_exact, best_cfg = val, cfg
        # 精确抛光（算法第二阶段）
        from gri_alns import exact_polish, exact_pair_polish
        if best_cfg is not None:
            def ex(c):
                return A.evaluate(inst, c, train, require_cover=True)
            pc, pv, nm = exact_polish(inst, best_cfg, train, ex, budget_s=180)
            pc, pv2, nm2 = exact_pair_polish(inst, pc, train, ex, budget_s=180)
            if pv2 is not None and pv2 < pv:
                pv = pv2
            if pv is not None and pv < best_exact:
                best_exact = pv
                if abs(pv - opt) / opt < 1e-6:
                    n_hit += 1
        gap = (best_exact / opt - 1) * 100 if best_exact else float('nan')
        rows.append(gap)
        print(f'seed {seed}: pool={len(pool)} LPfeas={n_cov} exact_hit={n_hit} '
              f'best={best_exact/1e6 if best_exact else float("nan"):.1f}M gap={gap:.2f}% '
              f't={time.time()-t0:.0f}s', flush=True)

    import numpy as np
    print(f'\n门栅: best-of-{NSEEDS} gap={min(rows):.2f}%  mean={np.mean(rows):.2f}%  '
          f'exact-hit seeds={sum(1 for g in rows if g < 1e-4)}/{NSEEDS}')


if __name__ == '__main__':
    main()
