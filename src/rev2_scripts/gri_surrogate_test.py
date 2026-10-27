# -*- coding: utf-8 -*-
"""对比贪心代理与锚点子集 LP 代理的搜索质量。"""
import json, sys, os, time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import gri_run, make_subset_eval, config_violations

    inst = json.load(open('instance_v1.json', encoding='utf-8'))
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000), k=10, seed=7)
    OPT = 853903997.648496
    sub = make_subset_eval(inst, train, 3)

    def ex(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)

    for label, ef in [('greedy', None), ('subset', sub)]:
        t0 = time.time()
        pool = gri_run(inst, train, n_islands=6, wall_budget=60, seed=0, eval_fn=ef)
        best = None
        for cfg, gv in pool[:8]:
            v = ex(cfg)
            if v is not None and (best is None or v < best):
                best = v
        g = (best / OPT - 1) * 100 if best else float('nan')
        print(f'{label:8s}: search 60s -> best exact {best/1e6 if best else 0:.2f}M '
              f'gap={g:.2f}%  pool={len(pool)}  t={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
