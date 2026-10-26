# -*- coding: utf-8 -*-
"""前置验证 1：m=4/m=6 上 CPLEX 未返回解，是"没找到"还是"模型不可行"？
方法：把满配解作为 MIP start 喂给 CPLEX。若立刻返回 → 可行但搜索失败；
      若仍不可行 → 模型在该规模下真的无解（需查容量）。
"""
import json, sys, os, time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios, build_and_solve_saa
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import initial_full, repair_monotone, config_violations

    for case in ['m4', 'm6']:
        fn = f'{D}/experiments/synth_instances/city_{case}_s1001.json'
        inst = json.load(open(fn, encoding='utf-8'))
        T = inst['meta']['periods']
        train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1001), k=10, seed=7)

        cfg = initial_full(inst)
        for j in inst['sorting']:
            cfg['zJ'][j] = [4] * T
        for k in inst['recycling']:
            cfg['zK'][k] = [4] * T
        cfg = repair_monotone(cfg, inst)
        lpfeas = A.evaluate(inst, cfg, train, require_cover=True)
        print(f'{case}: max-build LP cover-feasible = {lpfeas is not None}',
              flush=True)

        # 用固定约束直接检验该配置在 CPLEX 里是否可行
        mdl, sol, V = build_and_solve_saa(inst, train, f'feas_{case}', retire_lstar=True,
                                          t_star=2, log=False, require_cover=True,
                                          time_limit=300)
        print(f'  plain CPLEX(300s): sol={"None" if sol is None else round(sol.objective_value)}',
              flush=True)
        for key in ['xJ', 'xK', 'xL', 'zJ', 'zK', 'rL']:
            for f in cfg[key]:
                for t in range(1, T + 1):
                    mdl.add_constraint(V[key][f, t] == cfg[key][f][t - 1])
        sol2 = mdl.solve()
        print(f'  max-build FIXED in CPLEX: '
              f'{"infeasible/None" if sol2 is None else round(sol2.objective_value)}'
              f'  status={mdl.solve_details.status}', flush=True)


if __name__ == '__main__':
    main()
