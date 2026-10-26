# -*- coding: utf-8 -*-
"""GRI 统一入口的速度/质量测试（real + m2），ONLY 指定实例序号。"""
import json, sys, os, time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sys.path.insert(0, D); os.chdir(D)
    from cplex_saa import gen_scenarios
    from run_experiments import reduce_scenarios
    import alns as A
    from gri_alns import gri_solve, config_violations

    which = os.environ.get('INST', 'real')
    if which == 'real':
        fn, opt = 'instance_v1.json', 853903997.648496
    else:
        fn = f'experiments/synth_instances/city_{which}_s1001.json' if which.startswith('m') else which
        import pandas as pd
        opt = None
    inst = json.load(open(fn, encoding='utf-8'))
    seed = inst['meta'].get('gen_seed', 0)
    k = 10
    train = reduce_scenarios(inst, gen_scenarios(inst, 30, seed=1000 + seed % 100),
                             k=k, seed=7)

    def ex(c):
        if c is None or config_violations(inst, c):
            return None
        return A.evaluate(inst, c, train, require_cover=True)

    bsearch = float(os.environ.get('BS', '180'))
    bpolish = float(os.environ.get('BP', '90'))
    cfg, val, st = gri_solve(inst, train, ex, budget_search=bsearch,
                             budget_polish=bpolish, n_islands=6, seed=0)
    print(f"{which}: surrogate={st.get('surrogate')} "
          f"subset_eval={st.get('subset_eval_ms', 0):.1f}ms "
          f"t_search={st.get('t_search', 0):.0f}s t_total={st.get('time_s', 0):.0f}s "
          f"v_search={st.get('v_after_search', float('nan'))/1e6:.2f}M "
          f"v_final={st.get('v_final', float('nan'))/1e6:.2f}M "
          f"polish_evals={st.get('polish_exact_evals')} "
          f"status={st.get('status')}"
          + (f" gap={(val/opt-1)*100:.3f}%" if opt else ''), flush=True)
    if cfg is not None:
        json.dump(cfg, open(f"experiments/config_gri_solve_{which}.json", 'w'),
                  ensure_ascii=False)


if __name__ == '__main__':
    main()
