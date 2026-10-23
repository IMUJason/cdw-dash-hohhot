import json, os, sys, copy
import numpy as np, pandas as pd
D='/Users/cjx-ms/Documents/我的坚果云/科研/论文/2026 cdw 0825/hohhot case data/model'
sys.path.insert(0,D); os.chdir(D)
from cplex_saa import gen_scenarios
from run_experiments import run_method_det, run_method_alns, oos_eval, weighted_stats, EXP

base = json.load(open('instance_v1.json', encoding='utf-8'))
train = gen_scenarios(base, 30, seed=1000)
test  = gen_scenarios(base, 200, seed=20260901, include_anchor=True)

# 30 seeds for SAA and DASH(0.10) on real instance
rows = []
for seed in range(1, 31):
    for m in ['SAA', 'DRO(0.10)']:
        obj, cfg, wt = run_method_alns(base, m, train, alns_seed=seed)
        costs, probs, unmets, recycs = oos_eval(base, cfg, test)
        st = weighted_stats(costs, probs, obj, unmets, recycs)
        rows.append({'seed':seed, 'method':m, 'plan_obj':obj, 'oos_mean':st['oos_mean'],
                     'oos_p95':st['oos_p95'], 'disappointment':st['disappointment_rate'],
                     'service_fail':st.get('service_fail_rate',0), 'recycled':st['recycled_mean_t']})
    print(f'seed {seed}: SAA {rows[-2]["oos_mean"]/1e9:.3f}  DRO {rows[-1]["oos_mean"]/1e9:.3f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{EXP}/results_seeds30.csv', index=False)

# DET as reference (deterministic, no seed effect)
obj_det, cfg_det, wt_det = run_method_det(base)
c_det, p_det, u_det, r_det = oos_eval(base, cfg_det, test)
st_det = weighted_stats(c_det, p_det, obj_det, u_det, r_det)
print(f'\nDET (no seed): plan={obj_det/1e9:.3f} oos={st_det["oos_mean"]/1e9:.3f} p95={st_det["oos_p95"]/1e9:.3f} disc={st_det["disappointment_rate"]:.3f} sfr={st_det.get("service_fail_rate",0):.3f}')

for m in ['SAA', 'DRO(0.10)']:
    g = df[df.method==m]
    print(f'\n{m}: plan {g.plan_obj.min()/1e9:.3f}–{g.plan_obj.max()/1e9:.3f} (mean {g.plan_obj.mean()/1e9:.3f}±{g.plan_obj.std()/1e9:.3f})')
    print(f'      oos_mean {g.oos_mean.min()/1e9:.3f}–{g.oos_mean.max()/1e9:.3f} (mean {g.oos_mean.mean()/1e9:.3f}±{g.oos_mean.std()/1e9:.3f})')
    print(f'      disappointment {g.disappointment.min():.3f}–{g.disappointment.max():.3f} (mean {g.disappointment.mean():.3f}±{g.disappointment.std():.3f})')
    print(f'      service_fail mean {g.service_fail.mean():.3f}')
    # beats DET?
    beat_det = (g.oos_mean < st_det['oos_mean']).mean()
    print(f'      oos_mean < DET in {beat_det*100:.0f}% of seeds')
