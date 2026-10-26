# -*- coding: utf-8 -*-
"""补充种子战役：6 个未命中实例 × 10 新种子（5-14）。"""
import os, subprocess, time, glob
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
PY = '/opt/anaconda3/envs/env310/bin/python'
EXP = f'{D}/experiments'
LOGD = f'{EXP}/gri_logs'
os.makedirs(LOGD, exist_ok=True)

INSTANCES = [0, 1, 4, 5, 6, 9]
pending = list(INSTANCES)
running = {}

def launch(i):
    env = dict(os.environ)
    env.update(ONLY=str(i), NSEEDS='10', SEED0='5', TAG='_extra',
               BUDGET='180', POLISH='120', PAIR='180')
    log = open(f'{LOGD}/extra{i}.log', 'w')
    p = subprocess.Popen([PY, 'rev2_scripts/gri_small_verify.py'], env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    running[i] = (p, log)
    print(f'[{time.strftime("%H:%M")}] extra inst {i}', flush=True)

while pending or running:
    while pending and len(running) < 5:
        launch(pending.pop(0))
    time.sleep(5)
    for i, (p, log) in list(running.items()):
        if p.poll() is not None:
            log.close()
            del running[i]
            print(f'[{time.strftime("%H:%M")}] extra {i} exit={p.returncode}', flush=True)

files = sorted(glob.glob(f'{EXP}/results_gri_small_verify_inst*_extra.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print('\n=== 补充种子汇总（每实例 10 种子）===')
for i, g in df.groupby('instance'):
    print(f'inst={i}: best={g.gap_pct.min():.4f}% hits={g.exact_hit.sum()}/10 '
          f'mean={g.gap_pct.mean():.3f}%')
