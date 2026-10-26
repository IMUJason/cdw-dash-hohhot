# -*- coding: utf-8 -*-
"""小规模验证驱动：最多 3 个独立实例进程，避免 daemon 嵌套限制。"""
import os, subprocess, sys, time, glob
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
PY = '/opt/anaconda3/envs/env310/bin/python'
EXP = f'{D}/experiments'
LOGD = f'{EXP}/gri_logs'
os.makedirs(LOGD, exist_ok=True)
procs = []
pending = list(range(11))
running = {}

def launch(i):
    env = dict(os.environ)
    env['ONLY'] = str(i)
    log = open(f'{LOGD}/inst{i}.log', 'w')
    p = subprocess.Popen([PY, 'rev2_scripts/gri_small_verify.py'], env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    running[i] = (p, log)
    print(f'[{time.strftime("%H:%M")}] launch inst {i}', flush=True)

while pending or running:
    while pending and len(running) < 3:
        launch(pending.pop(0))
    time.sleep(5)
    for i, (p, log) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            log.close()
            del running[i]
            print(f'[{time.strftime("%H:%M")}] inst {i} exit={rc}', flush=True)

files = sorted(glob.glob(f'{EXP}/results_gri_small_verify_inst*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(f'{EXP}/results_gri_small_verify.csv', index=False)
print('\n=== 小规模验证汇总 ===')
for i, g in df.groupby('instance'):
    print(f'inst={i}: best={g.gap_pct.min():.3f}% mean={g.gap_pct.mean():.3f}% '
          f'hits={g.exact_hit.sum()}/{len(g)} time_mean={g.time_s.mean():.0f}s')
