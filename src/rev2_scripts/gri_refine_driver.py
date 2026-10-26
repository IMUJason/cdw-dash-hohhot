# -*- coding: utf-8 -*-
"""refine 驱动：11 实例独立进程、3 路并行。"""
import os, subprocess, time, glob
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
PY = '/opt/anaconda3/envs/env310/bin/python'
EXP = f'{D}/experiments'
LOGD = f'{EXP}/gri_logs'
os.makedirs(LOGD, exist_ok=True)

pending = list(range(11))
running = {}

def launch(i):
    env = dict(os.environ)
    env['ONLY'] = str(i)
    log = open(f'{LOGD}/refine{i}.log', 'w')
    p = subprocess.Popen([PY, 'rev2_scripts/gri_refine_saved.py'], env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    running[i] = (p, log)
    print(f'[{time.strftime("%H:%M")}] refine inst {i}', flush=True)

while pending or running:
    while pending and len(running) < 3:
        launch(pending.pop(0))
    time.sleep(5)
    for i, (p, log) in list(running.items()):
        if p.poll() is not None:
            log.close()
            del running[i]
            print(f'[{time.strftime("%H:%M")}] refine {i} exit={rc if (rc:=p.returncode) is not None else "?"}', flush=True)

files = sorted(glob.glob(f'{EXP}/results_gri_refine_inst*.csv'))
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print('\n=== refine 汇总 ===')
    print(df[['instance', 'gap_pct', 'exact_hit', 'time_s']].to_string(index=False))
    print(f'命中 {df.exact_hit.sum()}/{len(df)}')
