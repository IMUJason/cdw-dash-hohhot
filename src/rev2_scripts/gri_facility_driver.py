# -*- coding: utf-8 -*-
"""设施轴 GRI 驱动：逐实例独立进程，3 路并行。"""
import os, subprocess, time, glob
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
PY = '/opt/anaconda3/envs/env310/bin/python'
EXP = f'{D}/experiments'
LOGD = f'{EXP}/gri_logs'
os.makedirs(LOGD, exist_ok=True)

pending = list(range(12))     # real + 11 synth
running = {}

def launch(i):
    env = dict(os.environ)
    env['ONLY'] = str(i)
    log = open(f'{LOGD}/fac{i}.log', 'w')
    p = subprocess.Popen([PY, 'rev2_scripts/gri_facility.py'], env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    running[i] = (p, log)
    print(f'[{time.strftime("%H:%M")}] facility inst {i}', flush=True)

while pending or running:
    while pending and len(running) < 3:
        launch(pending.pop(0))
    time.sleep(5)
    for i, (p, log) in list(running.items()):
        if p.poll() is not None:
            log.close()
            del running[i]
            print(f'[{time.strftime("%H:%M")}] facility {i} exit={p.returncode}', flush=True)

files = sorted(glob.glob(f'{EXP}/results_gri_facility_inst*.csv'))
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(['m', 'instance'])
    df.to_csv(f'{EXP}/results_gri_facility.csv', index=False)
    print('\n=== 设施轴 GRI vs CPLEX ===')
    print(df[['instance', 'm', 'n_discrete', 'cplex_time', 'cplex_proved',
              'gri_time', 'gri_vs_cplex_pct']].to_string(index=False))
