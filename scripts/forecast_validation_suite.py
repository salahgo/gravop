"""Run F2 training jobs on explicit devices, with resumable completed stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_validation'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['controls','application'])
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    a=p.parse_args()
    jobs=([(k,s) for k in ('sidm','rd','ks','film','burgers') for s in range(3)]
          if a.stage=='controls' else [('sidm_full',s) for s in range(3)])
    def execute(cmd,label):
        print('start',label,flush=True)
        with (ROOT/'logs'/f'f2_{label}.log').open('w') as f:
            subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('done',label,flush=True)
    def worker(device,group):
        for system,seed in group:
            if a.stage=='controls':
                folder=OUT/'controls'/f'{system}_s{seed}'
                hist=folder/'training.json'
                if hist.exists() and json.loads(hist.read_text())['history'][-1]['epoch']==200:
                    print('complete',folder.name,flush=True); continue
                execute(['scripts/forecast_validate.py','control','--system',system,
                         '--seed',str(seed),'--device',device],f'control_{system}_s{seed}')
            else:
                cp=OUT/'application'/f'clock_seed{seed}.pt'
                if not cp.with_suffix('.json').exists():
                    execute(['scripts/forecast_validate.py','clock','--seed',str(seed),
                             '--device',device],f'full_clock_s{seed}')
                folder=OUT/'application'/f'sidm_full_s{seed}'
                hist=folder/'training.json'
                if hist.exists() and json.loads(hist.read_text())['history'][-1]['epoch']==200:
                    print('complete',folder.name,flush=True); continue
                execute(['scripts/sc_forecast.py','train','--system','sidm_full',
                         '--seed',str(seed),'--epochs','200','--out',str(folder),
                         '--clock',str(cp),'--device',device],f'full_field_s{seed}')
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        fs=[pool.submit(worker,d,jobs[i::len(a.devices)]) for i,d in enumerate(a.devices)]
        for f in fs: f.result()


if __name__=='__main__': main()
