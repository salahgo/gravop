"""Run the frozen five-system forecasting recipe and summarize paired results.

Training and test evaluation are separate stages. --resume skips only completed
training jobs with the requested epoch budget. Workers are independent training
processes assigned to explicit devices; they do not change published artifacts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['train','evaluate','report'])
    p.add_argument('--systems',nargs='+',default=['sidm','rd','ks','film','burgers'])
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--devices',nargs='+',default=['cuda:0'])
    p.add_argument('--root',type=Path,default=ROOT/'results/forecast')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    jobs=[(k,s) for k in a.systems for s in range(a.seeds)]
    def worker(device,group):
        for k,s in group:
            run=a.root/f'{k}_aligned_s{s}'
            if a.stage=='train':
                if a.resume and (run/'training.json').exists() and (run/'model.pt').exists():
                    history=json.loads((run/'training.json').read_text())['history']
                    cfg=json.loads((run/'config.json').read_text())
                    if (history[-1]['epoch']==a.epochs and
                        cfg['args']['epochs']==a.epochs and
                        cfg['model']['aligned'] and cfg['model']['local']):
                        print('complete',run.name,flush=True)
                        continue
                cmd=[sys.executable,str(ROOT/'scripts/sc_forecast.py'),'train',
                     '--system',k,'--seed',str(s),'--epochs',str(a.epochs),
                     '--out',str(run),'--device',device]
            else:
                cmd=[sys.executable,str(ROOT/'scripts/sc_forecast.py'),'evaluate',
                     '--run',str(run),'--device',device]
            log=ROOT/'logs'/f'forecast_{run.name}_{a.stage}.log'
            print('start',run.name,device,flush=True)
            with log.open('w') as f:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            print('finished',run.name,flush=True)
    if a.stage!='report':
        with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
            futures=[pool.submit(worker,dev,jobs[i::len(a.devices)])
                     for i,dev in enumerate(a.devices)]
            for f in futures: f.result()
        return
    rows={}
    for k in a.systems:
        runs=[a.root/f'{k}_aligned_s{s}' for s in range(a.seeds)]
        rec=[json.loads((r/'evaluation.json').read_text()) for r in runs]
        if any(r['test_index']!=rec[0]['test_index'] for r in rec):
            raise ValueError('incompatible test indices')
        best=min(rec[0]['references'],key=lambda b:np.mean([
            r['references'][b]['validation']['nrmse'] for r in rec]))
        ref=np.asarray([r['references'][best]['test']['per_trajectory'] for r in rec])
        new=np.asarray([r['test']['per_trajectory'] for r in rec])
        rng=np.random.default_rng(20260909)
        ix=rng.integers(0,new.shape[1],(5000,new.shape[1]))
        ratio=ref.mean(0)[ix].mean(1)/new.mean(0)[ix].mean(1)
        rows[k]=dict(reference=best,baseline_nrmse=float(ref.mean()),
            model_nrmse=float(new.mean()),improvement=float(ref.mean()/new.mean()),
            reduction_percent=float(100*(1-new.mean()/ref.mean())),
            paired_trajectory_ci95=np.quantile(ratio,[.025,.975]).tolist(),
            seed_model_nrmse=new.mean(1).tolist(),seed_baseline_nrmse=ref.mean(1).tolist(),
            model_late_nrmse=float(np.mean([r['test']['late_nrmse'] for r in rec])),
            baseline_late_nrmse=float(np.mean([r['references'][best]['test']['late_nrmse'] for r in rec])),
            n_test=new.shape[1],seeds=a.seeds,
            reference_selection='lowest mean validation error across seeds: converged-clock FNO/CNO/Transformer or direct-time FNO')
    (a.root/'summary.json').write_text(json.dumps(rows,indent=2))
    print('| system | reference | baseline nRMSE | model nRMSE | improvement | paired 95% CI |')
    print('|---|---|---:|---:|---:|---|')
    for k,r in rows.items():
        lo,hi=r['paired_trajectory_ci95']
        print(f"| {k} | {r['reference']} | {r['baseline_nrmse']:.6f} | "
              f"{r['model_nrmse']:.6f} | {r['improvement']:.2f}x | [{lo:.2f}, {hi:.2f}] |")


if __name__=='__main__': main()
