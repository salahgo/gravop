"""Freeze and execute the prerequisite diagnostics for final inference validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from forecast_final_case import BASE_DATA as DATA, BASE_OUT as OUT, ROOT, arrays, worker_init
from forecast_numerics import refined_row
from forecast_calibrate import write_npz
from forecast_validate import digest, write_json

ORDER = [0, 4, 3, 1, 5, 2]


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'results/forecast_posterior_refine/source_snapshot.json').read_text())
    for name, h in old['files'].items():
        assert digest(ROOT/name) == h, f'F7 dependency changed: {name}'
    inputs = {}
    for name in ['forecast_conditional', 'forecast_posterior_refine']:
        manifest = ROOT/'results'/name/'data_manifest.json'
        inputs[str(manifest.relative_to(ROOT))] = digest(manifest)
        for rel, h in json.loads(manifest.read_text())['files'].items():
            assert digest(ROOT/rel) == h, rel
            inputs[rel] = h
    for name in ['pilot_bank.npz','pilot_truth.npz','pilot_observations.npz']:
        p = ROOT/'data/d1/forecast_numerics'/name
        inputs[str(p.relative_to(ROOT))] = digest(p)
    model_paths = sorted((ROOT/'results/forecast_validation/application').glob('sidm_full_s*/model.pt'))
    assert len(model_paths) == 3
    inputs.update({str(p.relative_to(ROOT)):digest(p) for p in model_paths})
    names = list(old['files']) + ['scripts/forecast_final_case.py', 'scripts/forecast_final_case_report.py',
        'scripts/forecast_final_validate.py','tests/test_forecast_final.py','docs/FORECAST_FINAL_PROTOCOL.md']
    config = dict(order=ORDER, sources={n:digest(ROOT/n) for n in names}, inputs=inputs,
                  decision='stop at first prerequisite failure; no population claim from passing cases alone')
    path = OUT/'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'F8 frozen configuration changed'
    else:
        write_json(path, config)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in names: z.write(ROOT/name,name)
        obs = arrays(ROOT/'data/d1/forecast_numerics/pilot_observations.npz')
        for i in range(6):
            directory = DATA/f'case{i}'
            directory.mkdir(parents=True,exist_ok=True)
            write_npz(directory/'observation.npz', **{k:v[i:i+1] for k,v in obs.items()})
    return config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        if args.prepare: return
        if (OUT/'validation.json').exists():
            print('Final validation already stopped; see validation.json',flush=True)
            return
        truth=arrays(ROOT/'data/d1/forecast_numerics/pilot_truth.npz')
        strict_path=DATA/'truth_level2.npz'
        if not strict_path.exists():
            write_json(OUT/'truth_selection.json',dict(params=truth['params'].tolist(),level=2,
                selection='all six F5 diagnostic truths, unchanged'))
            with ProcessPoolExecutor(max_workers=min(args.workers,6),initializer=worker_init,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                rows=list(pool.map(refined_row,[(i,p,2) for i,p in enumerate(truth['params'])]))
            failures=[r[-1] for r in rows if r[-1]]
            write_npz(strict_path,profiles=np.stack([r[1] for r in rows]),valid=np.stack([r[2] for r in rows]))
            write_json(OUT/'truth_generation.json',dict(failures=failures,sha256=digest(strict_path)))
        strict=arrays(strict_path)
        common=truth['valid']&strict['valid']
        delta=np.abs(truth['profiles']-strict['profiles'])
        maximum=float(delta[common].max())
        disagreements=np.argwhere(truth['valid']!=strict['valid']).tolist()
        failures=json.loads((OUT/'truth_generation.json').read_text())['failures']
        temporal=dict(max_profile_dex=maximum, per_case_max=[float(delta[i][common[i]].max()) if common[i].any() else None for i in range(6)],
            support_disagreements=disagreements,failures=failures,
            passed=bool(not failures and not disagreements and maximum<=.00434))
        write_json(OUT/'truth_temporal.json',temporal)
        print('F8 diagnostic truth temporal screen',temporal,flush=True)
        results=[]
        if temporal['passed']:
            for i in ORDER:
                if i==2:
                    result=json.loads((ROOT/'results/forecast_posterior_refine/validation.json').read_text())
                    results.append(dict(case=i,passed=result['passed'],reused='F7'))
                else:
                    for script in ['forecast_final_case.py','forecast_final_case_report.py']:
                        cmd=[sys.executable,str(ROOT/'scripts'/script),'--case',str(i)]
                        if script=='forecast_final_case.py': cmd+=['--workers',str(args.workers)]
                        log=ROOT/'logs'/f'f8_case{i}_{script.removesuffix(".py")}.log'
                        with log.open('a') as f: subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                    result=json.loads((OUT/f'case{i}/validation.json').read_text())
                    results.append(dict(case=i,passed=result['passed'],validation=f'case{i}/validation.json'))
                write_json(OUT/'diagnostic_progress.json',dict(results=results))
                print('F8 case',i,'passed',results[-1]['passed'],flush=True)
                if not results[-1]['passed']: break
        all_pass=temporal['passed'] and len(results)==6 and all(r['passed'] for r in results)
        result=dict(complete=not all_pass,diagnostic_truth_temporal=temporal,diagnostics=results,
            diagnostics_passed=all_pass,population_calibration_passed=False,
            population_status='ready for fresh confirmation' if all_pass else 'not run: required diagnostic gate failed',
            unrun_diagnostics=[i for i in ORDER if i not in [r['case'] for r in results]],
            inference_claim='No continuous-prior population calibration established')
        write_json(OUT/('diagnostics_passed.json' if all_pass else 'validation.json'),result)
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()
