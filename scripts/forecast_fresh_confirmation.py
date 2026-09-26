"""Prospective confirmation, executable only after every F8 prerequisite passes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import forecast_final_case as f8
import forecast_final_case_report as audit_case
from forecast_calibrate import LO,HI,NOISE_DEX,TIMES,write_npz
from forecast_validate import digest,write_json
from forecast_numerics import refined_row
from gravop.calibration import pit_summary,holm,wilson

ROOT=f8.ROOT
OUT=ROOT/'results/forecast_fresh_confirmation'
DATA=ROOT/'data/d1/forecast_fresh_confirmation'
SEED,NOISE_SEED,N=990000000,990000001,256


def prerequisite(root=ROOT):
    path=root/'results/forecast_final/diagnostics_passed.json'
    if not path.exists(): raise RuntimeError('fresh confirmation blocked: F8 diagnostics have not all passed')
    d=json.loads(path.read_text())
    if (not d['diagnostics_passed'] or len(d['diagnostics'])!=6 or
            not all(r['passed'] for r in d['diagnostics'])):
        raise RuntimeError('fresh confirmation blocked: a required diagnostic failed')
    return path


def overlaps(params, root=ROOT):
    targets={dtype:{p.astype(dtype).tobytes() for p in params} for dtype in ('f8','f4')}
    checks={}
    folders=['forecast_calibration','forecast_integration','forecast_numerics',
             'forecast_conditional','forecast_posterior_refine','forecast_final']
    for folder in folders:
        for path in (root/'data/d1'/folder).rglob('*.npz'):
            with np.load(path) as f:
                if 'params' in f: values=f['params'][:,:2]
                elif 'x' in f and 'y' in f: values=np.column_stack([f['x'],np.full(len(f['x']),float(f['y']))])
                elif 'theta' in f: values=f['theta'][:,:2]
                else: continue
                checks[str(path.relative_to(root))]={t:len(s&{p.astype(t).tobytes() for p in values}) for t,s in targets.items()}
    with h5py.File(root/'data/d1/train.h5') as f:
        values=np.column_stack([f['log_sigma0'][:],f['log_w'][:]])
    checks['data/d1/train.h5']={t:len(s&{p.astype(t).tobytes() for p in values}) for t,s in targets.items()}
    gpu_bank=root/'results/gpu_solver/development_params.npz'
    if gpu_bank.exists():
        with np.load(gpu_bank) as f: values=f['params']
        checks[str(gpu_bank.relative_to(root))]={t:len(s&{p.astype(t).tobytes() for p in values}) for t,s in targets.items()}
    if any(n for d in checks.values() for n in d.values()): raise ValueError('confirmation parameters overlap earlier data')
    return checks


def prepare():
    gate=prerequisite()
    base=json.loads((f8.BASE_OUT/'config.json').read_text())
    for group in ('sources','inputs'):
        for name,h in base[group].items():
            if digest(ROOT/name)!=h: raise RuntimeError(f'F8 {group} changed: {name}')
    paths=set(base['sources'])|set(base['inputs'])
    paths.update(['scripts/forecast_fresh_confirmation.py','docs/FORECAST_FRESH_CONFIRMATION_PROTOCOL.md',
                  str(gate.relative_to(ROOT)), 'tests/test_fresh_confirmation.py'])
    for p in f8.BASE_OUT.glob('case*/validation.json'): paths.add(str(p.relative_to(ROOT)))
    for p in f8.BASE_DATA.rglob('*.npz'): paths.add(str(p.relative_to(ROOT)))
    gpu_bank=ROOT/'results/gpu_solver/development_params.npz'
    if gpu_bank.exists(): paths.add(str(gpu_bank.relative_to(ROOT)))
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    path=OUT/'config.json'
    if not path.exists():
        params=np.random.default_rng(SEED).uniform(LO,HI,(N,2))
        checked=overlaps(params)
        write_json(OUT/'overlap.json',checked)
        write_npz(DATA/'params.npz',params=params)
    paths.update(str(p.relative_to(ROOT)) for p in (OUT/'overlap.json',DATA/'params.npz'))
    frozen=dict(seed=SEED,noise_seed=NOISE_SEED,n=N,files={p:digest(ROOT/p) for p in sorted(paths)})
    if path.exists():
        if json.loads(path.read_text())!=frozen: raise RuntimeError('fresh confirmation inputs or sources changed')
    else:
        write_json(path,frozen)
    np.testing.assert_array_equal(f8.arrays(DATA/'params.npz')['params'],np.random.default_rng(SEED).uniform(LO,HI,(N,2)))


def generate(workers):
    params=f8.arrays(DATA/'params.npz')['params']
    for level in (1,2):
        folder=DATA/f'level{level}';folder.mkdir(exist_ok=True)
        pending=[(i,p,level) for i,p in enumerate(params) if not (folder/f'{i}.npz').exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=f8.worker_init,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,profile,valid,status,end,error in pool.map(refined_row,pending):
                if error:
                    write_json(OUT/'generation_failure.json',dict(index=i,level=level,error=error))
                    return False
                write_npz(folder/f'{i}.npz',profiles=profile,valid=valid,status=status,t_end=end)
                write_json(folder/f'{i}.json',dict(sha256=digest(folder/f'{i}.npz')))
        for i in range(N):
            assert digest(folder/f'{i}.npz')==json.loads((folder/f'{i}.json').read_text())['sha256']
    maximum=0.;disagreements=[];base=[]
    for i in range(N):
        a,b=[f8.arrays(DATA/f'level{level}/{i}.npz') for level in (1,2)]
        common=a['valid']&b['valid']
        if common.any(): maximum=max(maximum,float(abs(a['profiles'][common]-b['profiles'][common]).max()))
        if (a['valid']!=b['valid']).any(): disagreements.append(i)
        base.append(a)
    temporal=dict(max_profile_dex=maximum,support_disagreements=disagreements,passed=not disagreements and maximum<=.00434)
    write_json(OUT/'truth_temporal.json',temporal)
    if not temporal['passed']: return False
    rng=np.random.default_rng(NOISE_SEED); rows=[];times=[];observed=[]
    for j in range(3):
        for i,a in enumerate(base):
            if a['valid'][j]:
                rows.append(i);times.append(j);observed.append(a['profiles'][j]+rng.normal(0,NOISE_DEX,20))
    observations=dict(row=np.array(rows),time_index=np.array(times),observations=np.array(observed),params=params[rows])
    observations['truth_log_likelihood']=np.array([-.5*np.square((y-base[i]['profiles'][j])/NOISE_DEX).sum() for i,j,y in zip(rows,times,observed)])
    path=DATA/'observations.npz'
    if path.exists():
        for k,v in observations.items(): np.testing.assert_array_equal(f8.arrays(path)[k],v)
    else: write_npz(path,**observations)
    write_json(OUT/'generation.json',dict(n=N,observations=len(rows),survivors=[times.count(j) for j in range(3)],observations_sha256=digest(path)))
    return True


def configure_case(i):
    prepare()
    generation=json.loads((OUT/'generation.json').read_text())
    assert digest(DATA/'observations.npz')==generation['observations_sha256']
    obs=f8.arrays(DATA/'observations.npz')
    f8.DATA=DATA/f'case{i}';f8.OUT=OUT/f'case{i}'
    f8.DATA.mkdir(exist_ok=True);f8.OUT.mkdir(exist_ok=True)
    path=f8.DATA/'observation.npz'
    values={k:v[i:i+1] for k,v in obs.items()}
    if path.exists():
        for k,v in values.items(): np.testing.assert_array_equal(f8.arrays(path)[k],v)
    else: write_npz(path,**values)
    audit_case.DATA,audit_case.OUT=f8.DATA,f8.OUT


def numerical_summary(completed):
    """Convergence and failure rate over every ATTEMPTED observation.

    REPORTING AMENDMENT (2026-09-11). The frozen rule stopped at the first
    required numerical failure, so its most likely outcome was no reportable
    quantity at all. This reports the rate instead, with a Wilson interval, and
    is the amendment's primary new quantity.
    """
    attempted=len(completed)
    if not attempted: raise ValueError('no attempted observation to summarize')
    ok=sorted(int(c['case']) for c in completed if c['passed'])
    failed=sorted(int(c['case']) for c in completed if not c['passed'])
    lo,hi=wilson(len(ok),attempted)
    return dict(attempted=attempted,converged=len(ok),failed_cases=failed,
        convergence_rate=len(ok)/attempted,convergence_rate_ci95=[lo,hi],
        failure_rate=1-len(ok)/attempted,failure_rate_ci95=[1-hi,1-lo],
        rule='reporting amendment 2026-09-11: attempt every observation, report the rate')


def summarize(n, completed):
    """Screens on the numerically successful subset, plus the failure rate.

    REPORTING AMENDMENT (2026-09-11). The frozen rule stopped the experiment at
    the first required numerical failure, so its most likely outcome was no
    reportable quantity at all after weeks of compute. The amendment attempts
    every observation and reports how many converged, with a Wilson interval.

    The screens below are therefore CONDITIONAL ON NUMERICAL SUCCESS, which is
    a selection: a case that fails the integration gates is plausibly one whose
    posterior is hardest, so dropping it can bias coverage optimistically. That
    is why `population_calibration_passed` still requires zero failures and the
    subset result is reported separately as the weaker, conditional claim.
    """
    obs=f8.arrays(DATA/'observations.npz'); methods={}
    ok=np.array(sorted(c['case'] for c in completed if c['passed']),dtype=int)
    failed=sorted(c['case'] for c in completed if not c['passed'])
    if not len(ok): raise RuntimeError('no observation passed its numerical gates')
    time_index=obs['time_index'][ok]
    for method in ('solver','candidate'):
        rows=[f8.arrays(OUT/f'case{i}/latest_{method}.npz') for i in ok]
        cdf=np.concatenate([r['cdf'] for r in rows]);tv=np.concatenate([r['tv'] for r in rows])
        by_time={};pvalues=[]
        for j,t in enumerate(TIMES):
            ix=time_index==j
            quantities={q:pit_summary(cdf[ix,k]) for k,q in enumerate(('log_sigma0','log_w','solver_log_likelihood'))}
            pvalues.extend(q['ks_pvalue'] for q in quantities.values())
            by_time[str(int(t))]=dict(quantities=quantities,median_tv=float(np.median(tv[ix])),
                mean_false_support_mass=float(np.mean([rows[i]['false_support_mass'][0] for i in np.flatnonzero(ix)])),
                mean_rejected_reference_mass=float(np.mean([rows[i]['rejected_reference_mass'][0] for i in np.flatnonzero(ix)])))
        adjusted=holm(pvalues)
        coverage=all(.85<=r['quantities'][q]['coverage']['0.9']['estimate']<=.95 for r in by_time.values() for q in ('log_sigma0','log_w'))
        methods[method]=dict(times=by_time,holm_pvalues=adjusted.tolist(),coverage_passed=coverage,
            uniformity_passed=bool((adjusted>=.05).all()),tv_passed=all(r['median_tv']<=.1 for r in by_time.values()))
    controls=negative_control(obs,ok)
    screens=all(m['coverage_passed'] and m['uniformity_passed'] and m['tv_passed'] for m in methods.values()) and controls['detected_at_all_times']
    numerical=numerical_summary(completed)
    return dict(complete=True,methods=methods,negative_control=controls,numerical=numerical,
                screens_passed_on_converged_subset=screens,
                population_calibration_passed=bool(screens and not failed),
                population_calibration_passed_conditional=bool(screens),
                selection_caveat=('the screens are conditional on numerical success; a case that fails '
                    'its integration gates is plausibly one whose posterior is hardest, so the '
                    'conditional result can be optimistically biased and is not the frozen claim'))


def single_draw_rank(value, truth, uniform):
    """Randomized rank with one independent posterior draw, including ties."""
    value, truth, uniform = np.asarray(value), np.asarray(truth), np.asarray(uniform)
    if ((uniform<0)|(uniform>1)).any(): raise ValueError('invalid rank randomizer')
    return ((value<truth).astype(float)+uniform*(1+(value==truth).astype(float)))/2


def control_one(task):
    i,j,observation,truth,truth_lp=task
    rng=np.random.default_rng(np.random.SeedSequence([991000000,i]))
    folder=DATA/'prior_control';folder.mkdir(exist_ok=True)
    trial=0
    while True:
        theta=rng.uniform(LO,HI)
        candidate=folder/f'{i}_{trial}.npz'
        if candidate.exists():
            row=f8.arrays(candidate)
            np.testing.assert_array_equal(theta,row['params'])
        else:
            _,profile,valid,status,end,error=refined_row((0,theta,1))
            if error:
                write_json(folder/f'{i}_failure.json',dict(params=theta.tolist(),error=error))
                raise RuntimeError(f'prior-control numerical failure: {error}')
            row=dict(params=theta,profiles=profile,valid=valid,status=status,t_end=end)
            write_npz(candidate,**row)
        trial+=1
        if row['valid'][j]: break
    lp=-.5*np.square((row['profiles'][j]-observation)/NOISE_DEX).sum()
    u=np.random.default_rng(np.random.SeedSequence([991000001,i])).uniform(size=3)
    rank=single_draw_rank(np.r_[theta,lp],np.r_[truth,truth_lp],u)
    write_npz(folder/f'{i}.npz',rank=rank,params=theta,attempts=trial)
    return rank,trial


def negative_control(obs,ok=None):
    # Each observation has its OWN independent prior draw. Shared finite-bank
    # ranks would be dependent and would not have the ordinary iid null.
    # `ok` restricts the control to the same observations the method screens
    # used, so the comparison stays on identical data. Each draw keeps its
    # original observation-indexed seed, so restricting does not reseed it.
    index=range(len(obs['time_index'])) if ok is None else [int(i) for i in ok]
    tasks=[(i,int(obs['time_index'][i]),obs['observations'][i],obs['params'][i],obs['truth_log_likelihood'][i])
           for i in index]
    with ProcessPoolExecutor(max_workers=24,initializer=f8.worker_init,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(control_one,tasks))
    ranks=[r[0] for r in rows];attempts=sum(r[1] for r in rows)
    ranks=np.array(ranks);times={};pvalues=[]
    time_index=obs['time_index'][np.asarray(index,dtype=int)]
    for j,t in enumerate(TIMES):
        quantities={q:pit_summary(ranks[time_index==j,k]) for k,q in enumerate(('log_sigma0','log_w','solver_log_likelihood'))}
        times[str(int(t))]=quantities
        pvalues.extend(q['ks_pvalue'] for q in quantities.values())
    adjusted=holm(pvalues)
    return dict(kind='one independent survival-conditioned prior draw per observation; exact randomized finite-sample rank',
        times=times,holm_pvalues=adjusted.tolist(),solver_evaluations=attempts,
        detected_at_all_times=bool((adjusted[[2,5,8]]<.05).all()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--case',type=int)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    prerequisite()
    if args.case is not None:
        configure_case(args.case)
        with (f8.OUT/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            f8.adaptive(SimpleNamespace(workers=args.workers,columns=24))
            audit_case.main()
        return
    prepare()
    if args.prepare: return
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'validation.json').exists(): return
        if not generate(args.workers):
            write_json(OUT/'validation.json',dict(complete=True,population_calibration_passed=False,reason='required generating-truth solver/temporal check failed'))
            return
        n=len(f8.arrays(DATA/'observations.npz')['row']);completed=[]
        for i in range(n):
            log=ROOT/'logs'/f'f9_case{i}.log'
            # REPORTING AMENDMENT (2026-09-11): a case that raises is recorded as
            # a numerical failure and the sweep continues, where the frozen rule
            # returned here. The exception text is kept so a crash is never
            # silently indistinguishable from a clean gate failure.
            try:
                with log.open('a') as f:
                    subprocess.run([sys.executable,__file__,'--case',str(i),'--workers',str(args.workers)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                result=json.loads((OUT/f'case{i}/validation.json').read_text())
                entry=dict(case=i,passed=bool(result['passed']))
            except (subprocess.CalledProcessError,OSError,ValueError,KeyError) as exc:
                entry=dict(case=i,passed=False,error=f'{type(exc).__name__}: {exc}')
            completed.append(entry)
            write_json(OUT/'progress.json',dict(n=n,completed=completed))
        try:
            result=summarize(n,completed)
        except RuntimeError as exc:
            result=dict(complete=True,population_calibration_passed=False,reason=str(exc),stage='negative control')
        write_json(OUT/'validation.json',result)
        print('Fresh confirmation complete; see validation.json',flush=True)


if __name__=='__main__': main()
