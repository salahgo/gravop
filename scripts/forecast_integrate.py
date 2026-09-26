"""F4 adaptive direct-solver integration; frozen models and fresh observations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gravop.quadrature import DyadicMesh
from gravop.calibration import posterior_weights,randomized_pit,pit_summary,holm
from forecast_calibrate import solver_row,write_npz,LO,HI,TIMES,NOISE_DEX,marginal_summary
from forecast_validate import write_json,digest,from_d1
from forecast_application import Ensemble,initial_nfw,interpolate
from e9_inference import R_OBS

OUT=ROOT/'results/forecast_integration'
DATA=ROOT/'data/d1/forecast_integration'
OLD=ROOT/'data/d1/forecast_calibration'
LATTICE=4096
BUDGET=8000


def solve(params,workers):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(solver_row,[(i,p,False) for i,p in enumerate(params)],chunksize=4):rows.append(row)
    rows.sort(key=lambda x:x[0])
    failures=[dict(index=r[0],error=r[-1]) for r in rows if r[-1]]
    if failures:
        write_json(OUT/'solver_failures.json',failures)
        raise RuntimeError(f'{len(failures)} solver failures; numerical reference blocked')
    return dict(params=params,profiles=np.stack([r[1] for r in rows]),
                valid=np.stack([r[2] for r in rows]),status=np.array([r[3] for r in rows]),
                t_end=np.array([r[4] for r in rows]))


def generate(args):
    DATA.mkdir(parents=True,exist_ok=True)
    path=DATA/'confirmation.npz'
    if path.exists():raise SystemExit('fresh F4 data already exists')
    params=np.random.default_rng(980000000).uniform(LO,HI,(256,2))
    start=time.perf_counter();d=solve(params,args.workers);write_npz(path,**d)
    rng=np.random.default_rng(980100000);ids=[];times=[];obs=[]
    for j in range(3):
        ix=np.flatnonzero(d['valid'][:,j]);ids.extend(ix.tolist());times.extend([j]*len(ix))
        obs.extend(d['profiles'][ix,j]+rng.normal(0,NOISE_DEX,(len(ix),20)))
    ids=np.array(ids);times=np.array(times);obs=np.array(obs)
    lp=-.5*np.square((obs-d['profiles'][ids,times])/NOISE_DEX).sum(1)
    write_npz(DATA/'observations.npz',row=ids,time_index=times,observations=obs,
              params=params[ids],truth_log_likelihood=lp)
    # Compare the actual float32 parameter identities consumed by the models.
    identities={r.astype('f4').tobytes() for r in params};overlaps={}
    for name in ('calibration','confirmation'):
        old=np.load(OLD/f'{name}.npz')['params']
        overlaps['F3_'+name]=len(identities&{r.astype('f4').tobytes() for r in old})
    old=from_d1(ROOT/'data/d1/train.h5')['par'];old=old[old[:,2]==1,:2]
    overlaps['D1']=len(identities&{r.tobytes() for r in old})
    assert not any(overlaps.values()),overlaps
    write_json(OUT/'generation.json',dict(n=256,survivors=d['valid'].sum(0).tolist(),
        statuses={str(v):int((d['status']==v).sum()) for v in np.unique(d['status'])},
        failures=0,overlap=overlaps,seconds=time.perf_counter()-start,
        data_sha256=digest(path),observations_sha256=digest(DATA/'observations.npz'),
        protocol_sha256=digest(ROOT/'docs/FORECAST_INTEGRATION_PROTOCOL.md')))
    print('generated F4',d['valid'].sum(0),'observations',len(obs),flush=True)


class Bank:
    def __init__(self):
        path=DATA/'bank.npz'
        if path.exists():
            with np.load(path) as f:self.d={k:f[k] for k in f.files}
        else:
            old=np.load(OLD/'grid.npz');n=len(old['params'])
            self.d={k:old[k].copy() for k in ('params','profiles','valid','status','t_end')}
            self.d['keys']=np.rint((self.d['params']-LO)/(HI-LO)*LATTICE).astype(int)
            assert n==4225
            write_npz(path,**self.d)
        self.index={tuple(k):i for i,k in enumerate(self.d['keys'])}

    def ensure(self,keys,workers):
        missing=np.array([k for k in keys if tuple(k) not in self.index],dtype=int)
        if not len(missing):return True
        if len(self.d['keys'])+len(missing)>4225+BUDGET:return False
        params=LO+(missing/LATTICE)*(HI-LO)
        print('solve',len(missing),'new nodes; total new',len(self.d['keys'])-4225+len(missing),flush=True)
        new=solve(params,workers);new['keys']=missing
        self.d={k:np.concatenate([self.d[k],new[k]]) for k in self.d}
        self.index={tuple(k):i for i,k in enumerate(self.d['keys'])}
        write_npz(DATA/'bank.npz',**self.d)
        return True

    def ids(self,keys):return np.array([self.index[tuple(k)] for k in keys])


def log_likelihood(profiles,valid,observations):
    n=len(profiles);nc=len(observations['row']);out=np.full((n,nc),-np.inf)
    for j in range(3):
        ix=np.flatnonzero(observations['time_index']==j)
        means=np.nan_to_num(profiles[:,j]);ys=observations['observations'][ix]
        # Float64 squared-distance identity avoids a nodes x mocks x radii tensor.
        sq=np.square(means).sum(1)[:,None]+np.square(ys).sum(1)[None]-2*means@ys.T
        lp=-.5*np.maximum(sq,0)/NOISE_DEX**2;lp[~valid[:,j]]=-np.inf
        out[:,ix]=lp
    return out


def summarize(theta,weights,lp,reference_lp,observations):
    good=weights>0;theta=theta[good];weights=weights[good]
    lp=lp[good];reference_lp=reference_lp[good]
    means=[];sds=[];widths=[];cdf=[];covered=[];tvs=[];false_mass=[];rejected_mass=[]
    logw=np.log(weights)
    for i in range(lp.shape[1]):
        q=posterior_weights(lp[:,i]+logw);ref=posterior_weights(reference_lp[:,i]+logw)
        mu,sd,interval=marginal_summary(theta,q)
        means.append(mu);sds.append(sd);widths.append(interval[:,1]-interval[:,0])
        truth=observations['params'][i]
        cdf.append([randomized_pit(theta[:,j],q,truth[j],.5) for j in range(2)]+
                   [randomized_pit(reference_lp[:,i],q,observations['truth_log_likelihood'][i],.5)])
        covered.append((truth>=interval[:,0])&(truth<=interval[:,1]))
        tvs.append(.5*np.abs(q-ref).sum())
        false_mass.append(q[~np.isfinite(reference_lp[:,i])].sum())
        rejected_mass.append(ref[~np.isfinite(lp[:,i])].sum())
    return {k:np.array(v) for k,v in dict(mean=means,sd=sds,width90=widths,cdf=cdf,
        covered90=covered,tv=tvs,false_support_mass=false_mass,rejected_reference_mass=rejected_mass).items()}


def comparison(a,b,obs):
    rows={}
    for j,t in enumerate(TIMES):
        ix=obs['time_index']==j
        shift=np.quantile(np.abs(a['mean'][ix]-b['mean'][ix])/np.maximum(b['sd'][ix],1e-12),.95,axis=0)
        width=np.median(np.abs(a['width90'][ix]-b['width90'][ix])/np.maximum(b['width90'][ix],1e-12),axis=0)
        cdf=np.quantile(np.abs(a['cdf'][ix]-b['cdf'][ix]),.95,axis=0)
        rows[str(int(t))]=dict(n=int(ix.sum()),mean_shift_sd_p95=shift.tolist(),width_relative_median=width.tolist(),
            cdf_difference_p95=cdf.tolist(),passed=bool((shift<=.1).all() and (width<=.1).all() and (cdf<=.02).all()))
    return dict(times=rows,passed=all(v['passed'] for v in rows.values()))


def adaptive(args):
    obs=np.load(DATA/'observations.npz');bank=Bank();state_path=DATA/'mesh.json'
    if not (DATA/'initial_rule.npz').exists():
        base=DyadicMesh();keys,indices=base.stencil();ids=bank.ids(keys)
        weights=np.zeros(len(bank.d['keys']))
        weights[ids]=base.weights(indices,len(keys))
        write_npz(DATA/'initial_rule.npz',weights=weights,keys=bank.d['keys'])
    if state_path.exists():
        state=json.loads(state_path.read_text());mesh=DyadicMesh(cells=state['cells'])
        stage=int(state['stage']);history=state['history']
        if state.get('complete') and not args.refine_next:
            print('reuse completed adaptive integration',flush=True);return
        if args.refine_next:
            if not state.get('complete') or stage>=2:
                raise ValueError('next tolerance requires a completed earlier stage')
            stage+=1
    else:mesh=DyadicMesh();stage=0;history=[]
    targets=[.02,.005,.00125];start=time.perf_counter()
    while True:
        keys,indices=mesh.stencil()
        if not bank.ensure(keys,args.workers):
            # The preceding mesh remains the last fully evaluated rule.
            print('node budget reached before requested refinement',flush=True);break
        ids=bank.ids(keys);lp=log_likelihood(bank.d['profiles'],bank.d['valid'],obs)
        scale=np.max(lp,axis=0);values=np.exp(lp[ids]-scale)
        assessment=mesh.assess(values,indices);worst=float(assessment['relative_error'].max())
        weights=np.zeros(len(bank.d['keys']));weights[ids]=mesh.weights(indices,len(keys))
        row=dict(iteration=len(history),nodes=len(bank.d['keys']),new_nodes=len(bank.d['keys'])-4225,
                 cells=len(mesh.cells),worst_relative_error=worst,target=targets[stage],
                 median_relative_error=float(np.median(assessment['relative_error'])),
                 p95_relative_error=float(np.quantile(assessment['relative_error'],.95)),
                 elapsed_this_session=time.perf_counter()-start)
        history.append(row);print('mesh',row,flush=True)
        write_npz(DATA/'latest_rule.npz',weights=weights,keys=bank.d['keys'])
        write_json(OUT/'progress.json',dict(history=history,converged=False))
        if worst<=targets[stage]:
            label=f'stage{stage}'
            write_npz(DATA/f'{label}.npz',weights=weights,keys=bank.d['keys'])
            estimate=summarize(bank.d['params'],weights,lp,lp,obs)
            write_npz(DATA/f'{label}_summary.npz',**estimate)
            write_json(OUT/f'{label}.json',dict(**row,converged=True))
            if stage>=1:
                prev=np.load(DATA/f'stage{stage-1}_summary.npz')
                comp=comparison(prev,estimate,obs);write_json(OUT/'solver_convergence.json',comp)
                if comp['passed'] or stage==2:
                    write_json(OUT/'progress.json',dict(history=history,converged=comp['passed'],final_stage=stage,
                        action='completed successive-rule check'))
                    write_json(state_path,dict(cells=mesh.cells.tolist(),stage=stage,history=history,complete=True))
                    return
            stage+=1
        eligible=np.flatnonzero(mesh.cells[:,2]>=4)
        rank=eligible[np.argsort(assessment['priority'][eligible])[::-1]]
        # Prevent a batch from overrunning the new-node budget. Splitting a
        # cell needs at most 16 additional nodes; neighbors often share them.
        slots=min(64,(4225+BUDGET-len(bank.d['keys']))//16)
        if not slots or not len(rank):
            print('refinement cap reached',flush=True);break
        mesh.refine(rank[:slots])
        write_json(state_path,dict(cells=mesh.cells.tolist(),stage=stage,history=history,complete=False))
    write_json(OUT/'progress.json',dict(history=history,converged=False,final_stage=stage,
        action='refinement cap reached; do not claim integration convergence'))


def evaluate(args):
    obs=np.load(DATA/'observations.npz');bank=Bank();progress=json.loads((OUT/'progress.json').read_text())
    done=sorted(DATA.glob('stage[0-9].npz'))
    if progress['action'].startswith('refinement cap'):
        rules=(done[-1:] or [DATA/'initial_rule.npz'])+[DATA/'latest_rule.npz']
    else:
        rules=done[-2:] if len(done)>=2 else done+[DATA/'latest_rule.npz']
    if len(rules)<2:raise RuntimeError('need two evaluated meshes for comparison')
    lp_ref=log_likelihood(bank.d['profiles'],bank.d['valid'],obs)
    prediction=DATA/'model_predictions.npz'
    hashes={}
    if prediction.exists():
        p=np.load(prediction);pred={k:p[k] for k in p.files}
        provenance=json.loads((OUT/'prediction_provenance.json').read_text())
        if provenance['bank_sha256']!=digest(DATA/'bank.npz'):
            raise ValueError('bank refined; archive the preceding prediction cache before reevaluating')
        hashes=provenance['checkpoint_hashes']
    else:
        pred={};profile=initial_nfw();par=np.column_stack([bank.d['params'],np.ones(len(bank.d['params']))])
        for kind in ('candidate','production'):
            ens=Ensemble(kind,args.device);hashes[kind]=ens.hashes
            means=[];valid=[]
            for t in TIMES:
                ys=[];vs=[]
                for off in range(0,len(par),256):
                    ps=par[off:off+256]
                    y,q=ens.predict(np.repeat(profile[None],len(ps),0),ps,t)
                    if not np.isfinite(y).all() or not np.isfinite(q).all():raise ValueError('nonfinite prediction')
                    ys.append(interpolate(y[:,0],R_OBS));vs.append(q<=1)
                means.append(np.concatenate(ys));valid.append(np.concatenate(vs))
            pred[kind+'_profiles']=np.stack(means,1);pred[kind+'_valid']=np.stack(valid,1)
        write_npz(prediction,**pred)
        write_json(OUT/'prediction_provenance.json',dict(bank_sha256=digest(DATA/'bank.npz'),
                    checkpoint_hashes=hashes,predictions_sha256=digest(prediction)))
    lps={'solver':lp_ref,'prior_only':np.where(np.isfinite(lp_ref),0.,-np.inf)}
    for kind in ('candidate','production'):
        lps[kind]=log_likelihood(pred[kind+'_profiles'],pred[kind+'_valid'],obs)
    summaries={};out={'convergence':{},'methods':{},'progress':progress,'checkpoint_hashes':hashes}
    for kind,lp in lps.items():
        estimates=[]
        for rule in rules:
            with np.load(rule) as f:
                assert np.array_equal(bank.d['keys'][:len(f['keys'])],f['keys'])
                weights=np.pad(f['weights'],(0,len(bank.d['keys'])-len(f['weights'])))
            estimates.append(summarize(bank.d['params'],weights,lp,lp_ref,obs))
        out['convergence'][kind]=comparison(*estimates,obs)
        estimate=estimates[-1];summaries.update({kind+'_'+k:v for k,v in estimate.items()})
        rows={}
        for j,t in enumerate(TIMES):
            ix=obs['time_index']==j;pit=estimate['cdf'][ix]
            rows[str(int(t))]=dict(n=int(ix.sum()),median_tv=float(np.median(estimate['tv'][ix])),
                p90_tv=float(np.quantile(estimate['tv'][ix],.9)),
                mean_false_support_mass=float(estimate['false_support_mass'][ix].mean()),
                mean_rejected_reference_mass=float(estimate['rejected_reference_mass'][ix].mean()),
                ordinary_coverage90=estimate['covered90'][ix].mean(0).tolist(),
                median_width90=np.median(estimate['width90'][ix],axis=0).tolist(),
                quantities={name:pit_summary(pit[:,k]) for k,name in enumerate(('log_sigma0','log_w','solver_log_likelihood'))})
        records=[q for r in rows.values() for q in r['quantities'].values()]
        for r,p in zip(records,holm([r['ks_pvalue'] for r in records])):r['holm_pvalue']=float(p)
        out['methods'][kind]=rows
    decisions={}
    for kind,rows in out['methods'].items():
        decisions[kind]=dict(uniformity=all(q['holm_pvalue']>=.05 for r in rows.values() for q in r['quantities'].values()),
            coverage90=all(.85<=r['quantities'][q]['coverage']['0.9']['estimate']<=.95 for r in rows.values() for q in ('log_sigma0','log_w')),
            median_tv=all(r['median_tv']<=.1 for r in rows.values()))
    out['decisions']=decisions
    out['surviving_prior_mass']={str(int(t)):float(weights[bank.d['valid'][:,j]].sum()) for j,t in enumerate(TIMES)}
    out['controls_passed']=bool(decisions['solver']['uniformity'] and
        all(r['quantities']['solver_log_likelihood']['holm_pvalue']<.05 for r in out['methods']['prior_only'].values()))
    out['integration_passed']=bool(progress['converged'] and
        all(out['convergence'][k]['passed'] for k in ('solver','candidate')))
    out['candidate_passed']=bool(out['integration_passed'] and out['controls_passed'] and all(decisions['candidate'].values()))
    out['rules']=[r.name for r in rules]
    out['action']='retain frozen raw candidate; no production replacement or manuscript change'
    out['provenance']=dict(bank_sha256=digest(DATA/'bank.npz'),observations_sha256=digest(DATA/'observations.npz'),
        protocol_sha256=digest(ROOT/'docs/FORECAST_INTEGRATION_PROTOCOL.md'),
        source_sha256={p:digest(ROOT/p) for p in ('scripts/forecast_integrate.py','src/gravop/quadrature.py')})
    write_npz(OUT/'diagnostics.npz',**summaries)
    write_json(OUT/'validation.json',out)
    print('F4 decision',decisions,'integration',out['integration_passed'],'candidate',out['candidate_passed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','adaptive','evaluate'])
    p.add_argument('--workers',type=int,default=20);p.add_argument('--device',default='cuda:0')
    p.add_argument('--refine-next',action='store_true',help='advance to the predefined third tolerance after candidate convergence fails')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);globals()[args.stage](args)


if __name__=='__main__':main()
