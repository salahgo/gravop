"""F3: independently calibrated discrepancy and solver-backed population SBC.

See docs/FORECAST_CALIBRATION_PROTOCOL.md for frozen choices and limits.
"""
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
from gravop import solver, dataset as ds
from gravop.calibration import (GaussianDiscrepancy, posterior_weights,
                                randomized_pit, select_discrepancy, pit_summary, holm)
from forecast_validate import write_json, digest
from forecast_application import Ensemble, initial_nfw, interpolate
from e9_inference import PRIOR, NOISE_DEX, R_OBS, SOLVER_KW

OUT=ROOT/'results/forecast_calibration'
DATA=ROOT/'data/d1/forecast_calibration'
TIMES=np.array([100.,400.,900.])
LO=np.array([PRIOR['lsig'][0],PRIOR['lw'][0]])
HI=np.array([PRIOR['lsig'][1],PRIOR['lw'][1]])


def write_npz(path, **arrays):
    """Avoid treating interrupted writes as completed, reusable solver data."""
    path=Path(path); temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def solver_row(task):
    index,theta,refine=task
    kw=dict(SOLVER_KW,t_end=900.,max_snaps=3000)
    if refine: kw.update(eps_t=1e-4,snap_dlnrho=.01,snap_dlnt=.025)
    try:
        result=solver.run_halo(sigma_m=10.**theta[0],w=10.**theta[1],**kw)
        pred=np.full((3,len(R_OBS)),np.nan); valid=np.zeros(3,bool)
        status=result['status']; ts=result['t']
        if status not in (0,1,2):
            return index,pred,valid,status,result['t_final'],f'numerical solver status {status}'
        for j,t in enumerate(TIMES):
            if t>ts[-1]+1e-9: continue
            i=int(np.clip(np.searchsorted(ts,t)-1,0,len(ts)-2))
            alpha=np.clip((t-ts[i])/max(ts[i+1]-ts[i],1e-300),0,1)
            vals=[]
            for k in (i,i+1):
                r=ds.shell_midpoints(result['r'][k])
                vals.append(np.interp(np.log10(R_OBS),np.log10(r),np.log10(result['rho'][k])))
            pred[j]=(1-alpha)*vals[0]+alpha*vals[1]
            valid[j]=np.isfinite(pred[j]).all()
            if not valid[j]: raise ValueError('nonfinite interpolated solver profile')
        return index,pred,valid,status,result['t_final'],None
    except Exception as exc:
        return index,np.full((3,len(R_OBS)),np.nan),np.zeros(3,bool),-1,np.nan,repr(exc)


def generate_group(name,params,workers,refine=False):
    path=DATA/f'{name}.npz'
    if path.exists():
        with np.load(path) as old: np.testing.assert_array_equal(old['params'],params)
        recorded=json.loads((OUT/f'{name}_generation.json').read_text())
        assert digest(path)==recorded['sha256'],'cached solver data changed'
        print('reuse',name,flush=True); return
    values=np.full((len(params),3,len(R_OBS)),np.nan); valid=np.zeros((len(params),3),bool)
    status=np.full(len(params),-1); end=np.full(len(params),np.nan); failures=[]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count,row in enumerate(pool.map(solver_row,[(i,p,refine) for i,p in enumerate(params)],chunksize=4),1):
            i,y,ok,st,tc,error=row
            values[i],valid[i],status[i],end[i]=y,ok,st,tc
            if error: failures.append(dict(index=i,error=error))
            if count%200==0: print(name,count,'/',len(params),'seconds',round(time.perf_counter()-start,1),flush=True)
    write_npz(path,params=params,profiles=values,valid=valid,status=status,t_end=end,times=TIMES)
    write_json(OUT/f'{name}_generation.json',dict(n=len(params),valid_per_time=valid.sum(0).tolist(),
        statuses={str(s):int((status==s).sum()) for s in np.unique(status)},failures=failures,
        seconds=time.perf_counter()-start,sha256=digest(path)))
    print('done',name,'valid',valid.sum(0),'failures',len(failures),flush=True)


def generate(args):
    DATA.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    x,y=np.meshgrid(np.linspace(LO[0],HI[0],65),np.linspace(LO[1],HI[1],65),indexing='ij')
    grid=np.stack([x.ravel(),y.ravel()],1)
    groups={'calibration':np.random.default_rng(930000000).uniform(LO,HI,(256,2)),
            'confirmation':np.random.default_rng(940000000).uniform(LO,HI,(128,2)), 'grid':grid}
    settings=dict(times=TIMES.tolist(),prior=[LO.tolist(),HI.tolist()],noise_dex=NOISE_DEX,
                  grid_n=65,n_sbc_per_time=512,solver=dict(SOLVER_KW,t_end=900.,max_snaps=3000),
                  protocol_sha256=digest(ROOT/'docs/FORECAST_CALIBRATION_PROTOCOL.md'))
    config=OUT/'config.json'
    if config.exists(): assert json.loads(config.read_text())==settings,'protocol/config changed'
    else: write_json(config,settings)
    for name,params in groups.items(): generate_group(name,params,args.workers)
    indices=np.random.default_rng(960000000).choice(len(grid),12,replace=False)
    generate_group('refinement',grid[indices],args.workers,True)
    write_json(OUT/'refinement_indices.json',indices.tolist())


def predict(args):
    ensembles={k:Ensemble(k,args.device) for k in ('candidate','production')}
    initial=initial_nfw()
    for name in args.groups:
        path=DATA/f'{name}_predictions.npz'
        if path.exists():
            rec=json.loads((OUT/f'{name}_prediction_provenance.json').read_text())
            assert rec['data_sha256']==digest(DATA/f'{name}.npz'),'prediction data changed'
            assert rec['predictions_sha256']==digest(path),'cached predictions changed'
            assert rec['checkpoint_hashes']=={k:e.hashes for k,e in ensembles.items()},'checkpoints changed'
            print('reuse predictions',name,flush=True);continue
        d=np.load(DATA/f'{name}.npz'); par=np.column_stack([d['params'],np.ones(len(d['params']))])
        values={}
        for kind,ens in ensembles.items():
            means=[];ratios=[]
            for t in TIMES:
                ys=[];qs=[]
                for off in range(0,len(par),256):
                    p=par[off:off+256];prof=np.repeat(initial[None],len(p),0)
                    pred,q=ens.predict(prof,p,t)
                    if not np.isfinite(pred).all() or not np.isfinite(q).all() or (q<0).any():
                        raise ValueError('invalid ensemble prediction or clock ratio')
                    ys.append(interpolate(pred[:,0],R_OBS));qs.append(q)
                means.append(np.concatenate(ys));ratios.append(np.concatenate(qs))
            values[kind+'_mean']=np.stack(means,1); values[kind+'_ratio']=np.stack(ratios,1)
        write_npz(path,**values)
        write_json(OUT/f'{name}_prediction_provenance.json',dict(data_sha256=digest(DATA/f'{name}.npz'),
            predictions_sha256=digest(path),checkpoint_hashes={k:e.hashes for k,e in ensembles.items()},
            source_sha256={p:digest(ROOT/p) for p in ('scripts/forecast_application.py','src/statecond/forecast.py')}))
        print('predicted',name,flush=True)


def calibrate(args):
    path=OUT/'discrepancy.json'
    if path.exists(): raise SystemExit('discrepancy already frozen')
    d=np.load(DATA/'calibration.npz'); pred=np.load(DATA/'calibration_predictions.npz')
    out={}; rng=np.random.default_rng(930100000)
    for kind in ('candidate','production'):
        out[kind]={}
        for j,t in enumerate(TIMES):
            delta=d['profiles'][:,j]-pred[kind+'_mean'][:,j]
            fit=np.flatnonzero(d['valid'][:192,j]); select=np.flatnonzero(d['valid'][192:,j])+192
            model,meta=select_discrepancy(delta[fit],delta[select],NOISE_DEX,rng)
            out[kind][str(int(t))]=dict(model=model.to_dict(),**meta,
                fit_indices=fit.tolist(),selection_indices=select.tolist(),
                fit_residual_rmse_dex=float(np.sqrt(np.square(delta[fit]).mean())))
            print('correction',kind,t,meta,flush=True)
    out['provenance']=dict(data_sha256=digest(DATA/'calibration.npz'),predictions_sha256=digest(DATA/'calibration_predictions.npz'))
    write_json(path,out)


def grid_prior(n):
    one=np.ones(n);one[[0,-1]]=.5
    p=np.outer(one,one).ravel();return p/p.sum()


def posteriors(obs,means,valid,ratios,prior,models):
    logprior=np.log(prior)
    truth=np.full(len(prior),-np.inf)
    truth[valid]=-.5*np.square((means['solver'][valid]-obs)/NOISE_DEX).sum(1)
    out={'solver':posterior_weights(truth+logprior),
         'prior_only':posterior_weights(np.where(valid,logprior,-np.inf))}
    for kind in ('candidate','production'):
        delta=obs-means[kind]
        for arm in ('raw','corrected'):
            lp=models[kind][arm].logpdf(delta)+logprior
            lp[ratios[kind]>1]=-np.inf
            out[kind+'_'+arm]=posterior_weights(lp)
    return out,truth


def marginal_summary(theta,q):
    mean=q@theta; sd=np.sqrt(q@np.square(theta-mean))
    intervals=[]
    for j in range(2):
        axis,inverse=np.unique(theta[:,j],return_inverse=True)
        weights=np.bincount(inverse,weights=q,minlength=len(axis))
        cdf=np.cumsum(weights)
        intervals.append(axis[np.minimum(np.searchsorted(cdf,[.05,.95]),len(axis)-1)].tolist())
    return mean,sd,np.array(intervals)


def paired_median_ci(a,b):
    a,b=np.asarray(a),np.asarray(b);rng=np.random.default_rng(970000000)
    ix=rng.integers(len(a),size=(2000,len(a)))
    vals=np.median(a[ix],1)-np.median(b[ix],1)
    return np.quantile(vals,[.025,.975]).tolist()


def evaluate(args):
    # Require calibration to be frozen before drawing any final observations.
    cal=json.loads((OUT/'discrepancy.json').read_text())
    for name in ('grid','calibration','confirmation','refinement'):
        rec=json.loads((OUT/f'{name}_generation.json').read_text())
        assert rec['sha256']==digest(DATA/f'{name}.npz'),'solver data changed'
        if name!='refinement':
            rec=json.loads((OUT/f'{name}_prediction_provenance.json').read_text())
            assert rec['data_sha256']==digest(DATA/f'{name}.npz'),'prediction source changed'
            assert rec['predictions_sha256']==digest(DATA/f'{name}_predictions.npz'),'predictions changed'
    assert cal['provenance']['data_sha256']==digest(DATA/'calibration.npz')
    assert cal['provenance']['predictions_sha256']==digest(DATA/'calibration_predictions.npz')
    from forecast_validate import from_d1
    old=from_d1(ROOT/'data/d1/train.h5')['par'];old=old[old[:,2]==1,:2]
    identities={k:{p.astype('f4').tobytes() for p in np.load(DATA/f'{k}.npz')['params']}
                for k in ('calibration','confirmation')}
    original={p.tobytes() for p in old}
    audit=dict(calibration_confirmation_parameter_overlap=len(identities['calibration']&identities['confirmation']),
        calibration_old_D1_parameter_overlap=len(identities['calibration']&original),
        confirmation_old_D1_parameter_overlap=len(identities['confirmation']&original))
    assert not any(audit.values()),audit
    write_json(OUT/'split_audit.json',audit)
    grid=np.load(DATA/'grid.npz'); pred=np.load(DATA/'grid_predictions.npz')
    cont=np.load(DATA/'confirmation.npz'); cont_pred=np.load(DATA/'confirmation_predictions.npz')
    theta=grid['params'];prior=grid_prior(65)
    coarse=np.arange(65*65).reshape(65,65)[::2,::2].ravel();cp=grid_prior(33)
    arms=('solver','candidate_raw','candidate_corrected','production_raw','production_corrected','prior_only')
    out={'discrepancy_sha256':digest(OUT/'discrepancy.json'),'times':{},'continuous':{},'numerics':{}}
    arrays={}; noise_rng=np.random.default_rng(940100000)
    raw=GaussianDiscrepancy(np.zeros(20),np.zeros((20,20)),NOISE_DEX)
    for j,t in enumerate(TIMES):
        tag=str(int(t)); valid=grid['valid'][:,j]
        means={'solver':grid['profiles'][:,j],**{k:pred[k+'_mean'][:,j] for k in ('candidate','production')}}
        ratios={k:pred[k+'_ratio'][:,j] for k in ('candidate','production')}
        models={k:dict(raw=raw,corrected=GaussianDiscrepancy(**cal[k][tag]['model'])) for k in ('candidate','production')}
        rng=np.random.default_rng(950000000+j)
        generating_prior=prior*valid; generating_prior/=generating_prior.sum()
        truths=rng.choice(len(theta),512,p=generating_prior)
        observations=means['solver'][truths]+rng.normal(0,NOISE_DEX,(512,20))
        randomizers=rng.random((512,3))
        pits={k:[] for k in arms};tvs={k:[] for k in arms}; widths={k:[] for k in arms}
        rejected={k:[] for k in ('candidate','production')};falsemass={k:[] for k in arms}
        for i,(truth,obs) in enumerate(zip(truths,observations)):
            post,likelihood=posteriors(obs,means,valid,ratios,prior,models)
            quantities=[theta[:,0],theta[:,1],likelihood]
            for kind,q in post.items():
                pits[kind].append([randomized_pit(v,q,v[truth],randomizers[i,k]) for k,v in enumerate(quantities)])
                tvs[kind].append(float(.5*np.abs(q-post['solver']).sum()))
                _,_,interval=marginal_summary(theta,q);widths[kind].append((interval[:,1]-interval[:,0]).tolist())
                falsemass[kind].append(float(q[~valid].sum()))
            for kind in rejected: rejected[kind].append(float(post['solver'][ratios[kind]>1].sum()))
        rec={'n':512,'solver_surviving_prior_mass':float((prior*valid).sum()),'methods':{}}
        for kind in arms:
            v=np.array(pits[kind]);arrays[f'{tag}_{kind}_pit']=v
            arrays[f'{tag}_{kind}_tv']=np.array(tvs[kind]);arrays[f'{tag}_{kind}_width90']=np.array(widths[kind])
            rec['methods'][kind]=dict(test_quantities={q:pit_summary(v[:,k]) for k,q in enumerate(('log_sigma0','log_w','solver_log_likelihood'))},
                median_tv=float(np.median(tvs[kind])),p90_tv=float(np.quantile(tvs[kind],.9)),
                median_width90=np.median(widths[kind],axis=0).tolist(),
                mean_false_support_mass=float(np.mean(falsemass[kind])))
        rec['candidate_corrected_minus_raw_median_tv_ci95']=paired_median_ci(tvs['candidate_corrected'],tvs['candidate_raw'])
        rec['support']={k:dict(prior_false_rejection_mass=float(generating_prior[ratios[k]>1].sum()),
                                mean_rejected_solver_posterior_mass=float(np.mean(rejected[k]))) for k in rejected}
        arrays[f'{tag}_truth_index']=truths; arrays[f'{tag}_observations']=observations
        out['times'][tag]=rec
        print('SBC',tag,{k:round(v['median_tv'],4) for k,v in rec['methods'].items()},flush=True)
        # Fresh continuous truths: separate diagnostic, not exact discrete SBC.
        ids=np.flatnonzero(cont['valid'][:,j]);rows=[]
        for index in ids:
            obs=cont['profiles'][index,j]+noise_rng.normal(0,NOISE_DEX,20)
            post,_=posteriors(obs,means,valid,ratios,prior,models)
            cm={k:v[coarse] for k,v in means.items()};cr={k:v[coarse] for k,v in ratios.items()}
            small,_=posteriors(obs,cm,valid[coarse],cr,cp,models)
            entry={'index':int(index),'methods':{}}
            for kind,q in post.items():
                mean,sd,interval=marginal_summary(theta,q)
                entry['methods'][kind]=dict(covered90=((cont['params'][index]>=interval[:,0])&(cont['params'][index]<=interval[:,1])).tolist())
                if kind=='solver':
                    cmean,_,cinterval=marginal_summary(theta[coarse],small[kind])
                    entry['grid_mean_shift_sd']=(np.abs(cmean-mean)/np.maximum(sd,1e-12)).tolist()
                    width=interval[:,1]-interval[:,0];cwidth=cinterval[:,1]-cinterval[:,0]
                    entry['grid_width_relative_change']=(np.abs(cwidth-width)/np.maximum(width,1e-12)).tolist()
            rows.append(entry)
        mean_shifts=np.array([r['grid_mean_shift_sd'] for r in rows]);width_changes=np.array([r['grid_width_relative_change'] for r in rows])
        out['continuous'][tag]=dict(n=len(ids),coverage90={k:np.mean([r['methods'][k]['covered90'] for r in rows],axis=0).tolist() for k in arms},
            grid_mean_shift_sd_p95=np.quantile(mean_shifts,.95,axis=0).tolist(),
            grid_width_relative_change_median=np.median(width_changes,axis=0).tolist(),
            grid_adequate=bool((np.quantile(mean_shifts,.95,axis=0)<=.1).all() and (np.median(width_changes,axis=0)<=.1).all()),rows=rows)
        write_json(OUT/'validation.json',out)
    for kind in arms:
        records=[r['methods'][kind]['test_quantities'][q] for r in out['times'].values()
                 for q in ('log_sigma0','log_w','solver_log_likelihood')]
        for rec,p in zip(records,holm([r['ks_pvalue'] for r in records])): rec['holm_pvalue']=float(p)
    refined=np.load(DATA/'refinement.npz'); indices=np.array(json.loads((OUT/'refinement_indices.json').read_text()))
    same=grid['valid'][indices]&refined['valid']; delta=np.abs(grid['profiles'][indices]-refined['profiles'])
    out['numerics']=dict(max_abs_profile_difference_dex=float(delta[same].max()),
        rmse_dex=float(np.sqrt(np.square(delta[same]).mean())),support_disagreement=int((grid['valid'][indices]!=refined['valid']).sum()),
        passed=bool(delta[same].max()<=.00434 and np.array_equal(grid['valid'][indices],refined['valid'])))
    out['generation_failures']={name:len(json.loads((OUT/f'{name}_generation.json').read_text())['failures'])
                                 for name in ('grid','calibration','confirmation','refinement')}
    decisions={}
    for kind in arms:
        rs=[r['methods'][kind] for r in out['times'].values()]
        decisions[kind]=dict(uniformity_passed=all(v['holm_pvalue']>=.05 for r in rs for v in r['test_quantities'].values()),
            coverage90_passed=all(.85<=r['test_quantities'][q]['coverage']['0.9']['estimate']<=.95 for r in rs for q in ('log_sigma0','log_w')),
            median_tv_passed=all(r['median_tv']<=.10 for r in rs))
    out['decisions']=decisions
    out['controls_passed']=bool(decisions['solver']['uniformity_passed'] and
        all(r['methods']['prior_only']['test_quantities']['solver_log_likelihood']['holm_pvalue']<.05 for r in out['times'].values()))
    out['continuous_reference_passed']=bool(out['numerics']['passed'] and not any(out['generation_failures'].values())
        and all(v['grid_adequate'] for v in out['continuous'].values()))
    out['candidate_can_advance']={k:bool(all(decisions[k].values()) and out['controls_passed'] and out['continuous_reference_passed'])
        for k in ('candidate_raw','candidate_corrected')}
    if any(r['methods']['candidate_corrected']['median_tv']>r['methods']['candidate_raw']['median_tv']+.02 for r in out['times'].values()):
        out['candidate_can_advance']['candidate_corrected']=False
    out['action']='retain production; report discrete SBC separately from continuous-prior and solver-accuracy checks'
    write_npz(OUT/'diagnostics.npz',**arrays)
    out['provenance']=dict(data_sha256={p.name:digest(p) for p in DATA.glob('*.npz')},
        source_sha256={p:digest(ROOT/p) for p in ('scripts/forecast_calibrate.py','src/gravop/calibration.py')},
        protocol_sha256=digest(ROOT/'docs/FORECAST_CALIBRATION_PROTOCOL.md'))
    write_json(OUT/'validation.json',out)
    print('decisions',decisions,'reference',out['continuous_reference_passed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','predict','calibrate','evaluate'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--groups',nargs='+',choices=['calibration','confirmation','grid'],
                   default=['calibration','confirmation','grid'],help='prediction stage data groups')
    args=p.parse_args();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](args)


if __name__=='__main__':main()
