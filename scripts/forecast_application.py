"""F2 full-D1, independent SIDM, N-body and solver-posterior transfer checks.

Evaluates experimental checkpoints without modifying the production MCMC code.
Posterior checks use identical observational-noise likelihoods to isolate the
forward approximation. They are fixed-mock grid checks, not population SBC.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gravop import solver, dataset as gds
from gravop.model import GravOpNet
from statecond import data
from statecond.forecast import ForecastPredictor
from forecast_validate import OUT, from_d1, write_json, digest, comparison


class Ensemble:
    def __init__(self,kind,device):
        self.kind,self.device=kind,device
        self.models=[]; self.norms=[]; self.hashes={}
        for s in range(3 if kind=='candidate' else 5):
            if kind=='candidate':
                path=OUT/'application'/f'sidm_full_s{s}'/'model.pt'
                obj=ForecastPredictor(path,device)
                self.models.append(obj.model); self.norms.append(obj)
            else:
                path=ROOT/'results/runs'/f'gravop_s{s}'/'model.pt'
                ck=torch.load(path,map_location=device,weights_only=False)
                a=ck['args']
                m=GravOpNet(width=a['width'],modes=a['modes'],n_layers=a['layers']).to(device)
                m.load_state_dict(ck['model']); self.models.append(m.eval()); self.norms.append(ck['norm'])
            self.hashes[str(path.relative_to(ROOT))]=digest(path)

    @torch.no_grad()
    def predict(self,prof,par,time):
        device=self.device
        prof=torch.as_tensor(prof,device=device,dtype=torch.float32)
        par=torch.as_tensor(par,device=device,dtype=torch.float32)
        t=torch.as_tensor(time,device=device,dtype=torch.float32).expand(len(prof))
        inputs=[]; log_ends=[]
        for m,n in zip(self.models,self.norms):
            if self.kind=='candidate':
                p=(prof-n.f_mu)/n.f_sd; c=(par-n.p_mu)/n.p_sd
                log_end=m.clock(p,c)*m.clock_sd+m.clock_mu
            else:
                mu=torch.tensor([n['rho_mu'],n['v_mu']],device=device)[None,:,None]
                sd=torch.tensor([n['rho_sd'],n['v_sd']],device=device)[None,:,None]
                p=(prof-mu)/sd
                c=torch.stack([(par[:,0]-n['sig_mu'])/n['sig_sd'],
                               (par[:,1]-n['w_mu'])/n['w_sd'],par[:,2]],1)
                c0=torch.cat([c,torch.zeros_like(t[:,None])],1)
                _,sc=m(p,c0); log_end=sc[:,0]*n['tc_sd']+n['tc_mu']
            inputs.append((p,c)); log_ends.append(log_end)
        ratio=t/10.**torch.stack(log_ends).mean(0)
        vals=[]
        for m,n,(p,c) in zip(self.models,self.norms,inputs):
            if self.kind=='candidate':
                y=m(p,c,t)*n.f_sd+n.f_mu
                y=torch.where((t==0)[:,None,None],prof,y)
            else:
                y,_=m(p,torch.cat([c,ratio.clamp(0,1)[:,None]],1))
                mu=torch.tensor([n['rho_mu'],n['v_mu']],device=device)[None,:,None]
                sd=torch.tensor([n['rho_sd'],n['v_sd']],device=device)[None,:,None]
                y=y*sd+mu
            vals.append(y)
        return torch.stack(vals).mean(0).cpu().numpy(),ratio.cpu().numpy()


def interpolate(rows,radii):
    grid=np.linspace(-2.5,2.,128)
    return np.stack([np.interp(np.log10(radii),grid,row) for row in rows])


def field_metrics(ensemble,d,idx):
    radii=np.logspace(np.log10(.02),np.log10(20.),20)
    ri=np.repeat(idx,48); ti=np.tile(np.arange(48),len(idx))
    times=10.**d['log_tsing'][:,None]*data.tau_grid()[None]
    vals={k:[] for k in ('density_relative','log_density_mse','observation_mse','past_predicted_horizon')}
    for start in range(0,len(ri),256):
        r,t=ri[start:start+256],ti[start:start+256]
        pred,q=ensemble.predict(d['u0'][r],d['par'][r],times[r,t])
        delta=pred[:,0]-d['u'][r,t,0]
        if not np.isfinite(delta).all(): raise ValueError('nonfinite field prediction')
        rel=np.abs(np.expm1(np.log(10.)*delta.astype('f8'))).mean(1)
        if not np.isfinite(rel).all(): raise ValueError('nonfinite relative density error')
        vals['density_relative'].extend(rel.tolist())
        vals['log_density_mse'].extend((delta.astype('f8')**2).mean(1).tolist())
        vals['observation_mse'].extend((interpolate(delta,radii).astype('f8')**2).mean(1).tolist())
        vals['past_predicted_horizon'].extend((q>1).astype(float).tolist())
    arrays={k:np.asarray(v).reshape(len(idx),48) for k,v in vals.items()}
    masks={'all':np.ones(len(idx),bool),'terminal':np.isin(d['status'][idx],[0,2]),
           'horizon_stop':d['status'][idx]==1}
    summary={}
    for label,mask in masks.items():
        if not mask.any(): continue
        summary[label]=dict(n=int(mask.sum()),density_mean_pct=float(100*arrays['density_relative'][mask].mean()),
              density_median_trajectory_pct=float(100*np.median(arrays['density_relative'][mask].mean(1))),
              log_density_rmse_dex=float(np.sqrt(arrays['log_density_mse'][mask].mean())),
              observation_rmse_dex=float(np.sqrt(arrays['observation_mse'][mask].mean())),
              predicted_horizon_rejection_fraction=float(arrays['past_predicted_horizon'][mask].mean()))
    return dict(summary=summary,per_trajectory={k:v.mean(1).tolist() for k,v in arrays.items()})


def initial_nfw():
    r,_,rho,p=solver.nfw_profile(n_shells=300)
    a,b=gds.regrid_snapshot(r,rho,p)
    return np.stack([a,b]).astype('f4')


def nbody(ensembles):
    from e11_emulator_vs_nbody import R_S,RHO_S,T0G,SIGMA_HAT,S_RESCALE,FIT_RANGE
    d=np.load(ROOT/'results/e2/nbody_profiles.npz')
    # Prespecified epochs in the existing comparison, with its fixed calibration.
    keys=['snapshot_010','snapshot_020','snapshot_030','snapshot_040','snapshot_045']
    prof=initial_nfw()[None]
    par=np.array([[np.log10(SIGMA_HAT),0.,0.]],dtype='f4')
    out={}
    for name,ens in ensembles.items():
        residuals=[]; rows=[]
        for key in keys:
            t=float(d[key+'_t'])/T0G*S_RESCALE
            pred,ratio=ens.predict(prof,par,np.array([t]))
            r=d[key+'_r']/R_S; rho=d[key+'_rho']/(RHO_S*1e-10)
            mask=(r>=FIT_RANGE[0])&(r<=FIT_RANGE[1])&(rho>0)
            e=interpolate(pred[:,0],r[mask])[0]-np.log10(rho[mask])
            residuals.extend(e.tolist())
            rows.append(dict(snapshot=key,rmse_dex=float(np.sqrt(np.mean(e**2))),
                             predicted_time_ratio=float(ratio[0])))
        out[name]=dict(rmse_dex=float(np.sqrt(np.mean(np.square(residuals)))),snapshots=rows)
    out['note']='Existing calibration epochs and fixed s=0.945; this is not an independent calibration validation.'
    return out


def posterior(ensembles):
    mock=json.loads((ROOT/'results/e9/mock.json').read_text())
    profile=initial_nfw()
    out={}
    for file in ('grid_logpost.npz','grid_zoom.npz'):
        source=ROOT/'results/e9'/file
        grid=np.load(source); x,y=grid['lsig'],grid['lw']
        xx,yy=np.meshgrid(x,y,indexing='ij'); params=np.stack([xx.ravel(),yy.ravel(),np.ones(xx.size)],1)
        truth=grid['logpost'].ravel(); w=np.exp(truth-logsumexp(truth))
        coords=params[:,:2]
        mean=(w[:,None]*coords).sum(0)
        sd=np.sqrt((w[:,None]*(coords-mean)**2).sum(0))
        rec={'solver_mean':mean.tolist(),'solver_sd':sd.tolist(),'source_sha256':digest(source)}
        for name,ens in ensembles.items():
            lp=[]
            for off in range(0,len(params),256):
                par=params[off:off+256]
                prof=np.broadcast_to(profile,(len(par),*profile.shape)).copy()
                pred,ratio=ens.predict(prof,par,np.array(mock['t_obs']))
                rho=interpolate(pred[:,0],mock['r_obs'])
                logp=-.5*np.sum((rho-np.asarray(mock['lrho_obs']))**2/float(mock['noise_dex'])**2,1)
                logp[ratio>1]=-np.inf; lp.extend(logp.tolist())
            lp=np.asarray(lp)
            if not np.isfinite(lp).any(): raise ValueError('posterior has no valid support')
            q=np.exp(lp-logsumexp(lp)); m=(q[:,None]*coords).sum(0)
            rec[name]=dict(total_variation=float(.5*np.abs(q-w).sum()),
                           mean=m.tolist(),mean_shift_in_solver_sd=((m-mean)/sd).tolist(),
                           solver_mass_rejected=float(w[~np.isfinite(lp)].sum()))
        out[file]=rec
    out['note']='Same fixed mock and observational-noise-only likelihood as solver grids; no tuned discrepancy or ensemble variance. Not SBC or MCMC convergence validation.'
    return out


def screening_decision(out):
    checks={}
    for stratum in ('terminal','horizon_stop'):
        for metric in ('density_mean_pct','observation_rmse_dex'):
            new=out['fresh']['candidate']['summary'][stratum][metric]
            old=out['fresh']['production']['summary'][stratum][metric]
            checks[f'fresh_{stratum}_{metric}']=dict(candidate=new,production=old,
                                                    passed=bool(new<=1.05*old))
    new=out['nbody']['candidate']['rmse_dex']; old=out['nbody']['production']['rmse_dex']
    checks['nbody']=dict(candidate=new,production=old,passed=bool(new<=old+.002))
    for grid in ('grid_logpost.npz','grid_zoom.npz'):
        new=out['posterior'][grid]['candidate']['total_variation']
        old=out['posterior'][grid]['production']['total_variation']
        checks[grid]=dict(candidate=new,production=old,passed=bool(new<=old+.02))
    return dict(checks=checks,passed=all(c['passed'] for c in checks.values()),
                action='retain production; these screens do not establish population-level posterior calibration')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); torch.set_num_threads(4)
    ensembles={k:Ensemble(k,a.device) for k in ('candidate','production')}
    old=from_d1(ROOT/'data/d1/train.h5'); _,_,te=data.split(len(old['u']))
    fresh=from_d1(ROOT/'data/d1/forecast_fresh.h5')
    out={'checkpoint_hashes':{k:v.hashes for k,v in ensembles.items()},
         'fresh_sha256':digest(ROOT/'data/d1/forecast_fresh.h5')}
    for label,d,idx in [('full_D1_test',old,te),('fresh',fresh,np.arange(len(fresh['u'])))]:
        out[label]={k:field_metrics(v,d,idx) for k,v in ensembles.items()}
        for metric in ('density_relative','observation_mse'):
            out[label][metric+'_comparison']=comparison(
                [out[label]['production']['per_trajectory'][metric]],
                [out[label]['candidate']['per_trajectory'][metric]])
        print(label,{k:v['summary'] for k,v in out[label].items() if 'summary' in v},flush=True)
        write_json(OUT/'application/validation.json',out)
    out['nbody']=nbody(ensembles); print('nbody',out['nbody'],flush=True)
    out['posterior']=posterior(ensembles); print('posterior',out['posterior'],flush=True)
    out['screening']=screening_decision(out)
    write_json(OUT/'application/validation.json',out)


if __name__=='__main__': main()
