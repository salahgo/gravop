"""F2 controls, independent data, and application validation.

Protocol: docs/FORECAST_VALIDATION_PROTOCOL.md. No manuscript or production
checkpoint is mutated. Fresh test data never determines normalization or fitting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from statecond import data
from statecond.model import BACKBONES
from statecond.forecast import ClockConditionedOperator
from sc_forecast import prepare, errors, summary, digest
from sc_experiment import Norm, tensors, load_checkpoint, save_checkpoint

OUT = ROOT/'results/forecast_validation'
REFERENCE = {'sidm': 'cno', 'rd': 'cno', 'ks': 'cno',
             'film': 'fno', 'burgers': 'fno'}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def from_d1(path, terminal_only=False):
    """Preserve row identities and statuses while exposing common field schema."""
    with h5py.File(path) as f:
        d = {k: f[k][:] for k in ('lrho','lv','log_sigma0','log_w','model_id',
                                  'log_tc','status','ic_type','seed')}
    ok = np.isin(d['status'], [0,2] if terminal_only else [0,1,2])
    ids = np.flatnonzero(ok)
    d = {k:v[ok] for k,v in d.items()}
    u = np.stack([d['lrho'],d['lv']],2).astype(np.float32)
    return dict(u=u,u0=u[:,0],par=np.stack([d['log_sigma0'],d['log_w'],d['model_id']],1).astype('f4'),
                log_tsing=d['log_tc'].astype('f4'),status=d['status'].astype('i1'),
                phi=np.zeros(u.shape[:2],dtype='f4'),flag=np.zeros(u.shape[:2],dtype='f4'),
                x=np.linspace(-2.5,2.,128).astype('f4'),system='sidm',
                param_names=['log_sigma0','log_w','model_id'],source_index=ids,
                seed=d['seed'].astype('i8'),ic_type=d['ic_type'].astype('i1'))


class Reference(torch.nn.Module):
    def __init__(self, net, clock, n, time_scale, direct=False):
        super().__init__()
        self.net, self.clock, self.n = net, clock, n
        self.time_scale, self.direct = time_scale, direct
        self.clock.requires_grad_(False)

    def forward(self, p, c, t):
        if self.direct:
            q = torch.log1p(t/self.time_scale)
        else:
            q = (t/10.**(self.clock(p,c)*self.n.t_sd+self.n.t_mu)).clamp(0,1)
        return self.net(p,torch.cat([c,q[:,None]],1))


def control(args):
    out = OUT/'controls'/f'{args.system}_s{args.seed}'
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    out.mkdir(parents=True)
    d,n,u,u0,par,times,tr,va,te = prepare(args.system,args.device)
    src = ROOT/'results/statecond/ckpt'/f'{args.system}_fno_time_conv_seed{args.seed}.pt'
    head,meta = load_checkpoint(src,args.device)
    for k in ('f_mu','f_sd','p_mu','p_sd','t_mu','t_sd'):
        np.testing.assert_allclose(meta['norm'][k],getattr(n,k),rtol=1e-6,atol=1e-7)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    backbone = REFERENCE[args.system]
    direct = args.system=='burgers'
    net = BACKBONES[backbone](n_fields=u.shape[2],n_par=par.shape[1],
                             cond_extra=1,n_x=u.shape[-1]).to(args.device)
    scale = float(10.**np.median(d['log_tsing'][tr]))
    wrapped = Reference(net,head,n,scale,direct).to(args.device)
    opt = torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    tau = torch.tensor(data.tau_grid(),dtype=torch.float32,device=args.device)
    ri = np.repeat(tr,48); ti = np.tile(np.arange(48),len(tr))
    budget = json.loads((ROOT/'results/forecast'/f'{args.system}_aligned_s{args.seed}'/'training.json').read_text())['seconds']
    cfg = dict(system=args.system,seed=args.seed,backbone=backbone,direct=direct,
               time_scale=scale,batch=512,epochs=200,updates=200*int(np.ceil(len(ri)/512)),
               clock_sha256=digest(src),data_sha256=digest(ROOT/'data/sc'/f'{args.system}.h5'),
               source_sha256=digest(__file__),time_budget_seconds=budget,
               train_index=tr.tolist(),validation_index=va.tolist(),test_index=te.tolist(),
               training_coordinate='physical log1p time' if direct else 'oracle tau',
               selection='deployed validation nRMSE')
    write_json(out/'config.json',cfg)
    hist=[]; best=float('inf'); best_time=float('inf'); start=time.perf_counter()
    for ep in range(200):
        net.train()
        order=rng.permutation(len(ri))
        for off in range(0,len(order),512):
            ix=order[off:off+512]; r,t=ri[ix],ti[ix]
            q=torch.log1p(times[r,t]/scale) if direct else tau[t]
            pred=net(u0[r],torch.cat([par[r],q[:,None]],1))
            loss=(pred-u[r,t]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        sch.step()
        if ep==0 or (ep+1)%20==0:
            val=summary(errors(wrapped,u,u0,par,times,va))
            elapsed=time.perf_counter()-start
            payload=dict(state=net.state_dict(),clock=head.state_dict(),norm=n.__dict__,
                         config=cfg,epoch=ep+1,validation=val)
            if val['nrmse']<best:
                best=val['nrmse']; torch.save(payload,out/'model.pt')
            if elapsed<=budget and val['nrmse']<best_time:
                best_time=val['nrmse']; torch.save(payload,out/'time_budget.pt')
            hist.append(dict(epoch=ep+1,validation=val['nrmse'],seconds=elapsed))
            write_json(out/'training.json',dict(history=hist,best_validation=best,
                       seconds=elapsed,parameters=sum(p.numel() for p in net.parameters())))
            print(args.system,args.seed,hist[-1],flush=True)


def generate(args):
    from sc_gen import generate as generate_pde
    from gen_dataset import one_run, SOLVER_KW
    folder=ROOT/'data/sc/forecast_fresh'
    if (folder/'manifest.json').exists():
        raise SystemExit('fresh manifest already exists; refusing to regenerate confirmation data')
    folder.mkdir(parents=True,exist_ok=True)
    manifest={}
    for i,name in enumerate(('rd','ks','film','burgers')):
        base=910000000+100000*i
        d,info=generate_pde(name,300,base,args.workers)
        path=folder/f'{name}.h5'; data.save(path,d)
        manifest[name]=dict(**info,seed_base=base,sha256=digest(path))
        print(name,manifest[name],flush=True)
    tasks=[(i,920000000+i,'train') for i in range(400)]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(one_run,tasks,chunksize=1))
    valid=[(i,s) for i,s,err in rows if s is not None]
    failures=[dict(index=i,error=err) for i,s,err in rows if s is None]
    path=ROOT/'data/d1/forecast_fresh.h5'
    if path.exists(): raise SystemExit('fresh SIDM file already exists')
    with h5py.File(path,'w') as f:
        for k in valid[0][1]: f[k]=np.stack([s[k] for _,s in valid])
        f['draw_index']=np.array([i for i,_ in valid])
        f.attrs['solver_kw']=json.dumps(SOLVER_KW)
        f.attrs['base_seed']=920000000
    full=from_d1(path)
    scpath=folder/'sidm.h5'; data.save(scpath,from_d1(path,True))
    fullpath=folder/'sidm_full.h5'; data.save(fullpath,full)
    statuses,counts=np.unique(full['status'],return_counts=True)
    manifest['sidm']=dict(requested=400,kept=len(full['u']),failures=failures,
                          statuses=dict(zip(map(str,statuses),map(int,counts))),
                          seed_base=920000000,seconds=time.perf_counter()-start,
                          sha256=digest(scpath),full_sha256=digest(fullpath),
                          d1_sha256=digest(path),solver=SOLVER_KW)
    # Verify exact parameter/initial-field identities do not duplicate old data.
    for name in ('rd','ks','film','burgers','sidm'):
        old=data.load(ROOT/'data/sc'/f'{name}.h5'); fresh=data.load(folder/f'{name}.h5')
        oldrows={np.concatenate([p,x.ravel()]).tobytes() for p,x in zip(old['par'],old['u0'])}
        overlap=sum(np.concatenate([p,x.ravel()]).tobytes() in oldrows
                    for p,x in zip(fresh['par'],fresh['u0']))
        if overlap: raise ValueError(f'fresh data overlaps old data: {name}')
        manifest[name]['exact_input_overlap']=overlap
    write_json(folder/'manifest.json',manifest)
    write_json(OUT/'fresh_manifest.json',manifest)
    print('fresh generation complete',flush=True)


def clock(args):
    from sc_timehead import train_head
    path=ROOT/'data/sc/sidm_full.h5'
    if not path.exists(): data.save(path,from_d1(ROOT/'data/d1/train.h5'))
    d,n,u,u0,par,times,tr,va,te=prepare('sidm_full',args.device)
    T=tensors(d,n,args.device)
    out=OUT/'application'/f'clock_seed{args.seed}.pt'
    if out.exists(): raise SystemExit(f'refusing to overwrite {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    head=train_head(T,tr,args.seed,args.device,20000,u.shape[2],par.shape[1])
    save_checkpoint(out,head,'time','fno',n,args.seed,20000,(u.shape[2],par.shape[1],u.shape[-1]))
    with torch.no_grad():
        val=100*np.abs(10.**(head(u0[va],par[va]).cpu().numpy()*n.t_sd+n.t_mu-d['log_tsing'][va])-1)
    write_json(out.with_suffix('.json'),dict(seconds=time.perf_counter()-start,
               validation_median_pct=float(np.median(val)),validation_mean_pct=float(val.mean()),
               train_index=tr.tolist(),data_sha256=digest(path),epochs=20000,
               source_sha256=digest(__file__),checkpoint_sha256=digest(out)))
    print('clock',args.seed,'validation median %',np.median(val),flush=True)


def normalized(d,n,device):
    u,u0,p,*_=tensors(d,n,device)
    times=torch.tensor(10.**d['log_tsing'][:,None]*data.tau_grid()[None],
                       device=device,dtype=torch.float32)
    return u,u0,p,times


def load_control(system,seed,device,filename='model.pt'):
    ck=torch.load(OUT/'controls'/f'{system}_s{seed}'/filename,map_location=device,weights_only=False)
    cfg=ck['config']; n=Norm.__new__(Norm); n.__dict__.update(ck['norm'])
    old=data.load(ROOT/'data/sc'/f'{system}.h5')
    head,_=load_checkpoint(ROOT/'results/statecond/ckpt'/f'{system}_fno_time_conv_seed{seed}.pt',device)
    head.load_state_dict(ck['clock'])
    net=BACKBONES[cfg['backbone']](n_fields=old['u'].shape[2],n_par=old['par'].shape[1],
                                  cond_extra=1,n_x=old['u'].shape[-1]).to(device)
    net.load_state_dict(ck['state'])
    return Reference(net,head,n,cfg['time_scale'],cfg['direct']).eval(),n,ck


def comparison(a,b):
    """Reference/new, paired over independent trajectories; retain seed variation."""
    a,b=np.asarray(a),np.asarray(b)
    rng=np.random.default_rng(20260910)
    ix=rng.integers(a.shape[1],size=(5000,a.shape[1]))
    ratio=a.mean(0)[ix].mean(1)/b.mean(0)[ix].mean(1)
    seeds=rng.integers(a.shape[0],size=(5000,a.shape[0]))
    nested=np.array([a[s][:,j].mean()/b[s][:,j].mean() for s,j in zip(seeds,ix)])
    return dict(reference=float(a.mean()),candidate=float(b.mean()),
                improvement=float(a.mean()/b.mean()),ci95=np.quantile(ratio,[.025,.975]).tolist(),
                seed_and_trajectory_ci95=np.quantile(nested,[.025,.975]).tolist(),
                reference_seed_means=a.mean(1).tolist(),candidate_seed_means=b.mean(1).tolist())


def confirm(args):
    for system in REFERENCE:
        old=data.load(ROOT/'data/sc'/f'{system}.h5')
        freshpath=ROOT/'data/sc/forecast_fresh'/f'{system}.h5'
        fresh=data.load(freshpath)
        tr,va,te=data.split(len(old['u']))
        out={}; arrays={}
        for label,d,idx in [('old_test',old,te),('fresh',fresh,np.arange(len(fresh['u'])))]:
            arrays[label]={'candidate':[],'control':[],'time_budget':[]}
            for seed in range(3):
                cp=ROOT/'results/forecast'/f'{system}_aligned_s{seed}'/'model.pt'
                ck=torch.load(cp,map_location=args.device,weights_only=False)
                n=Norm.__new__(Norm); n.__dict__.update(ck['norm'])
                u,u0,p,times=normalized(d,n,args.device)
                net=ClockConditionedOperator(**ck['config']['model']).to(args.device)
                net.load_state_dict(ck['state'])
                arrays[label]['candidate'].append(errors(net,u,u0,p,times,idx).mean(1).tolist())
                for kind,file in [('control','model.pt'),('time_budget','time_budget.pt')]:
                    ref,rn,_=load_control(system,seed,args.device,file)
                    ru,ru0,rp,rt=normalized(d,rn,args.device)
                    arrays[label][kind].append(errors(ref,ru,ru0,rp,rt,idx).mean(1).tolist())
            out[label]={kind:comparison(arrays[label][kind],arrays[label]['candidate'])
                        for kind in ('control','time_budget')}
        out['per_trajectory']=arrays
        out['fresh_sha256']=digest(freshpath)
        out['control_backbone']=REFERENCE[system]
        out['checkpoint_hashes']={str(p.relative_to(ROOT)):digest(p) for seed in range(3)
            for p in [ROOT/'results/forecast'/f'{system}_aligned_s{seed}'/'model.pt',
                      OUT/'controls'/f'{system}_s{seed}'/'model.pt',
                      OUT/'controls'/f'{system}_s{seed}'/'time_budget.pt']}
        write_json(OUT/'confirmation'/f'{system}.json',out)
        print(system,json.dumps({k:v for k,v in out.items() if k in ('old_test','fresh')}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['control','generate','clock','confirm'])
    p.add_argument('--system',choices=list(REFERENCE),default='sidm')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args(); torch.set_num_threads(4)
    globals()[a.stage](a)


if __name__=='__main__': main()
