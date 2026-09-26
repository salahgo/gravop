"""Train and evaluate physical-time forecasting; never select on the test set.

Examples:
  python scripts/sc_forecast.py train --system sidm --out results/forecast/pilot
  python scripts/sc_forecast.py evaluate --run results/forecast/pilot

Training writes validation-only metrics. Test evaluation is a separate command
after architecture/budget selection. Existing published artifacts are read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from statecond import data
from statecond.forecast import ClockConditionedOperator
from sc_experiment import Norm, tensors, load_checkpoint, phys_time, field_err


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def prepare(system, device):
    path = ROOT / 'data' / 'sc' / f'{system}.h5'
    d = data.load(path)
    tr, va, te = data.split(len(d['u']))
    n = Norm(d, tr)
    u, u0, par, *_ = tensors(d, n, device)
    times = torch.tensor(10. ** d['log_tsing'][:, None] * data.tau_grid()[None],
                         device=device, dtype=torch.float32)
    return d, n, u, u0, par, times, tr, va, te


@torch.no_grad()
def errors(model, u, u0, par, times, idx, bs=512):
    model.eval()
    ri = np.repeat(idx, u.shape[1])
    ti = np.tile(np.arange(u.shape[1]), len(idx))
    vals = []
    for start in range(0, len(ri), bs):
        r, t = ri[start:start+bs], ti[start:start+bs]
        p = model(u0[r], par[r], times[r, t])
        vals.append((p-u[r,t]).square().mean((1,2)).sqrt().cpu().numpy())
    return np.concatenate(vals).reshape(len(idx), -1)


def summary(e):
    return dict(nrmse=float(e.mean()), late_nrmse=float(e[:,data.tau_grid()>.92].mean()),
                p90_trajectory=float(np.quantile(e.mean(1), .9)),
                per_trajectory=e.mean(1).tolist(), per_time=e.mean(0).tolist())


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise SystemExit('Run already exists; choose a new --out directory.')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d,n,u,u0,par,times,tr,va,te = prepare(args.system,args.device)
    source = (Path(args.clock).resolve() if args.clock else
              ROOT/'results/statecond/ckpt'/f'{args.system}_fno_time_conv_seed{args.seed}.pt')
    head,meta = load_checkpoint(source,args.device)
    for key in ('f_mu','f_sd','p_mu','p_sd','t_mu','t_sd'):
        if not np.allclose(meta['norm'][key],getattr(n,key),rtol=1e-6,atol=1e-7):
            raise ValueError(f'clock normalization mismatch: {key}')
    cfg=dict(n_fields=u.shape[2],n_par=par.shape[1],n_x=u.shape[3],
             width=args.width,n_layers=args.layers,modes=32,
             clock_mu=n.t_mu,clock_sd=n.t_sd,
             time_scale=float(10.**np.median(d['log_tsing'][tr])),
             aligned=not args.direct,local=not args.no_local)
    model=ClockConditionedOperator(**cfg).to(args.device)
    model.clock.load_state_dict(head.state_dict())
    config=dict(args=vars(args),model=cfg,split_seed=1234,
                data_sha256=digest(ROOT/'data/sc'/f'{args.system}.h5'),
                clock_sha256=digest(source),clock_source=os.path.relpath(source,ROOT),
                source_sha256={p:digest(ROOT/p) for p in
                    ('src/statecond/forecast.py','scripts/sc_forecast.py')},
                train_index=tr.tolist(),validation_index=va.tolist(),test_index=te.tolist(),
                selection='minimum validation trajectory/time mean nRMSE; no test evaluation')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    ri=np.repeat(tr,u.shape[1]); ti=np.tile(np.arange(u.shape[1]),len(tr))
    best=float('inf'); hist=[]; start=time.time()
    for ep in range(args.epochs):
        model.train(); model.clock.eval()
        order=rng.permutation(len(ri)); total=0.
        for off in range(0,len(order),args.batch):
            sel=order[off:off+args.batch]; r,t=ri[sel],ti[sel]
            pred=model(u0[r],par[r],times[r,t])
            loss=(pred-u[r,t]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                           if p.requires_grad],1.)
            opt.step(); total+=float(loss.detach())*len(sel)
        sch.step()
        if ep==0 or (ep+1)%args.eval_every==0 or ep+1==args.epochs:
            val=summary(errors(model,u,u0,par,times,va))
            row=dict(epoch=ep+1,loss=total/len(ri),validation=val['nrmse'],
                     seconds=time.time()-start)
            hist.append(row)
            if val['nrmse']<best:
                best=val['nrmse']
                torch.save(dict(state=model.state_dict(),config=config,
                                 norm=n.__dict__,epoch=ep+1,validation=val),out/'model.pt')
            (out/'training.json').write_text(json.dumps(dict(history=hist,best_validation=best,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                seconds=time.time()-start),indent=2))
            print(args.system,args.seed,row,'best',best,flush=True)


def evaluate(args):
    out=Path(args.run)
    ck=torch.load(out/'model.pt',map_location=args.device,weights_only=False)
    cfg=ck['config']; system=cfg['args']['system']; seed=cfg['args']['seed']
    if digest(ROOT/'data/sc'/f'{system}.h5') != cfg['data_sha256']:
        raise ValueError('Dataset changed since training')
    d,n,u,u0,par,times,tr,va,te=prepare(system,args.device)
    model=ClockConditionedOperator(**cfg['model']).to(args.device)
    model.load_state_dict(ck['state']); model.eval()
    # Choose the reference on validation, independently of the new test score.
    class Baseline(torch.nn.Module):
        def __init__(self,kind,backbone):
            super().__init__(); self.kind=kind
            path=ROOT/'results/statecond/ckpt'/f'{system}_{backbone}_{kind}_seed{seed}.pt'
            self.net,self.meta=load_checkpoint(path,args.device)
            self.sha256=digest(path)
            if (self.meta.get('h1',0.) != 0. or
                (backbone=='fno' and kind=='tau' and 'h1' not in self.meta)):
                raise ValueError(f'reference is not an identified plain arm: {path}')
            if (self.meta['backbone']!=backbone or self.meta['kind']!=kind
                    or self.meta['seed']!=seed):
                raise ValueError(f'reference metadata mismatch: {path}')
            for key in ('f_mu','f_sd','p_mu','p_sd'):
                if not np.allclose(self.meta['norm'][key],getattr(n,key)):
                    raise ValueError(f'reference normalization mismatch: {path}: {key}')
            _,self.mu,self.sd=phys_time(d,tr,data.tau_grid())
        def forward(self,p,c,t):
            if self.kind=='tau': q=model.clock_ratio(p,c,t).clamp(0,1)
            else:
                # Positive-time direct reference exactly follows sc_direct.
                # At t=0 reproduce its original trajectory-dependent floor.
                # The separate reference below handles that historical point.
                q=(torch.log10(t.clamp_min(1e-30))-self.mu)/self.sd
            return self.net(p,torch.cat([c,q[:,None]],1))
    refs={}
    for backbone,kind in (('fno','tau'),('cno','tau'),('transformer','tau'),('fno','phys')):
        base=Baseline(kind,backbone).eval()
        scores={}
        if 'h1' not in base.meta:
            # Earlier CNO/Transformer files predate the loss tag. Identify them
            # by reproducing every stored per-trajectory oracle error, rather
            # than assuming a filename identifies the trained experiment.
            if kind=='phys':
                published=ROOT/'results/statecond/direct.json'
                stored=json.loads(published.read_text())[system]
                ltab,_,_=phys_time(d,tr,data.tau_grid())
                tq=torch.tensor(ltab,device=args.device,dtype=torch.float32)
                cond=lambda ri,ti:torch.cat([par[ri],tq[ri,ti,None]],1)
                arm='D_time_direct'
            else:
                published=ROOT/'results/statecond'/f'{system}_{backbone}.json'
                stored=json.loads(published.read_text())
                tq=torch.tensor(data.tau_grid(),device=args.device,dtype=torch.float32)
                cond=lambda ri,ti:torch.cat([par[ri],tq[ti,None]],1)
                arm='A_tau_oracle'
            measured=field_err(base.net,cond,u,u0,te).mean(1)
            expected=stored['per_trajectory'][arm][stored['seeds'].index(seed)]
            # Match the original per-time batching for identity checks: CUDA
            # convolution kernel selection also depends on the batch shape.
            # CPU gives ~0.3% relative differences in these small errors.
            identity_rtol=1e-2 if str(args.device).startswith('cpu') else 2e-4
            if stored['test_index']!=te.tolist() or not np.allclose(measured,expected,rtol=identity_rtol,atol=2e-6):
                raise ValueError(f'legacy checkpoint failed behavioral identity check: {backbone}')
            scores['legacy_identity_checked_against_sha256']=digest(published)
            scores['legacy_identity_rtol']=identity_rtol
        for label,idx in (('validation',va),('test',te)):
            # Legacy D's t=0 convention needs the dataset event time. This is
            # used only to reproduce the reference, never by the new model.
            rt=times.clone()
            if kind=='phys': rt[:,0]=times[:,1]*.5
            scores[label]=summary(errors(base,u,u0,par,rt,idx))
        scores['checkpoint_sha256']=base.sha256
        refs[f'{backbone}_{kind}']=scores
    selected=min(refs,key=lambda k:refs[k]['validation']['nrmse'])
    e=errors(model,u,u0,par,times,te)
    num=np.asarray(refs[selected]['test']['per_trajectory']); den=e.mean(1)
    rng=np.random.default_rng(20260909); draws=rng.integers(0,len(te),(5000,len(te)))
    ratios=num[draws].mean(1)/den[draws].mean(1)
    result=dict(system=system,seed=seed,selected_epoch=ck['epoch'],
                test=summary(e),references=refs,validation_selected_reference=selected,
                improvement=float(num.mean()/den.mean()),
                paired_trajectory_ci95=np.quantile(ratios,[.025,.975]).tolist(),
                test_index=te.tolist(),note='Warm-started frozen converged clock; additional field training budget.')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('references','test','test_index')},indent=2))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='stage',required=True)
    t=sub.add_parser('train'); t.add_argument('--system',required=True)
    t.add_argument('--out',required=True); t.add_argument('--seed',type=int,default=0)
    t.add_argument('--epochs',type=int,default=300); t.add_argument('--batch',type=int,default=512)
    t.add_argument('--width',type=int,default=64); t.add_argument('--layers',type=int,default=4)
    t.add_argument('--lr',type=float,default=1e-3); t.add_argument('--eval-every',type=int,default=20)
    t.add_argument('--direct',action='store_true'); t.add_argument('--no-local',action='store_true')
    t.add_argument('--clock', help='explicit training-partition-compatible clock checkpoint')
    e=sub.add_parser('evaluate'); e.add_argument('--run',required=True)
    for s in (t,e): s.add_argument('--device',default='cuda:0')
    args=p.parse_args(); torch.set_num_threads(4)
    train(args) if args.stage=='train' else evaluate(args)


if __name__=='__main__': main()
