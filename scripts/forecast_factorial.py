"""Validation-only development ablation; independent of frozen F9 sources."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from sc_forecast import ROOT, digest, prepare, errors, summary, load_checkpoint
from statecond.forecast import ClockConditionedOperator

OUT = ROOT / 'results/forecast/factorial_v1'
SEEDS = (0,)
ARMS = ('full', 'features_only', 'gate_only', 'direct')
SYSTEMS = ('sidm', 'rd', 'burgers')
SOURCES = ('scripts/forecast_factorial.py', 'scripts/sc_forecast.py',
           'scripts/sc_experiment.py', 'src/statecond/forecast.py',
           'src/statecond/data.py', 'src/statecond/model.py', 'src/gravop/model.py',
           'docs/FORECAST_FACTORIAL_PROTOCOL.md', 'tests/test_forecast_factorial.py')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def resources():
    allowed = set(range(6)) | set(range(12, 18))
    os.sched_setaffinity(0, os.sched_getaffinity(0) & allowed)
    os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


class FactorialOperator(ClockConditionedOperator):
    def __init__(self, arm, **kwargs):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(**kwargs)
        if not self.aligned:
            raise ValueError('All factorial arms require the same four time-feature slots')
        self.arm = arm

    def forward(self, prof, par, time):
        if prof.shape[-1] != self.n_x:
            raise ValueError(f'clock encoder requires {self.n_x} spatial points')
        lt = torch.log1p(time / self.time_scale)
        q = self.clock_ratio(prof, par, time)
        qc = q.clamp(0., 1.2)
        features = torch.stack((qc, -torch.log((1. - qc).clamp_min(1e-3)),
                                torch.asinh(10. * (qc - .9))), dim=1)
        if self.arm in ('gate_only', 'direct'):
            features = torch.zeros_like(features)
        c = torch.cat((par, lt[:, None], features), dim=1)
        ratio = qc if self.arm in ('full', 'gate_only') else time / self.time_scale
        gate = -torch.expm1(-10. * ratio)
        ce = self.condition(c)
        grid = self.coord[None, None].expand(prof.shape[0], 1, -1)
        x = self.lift(torch.cat((prof, grid), dim=1))
        for block in self.blocks:
            x = block(x, ce)
        return prof + gate[:, None, None] * self.project(x)


def identities():
    paths = list(SOURCES)
    for system in SYSTEMS:
        paths += [f'data/sc/{system}.h5',
                  f'results/statecond/ckpt/{system}_fno_time_conv_seed0.pt']
    return {p: digest(ROOT / p) for p in paths}


def train(system, arm, device, seed=0):
    frozen = json.loads((OUT / 'manifest.json').read_text())
    if frozen['files'] != identities():
        raise RuntimeError('Development source or input identity changed')
    out = OUT / f'{system}_{arm}_s{seed}'
    out.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, n, u, u0, par, times, tr, va, te = prepare(system, device)
    clock = ROOT / f'results/statecond/ckpt/{system}_fno_time_conv_seed0.pt'
    head, meta = load_checkpoint(clock, device)
    for key in ('f_mu', 'f_sd', 'p_mu', 'p_sd', 't_mu', 't_sd'):
        np.testing.assert_allclose(meta['norm'][key], getattr(n, key), rtol=1e-6, atol=1e-7)
    cfg = dict(n_fields=u.shape[2], n_par=par.shape[1], n_x=u.shape[3],
               width=64, n_layers=4, modes=32, clock_mu=n.t_mu, clock_sd=n.t_sd,
               time_scale=float(10. ** np.median(d['log_tsing'][tr])),
               aligned=True, local=True)
    model = FactorialOperator(arm, **cfg).to(device)
    model.clock.load_state_dict(head.state_dict())
    config = dict(system=system, arm=arm, seed=seed, model=cfg,
                  train_index=tr.tolist(), validation_index=va.tolist(),
                  test_evaluated=False, epochs=200, batch=512,
                  manifest_sha256=digest(OUT / 'manifest.json'),
                  device=device, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    write(out / 'config.json', config)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    ri = np.repeat(tr, u.shape[1]); ti = np.tile(np.arange(u.shape[1]), len(tr))
    best = float('inf'); hist = []; start = time.monotonic()
    for ep in range(200):
        model.train(); model.clock.eval()
        order = rng.permutation(len(ri)); total = 0.
        for off in range(0, len(order), 512):
            sel = order[off:off+512]; r, t = ri[sel], ti[sel]
            pred = model(u0[r], par[r], times[r, t])
            loss = (pred-u[r, t]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step(); total += float(loss.detach()) * len(sel)
        scheduler.step()
        if ep == 0 or (ep+1) % 20 == 0:
            val = summary(errors(model, u, u0, par, times, va))
            row = dict(epoch=ep+1, loss=total/len(ri), validation=val['nrmse'],
                       seconds=time.monotonic()-start)
            hist.append(row)
            if val['nrmse'] < best:
                best = val['nrmse']
                torch.save(dict(state=model.state_dict(), config=config, norm=n.__dict__,
                                epoch=ep+1, validation=val), out/'model.pt')
            write(out/'training.json', dict(history=hist, best_validation=best,
                trainable_parameters=sum(p.numel() for p in params),
                total_parameters=sum(p.numel() for p in model.parameters()),
                seconds=time.monotonic()-start))
            print(system, arm, row, flush=True)
    ck = torch.load(out/'model.pt', map_location='cpu', weights_only=False)
    write(out/'validation.json', dict(complete=True, selected_epoch=ck['epoch'],
        validation=ck['validation'], checkpoint_sha256=digest(out/'model.pt'),
        test_evaluated=False, manifest_sha256=digest(OUT/'manifest.json')))


def report():
    rows = {}
    for system in SYSTEMS:
        arms = {}
        for arm in ARMS:
            per_seed = []
            for seed in SEEDS:
                run = OUT/f'{system}_{arm}_s{seed}'
                value = json.loads((run/'validation.json').read_text())
                if not value['complete'] or value['test_evaluated']:
                    raise RuntimeError('Invalid completed validation record')
                if value['checkpoint_sha256'] != digest(run/'model.pt'):
                    raise RuntimeError('Checkpoint changed')
                if value['manifest_sha256'] != digest(OUT/'manifest.json'):
                    raise RuntimeError('Manifest changed')
                per_seed.append(value)
            arms[arm] = dict(
                nrmse=float(np.mean([v['validation']['nrmse'] for v in per_seed])),
                per_seed_nrmse=[v['validation']['nrmse'] for v in per_seed],
                late_nrmse=float(np.mean([v['validation']['late_nrmse'] for v in per_seed])),
                epoch=[v['selected_epoch'] for v in per_seed])
        e = {a: v['nrmse'] for a, v in arms.items()}
        rows[system] = dict(arms=arms, multiplicative_interaction=
                           e['full']*e['direct']/(e['features_only']*e['gate_only']))
    write(OUT/'summary.json', dict(complete=True, systems=rows, seeds=list(SEEDS),
        scope='Development validation only; no fresh-test or population-calibration claim.'))
    lines = ['# F10: clock-feature and gate development ablation', '',
             f'Seeds {list(SEEDS)}; previously used validation splits. Development results.', '',
             '| System | Full | Features only | Gate only | Direct | Interaction |',
             '|---|---:|---:|---:|---:|---:|']
    for k, v in rows.items():
        lines.append('| '+k+' | '+' | '.join(f"{v['arms'][a]['nrmse']:.6g}" for a in ARMS)+
                     f" | {v['multiplicative_interaction']:.4f} |")
    lines += ['', 'Lower nRMSE is better. Interaction below one is favorable on the multiplicative error scale.',
              'All arms use equal tensor shapes and training budgets; feature-zeroed weights are inactive.',
              'No model replaces the frozen F9 candidate. Replication and fresh confirmation are required',
              'before interpreting an apparent improvement as generalization.', '',
              '[Protocol](FORECAST_FACTORIAL_PROTOCOL.md). Raw records: `results/forecast/factorial_v1`.', '']
    # F10b: the results doc is per-run. Writing a fixed path let the
    # multi-seed replication silently overwrite the one-seed pilot's
    # record, which still said 'One seed' above three-seed means.
    name = ('docs/FORECAST_FACTORIAL_RESULTS_2026-09-11.md'
            if OUT.name == 'factorial_v1'
            else f'docs/FORECAST_{OUT.name.upper()}_RESULTS.md')
    (ROOT/name).write_text('\n'.join(lines))


def suite(device):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(files=identities(), systems=SYSTEMS, arms=ARMS, seeds=list(SEEDS), epochs=200)
    # JSON serialization normalizes tuples for exact resumed-manifest comparison.
    manifest = json.loads(json.dumps(manifest))
    path = OUT/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Existing factorial manifest differs')
    else:
        write(path, manifest)
    complete = []
    try:
        for system in SYSTEMS:
          for seed in SEEDS:
            for arm in ARMS:
                run = OUT/f'{system}_{arm}_s{seed}'
                if (run/'validation.json').exists():
                    v = json.loads((run/'validation.json').read_text())
                    if (not v['complete'] or v['manifest_sha256'] != digest(path)
                            or v['checkpoint_sha256'] != digest(run/'model.pt')):
                        raise RuntimeError('Completed run identity failed')
                else:
                    with (ROOT/f'logs/f10_{system}_{arm}_s{seed}.log').open('x') as log:
                        subprocess.run([sys.executable, __file__, 'train', '--system', system,
                                        '--arm', arm, '--device', device,
                                        '--seeds', str(seed),
                                        '--out', OUT.name], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                complete.append(f'{system}_{arm}')
                write(OUT/'progress.json', dict(completed=complete, total=len(SYSTEMS)*len(ARMS)*len(SEEDS)))
        if identities() != manifest['files']:
            raise RuntimeError('Inputs changed during the suite')
        report()
    except Exception as exc:
        write(OUT/'failure.json', dict(error=repr(exc), completed=complete,
                                      time=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('suite', 'train', 'report'))
    p.add_argument('--system', choices=SYSTEMS)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--out')
    a = p.parse_args()
    if a.seeds: SEEDS = tuple(a.seeds)
    if a.out: OUT = ROOT / 'results/forecast' / a.out
    resources()
    if a.stage == 'suite': suite(a.device)
    elif a.stage == 'train': train(a.system, a.arm, a.device, SEEDS[0])
    else: report()
