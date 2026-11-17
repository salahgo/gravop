#!/usr/bin/env python3
"""Baryon-aware emulator on dataset D_b, and the dark-matter-only emulator on it.

The production network (gravop.model.GravOpNet, unchanged; FNO width 64, four
layers of 32 modes) is trained with two more conditioning inputs, M_b-hat
(scaled by 0.05) and log10 a-hat (standardised), next to log sigma0, log w,
the model flag and tau. The physics penalties are off: their hydrostatic term
counts the dark-matter mass only and would penalise correct baryonic fields;
the penalty-free FNO matches the production one on D1 (Table of the
generalisation splits). Everything else follows scripts/train.py: 200 epochs,
AdamW, lr 1e-3, cosine decay, batch 1024, clock loss weight 0.1.

Splits: runs with held_out = 1 (|Phi_b(0)| > 0.6, fixed in the generator) are
never used for training or model selection. The rest are split 80/10/10 by run
with the production split seed.

Modes:
  train --seed S          one member -> results/baryons_20261107/runs/seed<S>
  evaluate                members 0-2 on the test and held-out runs, and the
                          production dark-matter-only ensemble (results/runs/
                          gravop_s0..4) zero-shot on the same runs
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
from gravop.model import GravOpNet  # noqa: E402
from gravop import dataset as ds  # noqa: E402
from train import split  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'baryons', 'train.h5')
OUT = os.path.join(ROOT, 'results', 'baryons_20261107')
MB_SCALE = 0.05


def load():
    with h5py.File(DATA, 'r') as h:
        d = {k: h[k][:] for k in h}
    return d


def splits(d):
    keep = np.where(d['held_out'] == 0)[0]
    tr, va, te = split(len(keep))
    return keep[tr], keep[va], keep[te], np.where(d['held_out'] == 1)[0]


class Norm:
    def __init__(self, d, tr):
        f = lambda k: (float(d[k][tr].mean()), float(d[k][tr].std()) or 1.0)
        self.rho_mu, self.rho_sd = f('lrho')
        self.v_mu, self.v_sd = f('lv')
        self.sig_mu, self.sig_sd = f('log_sigma0')
        self.w_mu, self.w_sd = f('log_w')
        self.tc_mu, self.tc_sd = f('log_tc')
        la = np.log10(d['ab'][tr])
        self.la_mu, self.la_sd = float(la.mean()), float(la.std())


def tensors(d, n, dev, baryon_inputs=True):
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    lrho = (T(d['lrho']) - n.rho_mu) / n.rho_sd
    lv = (T(d['lv']) - n.v_mu) / n.v_sd
    prof = torch.stack([(T(d['lrho0']) - n.rho_mu) / n.rho_sd,
                        (T(d['lv0']) - n.v_mu) / n.v_sd], 1)
    cols = [(T(d['log_sigma0']) - n.sig_mu) / n.sig_sd, (T(d['log_w']) - n.w_mu) / n.w_sd,
            T(d['model_id'])]
    if baryon_inputs:
        cols += [T(d['Mb']) / MB_SCALE, (T(np.log10(d['ab'])) - n.la_mu) / n.la_sd]
    par = torch.stack(cols, 1)
    tc = (T(d['log_tc']) - n.tc_mu) / n.tc_sd
    return lrho, lv, prof, par, tc, T(d['no_collapse'])


def train(seed, dev):
    d = load()
    tr, va, te, ho = splits(d)
    n = Norm(d, tr)
    torch.manual_seed(seed)
    np.random.seed(seed)
    lrho, lv, prof, par, tc, nc = tensors(d, n, dev)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)
    model = GravOpNet(n_r=ds.N_R, width=64, modes=32, n_layers=4, cond_dim=6).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    nt = taus.shape[0]
    pr, pt = np.repeat(tr, nt), np.tile(np.arange(nt), len(tr))
    t0 = time.time()
    for ep in range(200):
        model.train()
        perm = np.random.permutation(len(pr))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ti = pr[sel], pt[sel]
            cond = torch.cat([par[ri], taus[ti][:, None]], 1)
            f, sc = model(prof[ri], cond)
            tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], 1)
            loss = (F.mse_loss(f, tgt) + 0.1 * F.mse_loss(sc[:, 0], tc[ri])
                    + 0.1 * F.binary_cross_entropy_with_logits(sc[:, 1], nc[ri]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    out = os.path.join(OUT, 'runs', f'seed{seed}')
    os.makedirs(out, exist_ok=True)
    torch.save({'model': model.state_dict(), 'norm': n.__dict__, 'seed': seed},
               os.path.join(out, 'model.pt'))
    json.dump({'seed': seed, 'wall_s': time.time() - t0, 'n_train': int(len(tr)),
               'n_val': int(len(va)), 'n_test': int(len(te)), 'n_held_out': int(len(ho))},
              open(os.path.join(out, 'train.json'), 'w'), indent=1)
    print(f'seed {seed}: {time.time() - t0:.0f}s')


def predict(models, prof, par, taus, idx, extra_tau=None):
    """Ensemble fields (M, len(idx), n_tau, 2, N_R) and log t_c heads (M, len(idx))."""
    nt = taus.shape[0]
    F_, H = [], []
    with torch.no_grad():
        for m in models:
            fr, hr = [], []
            for r in idx:
                tt = taus if extra_tau is None else extra_tau[r]
                cond = torch.cat([par[[r] * nt], tt[:, None]], 1)
                f, sc = m(prof[[r] * nt], cond)
                fr.append(f.cpu().numpy())
                hr.append(sc[0, 0].item())
            F_.append(np.stack(fr))
            H.append(np.array(hr))
    return np.stack(F_), np.stack(H)


def scores(fields_ens, H, n, d, idx):
    """Median over (run, tau) of the radially averaged relative density error,
    the ensemble-mean prediction scored as in scripts/train.py; median clock
    error; median ensemble spread of log rho (dex)."""
    mu = fields_ens.mean(0)
    lr_pred = mu[:, :, 0] * n.rho_sd + n.rho_mu
    rel = np.abs(10 ** (lr_pred - d['lrho'][idx]) - 1).mean(-1)
    spread = (fields_ens[:, :, :, 0] * n.rho_sd).std(0).mean(-1)
    coll = d['no_collapse'][idx] == 0
    tc_pred = 10 ** (H.mean(0) * n.tc_sd + n.tc_mu)
    tc_rel = np.abs(tc_pred / 10 ** d['log_tc'][idx] - 1)[coll]
    return {'n_runs': int(len(idx)), 'median_rel_rho': float(np.median(rel)),
            'p90_rel_rho': float(np.percentile(rel, 90)),
            'median_clock_rel_err': float(np.median(tc_rel)) if coll.any() else None,
            'median_ensemble_spread_dex': float(np.median(spread))}


def evaluate(dev):
    from evaluate_all import load_model, NormObj
    d = load()
    tr, va, te, ho = splits(d)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)
    out = {'note': __doc__.strip(), 'sets': {}}
    # baryon-aware members
    bm, bn = [], None
    for s in range(3):
        ck = torch.load(os.path.join(OUT, 'runs', f'seed{s}', 'model.pt'), map_location=dev, weights_only=False)
        m = GravOpNet(n_r=ds.N_R, width=64, modes=32, n_layers=4, cond_dim=6).to(dev)
        m.load_state_dict(ck['model'])
        m.eval()
        bm.append(m)
        bn = type('N', (), ck['norm'])
    lrho, lv, prof, par, tc, nc = tensors(d, bn, dev)
    # production dark-matter-only members, zero-shot (no baryon inputs)
    pm, pn = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs', f'gravop_s{s}'))
        pm.append(m.to(dev).eval())
        pn = NormObj(nrm)
    _, _, pprof, ppar, _, _ = tensors(d, pn, dev, baryon_inputs=False)
    baryonic = d['Mb'] > 0
    for name, idx in [('test', te), ('test_baryonic', te[baryonic[te]]),
                      ('test_dmo', te[~baryonic[te]]), ('held_out', ho)]:
        fb, hb = predict(bm, prof, par, taus, idx)
        fp, hp = predict(pm, pprof, ppar, taus, idx)
        out['sets'][name] = {'baryon_aware': scores(fb, hb, bn, d, idx),
                             'dmo_emulator_zero_shot': scores(fp, hp, pn, d, idx)}
    json.dump(out, open(os.path.join(OUT, 'emulator.json'), 'w'), indent=1)
    print(json.dumps(out['sets'], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'evaluate'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    train(a.seed, dev) if a.mode == 'train' else evaluate(dev)


if __name__ == '__main__':
    main()
