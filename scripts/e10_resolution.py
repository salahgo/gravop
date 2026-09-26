"""E10 — zero-shot resolution transfer (discretization invariance).

The emulator is trained on a 128-point log-radial grid. Neural operators are
supposed to be discretization-invariant: the same trained weights should
evaluate on a finer grid of the same domain without retraining. We test this
by re-running the exact test-split solver trajectories (same seeds, hence same
parameters and initial conditions) and regridding them onto N_R = 64, 128, 256
and 512 points, then evaluating the 128-trained ensemble at each resolution.

Usage: e10_resolution.py [--n_runs 60]
Output: results/e10/resolution.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import solver, dataset as ds  # noqa: E402
from train import load_data, split, Norm, make_tensors  # noqa: E402
from evaluate_all import load_model, NormObj  # noqa: E402
from gen_dataset import draw_params, make_ic, SOLVER_KW  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e10')
os.makedirs(OUT, exist_ok=True)
RESOLUTIONS = (64, 128, 256, 512)


def regrid_at(res, n_r):
    """Regrid one solver result onto n_r radial points (same log-r interval)."""
    rg = np.logspace(ds.LOGR_MIN, ds.LOGR_MAX, n_r)
    taus = ds.tau_grid()
    t_norm = res['t_final']
    n_snap = len(res['t'])
    lrho_s = np.empty((n_snap, n_r), dtype=np.float32)
    lv_s = np.empty((n_snap, n_r), dtype=np.float32)
    for i in range(n_snap):
        a, b = ds.regrid_snapshot(res['r'][i], res['rho'][i], res['p'][i], rg)
        lrho_s[i] = a
        lv_s[i] = b
    ts = res['t']
    t_targets = np.clip(taus * t_norm, 0.0, ts[-1])
    idx = np.clip(np.searchsorted(ts, t_targets, side='right') - 1,
                  0, n_snap - 2)
    w = np.clip((t_targets - ts[idx])
                / np.maximum(ts[idx + 1] - ts[idx], 1e-300), 0, 1)
    w = w[:, None].astype(np.float32)
    lrho = (1 - w) * lrho_s[idx] + w * lrho_s[idx + 1]
    lv = (1 - w) * lv_s[idx] + w * lv_s[idx + 1]
    return lrho, lv, lrho_s[0], lv_s[0]


def one(args):
    seed, which = args
    rng = np.random.default_rng(int(seed))
    d, ic, rc = draw_params(rng, which)
    prof = make_ic(ic, rc, SOLVER_KW['n_shells'])
    kw = dict(SOLVER_KW)
    if prof is not None:
        kw['r'], kw['m'], kw['rho'], kw['p'] = prof
    res = solver.run_halo(**d, **kw)
    out = {'seed': int(seed), 'status': res['status'],
           'log_sigma0': float(np.log10(d['sigma_m'])),
           'log_w': float(np.log10(d['w'])),
           'model_id': 0 if d['model_lmfp'] == 'constant' else 1,
           'log_tc': float(np.log10(res['t_final'])),
           'no_collapse': int(res['status'] == 1)}
    for n_r in RESOLUTIONS:
        lrho, lv, lrho0, lv0 = regrid_at(res, n_r)
        out[f'lrho_{n_r}'] = lrho
        out[f'lv_{n_r}'] = lv
        out[f'lrho0_{n_r}'] = lrho0
        out[f'lv0_{n_r}'] = lv0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_runs', type=int, default=60)
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()

    with h5py.File(os.path.join(ROOT, 'data', 'd1', 'train.h5'), 'r') as f:
        seeds = f['seed'][:]
        status = f['status'][:]
    ok = status >= 0
    tr, va, te = split(int(ok.sum()))
    te_seeds = seeds[ok][te][:args.n_runs]
    print(f'regenerating {len(te_seeds)} test-split runs at resolutions '
          f'{RESOLUTIONS}', flush=True)

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(one, [(s, 'train') for s in te_seeds])
    print(f'solver regeneration: {time.time() - t0:.0f}s', flush=True)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models, norm = [], None
    for s in range(5):
        rd = os.path.join(ROOT, 'results', 'runs', f'gravop_s{s}')
        m, nrm, a = load_model(rd)
        models.append(m)
        norm = nrm
    n = NormObj(norm)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)

    summary = {'n_runs': len(rows), 'trained_at': ds.N_R,
               'resolutions': list(RESOLUTIONS), 'per_resolution': {}}
    for n_r in RESOLUTIONS:
        rel_all, rmse_all = [], []
        for row in rows:
            prof = torch.tensor(np.stack([
                (row[f'lrho0_{n_r}'] - n.rho_mu) / n.rho_sd,
                (row[f'lv0_{n_r}'] - n.v_mu) / n.v_sd]),
                dtype=torch.float32, device=dev)[None]
            par = torch.tensor([[(row['log_sigma0'] - n.sig_mu) / n.sig_sd,
                                 (row['log_w'] - n.w_mu) / n.w_sd,
                                 float(row['model_id'])]],
                               dtype=torch.float32, device=dev)
            pi = prof.expand(ds.N_TAU, -1, -1)
            ci = torch.cat([par.expand(ds.N_TAU, -1), taus[:, None]], dim=1)
            preds = []
            with torch.no_grad():
                for m in models:
                    f_, _ = m(pi, ci)
                    preds.append((f_[:, 0].cpu().numpy() * n.rho_sd
                                  + n.rho_mu))
            mu = np.mean(preds, axis=0)
            tgt = row[f'lrho_{n_r}']
            rel_all.append(np.abs(10 ** (mu - tgt) - 1).mean())
            rmse_all.append(np.sqrt(((mu - tgt) ** 2).mean()))
        summary['per_resolution'][str(n_r)] = {
            'mean_rel_rho': float(np.mean(rel_all)),
            'median_rel_rho': float(np.median(rel_all)),
            'rmse_dex': float(np.mean(rmse_all))}
        print(n_r, summary['per_resolution'][str(n_r)], flush=True)

    json.dump(summary, open(os.path.join(OUT, 'resolution.json'), 'w'),
              indent=1)
    print('saved results/e10/resolution.json')


if __name__ == '__main__':
    main()
