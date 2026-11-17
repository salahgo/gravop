#!/usr/bin/env python3
"""Dataset D_b: gravothermal trajectories in a static Hernquist potential.

Same cross-section distribution and production solver settings as D1
(scripts/gen_dataset.py), with gravop.baryons in place of gravop.solver:
  70% yukawa_tu: log10 sigma0 ~ U[-1.5, 0.7], log10 w ~ U[-1.0, 1.5]
  30% constant:  log10 sigma  ~ U[-2.0, 0.5]
  initial profile: NFW, in Jeans equilibrium in the total potential
  baryons: 15% none; otherwise log10 M_b-hat ~ U[-2.5, -1.3]  (M_b / M0)
                                log10 a-hat   ~ U[-1.5, -0.5]  (a / r_s)
  grid: 300 shells from r = 0.003 r_s (ten shells inside the most compact a)

The extrapolation hold-out is fixed here, before any training: runs with a
central baryonic potential |Phi_b(0)| = M_b-hat / a-hat > 0.6 (the deepest
~7 per cent of the box) are flagged held_out = 1 and are never used for
training or model selection.

Usage: gen_baryon_dataset.py <n_runs> [--seed S] [--workers W]
Output: data/baryons/train.h5 (same per-run fields as D1, plus Mb, ab, held_out)
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import baryons, dataset  # noqa: E402

SOLVER_KW = dict(n_shells=300, r_min=0.003, eps_t=3e-4, rho_c_factor=1e3,
                 t_end=1200.0, max_snaps=3000, timestep_mode='implicit',
                 kn_min=0.02, max_steps=100_000_000)
PHI_HOLD_OUT = 0.6
P_NO_BARYONS = 0.15


def draw_params(rng):
    if rng.random() < 0.7:
        d = dict(model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                 sigma_m=10 ** rng.uniform(-1.5, 0.7), w=10 ** rng.uniform(-1.0, 1.5))
    else:
        d = dict(model_lmfp='constant', model_smfp='constant',
                 sigma_m=10 ** rng.uniform(-2.0, 0.5), w=1.0)
    if rng.random() < P_NO_BARYONS:
        Mb, ab = 0.0, 0.1
    else:
        Mb, ab = 10 ** rng.uniform(-2.5, -1.3), 10 ** rng.uniform(-1.5, -0.5)
    return d, Mb, ab


def one_run(args):
    idx, seed = args
    rng = np.random.default_rng(seed)
    d, Mb, ab = draw_params(rng)
    t0 = time.time()
    try:
        res = baryons.run_halo_b(**d, Mb=Mb, ab=ab, **SOLVER_KW)
        if res['status'] in (3, 4) or res['t_final'] < 1e-3:
            raise RuntimeError(f"unusable run: status {res['status']} t {res['t_final']}")
        s = dataset.sample_from_run(res)
        s.update(Mb=np.float32(Mb), ab=np.float32(ab),
                 held_out=np.int8(Mb / ab > PHI_HOLD_OUT), seed=seed,
                 wall=np.float32(time.time() - t0))
        return idx, s, None
    except Exception:
        return idx, None, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n_runs', type=int)
    ap.add_argument('--seed', type=int, default=20261107)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'baryons')
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(i, a.seed + i) for i in range(a.n_runs)]
    rows, fails = {}, []
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, s, err in pool.imap_unordered(one_run, tasks):
            if s is None:
                fails.append((i, err))
            else:
                rows[i] = s
            if (len(rows) + len(fails)) % 100 == 0:
                print(f'{len(rows)} ok, {len(fails)} failed, {time.time() - t0:.0f}s', flush=True)
    keys = sorted(rows)
    scal = ['log_sigma0', 'log_w', 'model_id', 'log_tc', 'no_collapse', 'status',
            'Mb', 'ab', 'held_out', 'seed', 'wall']
    with h5py.File(os.path.join(out_dir, 'train.h5'), 'w') as h5:
        for k in ('lrho', 'lv', 'lrho0', 'lv0'):
            h5.create_dataset(k, data=np.stack([rows[i][k] for i in keys]).astype('f4'))
        for k in scal:
            h5.create_dataset(k, data=np.array([rows[i][k] for i in keys]))
        h5.attrs['solver_kw'] = str(SOLVER_KW)
        h5.attrs['phi_hold_out'] = PHI_HOLD_OUT
        h5.attrs['seed'] = a.seed
        h5.attrs['n_failed'] = len(fails)
    with open(os.path.join(out_dir, 'train.failures.log'), 'w') as f:
        for i, e in fails:
            f.write(f'--- run {i}\n{e}\n')
    print(f'done: {len(rows)} runs, {len(fails)} failed, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
