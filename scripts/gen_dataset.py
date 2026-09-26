"""Generate dataset D1: solver trajectories -> HDF5 samples.

Usage: gen_dataset.py <name> <n_runs> [--seed S] [--workers W] [--set train|ood_param|ood_ic]

Sampling (train/val/test pool, seed-controlled):
  70% yukawa_tu: log10 sigma0 ~ U[-1.5, 0.7], log10 w ~ U[-1.0, 1.5]
  30% constant:  log10 sigma  ~ U[-2.0, 0.5], w = 1 (unused)
  ICs: 35% NFW, 30% cored-NFW (log10 rc ~ U[log10 0.03, 0]), 35% gNFW
       (inner slope gamma ~ U[0.5, 1.5]; rc field stores gamma)
OOD sets:
  ood_param: yukawa_tu with log10 sigma0 ~ U[0.7, 1.0] (w in U[-1,1]) or
             log10 w ~ U[-1.5, -1.0] (sigma0 from train range)
  ood_ic:    Hernquist profile (unseen family), params from train ranges
Ranges trimmed 2026-08-27 (cost control; sigma0<=5 keeps LMFP->SMFP transition
during collapse; t_end=1200 t0 >> Hubble time for realistic halos).
Solver settings fixed by the E1 convergence study (results/e1/).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver, dataset  # noqa: E402

# production solver settings (locked after E1 + implicit convergence matrix:
# results/e1/implicit_convergence.json; explicit crosscheck vs GravothermalSIDM
# at N=150 in results/e1/crosscheck.json; implicit-vs-explicit agreement and
# Kn-floor rationale from the E1b study)
SOLVER_KW = dict(n_shells=300, eps_t=3e-4, rho_c_factor=1e3,
                 t_end=1200.0, max_snaps=3000, timestep_mode='implicit',
                 kn_min=0.02, max_steps=100_000_000)


def draw_params(rng, which):
    if which == 'train':
        if rng.random() < 0.7:
            d = dict(model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                     sigma_m=10 ** rng.uniform(-1.5, 0.7),
                     w=10 ** rng.uniform(-1.0, 1.5))
        else:
            d = dict(model_lmfp='constant', model_smfp='constant',
                     sigma_m=10 ** rng.uniform(-2.0, 0.5), w=1.0)
        u_ic = rng.random()
        ic = 'nfw' if u_ic < 0.35 else ('cored' if u_ic < 0.65 else 'gnfw')
        rc = (10 ** rng.uniform(np.log10(0.03), 0.0) if ic == 'cored'
              else rng.uniform(0.5, 1.5))  # gnfw: rc = inner slope gamma
    elif which == 'ood_param':
        if rng.random() < 0.5:
            d = dict(model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                     sigma_m=10 ** rng.uniform(0.7, 1.0),
                     w=10 ** rng.uniform(-1.0, 1.0))
        else:
            d = dict(model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                     sigma_m=10 ** rng.uniform(-1.5, 0.7),
                     w=10 ** rng.uniform(-1.5, -1.0))
        ic, rc = 'nfw', None
    elif which == 'ood_ic':
        d = dict(model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                 sigma_m=10 ** rng.uniform(-1.5, 0.7),
                 w=10 ** rng.uniform(-1.0, 1.5))
        ic, rc = 'hernquist', None
    else:
        raise ValueError(which)
    return d, ic, rc


def hernquist_rho(x):
    return 1.0 / (x * (1.0 + x) ** 3)


def gnfw_rho(x, gam):
    return 1.0 / (x ** gam * (1.0 + x) ** (3.0 - gam))


def make_ic(ic, rc, n_shells):
    if ic == 'nfw':
        return None  # solver default (analytic mass, fast)
    if ic == 'cored':
        f = lambda x: solver.cored_nfw_rho(x, rc)
    elif ic == 'gnfw':
        f = lambda x: gnfw_rho(x, rc)
    elif ic == 'hernquist':
        f = hernquist_rho
    return solver.profile_from_rho(f, n_shells=n_shells)


def one_run(args):
    idx, seed, which = args
    rng = np.random.default_rng(seed)
    d, ic, rc = draw_params(rng, which)
    t0 = time.time()
    try:
        prof = make_ic(ic, rc, SOLVER_KW['n_shells'])
        kw = dict(SOLVER_KW)
        if prof is not None:
            kw['r'], kw['m'], kw['rho'], kw['p'] = prof
        res = solver.run_halo(**d, **kw)
        if res['status'] in (3, 4) or res['t_final'] < 1.0:
            raise RuntimeError(
                f"unusable run: status {res['status']} t {res['t_final']}")
        s = dataset.sample_from_run(res)
        s['ic_type'] = {'nfw': 0, 'cored': 1, 'hernquist': 2, 'gnfw': 3}[ic]
        s['rc'] = np.float32(rc if rc else 0.0)
        s['seed'] = seed
        s['wall'] = np.float32(time.time() - t0)
        return idx, s, None
    except Exception:
        return idx, None, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('n_runs', type=int)
    ap.add_argument('--seed', type=int, default=20260827)
    ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--set', dest='which', default='train')
    ap.add_argument('--quick', action='store_true',
                    help='small/fast solver settings (pipeline testing only)')
    args = ap.parse_args()
    if args.quick:
        SOLVER_KW.update(n_shells=100, rho_c_factor=100.0, t_end=300.0)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'd1')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.name + '.h5')
    log_path = os.path.join(out_dir, args.name + '.failures.log')

    tasks = [(i, args.seed + i, args.which) for i in range(args.n_runs)]
    t0 = time.time()
    n_fail = 0
    scalar_keys = ['log_sigma0', 'log_w', 'model_id', 'log_tc', 'no_collapse',
                   'status', 'ic_type', 'rc', 'seed', 'wall']
    with h5py.File(out_path, 'w') as h5, mp.Pool(args.workers) as pool:
        h5.attrs['solver_kw'] = json.dumps(SOLVER_KW)
        h5.attrs['base_seed'] = args.seed
        h5.attrs['set'] = args.which
        h5.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        n = args.n_runs
        dsets = {
            'lrho': h5.create_dataset('lrho', (n, dataset.N_TAU, dataset.N_R), 'f4'),
            'lv': h5.create_dataset('lv', (n, dataset.N_TAU, dataset.N_R), 'f4'),
            'lrho0': h5.create_dataset('lrho0', (n, dataset.N_R), 'f4'),
            'lv0': h5.create_dataset('lv0', (n, dataset.N_R), 'f4'),
        }
        for k in scalar_keys:
            dsets[k] = h5.create_dataset(k, (n,), 'f8' if k != 'seed' else 'i8')
        done = 0
        for idx, s, err in pool.imap_unordered(one_run, tasks, chunksize=1):
            if err is not None:
                n_fail += 1
                with open(log_path, 'a') as f:
                    f.write(f'--- idx {idx}\n{err}\n')
                dsets['status'][idx] = -1
                continue
            dsets['lrho'][idx] = s['lrho']
            dsets['lv'][idx] = s['lv']
            dsets['lrho0'][idx] = s['lrho0']
            dsets['lv0'][idx] = s['lv0']
            for k in scalar_keys:
                dsets[k][idx] = s[k]
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                print(f'{done}/{n} done, {n_fail} failed, {el:.0f}s elapsed, '
                      f'{el / done:.1f}s/run avg', flush=True)
    print(f'FINISHED {done} ok, {n_fail} failed, {time.time() - t0:.0f}s total')


if __name__ == '__main__':
    main()
