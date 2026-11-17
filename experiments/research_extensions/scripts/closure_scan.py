#!/usr/bin/env python3
"""Identifiability scan of the generalised conductivity interpolation.

K = (K_s^-alpha + K_l^-alpha)^(-1/alpha) (gravop.closure.run_halo_g, unchanged)
on a grid of (alpha, C), for the three haloes with kinetic/N-body references:

  A  Fischer, Yu & Dolag (2025) N-body: fitted time registration s - 1 against
     the N-body profiles (scripts/e2_calibration_fit.py scorer, unchanged);
  B, C  NSphere: tenfold central-density time t_x10 and the whole central
     density trajectory (mean density inside 0.3% of the mass), on NSphere's
     exact initial conditions (scripts/independent_halo_analysis.py measures).

Writes experiments/research_extensions/learned_closures/closure_scan.npz and .json.
Usage: closure_scan.py [--workers 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = os.path.join(EXT, 'learned_closures')

ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
CS = list(np.round(np.linspace(0.55, 1.20, 14), 4))
TGRID = {'B': np.linspace(0, 4.5, 451), 'C': np.linspace(0, 1.6, 321)}   # in units of the 2e5 reference t_x10


def job(args):
    h, a, C = args
    import closure_transfer as CT
    import independent_halo_analysis as IH
    from gravop import closure, solver
    if h == 'A':
        try:
            r = CT.halo_a(a, C)
        except ValueError as e:      # best time registration outside the scorer's s window
            return h, a, C, dict(clock_error=None, error=f'halo A scorer: {e}')
        return h, a, C, dict(clock_error=r['clock_error'], rms_dex_at_s_best=r['rms_dex_at_s_best'],
                             rms_dex_direct=r['rms_dex_direct'], Kn_c=r['Kn_c_at_largest_core'],
                             status=r['status'])
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p,
                             eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e4,
                             kn_min=0.02, t_end=u['t_end'], max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])
            for i in range(len(run['t']))]
    mr = np.array([x[1] for x in meas])
    mk = IH.collapse_markers(run['t'], mr[:, 1])
    i0 = int(np.argmin(run['rho'][:, 3]))
    return h, a, C, dict(t_x10=mk['t_x10'], t_x3=mk['t_x3'], t_x30=mk['t_x30'], t_min=mk['t_min'],
                         rho_min=mk['rho_min'], t_final=run['t_final'], status=run['status'],
                         Kn_c=float(1.0 / (np.sqrt(run['p'][i0, 3]) * u['sigma_hat'])),
                         t=run['t'], mean_rho=mr, disp=np.array([x[2] for x in meas]),
                         lagr_r=np.array([x[0] for x in meas]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tasks = [(h, al, C) for h in ('B', 'C', 'A') for al in ALPHAS for C in CS]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(job, tasks, chunksize=1)
    summ, traj = {}, {}
    for h, al, C, d in res:
        key = f'{h}|{al}|{C}'
        summ[key] = {k: v for k, v in d.items() if k not in ('t', 'mean_rho', 'disp', 'lagr_r')}
        if h != 'A':
            traj[key] = {k: d[k] for k in ('t', 'mean_rho', 'disp', 'lagr_r')}
    rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = dict(alphas=ALPHAS, Cs=[float(c) for c in CS], git=rev, wall_s=time.time() - t0,
                command=' '.join(sys.argv), solver='gravop.closure.run_halo_g (unchanged)',
                settings=dict(n_shells=300, eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e4,
                              kn_min=0.02, ic='NSphere truncated NFW (B, C); e2.RUN_KW (A)'))
    json.dump(dict(meta=meta, results=summ), open(os.path.join(OUT, 'closure_scan.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'closure_scan_traj.npz'),
                        **{f'{k}|{q}': v[q] for k, v in traj.items() for q in v})
    print('done', len(res), 'in', time.time() - t0, 's')


if __name__ == '__main__':
    main()
