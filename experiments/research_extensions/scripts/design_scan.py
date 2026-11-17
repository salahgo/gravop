#!/usr/bin/env python3
"""Experiment A: where do the conductivity closures disagree, and what would a
particle reference there cost?

NSphere haloes are a one-shape family (truncated NFW, falloff c, cut at 85 r_s;
constant isotropic cross-section), so mass and r_s only set units and the
dimensionless design space is (sigma-hat, c). For each candidate the fluid
model (gravop.closure.run_halo_g, unchanged) is run under each closure on
NSphere's exact initial conditions; t_x10 is measured exactly as for B and C.

Discrimination statistic. Two closures H1, H2 predict T1, T2 for the same
observation y (the seed-mean particle collapse time). Writing
d = (y - (T1 + T2)/2) sign(T1 - T2), d has mean +-|T1 - T2|/2 under H1/H2 and
variance sigma_y^2 under either, so the separation is
    S = |ln T1 - ln T2| / sigma_y,
    sigma_y^2 = sigma_seed^2 / n_seeds + sigma_Nbias^2 + sigma_struct^2,
with the observation noise entering once, not twice (it is shared). The
expected log Bayes factor between the two point hypotheses is S^2 / 2.
sigma_seed, sigma_Nbias and sigma_struct are read from the multi-fidelity study
(or passed in); the scan itself stores the raw predictions.
Cost: thread-hours = 0.066 x (N / 1e5) x t_final[t_dyn], t_final = 1.6 t_x10
(measured: B1 0.065, C1 0.067 thread-h per t_dyn at 1e5 particles).
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
OUT = os.path.join(EXT, 'closure_design')
CLOSURES = {'harmonic': (1.0, 0.712), 'alpha_A': (0.5, 0.878), 'literature': (0.5, 0.9)}
SIGHATS = list(np.round(np.logspace(np.log10(0.15), np.log10(8.0), 15), 4))
FALLOFFS = [6.0, 10.0, 19.0, 30.0]
TH_PER_TDYN_1E5 = 0.066


def units(sig, c):
    import independent_halo_analysis as IH
    key = f'D_{sig}_{c}'
    IH.HALO[key] = dict(mass=1e10, rs=1.0, falloff=c, kappa=1.0, tfinal_tdyn=1.0)
    u = IH.halo_units(key)
    kappa = sig / (u['sigma_hat'] / 1.0)            # sigma_hat is linear in kappa
    IH.HALO[key]['kappa'] = kappa
    u = IH.halo_units(key)
    return IH, key, u


def job(args):
    sig, c, name, extra = args
    from gravop import closure, solver
    IH, key, u = units(sig, c)
    a, C = extra if extra else CLOSURES[name]
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    t_end = 300.0 / max(sig, 0.05) * 4      # generous: LMFP t_c ~ 150 / (C sigma-hat) t0
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p,
                             eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e4,
                             kn_min=0.02, t_end=t_end, max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])
            for i in range(len(run['t']))]
    mr = np.array([x[1] for x in meas])
    mk = IH.collapse_markers(run['t'], mr[:, 1])
    i0 = int(np.argmin(run['rho'][:, 3]))
    t0_tdyn = u['t0_myr'] / u['tdyn_myr']
    return dict(sigma_hat=sig, falloff=c, closure=name, alpha=a, C=C, t_x10=mk['t_x10'],
                t_min=mk['t_min'], status=run['status'], t_final=run['t_final'],
                Kn_c=float(1.0 / (np.sqrt(run['p'][i0, 3]) * u['sigma_hat'])),
                t0_over_tdyn=t0_tdyn, I=u['I'],
                t_x10_tdyn=None if mk['t_x10'] is None else mk['t_x10'] * t0_tdyn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--extra', default='', help='name:alpha:C,... additional closures')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    extra = {}
    for tok in filter(None, a.extra.split(',')):
        n_, al, C = tok.split(':')
        extra[n_] = (float(al), float(C))
    names = {**{k: None for k in CLOSURES}, **extra}
    tasks = [(s, c, n_, extra.get(n_)) for s in SIGHATS for c in FALLOFFS for n_ in names]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(job, tasks, chunksize=1)
    for r in res:
        if r['t_x10_tdyn']:
            r['thread_hours_1e5'] = TH_PER_TDYN_1E5 * 1.6 * r['t_x10_tdyn']
    rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(meta=dict(git=rev, wall_s=time.time() - t0, command=' '.join(sys.argv), sighats=SIGHATS,
                             falloffs=FALLOFFS, closures={k: (v or CLOSURES[k]) for k, v in names.items()}),
                   results=res), open(os.path.join(OUT, f'design_scan{a.tag}.json'), 'w'), indent=1)
    print('done', len(res), time.time() - t0)


if __name__ == '__main__':
    main()
