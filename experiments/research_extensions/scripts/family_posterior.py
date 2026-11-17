#!/usr/bin/env python3
"""Experiment B: posterior-predictive of closure families for the design haloes.

For each shape point of a family (alpha; rational [1/2] Pade (a, b); Gaussian
regime gate (eps, l0) with width 1.5 in ln y), C is solved so that halo B's
t_x10 matches its reference (B lies deep in the long-mean-free-path regime and
fixes C almost alone), then haloes A, C, D1, D2 are run at that (shape, C).
Stored per point: C_B, ln-residuals of A (time registration), C (t_x10) and the
raw D1/D2 predictions. The posterior can be recomputed for any reference of B
by re-solving C from the stored B response (C_B is also stored for the default
reference). Fits use A, B, C only; D1/D2 predictions are frozen output.
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
for p in (os.path.join(EXT, 'learned_closures'), HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')):
    sys.path.insert(0, p)
OUT = os.path.join(EXT, 'learned_closures')
REF_B = 545.38
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))


def table(fam, par):
    import gfamilies as F
    return getattr(F, fam)(*par)


def t_x10(h, gx, gf, C):
    import gclosure as G
    import independent_halo_analysis as IH
    from gravop import solver
    if h in ('D1', 'D2'):
        d = PRE['haloes'][h]
        IH.HALO[h] = dict(mass=d['mass'], rs=d['rs'], falloff=d['falloff'], kappa=d['kappa'], tfinal_tdyn=1.0)
        u = IH.halo_units(h)
        t_end = 3 * d['tfinal_tdyn'] * u['tdyn_myr'] / u['t0_myr']
    else:
        u = IH.halo_units(h)
        t_end = u['t_end'] * 1.5
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = G.run_halo_t(sigma_m=u['sigma_hat'], gx=gx, gf=gf, C=C, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                       timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=t_end, max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I']) for i in range(len(run['t']))]
    mk = IH.collapse_markers(run['t'], np.array([x[1] for x in meas])[:, 1])
    tt = mk['t_x10']
    return None if tt is None else tt * (u['t0_myr'] / u['tdyn_myr'] if h in ('D1', 'D2') else 1.0)


def halo_a(gx, gf, C):
    import gclosure as G
    import e2_calibration_fit as e2
    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    run = G.run_halo_t(gx=gx, gf=gf, C=C, **e2.RUN_KW)
    try:
        sc = e2.scan(run, d, np.arange(0.70, 1.3001, 0.0025))
        return float(np.log(sc['s_best']))
    except ValueError:
        return None


def job(args):
    fam, par = args
    gx, gf = table(fam, par)
    # solve C from halo B by secant in ln C (ln t roughly linear in ln C)
    lc = [np.log(0.75), np.log(0.85)]
    lt = []
    for x in lc:
        t = t_x10('B', gx, gf, float(np.exp(x)))
        lt.append(np.log(t) if t else np.nan)
    for _ in range(4):
        if not np.all(np.isfinite(lt)) or lt[-1] == lt[-2]:
            break
        x_new = lc[-1] + (np.log(REF_B) - lt[-1]) * (lc[-1] - lc[-2]) / (lt[-1] - lt[-2])
        x_new = float(np.clip(x_new, np.log(0.3), np.log(3.0)))
        t = t_x10('B', gx, gf, float(np.exp(x_new)))
        lc.append(x_new); lt.append(np.log(t) if t else np.nan)
        if abs(lt[-1] - np.log(REF_B)) < 2e-3:
            break
    k = int(np.nanargmin(np.abs(np.array(lt) - np.log(REF_B))))
    C = float(np.exp(lc[k]))
    # B response slope dlnt/dlnC (to re-solve C for other B references)
    ii = np.isfinite(lt)
    slope = float(np.polyfit(np.array(lc)[ii], np.array(lt)[ii], 1)[0]) if ii.sum() >= 2 else None
    out = dict(family=fam, params=list(par), C=C, B_lnres=float(lt[k] - np.log(REF_B)), B_slope=slope,
               A_lnres=halo_a(gx, gf, C))
    tc = t_x10('C', gx, gf, C)
    out['C_lnres'] = None if tc is None else float(np.log(tc / 139.53))
    for h in ('D1', 'D2'):
        out[f'{h}_t_x10_tdyn'] = t_x10(h, gx, gf, C)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    pts = [('alpha', (x,)) for x in (0.4, 0.45, 0.5, 0.55, 0.6, 0.67, 0.75, 0.85, 1.0, 1.2, 1.5, 2.0)]
    for A_ in np.exp(np.linspace(np.log(0.05), np.log(20), 9)):
        for bb in np.linspace(-1.8, 2.5, 9):
            b = float(bb * np.sqrt(A_) + (1 + A_) * 0.0) if False else float(bb * 2 * np.sqrt(A_))
            if b > -2 * np.sqrt(A_) + 1e-3:
                pts.append(('rational', (float(A_), b)))
    for eps in (-1.2, -0.8, -0.4, 0.4, 0.8, 1.2):
        for l0 in (-4.0, -2.5, -1.0, 0.5, 2.0, 3.5):
            pts.append(('gate', (eps, l0, 1.5)))
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(job, pts, chunksize=1)
    rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    json.dump(dict(meta=dict(git=rev, wall_s=time.time() - t0, ref_B=REF_B, ref_C=139.53,
                             created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             note='frozen before any D1/D2 particle data; fits use A, B, C only'),
                   points=res), open(os.path.join(OUT, 'family_posterior.json'), 'w'), indent=1)
    print('done', len(res), time.time() - t0)


if __name__ == '__main__':
    main()
