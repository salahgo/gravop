#!/usr/bin/env python3
"""Does one conductivity calibration transfer across haloes A, B and C?

Closures (gravop.closure; alpha = 1 is the production harmonic interpolation):
  harmonic    alpha = 1,   C = 0.712  calibrated on halo A (0.753 x s, s = 0.945)
  alpha_A     alpha = 1/2, C = 0.878  calibrated on halo A alone (direct scan, results/e2)
  alpha_lit   alpha = 1/2, C = 0.9    van den Bosch & Tokayer (2026), fitted to nothing here

Haloes B and C are blind for every closure. Halo A is the calibration halo for
the first two and blind for the third.

  halo A  N-body run of Fischer, Yu & Dolag (2025): RMS log-density residual
          over the calibration snapshots with no time rescaling, and the time
          rescaling s that minimises it; s - 1 is the closure's clock error
          against the N-body run (E2's scorer, scripts/e2_calibration_fit.py).
  haloes B, C  NSphere references (scripts/independent_halo_analysis.py):
          time at which the central density first exceeds ten times its
          pre-collapse minimum, against the 1e5-particle seed mean and against
          the 2e5-particle runs where present; profile, Lagrangian-radius and
          dispersion agreement over the halo body before collapse.
Transfer variable: sigma-hat, and the central Knudsen number Kn_c at the
time of the largest core (minimum central density) in the harmonic run.

Writes results/closure_20261108/transfer.json and the fluid trajectories
results/closure_20261108/fluid_<halo>_<closure>.npz.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
from gravop import closure  # noqa: E402
import e2_calibration_fit as e2  # noqa: E402
import independent_halo_analysis as IH  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'closure_20261108')
CLOSURES = {'harmonic': (1.0, 0.712), 'alpha_A': (0.5, 0.878), 'alpha_lit': (0.5, 0.9)}
S_GRID = np.arange(0.80, 1.2001, 0.0025)


def halo_a(alpha, C):
    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    kw = dict(e2.RUN_KW)
    run = closure.run_halo_g(C=C, alpha=alpha, **kw)
    direct = e2.score(run, d, 1.0)
    sc = e2.scan(run, d, S_GRID)
    p = run['p'][:, 3]
    i0 = int(np.argmin(run['rho'][:, 3]))
    return {'rms_dex_direct': e2.total(direct) if direct else None,
            's_best': sc['s_best'], 'clock_error': sc['s_best'] - 1.0,
            'rms_dex_at_s_best': sc['rms_dex_best'],
            'Kn_c_at_largest_core': float(1.0 / (np.sqrt(p[i0]) * e2.SIGMA_HAT)),
            't_final': run['t_final'], 'status': run['status']}


def halo_bc(h, alpha, C, refs):
    u = IH.halo_units(h)
    from gravop import solver
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=alpha, r=r, m=m, rho=rho, p=p,
                             eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e4,
                             kn_min=0.02, t_end=u['t_end'], max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])
            for i in range(len(run['t']))]
    traj = dict(t=run['t'], lagr_r=np.array([x[0] for x in meas]),
                mean_rho=np.array([x[1] for x in meas]), disp=np.array([x[2] for x in meas]),
                prof=np.array([x[3] for x in meas]))
    mk = IH.collapse_markers(traj['t'], traj['mean_rho'][:, 1])
    i0 = int(np.argmin(run['rho'][:, 3]))
    out = {'t_x10': mk['t_x10'], 'rho_min': mk['rho_min'], 't_min': mk['t_min'],
           't_final': run['t_final'], 'status': run['status'],
           'Kn_c_at_largest_core': float(1.0 / (np.sqrt(run['p'][i0, 3]) * u['sigma_hat']))}
    for lab, ts in refs.items():
        out[f'clock_error_vs_{lab}'] = mk['t_x10'] / ts - 1 if ts else None
    # structure before collapse against the 1e5-particle seeds (halo body)
    dp, dr, ds_ = [], [], []
    t_ref = refs['N1e5_seed_mean']
    vol = 4 / 3 * np.pi * (IH.RBINS[1:] ** 3 - IH.RBINS[:-1] ** 3)
    rc = np.sqrt(IH.RBINS[1:] * IH.RBINS[:-1])
    for run_id in [k for k, v in IH.RUNS.items() if v == h][:2]:
        nd = np.load(os.path.join(IH.OUT, f'nsphere_{run_id}.npz'))
        for k, tt in enumerate(nd['t']):
            if 0 < tt < 0.8 * t_ref and tt <= traj['t'][-1]:
                mp = IH.interp_at(traj['t'], traj['prof'], tt)
                cnt = 10 ** nd['prof'][k] * (vol / (4 * np.pi)) / (u['I'] / int(nd['N']))
                ok = (np.isfinite(nd['prof'][k]) & (cnt >= 100)
                      & (rc < 0.6 * IH.HALO[h]['falloff']))
                dp.append(np.sqrt(np.mean((mp[ok] - nd['prof'][k][ok]) ** 2)))
                dr.append(np.abs(np.log10(IH.interp_at(traj['t'], traj['lagr_r'], tt)[1:]
                                          / nd['lagr_r'][k][1:])))
                ds_.append(np.abs(IH.interp_at(traj['t'], traj['disp'], tt)[1:] / nd['disp'][k][1:] - 1))
    out.update(profile_rms_dex_median=float(np.median(dp)),
               lagrangian_radius_dex_median=float(np.median(dr)),
               dispersion_rel_median=float(np.median(ds_)))
    return out, traj


def references(h):
    """t_x10 of the NSphere runs: seed mean at 1e5 particles, and 2e5 runs."""
    js = json.load(open(os.path.join(IH.OUT, 'independent_halo.json')))['haloes'][h]['nsphere']
    lo = [v['t_x10'] for r, v in js.items() if v['N'] == 100000 and r in ('B1', 'B2', 'C1', 'C2')]
    hi = [v['t_x10'] for r, v in js.items() if v['N'] == 200000 and v['t_x10']]
    refs = {'N1e5_seed_mean': float(np.mean(lo)) if lo else None,
            'N2e5_mean': float(np.mean(hi)) if hi else None}
    spread = {'N1e5_seed_halfspread_rel': float(np.ptp(lo) / 2 / np.mean(lo)) if len(lo) > 1 else None,
              'N2e5_halfspread_rel': float(np.ptp(hi) / 2 / np.mean(hi)) if len(hi) > 1 else None,
              'N2e5_runs': len(hi)}
    return refs, spread


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'note': __doc__.strip(), 'closures': {k: {'alpha': a, 'C': c} for k, (a, c) in CLOSURES.items()},
           'haloes': {'A': {'sigma_hat': e2.SIGMA_HAT}, 'B': {}, 'C': {}}}
    for name, (a, c) in CLOSURES.items():
        out['haloes']['A'][name] = halo_a(a, c)
        print('A', name, {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in out['haloes']['A'][name].items()}, flush=True)
    for h in ('B', 'C'):
        refs, spread = references(h)
        out['haloes'][h].update(sigma_hat=IH.halo_units(h)['sigma_hat'], references=refs, **spread)
        for name, (a, c) in CLOSURES.items():
            res, traj = halo_bc(h, a, c, refs)
            out['haloes'][h][name] = res
            np.savez(os.path.join(OUT, f'fluid_{h}_{name}.npz'), **traj)
            print(h, name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
                  flush=True)
    json.dump(out, open(os.path.join(OUT, 'transfer.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
