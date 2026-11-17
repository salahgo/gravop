"""E2 calibration fit -- the producer of the isolated-halo N-body calibration.

`results/e2/compare_production.json` (s_best = 0.945, 0.0197 dex) predates any
script that writes it. This script regenerates its inputs and its numbers and
adds the two checks the calibration needs:

1. The production fluid run: NFW, sigma-hat = 80 cm^2/g * rho_s r_s, 300 shells,
   eps_t = 3e-4, implicit conduction, to rho_c = 3e3 rho_c(0), C = 0.753. It is
   compared with the stored results/e2/fluid_run_production.npz bit for bit.
2. The time rescaling: the fluid model is scored at t_fluid = s * t_nbody over
   the calibration snapshots 010-045 (E2's convention: time interpolation
   between bracketing stored profiles, shell mid-radii, 0.05 < r/r_s < 10), on
   the 0.015-step grid that selected s = 0.945 and on a 0.001-step grid.
3. The conductivity itself: since t scales as 1/C in the long-mean-free-path
   regime, C_eff = C * s. The solver is run at that C directly and scored with
   no rescaling (s = 1), which tests the equivalence rather than assuming it.

Snapshot 051 is not in the fitted set, which was fixed at 010-045 before the
fit. Under the adopted s = 0.945 it falls after the fluid run's termination at
collapse; under the finer optimum it falls just before it. Both times are
recorded.

Output: results/e2/calibration_fit.json. compare_production.json is checked,
not rewritten.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import solver  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e2')
R_S, RHO_S, T0G = 1.28, 4.42e7, 0.02045991174237841
SIGMA_HAT = 0.94553884672
C_PROD = 0.753
FIT_RANGE = (0.05, 10.0)
CALIBRATION_SNAPSHOTS = ('snapshot_010', 'snapshot_020', 'snapshot_030',
                         'snapshot_040', 'snapshot_045')
RUN_KW = dict(sigma_m=SIGMA_HAT, n_shells=300, eps_t=3e-4,
              timestep_mode='implicit', rho_c_factor=3e3)


def midpoints(r):
    return 0.5 * (np.r_[r[0], r[:-1]] + r)


def score(run, d, s):
    """Per-snapshot RMS log-density residual at t_fluid = s * t_nbody, or None
    if a calibration snapshot falls after the end of the run."""
    ts = run['t']
    per = []
    for k in CALIBRATION_SNAPSHOTS:
        t = float(d[k + '_t']) / T0G * s
        if t > ts[-1]:
            return None
        i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        w = float(np.clip((t - ts[i]) / max(ts[i + 1] - ts[i], 1e-300), 0, 1))
        r = d[k + '_r'] / R_S
        rho = d[k + '_rho'] / (RHO_S * 1e-10)
        sel = (r > FIT_RANGE[0]) & (r < FIT_RANGE[1]) & (rho > 0)
        x = np.log10(r[sel])
        br = []
        for j in (i, i + 1):
            ok = run['rho'][j] > 0
            br.append(np.interp(x, np.log10(midpoints(run['r'][j])[ok]),
                                np.log10(run['rho'][j][ok])))
        e = (1 - w) * br[0] + w * br[1] - np.log10(rho[sel])
        per.append(float(np.sqrt(np.mean(e ** 2))))
    return per


def total(per):
    return float(np.sqrt(np.mean(np.square(per))))


def scan(run, d, grid):
    rows = [(float(s), score(run, d, s)) for s in grid]
    rows = [(s, total(p), p) for s, p in rows if p is not None]
    best = min(rows, key=lambda z: z[1])
    return {'grid': [float(grid[0]), float(grid[-1]), float(grid[1] - grid[0])],
            's_best': best[0], 'rms_dex_best': best[1],
            'per_snapshot_best': dict(zip(CALIBRATION_SNAPSHOTS, best[2])),
            'C_eff': C_PROD * best[0],
            'curve': [[s, r] for s, r, _ in rows]}


def main():
    d = np.load(os.path.join(OUT, 'nbody_profiles.npz'))
    run = solver.run_halo(C=C_PROD, **RUN_KW)
    stored = np.load(os.path.join(OUT, 'fluid_run_production.npz'))
    identical = (run['t'].shape == stored['t'].shape
                 and all(np.array_equal(run[k], stored[k])
                         for k in ('t', 'r', 'rho', 'p')))
    if not identical:
        raise SystemExit('the production fluid run does not reproduce '
                         'results/e2/fluid_run_production.npz')

    coarse = scan(run, d, 0.90 + 0.015 * np.arange(8))
    fine = scan(run, d, np.round(np.arange(0.900, 1.0005, 0.001), 3))
    ref = json.load(open(os.path.join(OUT, 'compare_production.json')))
    reproduced = (abs(coarse['s_best'] - ref['s_best']) < 1e-12
                  and abs(coarse['rms_dex_best'] - ref['rms_dex_best']) < 1e-12)
    if not reproduced:
        raise SystemExit(f'coarse scan gives s = {coarse["s_best"]}, '
                         f'{coarse["rms_dex_best"]}; compare_production.json '
                         f'has {ref["s_best"]}, {ref["rms_dex_best"]}')

    direct = {}
    for name, C in (('C_eff_coarse', round(C_PROD * coarse['s_best'], 3)),
                    ('C_eff_fine', round(C_PROD * fine['s_best'], 3)),
                    ('C_production', C_PROD)):
        r = run if C == C_PROD else solver.run_halo(C=C, **RUN_KW)
        p = score(r, d, 1.0)
        direct[name] = {'C': C, 't_final': float(r['t_final']),
                        'status': int(r['status']),
                        'rms_dex_no_rescaling': None if p is None else total(p),
                        'per_snapshot': None if p is None
                        else dict(zip(CALIBRATION_SNAPSHOTS, p))}
        print(name, direct[name]['C'], direct[name]['rms_dex_no_rescaling'],
              flush=True)

    out = {'reference': 'Fischer, Yu & Dolag 2025, arXiv:2506.06269, N = 5e7',
           'run': dict(RUN_KW, C=C_PROD),
           'calibration_snapshots': list(CALIBRATION_SNAPSHOTS),
           'fit_range_rs': list(FIT_RANGE),
           'not_scored': {'snapshot_051': {
               'reason': 'not in the fitted set (010-045, fixed before the fit)',
               't_fluid_at_s_coarse': float(d['snapshot_051_t']) / T0G * coarse['s_best'],
               't_fluid_at_s_fine': float(d['snapshot_051_t']) / T0G * fine['s_best'],
               'fluid_run_end': float(run['t'][-1])}},
           'fluid_run_reproduces_stored': identical,
           'compare_production_reproduced_by_coarse_scan': reproduced,
           'time_rescaling_coarse': coarse,
           'time_rescaling_fine': fine,
           'direct_conductivity': direct}
    json.dump(out, open(os.path.join(OUT, 'calibration_fit.json'), 'w'),
              indent=1)
    print('coarse', coarse['s_best'], coarse['rms_dex_best'],
          '| fine', fine['s_best'], fine['rms_dex_best'])
    print('wrote calibration_fit.json')


if __name__ == '__main__':
    main()
