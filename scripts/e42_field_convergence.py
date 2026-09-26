#!/usr/bin/env python3
"""E42 -- does the reference FIELD converge as well as its clock?

E41 measured the convergence of one scalar, the collapse time, and found clean
second order with the working grid short by about 1.2%. That is not enough. The
displacement

    F = || G(v~) - G(v) ||

is a norm over the whole FIELD, not a clock error, so what it inherits is the
field's discretisation error at matched phase. A reference whose event time
converges beautifully can still carry a poorly converged profile, and the
diagnostic at the centre of this paper would inherit the second while the paper
quoted the first.

This is the check the 18 September audit named as its own immediate next task,
run here on our solver and at our grids.

METHOD
------
Run the same configuration at 150/300/600/1200 shells. Each run has its own
radial grid AND its own collapse time, so neither axis can be compared raw.
Both are put on common ground the same way the displacement itself is computed:

  * time   -- compare at matched PHASE tau = t/t_c, not matched t, because a
              coarse run collapses earlier and comparing at equal t would score
              two different evolutionary states against each other;
  * radius -- interpolate log10(rho) onto one log-spaced grid spanning the
              radii all four runs resolve, so no run is scored where it has no
              shells.

The reported error is the RMS of log10(rho) differences against the 1200-shell
run, in dex, which is the unit the paper reports density accuracy in.

Output: results/e40/field_convergence.json
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import solver                                         # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e40')
SHELLS = (150, 300, 600, 1200)
MAX_SNAPS = 6000          # must not bind; asserted in run()
MAX_TAU_GAP = 0.02        # refuse to interpolate across a wider gap
PHASES = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
CASES = [
    {'label': 'thin',   'sigma_m': 0.06, 'w': 1.5},
    {'label': 'middle', 'sigma_m': 0.50, 'w': 3.0},
    {'label': 'opaque', 'sigma_m': 2.00, 'w': 8.0},
]


def run(sh, wh, N):
    t0 = time.time()
    r = solver.run_halo(sigma_m=sh, w=wh, model_lmfp='yukawa_tu',
                        model_smfp='yukawa_tu', C=0.797, n_shells=N,
                        eps_t=3e-4, rho_c_factor=300.0, t_end=1e5,
                        kn_min=0.02, max_steps=20_000_000, max_snaps=MAX_SNAPS,
                        timestep_mode='implicit')
    # The snapshot budget is a trap, and the first version of this script fell
    # into it. Snapshots stop being written the moment `isnap` reaches
    # max_snaps - 1; the run continues but records nothing until the final
    # snapshot at collapse. With 400 the budget was exhausted mid-run and left
    # a gap of 0.32 in tau -- a third of the evolution -- across which a linear
    # interpolation produced a spurious 1.16 dex "convergence failure" that was
    # non-monotone in resolution, which is how it was caught. Refuse to report
    # a number when the budget was binding.
    assert len(r['t']) < MAX_SNAPS - 1, (
        f'snapshot budget binding at N={N}: {len(r["t"])} of {MAX_SNAPS}; '
        'raise MAX_SNAPS, do not interpolate across the gap')
    return r, time.time() - t0


def cadence_ok(res, phase):
    """Is the snapshot spacing at this phase small enough to interpolate?"""
    t = res['t']
    j = int(np.clip(np.searchsorted(t, phase * t[-1]), 1, len(t) - 1))
    return float((t[j] - t[j - 1]) / t[-1])


def profile_at_phase(res, phase, lr_grid):
    """log10(rho) on `lr_grid` at tau = phase, interpolated in time then radius.

    Time first: the two bracketing snapshots are interpolated in log density at
    each of that run's OWN shells, because a shell is a Lagrangian mass element
    and its radius moves. Radius second.
    """
    t = res['t']
    tc = t[-1]
    target = phase * tc
    j = int(np.clip(np.searchsorted(t, target), 1, len(t) - 1))
    t0, t1 = t[j - 1], t[j]
    f = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
    lrho = np.log10(np.maximum((1 - f) * res['rho'][j - 1] + f * res['rho'][j],
                               1e-300))
    lr = np.log10(np.maximum((1 - f) * res['r'][j - 1] + f * res['r'][j],
                             1e-300))
    o = np.argsort(lr)
    return np.interp(lr_grid, lr[o], lrho[o], left=np.nan, right=np.nan)


def main():
    out = {
        'question': 'the displacement is a field norm, so it inherits the '
                    'reference FIELD error at matched phase, not the clock '
                    'error E41 measured',
        'method': {
            'time': 'compared at matched phase tau = t/t_c, not matched t, '
                    'because a coarse run collapses earlier',
            'radius': 'interpolated onto one log grid spanning the radii all '
                      'four resolutions resolve',
            'metric': 'RMS of log10(rho) differences vs 1200 shells, in dex',
        },
        'phases': PHASES.tolist(),
        'cases': [],
    }
    for c in CASES:
        runs = {}
        for N in SHELLS:
            r, wall = run(c['sigma_m'], c['w'], N)
            runs[N] = r
        # a radial window every resolution actually covers, at every phase
        lo, hi = -np.inf, np.inf
        for N in SHELLS:
            for ph in PHASES:
                t = runs[N]['t']
                j = int(np.clip(np.searchsorted(t, ph * t[-1]), 1, len(t) - 1))
                lr = np.log10(np.maximum(runs[N]['r'][j], 1e-300))
                lo, hi = max(lo, lr.min()), min(hi, lr.max())
        grid = np.linspace(lo, hi, 200)
        row = {'label': c['label'], 'sigma_m': c['sigma_m'], 'w': c['w'],
               'log_r_window': [float(lo), float(hi)], 'by_phase': {}}
        ref = {ph: profile_at_phase(runs[1200], ph, grid) for ph in PHASES}
        for ph in PHASES:
            gaps = {N: cadence_ok(runs[N], ph) for N in SHELLS}
            if max(gaps.values()) > MAX_TAU_GAP:
                row['by_phase'][f'{ph:.2f}'] = {
                    'skipped': 'snapshot gap too wide to interpolate',
                    'max_tau_gap': float(max(gaps.values()))}
                continue
            e = {'max_tau_gap': float(max(gaps.values()))}
            for N in SHELLS[:-1]:
                d = profile_at_phase(runs[N], ph, grid) - ref[ph]
                d = d[np.isfinite(d)]
                e[str(N)] = float(np.sqrt(np.mean(d ** 2))) if d.size else None
            row['by_phase'][f'{ph:.2f}'] = e
        # convergence order per phase from 150/300/600
        orders = []
        for ph in PHASES:
            v = row['by_phase'][f'{ph:.2f}']
            if v.get('150') and v.get('300') and v.get('600') and v['600'] > 0:
                orders.append(float(np.log2(v['150'] / v['300'])))
        row['observed_order_150_300'] = float(np.median(orders)) if orders else None
        out['cases'].append(row)
        print(f"{c['label']:8s} " + '  '.join(
            f"tau={ph:.2f}: "
            + (f"{row['by_phase'][f'{ph:.2f}']['150']:.2e} dex"
               if row['by_phase'][f'{ph:.2f}'].get('150') else 'skipped')
            for ph in PHASES), flush=True)

    pick = lambda N, sel: [c['by_phase'][f'{ph:.2f}'][N] for c in out['cases']
                           for ph in PHASES
                           if sel(ph) and c['by_phase'][f'{ph:.2f}'].get(N)]
    e150 = pick('150', lambda ph: True)
    e300 = pick('300', lambda ph: True)
    late = pick('150', lambda ph: ph >= 0.9)
    out['phases_skipped_for_cadence'] = [
        f'{ph:.2f}' for c in out['cases'] for ph in PHASES
        if 'skipped' in c['by_phase'][f'{ph:.2f}']]
    out['summary'] = {
        'median_field_error_150_dex': float(np.median(e150)),
        'max_field_error_150_dex': float(np.max(e150)),
        'median_field_error_300_dex': float(np.median(e300)),
        'median_field_error_150_late_phase_dex': (float(np.median(late))
                                                  if late else None),
        'median_observed_order': float(np.median(
            [c['observed_order_150_300'] for c in out['cases']
             if c['observed_order_150_300'] is not None])),
        'compare_emulator_accuracy_dex': 0.0197,
        'compare_clock_error_at_150_shells': 0.0113,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'field_convergence.json'), 'w'),
              indent=1)
    s = out['summary']
    print(f"\nfield error at 150 shells: median {s['median_field_error_150_dex']:.2e} dex, "
          f"max {s['max_field_error_150_dex']:.2e}, late-phase median "
          f"{s['median_field_error_150_late_phase_dex']:.2e}")
    print(f"at 300 shells: median {s['median_field_error_300_dex']:.2e} dex")
    print(f"observed order {s['median_observed_order']:.2f}; "
          f"for scale the emulator reaches {s['compare_emulator_accuracy_dex']} dex")


if __name__ == '__main__':
    main()
