"""Grid convergence of t_sing for the four cross-domain systems.

Everything in the cross-domain study is measured relative to t_sing, so t_sing
has to be a property of the PDE and not of the discretisation. This script
re-integrates the same parameter draws at half and double the production
integration resolution and reports the relative change in t_sing and in the
stored fields.

The threshold amplification is fixed, so this is a convergence test of the time
at which a fixed, resolvable amplification is reached -- not of the blow-up time
itself, which for these systems is only approached asymptotically.

Two samples are reported per system: `--n` random draws from the same prior the
training data comes from, and a `--hard` subset selected as the most difficult
of `--screen` fresh candidates. The random sample says whether the reference is
good enough typically; the hard subset says how bad it gets where the solver
struggles, which is the question a median cannot answer.

Usage: sc_convergence.py [--n 24] [--screen 48] [--hard 8]
Output: results/statecond/convergence.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from statecond import data as scd            # noqa: E402
from statecond.pdes import SYSTEMS           # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'statecond')


def run_at(name, p, n_solve, tol_factor=1.0):
    s = SYSTEMS[name]()
    s.n_solve = n_solve
    if tol_factor != 1.0:
        s.rtol, s.atol = s.rtol * tol_factor, s.atol * tol_factor
    return s.solve(p)


def t_sing_at(tr, factor):
    """Recompute t_sing for a different event threshold, from a stored run.

    `_finish` defines t_sing by log-interpolating A(t) to `blowup_factor*A(0)`.
    Any other threshold is therefore readable off the SAME trajectory at no
    extra cost, which makes event-threshold independence free to test: if the
    convergence conclusion depended on where the threshold was placed, that
    would mean t_sing is an artefact of the stopping rule rather than a
    property of the dynamics.
    """
    a, ts = tr.a, tr.t
    thresh = a[0] * factor
    reached = np.nonzero(a >= thresh)[0]
    if not reached.size or reached[0] == 0:
        return None
    j = int(reached[0])
    l0, l1 = np.log(a[j - 1]), np.log(a[j])
    frac = (np.log(thresh) - l0) / max(l1 - l0, 1e-300)
    return float(ts[j - 1] + frac * (ts[j] - ts[j - 1]))


def _stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not v.size:
        return None
    return {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
            'max': float(v.max()), 'n': int(v.size)}


def measure(name, cls, base, p, trs=None):
    """Full three-grid + tolerance + threshold row for one parameter draw.

    `trs` lets a caller that has already paid for the cheap grids reuse them,
    which is what makes the difficulty screen affordable.
    """
    trs = dict(trs or {})
    for m in (base // 2, base, 2 * base):
        if m not in trs:
            trs[m] = run_at(name, p, m)
    if any(t is None or t.status != 0 for t in trs.values()):
        return None
    ref = trs[2 * base]
    row = {'t_sing_ref': ref.t_sing}
    for m in (base // 2, base):
        row[f'dt_{m}'] = abs(trs[m].t_sing - ref.t_sing) / ref.t_sing
        u_m, _ = scd.resample(trs[m])
        u_r, _ = scd.resample(ref)
        row[f'du_{m}'] = float(np.abs(u_m - u_r).mean())

    # (a) TEMPORAL/tolerance refinement, independent of the grid: same
    # production resolution, integrator tolerances 100x tighter. If t_sing
    # moved here the spatial study would be measuring the wrong thing.
    tt = run_at(name, p, base, tol_factor=1e-2)
    row['dt_tol'] = (abs(tt.t_sing - trs[base].t_sing) / trs[base].t_sing
                     if tt is not None and tt.status == 0 else None)

    # (b) EVENT-THRESHOLD independence, free: re-read t_sing off the same
    # trajectories at a third and at three times the threshold.
    bf = cls.blowup_factor
    for tag, fac in (('lo', bf / 3.0), ('hi', bf * 3.0)):
        x, y = t_sing_at(trs[base], fac), t_sing_at(ref, fac)
        row[f'dt_thresh_{tag}'] = (abs(x - y) / y
                                   if x is not None and y is not None else None)
    return row


def hard_subset(name, cls, base, rng, n_screen, n_hard):
    """The difficult end of the prior, not a representative sample of it.

    A random draw answers "is the reference good enough on a typical
    trajectory?". It does not answer "how bad does it get where the solver is
    struggling?", and a benchmark that only reports the typical case invites
    exactly that question. Difficulty is screened on the TWO CHEAP grids --
    disagreement between half and production resolution -- so that the
    expensive 2x reference is paid only for the worst `n_hard` of `n_screen`
    candidates. The screen never looks at the reference it is selecting for,
    so it cannot manufacture the answer.
    """
    cand = []
    for _ in range(n_screen):
        p = cls().sample(rng)
        trs = {m: run_at(name, p, m) for m in (base // 2, base)}
        if any(t is None or t.status != 0 for t in trs.values()):
            continue
        score = abs(trs[base].t_sing - trs[base // 2].t_sing) / trs[base].t_sing
        cand.append((score, p, trs))
    cand.sort(key=lambda c: -c[0])
    rows = []
    for _, p, trs in cand[:n_hard]:
        r = measure(name, cls, base, p, trs)
        if r is not None:
            rows.append(r)
    return rows, len(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--screen', type=int, default=48,
                    help='candidates screened for the difficult subset')
    ap.add_argument('--hard', type=int, default=8,
                    help='hardest candidates carried to the full study')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    out = {}
    for name, cls in SYSTEMS.items():
        base = cls().n_solve
        rng = np.random.default_rng(9999)
        rows = []
        for _ in range(args.n):
            row = measure(name, cls, base, cls().sample(rng))
            if row is not None:
                rows.append(row)
        if not rows:
            out[name] = {'n': 0}
            continue
        out[name] = {
            'n': len(rows),
            'n_solve_production': base,
            'reference_grid': 2 * base,
            'blowup_factor': float(cls.blowup_factor),
            'median_rel_dt_sing_half': float(
                np.median([r[f'dt_{base // 2}'] for r in rows])),
            'median_rel_dt_sing_production': float(
                np.median([r[f'dt_{base}'] for r in rows])),
            'median_field_dex_half': float(
                np.median([r[f'du_{base // 2}'] for r in rows])),
            'median_field_dex_production': float(
                np.median([r[f'du_{base}'] for r in rows])),
            # the referee asked for tails, not medians alone
            'rel_dt_sing_production_stats': _stats(
                [r[f'dt_{base}'] for r in rows]),
            'field_dex_production_stats': _stats(
                [r[f'du_{base}'] for r in rows]),
            'tolerance_refinement': _stats([r['dt_tol'] for r in rows]),
            'threshold_third': _stats([r['dt_thresh_lo'] for r in rows]),
            'threshold_triple': _stats([r['dt_thresh_hi'] for r in rows]),
        }

        # the difficult end of the prior, screened on the cheap grids
        if args.hard:
            hrows, n_ok = hard_subset(name, cls, base,
                                      np.random.default_rng(4242),
                                      args.screen, args.hard)
            if hrows:
                out[name]['hard_subset'] = {
                    'n': len(hrows),
                    'n_screened': n_ok,
                    'selection': ('largest half-vs-production disagreement; '
                                  'the 2x reference is never used to select'),
                    'rel_dt_sing_production_stats': _stats(
                        [r[f'dt_{base}'] for r in hrows]),
                    'field_dex_production_stats': _stats(
                        [r[f'du_{base}'] for r in hrows]),
                    'tolerance_refinement': _stats(
                        [r['dt_tol'] for r in hrows]),
                }
        o = out[name]
        print(f"{name:8s} n={o['n']:3d}  |dt_sing|/t_sing: "
              f"half {o['median_rel_dt_sing_half']:.2e} -> "
              f"production {o['median_rel_dt_sing_production']:.2e}   "
              f"field: {o['median_field_dex_half']:.2e} -> "
              f"{o['median_field_dex_production']:.2e}", flush=True)

    json.dump(out, open(os.path.join(OUT, 'convergence.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
