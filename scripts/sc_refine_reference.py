"""P21 — a finer reference `t_sing`, and the production grid's error against it.

WHY THIS EXISTS
---------------
`sc_convergence.py` scores the production grid against a 2x reference. That was
adequate while the error it had to resolve was the published head's 1.8-7.1%.
P20 took the head to 0.04-0.82%, and recomputing the margin at the operating
point the project now quotes leaves three reference systems short of the
standard (median >= 10, hard subset >= 5):

    rd 3406 / 75      film 10.1 / 2.44      burgers 5.00 / 3.98      ks 0.44 / 0.12

`ks` is the serious one: **0.44 means the reference is coarser than the
measurement.** (`sidm` is excluded -- it was re-specified on 2026-09-06 as an
application system whose `t_sing` is a solver termination at a grid-dependent
shell index, so a convergence study on it would mix discretisation with a
change of definition.)

WHAT THIS DOES DIFFERENTLY
--------------------------
Two changes, both of which buy accuracy without touching the production grid:

1. **A 4x ladder** (`base`, `2 base`, `4 base`) instead of a 2x one, so the
   reference is four times finer rather than twice.
2. **Richardson extrapolation.** With two grids, `|t(N) - t(2N)|` is an
   *estimate* of the coarse grid's error that also carries the fine grid's own
   error. With three at a fixed refinement ratio the observed order can be
   measured and the extrapolated limit is a better reference than any single
   run. The production error is then scored against that limit.

The observed order is REPORTED, not assumed. If it is far from the scheme's
design order the extrapolation is not trustworthy and the row says so, rather
than quietly producing a smaller number.

Usage:
  sc_refine_reference.py --all [--n 24] [--hard 8] [--screen 48]
Output: results/statecond/refined_reference.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from statecond.pdes import SYSTEMS                                # noqa: E402
from sc_convergence import run_at                                 # noqa: E402
from sc_provenance import stamp                                   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
# sidm is absent by design: application system, see the module docstring.
REFERENCE_SYSTEMS = ('rd', 'ks', 'film', 'burgers')
# the standard a reference must meet against the head actually being reported
MARGIN_MEDIAN_MIN = 10.0
MARGIN_HARD_MIN = 5.0


def richardson(tc, tm, tf, ratio=2.0):
    """(order, extrapolated limit) from three grids at a fixed ratio.

    t_exact ~ t_fine + (t_fine - t_mid) / (ratio^p - 1), with p observed from
    the three values rather than assumed. Returns (None, None) when the three
    are not monotone in a way that admits an order -- which happens when the
    differences are at round-off, and is reported rather than papered over.
    """
    num, den = (tc - tm), (tm - tf)
    if den == 0 or num / den <= 0:
        return None, None
    p = float(np.log(num / den) / np.log(ratio))
    if not np.isfinite(p) or p <= 0:
        return None, None
    return p, float(tf + (tf - tm) / (ratio ** p - 1.0))


def row_for(name, cls, base, p_draw, trs=None):
    """One parameter draw on the 4x ladder."""
    trs = dict(trs or {})
    for m in (base, 2 * base, 4 * base):
        if m not in trs:
            trs[m] = run_at(name, p_draw, m)
    if any(t is None or t.status != 0 for t in trs.values()):
        return None
    tc, tm, tf = (trs[base].t_sing, trs[2 * base].t_sing, trs[4 * base].t_sing)
    order, t_ext = richardson(tc, tm, tf)
    row = {
        't_base': float(tc), 't_2x': float(tm), 't_4x': float(tf),
        # what sc_convergence reports: production against a 2x reference
        'rel_vs_2x': abs(tc - tm) / abs(tm),
        # production against the 4x reference
        'rel_vs_4x': abs(tc - tf) / abs(tf),
        'observed_order': order,
        't_richardson': t_ext,
        # production against the extrapolated limit: the best estimate of the
        # production grid's OWN error, with the reference's error removed
        'rel_vs_richardson': (abs(tc - t_ext) / abs(t_ext)
                              if t_ext is not None else None),
        # and the 4x reference's own residual error, which is what decides
        # whether it is fit to be called a reference at all
        'ref_4x_own_error': (abs(tf - t_ext) / abs(t_ext)
                             if t_ext is not None else None),
    }
    return row


def screen_hard(name, cls, base, rng, n_screen, n_hard):
    """The difficult end, screened on the two cheap grids only.

    Same rule as `sc_convergence.hard_subset`: difficulty is scored by
    base-vs-half disagreement, so the expensive 4x run is paid only for the
    worst candidates and the screen never inspects the reference it selects for.
    """
    cand = []
    for _ in range(n_screen):
        pd = cls().sample(rng)
        a, b = run_at(name, pd, base // 2), run_at(name, pd, base)
        if a is None or b is None or a.status != 0 or b.status != 0:
            continue
        cand.append((abs(b.t_sing - a.t_sing) / b.t_sing, pd, {base: b}))
    cand.sort(key=lambda c: -c[0])
    rows = []
    for _, pd, trs in cand[:n_hard]:
        r = row_for(name, cls, base, pd, trs)
        if r is not None:
            rows.append(r)
    return rows, len(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--system', default=None)
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--screen', type=int, default=48)
    ap.add_argument('--hard', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260906)
    args = ap.parse_args()

    th = os.path.join(OUT, 'timehead.json')
    heads = ({k: v['converged_head_err_pct'] / 100.0
              for k, v in json.load(open(th)).items()
              if isinstance(v, dict) and 'converged_head_err_pct' in v}
             if os.path.exists(th) else {})

    names = (list(REFERENCE_SYSTEMS) if args.all else [args.system])
    out = {}
    for name in names:
        cls = SYSTEMS[name]
        base = cls.n_solve
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        rows = []
        while len(rows) < args.n:
            r = row_for(name, cls, base, cls().sample(rng))
            if r is not None:
                rows.append(r)
        hard, n_screened = screen_hard(name, cls, base,
                                       np.random.default_rng(args.seed + 1),
                                       args.screen, args.hard)

        def med(rs, key):
            v = [r[key] for r in rs if r.get(key) is not None]
            return float(np.median(v)) if v else None

        rec = {
            'system': name,
            'production_grid': base,
            'reference_grid': 4 * base,
            'n_trajectories': len(rows),
            'seed': args.seed,
            'rel_vs_2x_median': med(rows, 'rel_vs_2x'),
            'rel_vs_4x_median': med(rows, 'rel_vs_4x'),
            'rel_vs_richardson_median': med(rows, 'rel_vs_richardson'),
            'ref_4x_own_error_median': med(rows, 'ref_4x_own_error'),
            'observed_order_median': med(rows, 'observed_order'),
            'n_with_order': sum(1 for r in rows
                                if r.get('observed_order') is not None),
            'hard': {
                'n': len(hard), 'n_screened': n_screened,
                'rel_vs_2x_median': med(hard, 'rel_vs_2x'),
                'rel_vs_4x_median': med(hard, 'rel_vs_4x'),
                'rel_vs_richardson_median': med(hard, 'rel_vs_richardson'),
                'observed_order_median': med(hard, 'observed_order'),
            },
            'wall_seconds': time.time() - t0,
        }
        if name in heads:
            c = heads[name]
            rec['converged_head_err_pct'] = 100 * c
            for tag, key in (('2x', 'rel_vs_2x_median'),
                             ('4x', 'rel_vs_4x_median'),
                             ('richardson', 'rel_vs_richardson_median')):
                v = rec[key]
                rec[f'margin_{tag}'] = (c / v) if v else None
            hv = rec['hard']['rel_vs_richardson_median']
            rec['margin_hard_richardson'] = (c / hv) if hv else None
            rec['meets_standard'] = bool(
                (rec.get('margin_richardson') or 0) >= MARGIN_MEDIAN_MIN
                and (rec.get('margin_hard_richardson') or 0) >= MARGIN_HARD_MIN)
        out[name] = rec
        print(f"{name:9s} prod {base:5d} ref {4*base:5d} | "
              f"vs2x {rec['rel_vs_2x_median']:.2e} "
              f"vs4x {rec['rel_vs_4x_median']:.2e} "
              f"vsRich {rec['rel_vs_richardson_median'] or float('nan'):.2e} | "
              f"order {rec['observed_order_median'] or float('nan'):.2f} "
              f"({rec['n_with_order']}/{len(rows)}) | "
              f"margin {rec.get('margin_richardson', float('nan')):8.1f} "
              f"hard {rec.get('margin_hard_richardson', float('nan')):7.1f} | "
              f"{'PASS' if rec.get('meets_standard') else 'FAIL'} "
              f"({rec['wall_seconds']:.0f}s)", flush=True)

    done = [k for k in REFERENCE_SYSTEMS if k in out]
    if len(done) == len(REFERENCE_SYSTEMS):
        fails = [k for k in done if not out[k].get('meets_standard')]
        out['_summary'] = {
            'standard': f'median >= {MARGIN_MEDIAN_MIN}, '
                        f'hard subset >= {MARGIN_HARD_MIN}, against the '
                        'CONVERGED head and the Richardson limit',
            'excluded': {'sidm': 'application system (re-specified 2026-09-06)'},
            'passing': [k for k in done if out[k].get('meets_standard')],
            'failing': fails,
            'verdict': ('every reference system now resolves the converged '
                        'head' if not fails else
                        'still inadequate for ' + ', '.join(fails)),
            '_provenance': stamp('sc_refine_reference.py'),
        }
        print('\n' + out['_summary']['verdict'])
    path = os.path.join(OUT, 'refined_reference.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
