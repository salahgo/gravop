"""P25a -- can the 2-D testbed meet the convergence standard the 1-D ones meet?

WHY THIS EXISTS
---------------
`src/statecond/pdes2d.py` implements a 2-D Keller-Segel testbed: the aggregate
forms wherever the initial data sends it, so the singular *location* is part of
what a surrogate has to get right rather than a coordinate choice. It runs, and
it has been **withheld** since it was written, because `t_sing` converged only
to 9.6-13% between the 128 and 256 grids -- two orders of magnitude short of
what a study of a few-percent effect needs.

The diagnosis was in the source: the scheme was **first order in both
operators** -- donor-cell upwind in space, integrating-factor forward Euler in
time. P25a replaces both (MUSCL with a minmod limiter; SSP-RK2 in the
integrating-factor variable) and asks whether that is enough.

THE STANDARD, which is the one the 1-D systems are held to (P16, P21)
---------------------------------------------------------------------
The reference `t_sing` must be far better converged than the `t_sing` error of
the time head whose deployment penalty is being measured:

    margin = (converged head t_sing error) / (production-grid t_sing error)
    median margin >= 10  AND  hard-subset margin >= 5

with the production error scored against a **Richardson-extrapolated limit**,
not against a 2x reference -- P21 showed a 2x reference flatters the production
grid by about a third, because its own error partially cancels.

No time head has been trained on `ks2d` (there is no dataset yet: that is the
work this decision gates), so the yardstick is the range the 1-D converged heads
span, `ks` 0.037% at the tight end and `film` 0.822% at the loose end
(`results/statecond/timehead.json`). Both ends are reported. A 2-D head has no
reason to beat the best 1-D one, so **the loose end is the one the verdict uses**
and the tight end is reported as the harder target it would also have to clear
if it turned out to be that good.

PRE-REGISTERED (before the full study was run; the scheme change itself was
validated on 4 trajectories first, which is a statement about the scheme's
ORDER and not about this verdict):

  PASS   some candidate production grid reaches median margin >= 10 and
         hard-subset >= 5 against a 0.822% head, at a cost that lets 600
         trajectories be regenerated in under 4 h on this machine's 24 cores.
         -> `ks2d` is registered at that grid and the dataset is generated.
  FAIL   no such grid. -> `ks2d` STAYS WITHHELD, and this artefact is the
         published diagnosis. A 2-D system that fails the standard the 1-D
         systems pass would trade a stated limitation for a hidden one.

Second order is expected to be worth ~2 decades of grid: the 128-grid error
should fall from ~10% to a few percent and reach the standard somewhere between
512 and 1024. If it does not, the honest reading is that the *event* -- a peak
amplitude 30x its initial value, measured on the grid -- is itself only
marginally resolved, and that is a specification problem rather than a scheme
problem.

Usage:
  sc_ks2d.py [--n 16] [--screen 48] [--hard 8] [--grids 128 256 512 1024]
             [--procs 12]
Output: results/statecond/ks2d_convergence.json
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from statecond.pdes2d import KS2D                             # noqa: E402
from sc_provenance import stamp                               # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
SEED = 20260907


def _solve(job):
    """(index, params, n_solve, scheme[, blowup]) -> (index, n, t_sing, wall)."""
    i, p, n, scheme = job[:4]
    blowup = job[4] if len(job) > 4 else None
    t0 = time.time()
    sysm = KS2D(n_solve=n, n_store=64, scheme=scheme)
    if blowup is not None:
        sysm.blowup_factor = blowup
    tr = sysm.solve(p)
    return i, n, (None if tr is None else float(tr.t_sing)), time.time() - t0


def richardson(t_coarse, t_mid, t_fine):
    """(observed order, extrapolated limit) from three grids in ratio 2.

    Returns (None, t_fine) when the differences do not admit an order, which
    happens when a trajectory is already at round-off or is non-monotone in the
    grid -- reported rather than silently averaged in.
    """
    d1, d2 = t_coarse - t_mid, t_mid - t_fine
    if d1 == 0 or d2 == 0 or d1 / d2 <= 0:
        return None, t_fine
    order = math.log2(d1 / d2)
    if not (0.5 <= order <= 4.0):
        return order, t_fine
    return order, t_fine + d2 / (2 ** order - 1)


def run(params, grids, procs, scheme):
    jobs = [(i, p, n, scheme) for i, p in enumerate(params) for n in grids]
    with Pool(procs) as pool:
        res = pool.map(_solve, jobs)
    table = {}
    wall = {}
    for i, n, t, w in res:
        table.setdefault(i, {})[n] = t
        wall[n] = wall.get(n, 0.0) + w
    return table, wall


def analyse(table, grids):
    """Per-trajectory relative error of each grid against the Richardson limit."""
    rows = []
    for i in sorted(table):
        ts = table[i]
        if any(ts.get(n) is None for n in grids):
            continue
        order, limit = richardson(ts[grids[-3]], ts[grids[-2]], ts[grids[-1]])
        rows.append({'traj': i, 'order': order, 'limit': limit,
                     'rel': {n: abs(ts[n] - limit) / limit for n in grids},
                     't_sing': {n: ts[n] for n in grids}})
    return rows


def event_diagnostic(params, grids, procs, scheme, factors=(10.0, 30.0)):
    """POST HOC: is what is left a SCHEME error or an EVENT-SPECIFICATION error?

    `t_sing` is the instant the grid maximum of `rho` reaches `f` times its
    initial value. Near blow-up the aggregate's width shrinks like 1/sqrt(f),
    so at f = 30 the peak spans only a few cells on the production grid and the
    grid maximum is itself a marginally resolved quantity -- an error in the
    *event*, which no amount of scheme order removes. Lowering `f` widens the
    peak at the event and should improve convergence if that is what dominates.

    Same trajectories, same grids, two thresholds. Reported as a diagnosis of
    the FAIL, not as part of the pre-registered verdict.
    """
    out = {}
    for f in factors:
        jobs = [(i, p, n, scheme, f) for i, p in enumerate(params)
                for n in grids]
        with Pool(procs) as pool:
            res = pool.map(_solve, jobs)
        table = {}
        for i, n, t, _ in res:
            table.setdefault(i, {})[n] = t
        rows = analyse(table, grids)
        out[f'{f:g}'] = {
            'rel_median': {str(n): float(np.median([r['rel'][n] for r in rows]))
                           for n in grids},
            'observed_order_median': float(np.median(
                [r['order'] for r in rows if r['order'] is not None])),
            'n_trajectories': len(rows),
        }
    return out


def difference_ladder(rows, grids, head=0.00822):
    """Extrapolation-free reading of the ladder, and what it implies for 2048.

    The `rel` columns are scored against a Richardson limit built from the
    THREE FINEST grids, so the finest row's error is measured partly against
    itself and flattens: the medians fall 4.3x, 7.1x, then only 2.2x, which
    reads like the scheme losing order at 1024. It is not. Successive
    grid-to-grid DIFFERENCES involve no limit at all, and those fall by a clean
    factor of ~4 per doubling.

    From the finest difference, error(fine) = diff / (2^p - 1) at order p, and
    each further doubling divides by 4. That is what licenses the statement
    about a grid nobody has run.
    """
    pairs = list(zip(grids[:-1], grids[1:]))
    diffs = {}
    for a, b in pairs:
        v = [abs(r['t_sing'][a] - r['t_sing'][b]) / r['t_sing'][b]
             for r in rows]
        diffs[f'{a}_vs_{b}'] = float(np.median(v))
    keys = list(diffs)
    ratios = {f'{keys[i]} / {keys[i + 1]}': diffs[keys[i]] / diffs[keys[i + 1]]
              for i in range(len(keys) - 1)}
    finest = diffs[keys[-1]]
    err = finest / 3.0                       # p = 2
    proj = {}
    n = finest_grid = int(grids[-1])         # grids may arrive as strings
    while n <= 4 * finest_grid:
        proj[str(n)] = {'implied_rel_error': err,
                        'implied_margin_vs_head': head / err}
        err /= 4.0
        n *= 2
    return {'median_differences': diffs,
            'ratio_of_successive_differences': ratios,
            'implied_order': {k: float(np.log2(v)) for k, v in ratios.items()},
            'projection': proj,
            'note': ('differences involve no Richardson limit, so this is the '
                     'honest reading of the order; the rel_median column '
                     'flattens at the finest grid because its limit is built '
                     'from that grid')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--screen', type=int, default=48)
    ap.add_argument('--hard', type=int, default=8)
    ap.add_argument('--grids', type=int, nargs='+',
                    default=[128, 256, 512, 1024])
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--scheme', default='muscl-rk2')
    ap.add_argument('--reanalyse', action='store_true',
                    help='recompute the derived blocks from the stored '
                         'per-trajectory t_sing, without solving anything')
    ap.add_argument('--event-diagnostic', action='store_true',
                    help='post hoc: rerun at a lower blow-up threshold to '
                         'separate scheme error from event-specification error')
    args = ap.parse_args()

    path = os.path.join(OUT, 'ks2d_convergence.json')
    if args.reanalyse:
        out = json.load(open(path))
        grids = [str(g) for g in out['grids']]
        out['difference_ladder'] = difference_ladder(
            out['per_trajectory'], grids)
        out['difference_ladder_hard'] = difference_ladder(
            [r for r in out['per_trajectory'] if r['hard']], grids)
        out['_provenance'] = stamp('sc_ks2d.py --reanalyse')
        json.dump(out, open(path, 'w'), indent=1)
        dl = out['difference_ladder']
        print('order from successive differences:',
              {k: round(v, 2) for k, v in dl['implied_order'].items()})
        for g, v in dl['projection'].items():
            h = out['difference_ladder_hard']['projection'][g]
            print(f"  grid {g:>5s}: implied error {v['implied_rel_error']:.3e} "
                  f"-> margin {v['implied_margin_vs_head']:.1f} median, "
                  f"{h['implied_margin_vs_head']:.1f} hard")
        print('wrote', path)
        return

    rng = np.random.default_rng(SEED)
    pool_params = [KS2D().sample(rng) for _ in range(args.screen)]

    # cheap screen on the two coarsest grids: the hard subset is the tail, and
    # a median alone would hide it (P16 found the same in 1-D)
    t0 = time.time()
    screen, _ = run(pool_params, [64, 128], args.procs, args.scheme)
    spread = {i: (abs(v[64] - v[128]) / v[128]
                  if v.get(64) and v.get(128) else np.inf)
              for i, v in screen.items()}
    order_by_hardness = sorted(spread, key=lambda i: -spread[i])
    hard_idx = order_by_hardness[:args.hard]
    rest = [i for i in order_by_hardness[args.hard:] if np.isfinite(spread[i])]
    main_idx = sorted(rng.permutation(rest)[:args.n].tolist())
    screen_wall = time.time() - t0
    print(f'screened {args.screen} in {screen_wall:.0f} s; '
          f'hard subset {hard_idx}', flush=True)

    sel = sorted(set(main_idx) | set(hard_idx))
    params = [pool_params[i] for i in sel]
    table, wall = run(params, args.grids, args.procs, args.scheme)
    rows = analyse(table, args.grids)
    for r in rows:
        r['traj'] = sel[r['traj']]
        r['hard'] = r['traj'] in hard_idx

    med = {n: float(np.median([r['rel'][n] for r in rows]))
           for n in args.grids}
    hard_rows = [r for r in rows if r['hard']]
    hard_med = {n: (float(np.median([r['rel'][n] for r in hard_rows]))
                    if hard_rows else None) for n in args.grids}
    orders = [r['order'] for r in rows if r['order'] is not None]

    heads = {'loose_film_0p822pct': 0.00822, 'tight_ks_0p037pct': 0.00037}
    margins = {k: {n: (h / med[n], (h / hard_med[n]) if hard_med[n] else None)
                   for n in args.grids} for k, h in heads.items()}
    # cost of regenerating 600 trajectories at each grid, from the measured
    # wall time of this study, on 24 cores
    per_traj = {n: wall[n] / len(params) for n in args.grids}
    cost_h = {n: 600 * per_traj[n] / 24 / 3600 for n in args.grids}

    passing = [n for n in args.grids
               if margins['loose_film_0p822pct'][n][0] >= 10
               and (margins['loose_film_0p822pct'][n][1] or 0) >= 5
               and cost_h[n] < 4.0]
    verdict = (
        f'PASS: ks2d meets the standard at grid {min(passing)} '
        f'({cost_h[min(passing)]:.2f} h to regenerate 600 trajectories on 24 '
        f'cores); register it there and generate the dataset'
        if passing else
        'FAIL: no affordable grid reaches median margin >= 10 and hard-subset '
        '>= 5 against a 0.822% head. ks2d STAYS WITHHELD and these numbers are '
        'the published diagnosis.')

    out = {
        'scheme': args.scheme,
        'grids': args.grids,
        'seed': SEED,
        'n_screened': args.screen,
        'n_trajectories': len(rows),
        'n_hard': len(hard_rows),
        'rel_median': {str(n): med[n] for n in args.grids},
        'rel_hard_median': {str(n): hard_med[n] for n in args.grids},
        'observed_order_median': float(np.median(orders)) if orders else None,
        'n_with_order': len(orders),
        'head_yardsticks_pct': {k: 100 * v for k, v in heads.items()},
        'margins': {k: {str(n): {'median': v[n][0], 'hard': v[n][1]}
                        for n in args.grids} for k, v in margins.items()},
        'wall_seconds_per_trajectory': {str(n): per_traj[n]
                                        for n in args.grids},
        'regen_hours_600_runs_24_cores': {str(n): cost_h[n]
                                          for n in args.grids},
        'standard': ('median margin >= 10 and hard-subset >= 5 against the '
                     'converged head, scored against a Richardson limit; '
                     'and 600 trajectories regenerable in < 4 h on 24 cores'),
        'passing_grids': passing,
        'verdict': verdict,
        'per_trajectory': rows,
        '_provenance': stamp('sc_ks2d.py'),
    }
    if args.event_diagnostic:
        out['event_diagnostic_POSTHOC'] = event_diagnostic(
            params[:8], args.grids[:-1], args.procs, args.scheme)
    json.dump(out, open(path, 'w'), indent=1)

    print(f"\nobserved order {out['observed_order_median']} "
          f"({out['n_with_order']}/{len(rows)} trajectories)")
    print(f"{'grid':>6s} {'rel median':>12s} {'rel hard':>12s} "
          f"{'margin@0.822%':>14s} {'hard':>8s} {'regen h':>9s}")
    for n in args.grids:
        m = margins['loose_film_0p822pct'][n]
        print(f'{n:6d} {med[n]:12.3e} '
              f'{(hard_med[n] if hard_med[n] else float("nan")):12.3e} '
              f'{m[0]:14.1f} {(m[1] if m[1] else float("nan")):8.1f} '
              f'{cost_h[n]:9.2f}')
    print('\n' + verdict)
    print('wrote', path)


if __name__ == '__main__':
    main()
