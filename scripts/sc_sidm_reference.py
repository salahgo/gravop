"""P21 — a real convergence study for the gravothermal reference `t_sing`.

WHY THIS EXISTS
---------------
`sc_reference_accuracy.py` puts all five systems on one footing, and `sidm` is
by far the weakest entry in it:

  * the other four use 24 sampled trajectories at a **2x** grid refinement;
  * `sidm` uses **2** cross-sections at a **1.33x** refinement (N = 300 vs 400),
    borrowed from the E1 sweep, and reports no field error at all.

That was tolerable while the number it had to resolve was the published head's
4.67%. P20 took the head to 0.042%, and the margin went from 35x to **0.32** --
the reference `t_sing` for the application system is now three times COARSER
than the quantity being measured through it. A margin below one means the
benchmark cannot support its own converged-clock number in `sidm`.

WHAT THIS RUNS
--------------
The same protocol the other four systems get: trajectories sampled from the
PRODUCTION prior (`gen_dataset.draw_params(..., 'train')`, so the distribution
is the one the emulator is trained on, not two hand-picked cross-sections),
each solved at the production grid and again at 2x, plus a tolerance check at
fixed grid. Reports the relative disagreement in `t_sing`, the field
disagreement in dex, and a Richardson estimate of the production grid's own
error where the scheme's order can be established from three grids.

Richardson matters here. With two grids the disagreement `|t(N) - t(2N)|` is an
*estimate* of the coarse grid's error and inherits the fine grid's error too.
With three (N, 2N, 4N) the observed order can be measured and the extrapolated
value is a better reference than any single run, which is the cheapest route to
a margin the converged head can be measured against.

Usage:
  sc_sidm_reference.py --n 16 --grids 300 600            # two-grid, ~minutes
  sc_sidm_reference.py --n 8 --grids 300 600 1200        # + Richardson
Output: results/statecond/sidm_reference.json
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

from gravop import solver                                        # noqa: E402
from gen_dataset import SOLVER_KW, draw_params, make_ic          # noqa: E402
from sc_provenance import stamp                                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')


def run_at(d, ic, rc, n_shells, eps_t):
    kw = dict(SOLVER_KW)
    kw['n_shells'] = n_shells
    kw['eps_t'] = eps_t
    prof = make_ic(ic, rc, n_shells)
    if prof is not None:
        kw['r'], kw['m'], kw['rho'], kw['p'] = prof
    res = solver.run_halo(**d, **kw)
    if res['status'] in (3, 4) or res['t_final'] < 1.0:
        return None
    return res


def _safe_rel(a, b):
    return abs(a - b) / abs(b) if (b not in (None, 0)) else float('nan')


def t_c_at_radius(res, r_ref, factor):
    """Collapse time defined at a FIXED PHYSICAL RADIUS, not a shell index.

    The solver terminates on `rho[3] >= factor * rho[3](0)` -- shell index 3.
    The shells are Lagrangian, so index 3 is a fixed *mass* shell, but the mass
    it encloses depends on `n_shells`: at N = 600 it is roughly half the mass it
    is at N = 300, sitting at a smaller radius where the density is higher and
    rises sooner. Differencing `t_final` between two grids therefore mixes a
    discretisation error with a change of DEFINITION, and the definitional part
    does not vanish as the grid refines.

    This evaluates the same physical criterion on both grids: interpolate log
    density at a fixed radius `r_ref`, and find when it first exceeds `factor`
    times its own initial value there, by log-interpolation in time. `r_ref` is
    anchored to the production grid's shell 3 at t = 0, so the meaning of `t_c`
    is unchanged -- only its grid dependence is removed.

    Both definitions are reported, so the size of the artefact is measured
    rather than assumed away.
    """
    lr = np.log10(r_ref)
    rc = np.array([np.interp(lr, np.log10(res['r'][i]),
                             np.log10(res['rho'][i]))
                   for i in range(len(res['t']))])
    target = rc[0] + np.log10(factor)
    hit = np.nonzero(rc >= target)[0]
    if hit.size == 0 or hit[0] == 0:
        return None
    j = int(hit[0])
    t0, t1 = res['t'][j - 1], res['t'][j]
    l0, l1 = rc[j - 1], rc[j]
    if l1 == l0:
        return float(t1)
    return float(t0 + (t1 - t0) * (target - l0) / (l1 - l0))


def mid_mass(res):
    """Mass coordinate of each shell's centre: shell i holds (m[i-1], m[i]]."""
    m = np.asarray(res['m'])
    return 0.5 * (np.r_[0.0, m[:-1]] + m)


def t_c_at_mass(res, m_ref, factor):
    """Collapse time at a FIXED ENCLOSED MASS (added 2026-10-15).

    `t_c_at_radius` never fires: the collapsing core contracts inside any fixed
    radius, so the density there does not reach `factor` times its initial
    value even when the solver's own criterion does (0 usable trajectories of
    96 draws with the radius criterion). The solver's criterion is a fixed
    Lagrangian shell, i.e. a fixed enclosed mass on a given grid, so the
    grid-independent version of it is the density at a fixed mass coordinate:
    log density is interpolated in log mass at `m_ref`, anchored to the
    production grid's shell 3, where it reproduces the solver's criterion.
    """
    lm = np.log10(mid_mass(res))
    lr = np.log10(m_ref)
    rc = np.array([np.interp(lr, lm, np.log10(res['rho'][i]))
                   for i in range(len(res['t']))])
    target = rc[0] + np.log10(factor)
    hit = np.nonzero(rc >= target - 1e-12)[0]
    if hit.size == 0 or hit[0] == 0:
        return None
    j = int(hit[0])
    t0, t1 = res['t'][j - 1], res['t'][j]
    l0, l1 = rc[j - 1], rc[j]
    if l1 == l0:
        return float(t1)
    return float(t0 + (t1 - t0) * (target - l0) / (l1 - l0))


def field_dex(a, b):
    """RMS log10 density disagreement at matched normalised time.

    The two runs carry different shell counts and different snapshot times, so
    both are interpolated onto a common (tau, log r) grid before differencing.
    Without that the comparison would be dominated by where the snapshots fell,
    not by the discretisation.
    """
    ta, tb = a['t'] / a['t_c'], b['t'] / b['t_c']
    grid_t = np.linspace(0.05, 0.95, 19)
    lr = np.linspace(np.log10(3e-3), np.log10(3.0), 64)
    out = []
    for tt in grid_t:
        ia = int(np.argmin(np.abs(ta - tt)))
        ib = int(np.argmin(np.abs(tb - tt)))
        la = np.interp(lr, np.log10(a['r'][ia]), np.log10(a['rho'][ia]))
        lb = np.interp(lr, np.log10(b['r'][ib]), np.log10(b['rho'][ib]))
        out.append(la - lb)
    return float(np.sqrt(np.mean(np.asarray(out) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--grids', type=int, nargs='+', default=[300, 600])
    ap.add_argument('--eps-t', type=float, default=SOLVER_KW['eps_t'])
    ap.add_argument('--eps-tol', type=float, default=1e-4,
                    help='tolerance-refinement check at the production grid')
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--criterion', choices=('mass', 'radius'), default='mass',
                    help='where the density criterion is evaluated; radius '
                         'is kept for the record and never fires')
    ap.add_argument('--max-attempts', type=int, default=None,
                    help='draws to try (default 4 n)')
    args = ap.parse_args()

    grids = sorted(args.grids)
    prod = grids[0]
    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    attempted = 0
    max_attempts = args.max_attempts or 4 * args.n
    n_not_density = 0
    while len(rows) < args.n and attempted < max_attempts:
        attempted += 1
        d, ic, rc = draw_params(rng, 'train')
        runs = {}
        rp = run_at(d, ic, rc, prod, args.eps_t)
        if rp is None:
            continue
        # anchor the criterion to the production grid's shell 3, so it means
        # what it has always meant, evaluated identically on every grid
        if args.criterion == 'mass':
            ref = float(mid_mass(rp)[3])
            crit = t_c_at_mass
        else:
            ref = float(rp['r'][0][3])
            crit = t_c_at_radius
        if crit(rp, ref, SOLVER_KW['rho_c_factor']) is None:
            # ended on the Knudsen floor or the horizon: no density event
            n_not_density += 1
            continue
        runs[prod] = rp
        ok = True
        for g in grids[1:]:
            r = run_at(d, ic, rc, g, args.eps_t)
            if r is None:
                ok = False
                break
            runs[g] = r
        if not ok:
            continue
        rtol = run_at(d, ic, rc, prod, args.eps_tol)
        if rtol is None:
            continue
        r_ref = ref
        tc = {}
        for g in grids:
            v = crit(runs[g], ref, SOLVER_KW['rho_c_factor'])
            if v is None:
                tc = None
                break
            tc[g] = v
            runs[g]['t_c'] = v
        if tc is None:
            continue
        # the shell-index definition, for comparison only
        tc_shell = {g: float(runs[g]['t_final']) for g in grids}
        row = {
            'params': {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in d.items()},
            'ic': ic, 'rc': float(rc or 0.0),
            'criterion': args.criterion,
            'reference_coordinate': r_ref,
            't_c': tc,
            't_c_shell_index_definition': tc_shell,
            'rel_dt_sing_shell_definition':
                abs(tc_shell[prod] - tc_shell[grids[-1]]) / tc_shell[grids[-1]],
            # production vs the finest grid available: the quantity the margin
            # is computed from
            'rel_dt_sing': abs(tc[prod] - tc[grids[-1]]) / tc[grids[-1]],
            'rel_dt_tol': _safe_rel(
                tc[prod], crit(rtol, ref, SOLVER_KW['rho_c_factor'])),
            'field_dex': field_dex(runs[prod], runs[grids[-1]]),
        }
        if len(grids) >= 3:
            # observed order p from three grids at fixed refinement ratio,
            # then Richardson: t_exact ~ t_fine + (t_fine - t_mid)/(r^p - 1)
            c, m, f = (tc[grids[0]], tc[grids[1]], tc[grids[2]])
            r_ref = grids[1] / grids[0]
            num, den = (c - m), (m - f)
            if den != 0 and num / den > 0:
                p_obs = float(np.log(num / den) / np.log(r_ref))
                t_ext = f + (f - m) / (r_ref ** p_obs - 1)
                row['observed_order'] = p_obs
                row['t_richardson'] = float(t_ext)
                row['rel_dt_sing_richardson'] = abs(f - t_ext) / abs(t_ext)
        rows.append(row)
        print(f"  [{len(rows):2d}/{args.n}] "
              + '  '.join(f'N{g}: {tc[g]:.4f}' for g in grids)
              + f"  rel {row['rel_dt_sing']:.2e}"
              + f"  field {row['field_dex']:.2e}"
              + (f"  p {row['observed_order']:.2f}"
                 if 'observed_order' in row else ''), flush=True)

    if not rows:
        raise SystemExit('no usable trajectories')

    rel = np.array([r['rel_dt_sing'] for r in rows])
    fld = np.array([r['field_dex'] for r in rows])
    tol = np.array([r['rel_dt_tol'] for r in rows])
    rec = {
        'n_trajectories': len(rows),
        'n_attempted': attempted,
        'n_without_density_event': n_not_density,
        'criterion': args.criterion,
        'scope': ('runs that end on the density criterion; Knudsen-floor and '
                  'horizon terminations have no density event to converge'),
        'grids': grids,
        'production_grid': prod,
        'reference_grid': grids[-1],
        'eps_t': args.eps_t,
        'eps_tol': args.eps_tol,
        'seed': args.seed,
        'prior': "gen_dataset.draw_params(rng, 'train')",
        'rel_dt_sing_median': float(np.median(rel)),
        'rel_dt_sing_p90': float(np.quantile(rel, 0.9)),
        'rel_dt_sing_max': float(rel.max()),
        'field_dex_median': float(np.median(fld)),
        'tolerance_refinement_median': float(np.median(tol)),
        'wall_seconds': time.time() - t0,
        'rows': rows,
        '_provenance': stamp('sc_sidm_reference.py'),
    }
    if 'rel_dt_sing_richardson' in rows[0]:
        rr = np.array([r['rel_dt_sing_richardson'] for r in rows
                       if 'rel_dt_sing_richardson' in r])
        po = np.array([r['observed_order'] for r in rows
                       if 'observed_order' in r])
        rec['rel_dt_sing_richardson_median'] = float(np.median(rr))
        rec['observed_order_median'] = float(np.median(po))
        rec['n_richardson'] = int(len(rr))

    th = os.path.join(OUT, 'timehead.json')
    if os.path.exists(th):
        c = json.load(open(th))['sidm']['converged_head_err_pct'] / 100.0
        rec['converged_head_err_pct'] = c * 100
        rec['margin_median_converged'] = c / rec['rel_dt_sing_median']
        if 'rel_dt_sing_richardson_median' in rec:
            rec['margin_median_converged_richardson'] = (
                c / rec['rel_dt_sing_richardson_median'])

    path = os.path.join(OUT, 'sidm_reference.json')
    json.dump(rec, open(path, 'w'), indent=1)
    print(f"\nn={len(rows)}  rel dt_sing median {rec['rel_dt_sing_median']:.2e}"
          f"  field {rec['field_dex_median']:.2e} dex"
          f"  margin vs converged head "
          f"{rec.get('margin_median_converged', float('nan')):.2f}")
    if 'margin_median_converged_richardson' in rec:
        print(f"  Richardson (order {rec['observed_order_median']:.2f}): "
              f"margin {rec['margin_median_converged_richardson']:.2f}")
    print('wrote', path)


if __name__ == '__main__':
    main()
