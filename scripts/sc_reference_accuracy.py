"""P16 — reference-solution accuracy for all FIVE systems, in one place.

WHY THIS EXISTS
---------------
Table 2 of the manuscript reports a single median relative change in t_sing for
four systems and claims it is "between two and four orders of magnitude below"
the 1.8-7.1% t_sing errors whose consequences the paper measures. Three things
are wrong with that as it stands:

1. **`sidm` is missing entirely** -- the fifth system, and the application the
   project exists for. Its convergence study was run (E1,
   `results/e1/implicit_convergence.json`: 5 shell counts x 3 tolerances x 2
   cross-sections) and simply never reached the table.
2. **The claimed margin is wrong.** Per system the true factors are 22x (`ks`),
   63x (`film`), 87x (`burgers`) and 1.1e5x (`rd`) -- that is 1.3 to 5.0 orders,
   not "two to four". And those are medians; on the worst of the 24 sampled
   trajectories `ks` has only a 3.6x margin, and on the difficulty-stratified
   subset its median margin is 5.9x. The numbers here are printed by the script
   rather than trusted from this docstring; see the artefact.
3. **The protocol is described backwards.** The caption says each system is
   solved "at its production resolution and again at half that resolution", but
   the numbers quoted are `median_rel_dt_sing_production` -- production against
   a 2x FINER reference. The study is better than its own description.

This script pulls the four PDE systems from `sc_convergence.py`'s output and
`sidm` from the E1 grid, puts them on one footing, and computes the margin
against the measured t_sing errors so the manuscript can quote it instead of
asserting it.

Output: results/statecond/reference_accuracy.json
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RS = os.path.join(ROOT, 'results', 'statecond')

SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')


def tsing_err_pct():
    """Median t_sing error of the time head, read from the arm results.

    Hardcoding these invites exactly the drift this script exists to catch:
    the margin is a RATIO, so a stale numerator is as wrong as a stale
    denominator. They come from `<system>.json`, mean over seeds of the
    per-seed median, which is what Table 3 reports.
    """
    return {s: json.load(open(os.path.join(RS, f'{s}.json')))
            ['tsing_median_err_pct']['mean'] for s in SYSTEMS}


def converged_head_err_pct():
    """Median t_sing error of the CONVERGED head (P20), or {} if not run.

    The margin is a ratio, and P20 moved its numerator by 6-110x. Reporting the
    margin only against the published head -- which is what this script did
    until v7/P21 -- says the reference is comfortable when against the head the
    paper now quotes it is not: `ks` 0.44 and `sidm` 0.32, i.e. the ruler is
    coarser than the thing being measured. Both operating points are reported
    so the gap cannot go unnoticed again.
    """
    p = os.path.join(RS, 'timehead.json')
    if not os.path.exists(p):
        return {}
    th = json.load(open(p))
    return {s: th[s]['converged_head_err_pct'] for s in SYSTEMS
            if s in th and 'converged_head_err_pct' in th[s]}
# the gravothermal production configuration (scripts/gen_dataset.py SOLVER_KW)
SIDM_PROD = {'n_shells': 300, 'eps_t': 3e-4}
SIDM_REF = {'n_shells': 400, 'eps_t': 3e-4}
SIDM_TOL = {'n_shells': 300, 'eps_t': 1e-4}


def sidm_rows():
    """Spatial and temporal convergence of the gravothermal event time.

    E1 swept N in {100,150,200,300,400} x eps_t in {1e-3,3e-4,1e-4} at two
    cross-sections. Production is N = 300, eps_t = 3e-4, so the finer reference
    is N = 400 at the same tolerance and the tolerance check is eps_t = 1e-4 at
    the same N. Every run in that sweep terminated normally (status 2).

    NOTE: `results/e1/conv_N*.json` is a DIFFERENT sweep, run to 1000x central
    density rather than ~300x, and there N = 200 and N = 300 hit the step cap
    without collapsing (status 3). That is why sidm looked unusable for a
    convergence table; the E1 grid at the production threshold is complete.
    """
    d = json.load(open(os.path.join(ROOT, 'results', 'e1',
                                    'implicit_convergence.json')))

    def get(N, eps, sig):
        for r in d:
            if r['N'] == N and abs(r['eps'] - eps) < 1e-12 and r['sig'] == sig:
                return r
        raise KeyError((N, eps, sig))

    spatial, temporal, drift = [], [], []
    for sig in sorted({r['sig'] for r in d}):
        prod = get(SIDM_PROD['n_shells'], SIDM_PROD['eps_t'], sig)
        ref = get(SIDM_REF['n_shells'], SIDM_REF['eps_t'], sig)
        tol = get(SIDM_TOL['n_shells'], SIDM_TOL['eps_t'], sig)
        spatial.append(abs(prod['t_final'] - ref['t_final']) / ref['t_final'])
        temporal.append(abs(prod['t_final'] - tol['t_final']) / tol['t_final'])
        drift.append(abs(prod['drift']))
    return spatial, temporal, drift


def _converged_summary(out):
    """The same margins against the converged head, and the verdict.

    Threshold: median >= 10 and hard-subset >= 5. Below that the reference is
    not fine enough to resolve the head error being reported, and the affected
    system's converged-clock number is a statement the benchmark cannot
    support -- which must be said, not smoothed over.
    """
    rows = {k: v for k, v in out.items()
            if isinstance(v, dict) and 'margin_median_converged' in v}
    if not rows:
        return None
    # `sidm` was re-specified on 2026-09-06 as an APPLICATION system: its
    # t_sing is a solver termination evaluated at shell INDEX 3, which encloses
    # a different mass fraction on every grid, so its "convergence" figure
    # mixes discretisation with a change of definition and cannot be held to
    # the reference standard. It is reported and excluded from the verdict
    # rather than dropped, because the number is still the honest description
    # of what the application system's clock is worth.
    reference = [k for k in rows if k != 'sidm']
    fails = [k for k in reference
             if not rows[k].get('reference_adequate_for_converged_head', False)]
    below_one = [k for k in reference if rows[k]['margin_median_converged'] < 1.0]
    return {
        'threshold': 'median >= 10 and hard-subset >= 5',
        'margin_median': {k: v['margin_median_converged']
                          for k, v in rows.items()},
        'margin_hard_median': {k: v['margin_hard_median_converged']
                               for k, v in rows.items()
                               if 'margin_hard_median_converged' in v},
        'reference_systems': sorted(reference),
        'application_systems': ['sidm'] if 'sidm' in rows else [],
        'inadequate': fails,
        'ruler_coarser_than_measurement': below_one,
        'sidm_excluded_because': (
            'application system, re-specified 2026-09-06: t_sing is a solver '
            'termination at shell INDEX 3, whose enclosed mass fraction is '
            'grid-dependent, so a convergence study on it would mix '
            'discretisation with a change of definition. Its margin '
            + (f"({rows['sidm']['margin_median_converged']:.2f}) is reported "
               'but is not a reference-standard number.'
               if 'sidm' in rows else 'is not computed.')),
        'verdict': (
            'the reference is adequate for the converged head in every system'
            if not fails else
            'INADEQUATE for ' + ', '.join(sorted(fails)) +
            ('; the reference is COARSER than the measurement in ' +
             ', '.join(sorted(below_one)) if below_one else '')),
    }


def main():
    conv = json.load(open(os.path.join(RS, 'convergence.json')))
    out = {}

    for name, v in conv.items():
        st = v['rel_dt_sing_production_stats']
        tol = v['tolerance_refinement']
        out[name] = {
            'production_grid': v['n_solve_production'],
            'reference_grid': v['reference_grid'],
            'n_trajectories': v['n'],
            'rel_dt_sing_median': st['median'],
            'rel_dt_sing_p90': st['p90'],
            'rel_dt_sing_max': st['max'],
            'tolerance_refinement_median': tol['median'] if tol else None,
            'threshold_third_median': (v['threshold_third']['median']
                                       if v['threshold_third'] else None),
            'field_dex_median': v['field_dex_production_stats']['median'],
            'energy_drift': None,
        }
        # the difficulty-stratified subset, when the convergence run produced
        # one. A median over the prior answers a different question from "how
        # bad is it where the solver struggles", and only the second bounds
        # what an unlucky trajectory suffers.
        h = v.get('hard_subset')
        if h:
            out[name]['hard_n'] = h['n']
            out[name]['hard_n_screened'] = h['n_screened']
            out[name]['hard_rel_dt_sing_median'] = (
                h['rel_dt_sing_production_stats']['median'])
            out[name]['hard_rel_dt_sing_max'] = (
                h['rel_dt_sing_production_stats']['max'])

    sp, tp, dr = sidm_rows()
    out['sidm'] = {
        'production_grid': SIDM_PROD['n_shells'],
        'reference_grid': SIDM_REF['n_shells'],
        'n_trajectories': len(sp),
        'rel_dt_sing_median': float(np.median(sp)),
        'rel_dt_sing_p90': float(np.max(sp)),
        'rel_dt_sing_max': float(np.max(sp)),
        'tolerance_refinement_median': float(np.median(tp)),
        'threshold_third_median': None,
        'field_dex_median': None,
        'energy_drift': float(np.max(dr)),
        'note': ('two cross-sections (sigma-hat 0.5, 2.0) rather than 16 '
                 'sampled trajectories, so p90 is reported as the max of two'),
    }

    err = tsing_err_pct()
    conv = converged_head_err_pct()
    for k, v in out.items():
        e = err[k] / 100.0
        v['tsing_err_pct'] = round(err[k], 2)
        v['margin_median'] = e / v['rel_dt_sing_median']
        v['margin_worst'] = e / v['rel_dt_sing_max']
        v['margin_orders_median'] = float(np.log10(v['margin_median']))
        # the same margin against the operating point the paper now quotes
        if k in conv:
            c = conv[k] / 100.0
            v['converged_head_err_pct'] = conv[k]
            v['margin_median_converged'] = c / v['rel_dt_sing_median']
            v['margin_worst_converged'] = c / v['rel_dt_sing_max']
            if 'hard_rel_dt_sing_median' in v:
                v['margin_hard_median_converged'] = (
                    c / v['hard_rel_dt_sing_median'])
            v['reference_adequate_for_converged_head'] = bool(
                v['margin_median_converged'] >= 10.0
                and v.get('margin_hard_median_converged', 1e9) >= 5.0)
        if v.get('hard_rel_dt_sing_median'):
            v['margin_hard_median'] = e / v['hard_rel_dt_sing_median']
            v['margin_hard_worst'] = e / v['hard_rel_dt_sing_max']

    mm = [v['margin_median'] for v in out.values()]
    mw = [v['margin_worst'] for v in out.values()]
    hm = [v['margin_hard_median'] for v in out.values()
          if 'margin_hard_median' in v]
    hw = [v['margin_hard_worst'] for v in out.values()
          if 'margin_hard_worst' in v]
    out['_summary'] = {
        'margin_hard_median_range': ([float(min(hm)), float(max(hm))]
                                     if hm else None),
        'margin_hard_worst_case': float(min(hw)) if hw else None,
        'margin_median_range': [float(min(mm)), float(max(mm))],
        'margin_orders_range': [float(np.log10(min(mm))),
                                float(np.log10(max(mm)))],
        'margin_worst_case': float(min(mw)),
        'converged': _converged_summary(out),
        'claim_in_manuscript': 'between two and four orders of magnitude',
        'verdict': ('FALSE as written: the true median range is '
                    f'{min(mm):.0f}x to {max(mm):.0f}x, i.e. '
                    f'{np.log10(min(mm)):.1f} to {np.log10(max(mm)):.1f} '
                    f'orders. On the worst single trajectory the margin falls '
                    f'to {min(mw):.1f}x, which must be stated.'),
        'hard_subset_verdict': (
            'on the difficulty-stratified subset the tightest median margin is '
            f'{min(hm):.1f}x and the worst single trajectory {min(hw):.1f}x, '
            'so the reference is adequate but not comfortable where the solver '
            'struggles' if hm else 'no hard subset in convergence.json'),
    }

    json.dump(out, open(os.path.join(RS, 'reference_accuracy.json'), 'w'),
              indent=1)

    print(f"{'system':9s} {'prod':>6s} {'ref':>6s} {'median':>10s} {'p90':>10s} "
          f"{'max':>10s} {'tol':>10s} {'margin':>9s}")
    for k in ('rd', 'ks', 'film', 'burgers', 'sidm'):
        v = out[k]
        t = v['tolerance_refinement_median']
        print(f"{k:9s} {v['production_grid']:6d} {v['reference_grid']:6d} "
              f"{v['rel_dt_sing_median']:10.2e} {v['rel_dt_sing_p90']:10.2e} "
              f"{v['rel_dt_sing_max']:10.2e} "
              f"{(f'{t:.2e}' if t else 'n/a'):>10s} "
              f"{v['margin_median']:8.0f}x")
    print('\n' + out['_summary']['verdict'])
    if out['_summary'].get('margin_hard_median_range'):
        print()
        for k in ('rd', 'ks', 'film', 'burgers'):
            v = out[k]
            print(f"{k:9s} hard subset {v['hard_n']}/{v['hard_n_screened']}: "
                  f"median {v['hard_rel_dt_sing_median']:.2e} "
                  f"(margin {v['margin_hard_median']:.0f}x), "
                  f"worst {v['hard_rel_dt_sing_max']:.2e} "
                  f"(margin {v['margin_hard_worst']:.1f}x)")
        print('\n' + out['_summary']['hard_subset_verdict'])


if __name__ == '__main__':
    main()
