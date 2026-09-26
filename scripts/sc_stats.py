"""P18 — the statistics repair: what predicts the deployment penalty out of sample.

WHY THIS EXISTS
---------------
`sc_law.py` fits `penalty ~ s^0.81` over 25 points and reports r = 0.927 in
logs, plus a held-out-BACKBONE test. P12 showed why that is not enough:

  s      = E(10% imposed)/E_A
  penalty = E_B/E_A

share E_A in the denominator, and P12 established that both numerators are
essentially the same system-level quantity — the clock-error floor. So a strong
pooled correlation is close to guaranteed, and the held-out-backbone test does
not probe the part a reader cares about: **would this predict a system I have
not seen?**

This script asks that, and pits the law against the floor as competitors:

  LAW    2 fitted coefficients, and to use it on a new system you must ALREADY
         have trained an operator there, because s is measured on that operator.
  FLOOR  ZERO fitted coefficients: penalty_pred = F_deployed / E_A, where F is
         computed from the reference solver alone. On a new system you need only
         the solver and the time head's error.

The floor therefore needs strictly less information. If it also predicts better
out of sample, the law should not be the paper's mechanism.

Also reported, because the referee report was right that ratios alone are not
enough: the ABSOLUTE deployed error alongside every ratio, and a system-aware
(hierarchical) bootstrap that resamples systems rather than points.

Usage:  sc_stats.py [--out results/statecond/stats.json]
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')


def loso_law(points):
    """Leave-one-SYSTEM-out: fit log-log on 4 systems, predict the 5th."""
    systems = sorted({p['system'] for p in points})
    out = {}
    for held in systems:
        fit = [p for p in points if p['system'] != held]
        tst = [p for p in points if p['system'] == held]
        x = np.log10([p['sensitivity'] for p in fit])
        y = np.log10([p['penalty'] for p in fit])
        b, a = np.polyfit(x, y, 1)
        xt = np.log10([p['sensitivity'] for p in tst])
        yt = np.log10([p['penalty'] for p in tst])
        pred = a + b * xt
        out[held] = {
            'n_fit': len(fit), 'n_test': len(tst),
            'exponent_on_the_other_four': float(b),
            'rms_log_residual': float(np.sqrt(np.mean((pred - yt) ** 2))),
            'median_factor_error': float(
                10 ** np.median(np.abs(pred - yt))),
        }
    return out


def loso_floor(points, floor_dep, mode='quadrature'):
    """The floor as a ZERO-parameter predictor of the deployment penalty.

    mode='bare'        penalty = F / E_A. Exact when the floor dominates, but it
                       ignores the operator's own error, so it under-predicts
                       wherever E_A is not small against F -- which is `burgers`,
                       the regularised control with the smallest floor of the
                       five.
    mode='quadrature'  penalty = sqrt(E_A^2 + F^2) / E_A. Still nothing fitted;
                       it just adds back the operator error that P12 showed is
                       the other term. This is the predictor to report.

    Nothing is fitted either way, so there is no train/test split to make --
    every system is held out by construction. Reported per system for
    comparability with `loso_law`.
    """
    out = {}
    for s in sorted({p['system'] for p in points}):
        tst = [p for p in points if p['system'] == s]
        F = floor_dep[s]
        if mode == 'bare':
            pred = np.log10([F / p['oracle'] for p in tst])
        else:
            pred = np.log10([np.hypot(p['oracle'], F) / p['oracle']
                             for p in tst])
        yt = np.log10([p['penalty'] for p in tst])
        out[s] = {
            'n_test': len(tst),
            'n_fitted_coefficients': 0,
            'mode': mode,
            'rms_log_residual': float(np.sqrt(np.mean((pred - yt) ** 2))),
            'median_factor_error': float(10 ** np.median(np.abs(pred - yt))),
        }
    return out


def system_bootstrap(points, n_boot=10000, seed=20260904):
    """Resample SYSTEMS, not points.

    The 25 points are 5 systems x 5 operators, not 25 independent experiments.
    `sc_law.py` bootstraps the (system x backbone) point, which understates the
    interval because the systems are the level the law is meant to generalise
    over. This resamples whole systems with replacement.
    """
    systems = sorted({p['system'] for p in points})
    by = {s: [p for p in points if p['system'] == s] for s in systems}
    rng = np.random.default_rng(seed)
    exps = []
    for _ in range(n_boot):
        pick = rng.choice(systems, size=len(systems), replace=True)
        sel = [p for s in pick for p in by[s]]
        x = np.log10([p['sensitivity'] for p in sel])
        y = np.log10([p['penalty'] for p in sel])
        if np.ptp(x) < 1e-9:
            continue
        exps.append(np.polyfit(x, y, 1)[0])
    exps = np.array(exps)
    return {'exponent_median': float(np.median(exps)),
            'ci95': [float(np.percentile(exps, 2.5)),
                     float(np.percentile(exps, 97.5))],
            'n_boot': int(len(exps)),
            'unit': 'system (5 drawn with replacement)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        ROOT, 'results', 'statecond', 'stats.json'))
    a = ap.parse_args()

    law = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                      'law.json')))
    flo = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                      'floor.json')))
    pts = law['points']
    fdep = {s: v['floor_deployed'] for s, v in flo['floor_deployed'].items()}

    L = loso_law(pts)
    Fb = loso_floor(pts, fdep, 'bare')
    F = loso_floor(pts, fdep, 'quadrature')

    print('Leave-one-SYSTEM-out prediction of the deployment penalty')
    print(f"{'held-out':10s} {'law':>8s} {'floor bare':>11s} "
          f"{'floor quad':>11s}   | median factor error")
    for s in sorted(L):
        print(f'{s:10s} {L[s]["rms_log_residual"]:8.3f} '
              f'{Fb[s]["rms_log_residual"]:11.3f} '
              f'{F[s]["rms_log_residual"]:11.3f}   | '
              f'law {L[s]["median_factor_error"]:.2f}x  '
              f'floor {F[s]["median_factor_error"]:.2f}x')
    lm = np.mean([L[s]['rms_log_residual'] for s in L])
    fbm = np.mean([Fb[s]['rms_log_residual'] for s in Fb])
    fm = np.mean([F[s]['rms_log_residual'] for s in F])
    print(f'{"MEAN":10s} {lm:8.3f} {fbm:11.3f} {fm:11.3f}   (dex)')
    print(f'\nlaw: 2 fitted coefficients, and needs an operator already trained '
          f'on the new system.\nfloor: 0 fitted coefficients, needs only the '
          f'solver and the time head.')

    sb = system_bootstrap(pts)
    print(f"\nExponent, resampling SYSTEMS (the level the law generalises over):"
          f"\n  {sb['exponent_median']:.3f}  95% CI "
          f"[{sb['ci95'][0]:.3f}, {sb['ci95'][1]:.3f}]   "
          f"(sc_law.py, resampling points: "
          f"[{law['exponent_ci']['lo']:.3f}, {law['exponent_ci']['hi']:.3f}])")

    # absolute deployed error alongside every ratio
    absol = {}
    for s in sorted({p['system'] for p in pts}):
        sel = [p for p in pts if p['system'] == s]
        dep = np.array([p['oracle'] * p['penalty'] for p in sel])
        orc = np.array([p['oracle'] for p in sel])
        absol[s] = {
            'oracle_spread': float(orc.max() / orc.min()),
            'deployed_spread': float(dep.max() / dep.min()),
            'deployed_mean': float(dep.mean()),
            'floor_deployed': fdep[s],
            'deployed_over_floor': float(dep.mean() / fdep[s]),
        }

    json.dump({'loso_law': L, 'loso_floor': F, 'loso_floor_bare': Fb,
               'loso_mean_rms': {'law': float(lm), 'floor': float(fm),
                                 'floor_bare': float(fbm)},
               'system_bootstrap': sb, 'absolute': absol,
               'note': ('The law needs 2 fitted coefficients AND an operator '
                        'already trained on the target system (s is measured '
                        'on it). The floor needs neither.')},
              open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
