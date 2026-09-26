#!/usr/bin/env python3
"""P36 — the deployed-accuracy predictor, specified and tested out of sample.

Protocol and fixed consequences: docs/PREDICTOR_PREREGISTRATION.md.

The manuscript claims a predictor of deployed accuracy. A predictor is a claim
only if the rule, its inputs and its regime of validity are stated, so this
script states them and scores them leave-one-system-out at both operating
points. The target is the deployment penalty p = E_dep / E_A, scored in log10.

Reads only stored artefacts. No training, no GPU, no solver call.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def score(points, predict, fit_needed):
    """Leave-one-system-out. `points` is a list of dicts with system/EA/F/p."""
    systems = sorted({p['system'] for p in points})
    per, pooled = {}, []
    for held in systems:
        fit = [p for p in points if p['system'] != held]
        tst = [p for p in points if p['system'] == held]
        state = fit_needed(fit) if fit_needed else None
        res = [np.log10(predict(p, state)) - np.log10(p['p']) for p in tst]
        per[held] = {'n_test': len(tst), 'rms_log_residual': rms(res),
                     'median_factor_error': float(10 ** np.median(np.abs(res)))}
        pooled += res
    return {'per_system': per,
            'mean_of_system_rms': float(np.mean(
                [v['rms_log_residual'] for v in per.values()])),
            'pooled_rms': rms(pooled)}


# ------------------------------------------------------------------ the rules
def r0_null(p, state):
    return 10 ** state


def r0_fit(fit):
    return float(np.mean(np.log10([q['p'] for q in fit])))


def r_unit(p, state):
    """The no-penalty baseline: predict that deployment costs nothing.

    Free in every sense -- no coefficients, no operator, no reference
    trajectories. The referee's point is that it must be beaten rather than
    assumed beatable, because a clock trained to convergence makes several
    penalties genuinely small, and a rule that cannot beat "no penalty" is not
    carrying information.
    """
    return 1.0


def r1_quadrature(p, state):
    return float(np.hypot(p['EA'], p['F']) / p['EA'])


def r2_bare(p, state):
    return float(p['F'] / p['EA'])


def r3_powerlaw(p, state):
    a, b = state
    return float(10 ** (a + b * np.log10(p['s'])))


def r3_fit(fit):
    x = np.log10([q['s'] for q in fit])
    y = np.log10([q['p'] for q in fit])
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def main():
    out = {'protocol': 'docs/PREDICTOR_PREREGISTRATION.md',
           'target': 'log10(E_dep / E_A), the deployment penalty',
           'aggregation': ('RMS of the log10 residual within the held-out '
                           'system, then the unweighted mean over systems')}

    # ---- published clock: 25 points, s available, so R3 can be scored
    law = json.load(open(os.path.join(RES, 'law.json')))['points']
    flo = json.load(open(os.path.join(RES, 'floor.json')))['floor_deployed']
    pub = [{'system': q['system'], 'EA': q['oracle'], 'p': q['penalty'],
            's': q['sensitivity'], 'F': flo[q['system']]['floor_deployed']}
           for q in law]
    out['published_clock'] = {
        'n_points': len(pub),
        'R0_null': score(pub, r0_null, r0_fit),
        'R_unit_no_penalty': score(pub, r_unit, None),
        'R1_quadrature': score(pub, r1_quadrature, None),
        'R2_bare': score(pub, r2_bare, None),
        'R3_power_law': score(pub, r3_powerlaw, r3_fit),
    }

    # ---- converged clock: 15 points measured by P33. s was never measured
    # here, so R3 cannot be scored and is reported as unavailable rather than
    # carried over from the other operating point.
    dec = json.load(open(os.path.join(RES, 'decomposition.json')))
    conv = []
    for key, v in dec.items():
        system = key.rsplit('_', 1)[0]
        conv.append({'system': system, 'EA': v['E_A'], 'F': v['F'],
                     'p': v['E_dep'] / v['E_A']})
    out['converged_clock'] = {
        'n_points': len(conv),
        'R0_null': score(conv, r0_null, r0_fit),
        'R_unit_no_penalty': score(conv, r_unit, None),
        'R1_quadrature': score(conv, r1_quadrature, None),
        'R2_bare': score(conv, r2_bare, None),
        'R3_power_law': 'not scored: phase sensitivity s was never measured '
                        'at the converged clock',
    }

    # ---- the information budget, stated rather than implied
    out['information_budget'] = {
        'R_unit_no_penalty': dict(
            fitted=0, needs_operator_on_target_system=False,
            needs_reference_trajectories=False,
            note='predicts no deployment penalty; consumes nothing at all'),
        'R0_null': dict(fitted=1, needs_operator_on_target_system=False,
                        needs_imposed_error_sweep=False),
        'R1_quadrature': dict(fitted=0, needs_operator_on_target_system=True,
                              detail='E_A, an oracle evaluation only',
                              needs_imposed_error_sweep=False),
        'R2_bare': dict(fitted=0, needs_operator_on_target_system=True,
                        detail='E_A, an oracle evaluation only',
                        needs_imposed_error_sweep=False),
        'R3_power_law': dict(fitted=2, needs_operator_on_target_system=True,
                             needs_imposed_error_sweep=True),
    }

    for regime in ('published_clock', 'converged_clock'):
        b = out[regime]
        print(f"\n{regime}  ({b['n_points']} points)")
        print(f"  {'rule':16s} {'mean sys RMS':>13s} {'pooled RMS':>11s}")
        for rule in ('R_unit_no_penalty', 'R0_null', 'R1_quadrature',
                     'R2_bare', 'R3_power_law'):
            v = b[rule]
            if isinstance(v, str):
                print(f"  {rule:16s}   {v}")
                continue
            print(f"  {rule:16s} {v['mean_of_system_rms']:13.3f} "
                  f"{v['pooled_rms']:11.3f}")

    p = os.path.join(RES, 'predictor.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('\nwrote', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
