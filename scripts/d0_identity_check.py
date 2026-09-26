#!/usr/bin/env python3
"""The quadrature rule's residual, in closed form, checked on the archived points.

The disjoint-system experiment rests on one claim: the error the quadrature predictor
makes is not an empirical property of the five systems but an exact function of
three quantities the decomposition already measures. This script checks that
claim on the fifteen (system x backbone) points in
`results/statecond/decomposition.json`, before any new system is generated.

Writing the exact decomposition of Section 5 as

    E_dep^2 = F^2 + ||e||^2 + 2<Delta, e>

and defining

    c = 2<Delta,e> / E_dep^2     the cross-term share
    k = ||e|| / E_A              the inflation at the corrupted coordinate
    g = F / E_A                  the displacement in units of the operator error

gives E_dep^2 (1 - c) = F^2 + k^2 E_A^2, so the rule's residual

    r = log10(p_hat / p),   p_hat = sqrt(E_A^2 + F^2)/E_A,   p = E_dep/E_A

is exactly

    r = 0.5 * log10[ (1 + g^2)(1 - c) / (g^2 + k^2) ].

It is zero when c = 0 and k = 1: the two approximations the rule makes are
exactly its two error terms and nothing else enters.

This is a decomposition, not a forecast. Of the three inputs only g is
available before deployment is measured -- F comes from the reference solutions
and E_A from one oracle evaluation -- while c and k both need e(v~), the error
at the corrupted coordinate, and computing that is the same work as measuring
E_dep itself. What the identity buys is a sharp question in place of a vague
one: the disjoint-system experiment asks whether c and k concentrate across
equations that share nothing, because if they do then g alone carries a
computable error bar on the rule, and if they do not the identity says which of
the two approximations broke. Either way it is worth establishing on data
already in hand rather than discovering on data that does not exist yet.

A discrepancy here would mean one of the four measured quantities is wrong, not
that the algebra is approximate, so the gate compares against machine epsilon
rather than against a tolerance chosen to pass.
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'results', 'statecond', 'decomposition.json')
OUT = os.path.join(ROOT, 'results', 'd0', 'identity.json')


def diagnostics(v):
    """(c, k, g) from one decomposition row, recomputed rather than read."""
    EA, F, Edep, edep = v['E_A'], v['F'], v['E_dep'], v['e_dep']
    c = (Edep ** 2 - F ** 2 - edep ** 2) / Edep ** 2
    return c, edep / EA, F / EA


def residual_from_identity(c, k, g):
    return 0.5 * np.log10((1.0 + g ** 2) * (1.0 - c) / (g ** 2 + k ** 2))


def residual_measured(v):
    EA, F, Edep = v['E_A'], v['F'], v['E_dep']
    return np.log10(np.sqrt(EA ** 2 + F ** 2) / Edep)


def main():
    d = json.load(open(SRC))
    rows = {k: v for k, v in d.items() if isinstance(v, dict) and 'E_A' in v}
    out, worst = {}, 0.0
    for name, v in rows.items():
        c, k, g = diagnostics(v)
        ri, rm = residual_from_identity(c, k, g), residual_measured(v)
        worst = max(worst, abs(ri - rm))
        out[name] = {'cross_share': c, 'inflation': k, 'displacement_ratio': g,
                     'residual_identity': float(ri),
                     'residual_measured': float(rm),
                     'difference': float(ri - rm)}
    cs = [r['cross_share'] for r in out.values()]
    summary = {
        'n_points': len(out),
        'max_abs_difference_dex': float(worst),
        'median_cross_share': float(np.median(cs)),
        'n_negative_cross': int(sum(x < 0 for x in cs)),
        'median_quadrature_over_measured': float(np.median(
            [v['quad_over_measured'] for v in rows.values()])),
        'what': 'r = 0.5*log10[(1+g^2)(1-c)/(g^2+k^2)] against the directly '
                'measured log10(quadrature/E_dep), per (system x backbone) point',
    }
    json.dump({'per_point': out, '_summary': summary}, open(OUT, 'w'), indent=1)
    print(f"{summary['n_points']} points, "
          f"max |identity - measured| = {worst:.2e} dex")
    print(f"median cross share {summary['median_cross_share']:+.4f}, "
          f"negative in {summary['n_negative_cross']} of {summary['n_points']}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == '__main__':
    main()
