"""P11 — accuracy transfer efficiency: does operator accuracy reach deployment?

Section 5 establishes that a more accurate operator pays a *larger* deployment
penalty. That is a statement about a ratio, and it leaves the obvious objection
open: the penalty is larger, but the deployed error might still be smaller, in
which case one should simply buy the better operator.

This script asks the question the objection deserves, of data already committed
in `results/statecond/law.json`. Define, within a system, across backbones,

    deployed = oracle x penalty
    eta      = d(log10 deployed) / d(log10 oracle)

    eta = 1   operator accuracy transfers fully to deployment
    eta = 0   a better operator buys nothing deployed
    eta < 0   a better operator is worse deployed

eta is a slope over the backbones available in each system -- four now, five if
P10 lands -- so the per-system intervals are wide and are reported rather than
hidden. It is NOT an ordering statistic: eta is not cleanly ordered by
singularity strength (see `ordering_test`), and claiming that it is would be an
overreach. What it supports is a split -- the regularised control is the one
system where accuracy still clearly pays -- plus the fit-free range comparison,
which needs no slope at all.

Because there are so few points per system, the interval is computed two ways
and both are reported. A bootstrap over backbones is the honest resampling of
the unit the slope is taken over, but with n = 4 it can only produce a handful
of distinct resamples; a jackknife (leave-one-backbone-out) says something
different and more stable, namely how much any single backbone is carrying the
result. Where the two disagree, believe the weaker one.

Usage:  python scripts/sc_transfer.py
Output: results/statecond/transfer.json
"""
import itertools
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')


def slope(oracle, deployed):
    return float(np.polyfit(np.log10(oracle), np.log10(deployed), 1)[0])


def analyse(pts):
    out = {}
    for s in SYSTEMS:
        sub = [p for p in pts if p['system'] == s]
        if len(sub) < 3:
            continue
        o = np.array([p['oracle'] for p in sub])
        d = np.array([p['oracle'] * p['penalty'] for p in sub])
        names = [p['backbone'] for p in sub]
        eta = slope(o, d)

        # bootstrap over backbones: the honest unit, but n is tiny
        rng = np.random.default_rng(20260904)
        boot = []
        for _ in range(10000):
            i = rng.integers(0, len(sub), len(sub))
            if len(np.unique(o[i])) < 2:
                continue
            boot.append(slope(o[i], d[i]))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # jackknife: how much is any single backbone carrying this?
        jack = {}
        for k in range(len(sub)):
            m = np.ones(len(sub), bool)
            m[k] = False
            jack[names[k]] = slope(o[m], d[m])

        # the raw statement a reader can check without trusting any fit
        best, worst = int(np.argmin(o)), int(np.argmax(o))
        out[s] = {
            'eta': eta,
            'eta_ci95_bootstrap': [float(lo), float(hi)],
            'eta_jackknife_leave_one_out': jack,
            'eta_jackknife_range': [float(min(jack.values())),
                                    float(max(jack.values()))],
            'n_backbones': len(sub),
            'oracle_range_factor': float(o.max() / o.min()),
            'deployed_range_factor': float(d.max() / d.min()),
            'most_accurate': {'backbone': names[best], 'oracle': float(o[best]),
                              'deployed': float(d[best])},
            'least_accurate': {'backbone': names[worst], 'oracle': float(o[worst]),
                               'deployed': float(d[worst])},
            'best_accuracy_is_worse_deployed': bool(d[best] > d[worst]),
        }

    o = np.array([p['oracle'] for p in pts])
    d = np.array([p['oracle'] * p['penalty'] for p in pts])
    rng = np.random.default_rng(20260904)
    pooled_boot = []
    for _ in range(10000):
        i = rng.integers(0, len(pts), len(pts))
        if len(np.unique(o[i])) < 2:
            continue
        pooled_boot.append(slope(o[i], d[i]))
    lo, hi = np.percentile(pooled_boot, [2.5, 97.5])
    pooled = {'eta': slope(o, d), 'eta_ci95_bootstrap': [float(lo), float(hi)],
              'n_points': len(pts),
              'CONFOUNDED': ('pooling across systems mixes the within-system '
                             'slope with the fact that systems with large '
                             'oracle error also have large deployed error. '
                             'The interval includes 1.0 for that reason. Do '
                             'not quote this as the result -- it is the same '
                             'confound sc_law.py warns about for the pooled '
                             'rank correlation.')}
    return out, pooled


def ordering_test(per_system, pts):
    """What survives, and what does not.

    The tempting claim is that eta is *ordered* by singularity strength. It is
    not: against the mean deployment penalty the correlation is about -0.69,
    a trend rather than an ordering, and `ks` and `film` swap. Claiming an
    ordering here would be the same overreach the pooled rank correlation is
    warned against in sc_law.py, so it is recorded as refuted.

    What does survive is a split: the viscously regularised control is the one
    system where operator accuracy clearly still pays, and every genuinely
    singular system sits at eta <= 0.15.
    """
    import numpy as np
    order = sorted(per_system, key=lambda s: -per_system[s]['eta'])
    pen = {s: float(np.mean([p['penalty'] for p in pts if p['system'] == s]))
           for s in per_system}
    r = float(np.corrcoef([pen[s] for s in per_system],
                          [per_system[s]['eta'] for s in per_system])[0, 1])
    singular = [s for s in per_system if s != 'burgers']
    return {
        'eta_descending': order,
        'burgers_rank': order.index('burgers') + 1 if 'burgers' in order else None,
        'n_systems': len(order),
        'corr_meanpenalty_eta': r,
        'ordering_by_singularity_strength': 'REFUTED -- a trend, not an ordering',
        'control_eta': per_system['burgers']['eta'] if 'burgers' in per_system else None,
        'max_eta_among_singular_systems': max(per_system[s]['eta']
                                              for s in singular),
        'n_singular_with_negative_eta': sum(per_system[s]['eta'] < 0
                                            for s in singular),
    }


def main():
    law = json.load(open(os.path.join(RES, 'law.json')))
    per_system, pooled = analyse(law['points'])
    order = ordering_test(per_system, law['points'])

    out = {
        'definition': ('eta = d(log10 deployed) / d(log10 oracle) within a '
                       'system across backbones, deployed = oracle x penalty'),
        'reading': {'1': 'accuracy transfers fully', '0': 'accuracy buys '
                    'nothing deployed', '<0': 'accuracy is counterproductive'},
        'per_system': per_system,
        'pooled': pooled,
        'ordering': order,
        'caveat': (f"eta is a slope over "
                   f"{per_system['rd']['n_backbones']} backbones per system. "
                   'The defensible claims are (a) the fit-free comparison of '
                   'oracle_range_factor against deployed_range_factor, which '
                   'assumes no model at all, and (b) the control-vs-singular '
                   'split. NOT an ordering by singularity strength (refuted), '
                   'NOT the pooled slope (confounded), and NOT any single eta '
                   'to three decimals. At n = 4 the bootstrap is badly behaved '
                   'in the low-|eta| systems -- see sidm, where near-degenerate '
                   'resamples drag the lower bound to -2.2 while the jackknife '
                   'stays inside [0.002, 0.034]. Prefer the jackknife there.'),
    }
    p = os.path.join(RES, 'transfer.json')
    json.dump(out, open(p, 'w'), indent=1)

    print('ACCURACY TRANSFER EFFICIENCY (P11)')
    print(f"{'system':9} {'eta':>7} {'bootstrap 95%':>18} "
          f"{'jackknife range':>18}  {'oracle':>7} {'deployed':>9}")
    for s in order['eta_descending']:
        v = per_system[s]
        ci = v['eta_ci95_bootstrap']
        jr = v['eta_jackknife_range']
        flag = '  <-- best-accuracy backbone is WORSE deployed' \
            if v['best_accuracy_is_worse_deployed'] else ''
        print(f"{s:9} {v['eta']:7.3f} [{ci[0]:7.3f},{ci[1]:7.3f}] "
              f"[{jr[0]:7.3f},{jr[1]:7.3f}]  "
              f"{v['oracle_range_factor']:6.1f}x {v['deployed_range_factor']:8.2f}x"
              f"{flag}")
    pc = pooled['eta_ci95_bootstrap']
    print(f"\npooled eta {pooled['eta']:.3f}  95% [{pc[0]:.3f}, {pc[1]:.3f}]"
          f"  over {pooled['n_points']} points  -- CONFOUNDED, do not quote")
    print(f"\ncontrol (burgers) eta        {order['control_eta']:+.3f}")
    print(f"largest eta among singular   "
          f"{order['max_eta_among_singular_systems']:+.3f}")
    print(f"singular systems with eta<0  "
          f"{order['n_singular_with_negative_eta']}/4")
    print(f"corr(mean penalty, eta)      {order['corr_meanpenalty_eta']:+.3f}"
          f"  -- a trend; ordering by singularity strength is REFUTED")
    print(f'-> {p}')


if __name__ == '__main__':
    main()
