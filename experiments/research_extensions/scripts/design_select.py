#!/usr/bin/env python3
"""Freeze the discriminating haloes D1, D2 and every closure's prediction for
them BEFORE any particle data exist (preregistration), and register their
NSphere runs.

Selection (from closure_design/design_scan.json): the valley of closures that
fit haloes A, B, C (learned_closures/identifiability_ref2e5.json) spreads by
+-30% at sigma-hat ~ 3.4 and +-15% at ~1.9 (core Kn < ~1), where no reference
exists; c = 6 is the cheapest truncation there. Closure parameters may be
fitted only on A, B, C (and their resolution ladders); D1, D2 are tests.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import design_scan as DS  # noqa: E402

PICK = {'D1': (3.412, 6.0), 'D2': (1.933, 6.0)}
MASS, RS = 1.0e10, 3.0          # units only; the dimensionless halo is (sigma-hat, c)
RUN_PLAN = [('s21', 100_000, 21, 1), ('s22', 100_000, 22, 1), ('s23', 100_000, 23, 1),
            ('N2e5_s24', 200_000, 24, 1), ('N2e5_s25', 200_000, 25, 1), ('dt2_s21', 100_000, 21, 2)]


def main():
    R = json.load(open(os.path.join(EXT, 'closure_design', 'design_scan.json')))['results']
    pre = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               rule='Closure parameters may be fitted only on haloes A, B, C and their ladders; D1/D2 are held-out tests. '
                    'Observable: t_x10 (first time the mean density inside 0.3% of the mass exceeds 10x its pre-collapse '
                    'minimum), seed mean at 1e5 and 2e5 particles, same pipeline as B and C.',
               haloes={})
    extra = {'haloes': {}, 'runs': {}}
    for name, (sig, c) in PICK.items():
        rows = [r for r in R if abs(r['sigma_hat'] - sig) < 2e-3 and r['falloff'] == c]
        preds = {r['closure']: dict(alpha=r['alpha'], C=r['C'], t_x10_t0=r['t_x10'], t_x10_tdyn=r['t_x10_tdyn'],
                                     Kn_c=r['Kn_c']) for r in rows}
        tmax = max(p['t_x10_tdyn'] for p in preds.values())
        tfinal = int(np.ceil(1.5 * tmax / 10.0) * 10)
        # exact kappa for this mass and radius
        import independent_halo_analysis as IH
        IH.HALO['tmp'] = dict(mass=MASS, rs=RS, falloff=c, kappa=1.0, tfinal_tdyn=1.0)
        kappa = sig / IH.halo_units('tmp')['sigma_hat']
        extra['haloes'][name] = dict(mass=MASS, rs=RS, falloff=c, kappa=round(kappa, 4))
        IH.HALO['tmp']['kappa'] = round(kappa, 4)
        sig_exact = IH.halo_units('tmp')['sigma_hat']
        dtw = int(round(DS_STEPS(tfinal) / 500))
        for tag, n, seed, dtdiv in RUN_PLAN:
            run = f'{name}_{tag}'
            dw = dtw * dtdiv
            extra['runs'][run] = dict(group='closure_design', halo=name, n=n, seed=seed, tfinal=tfinal,
                                      dtwrite=dw, steps=500 * dw + 1)
        pre['haloes'][name] = dict(sigma_hat=sig_exact, falloff=c, mass=MASS, rs=RS, kappa=round(kappa, 4),
                                   tfinal_tdyn=tfinal, predictions=preds,
                                   thread_hours_estimate={t: 0.066 * n / 1e5 * tfinal * d for t, n, _, d in RUN_PLAN})
    s = json.dumps(pre, indent=1, sort_keys=True)
    pre['sha256_of_content_above'] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(pre, open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json'), 'w'), indent=1)
    json.dump(extra, open(os.path.join(EXT, 'scripts', 'nsphere_runs_extra.json'), 'w'), indent=1)
    for k, v in pre['haloes'].items():
        print(k, round(v['sigma_hat'], 4), v['kappa'], v['tfinal_tdyn'],
              {c: round(p['t_x10_tdyn'], 1) for c, p in v['predictions'].items()},
              'thread-h', round(sum(v['thread_hours_estimate'].values()), 1))


def DS_STEPS(tfinal):
    return 12_500_000 / 1100.0 * tfinal


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(DS.ROOT, 'scripts'))
    main()
