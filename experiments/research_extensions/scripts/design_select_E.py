#!/usr/bin/env python3
"""Pre-registration of an untouched halo family E for the frozen closure.

After the five-halo fit (learned_closures/closure_fit5.json) was final, two
haloes from a profile family not used in any fit or design choice (NSphere
truncated NFW with concentration 19 and 10; D1/D2 used 6, and B used 19 at a
different Knudsen number) are predicted with the frozen closure, before any
particle run:
  E1  c = 19, sigma-hat 1.933: Kn_c ~ 1.1, matched to D2 at another concentration
      (a particle test of the single-coordinate dependence on Kn_c);
  E2  c = 10, sigma-hat 4.533: Kn_c ~ 0.46, below the fitted range 0.6-6.7
      (an extrapolation test).
Predictions by direct solver runs (no grid interpolation) for the frozen fit,
the harmonic closure and van den Bosch & Tokayer's closure. Observable t_x10 in
t_dyn, three seeds at 1e5 particles, scored with score rule below. Also adds
the halo B half-time-step control B_dt2_s1 (repeat of B1's seed).

Writes closure_design/PREREGISTRATION_E.json and appends the runs to
scripts/nsphere_runs_extra.json.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import independent_halo_analysis as IH  # noqa: E402
from gravop import closure, solver  # noqa: E402

MASS, RS = 1.0e10, 3.0
PICK = {'E1': (1.9333, 19.0), 'E2': (4.5329, 10.0)}
SEEDS = (31, 32, 33)
STEPS_PER_TDYN = 12_500_000 / 1100.0


def predict(h, a, C):
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                             timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=2000.0, max_snaps=3000)
    mr = np.array([IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])[1]
                   for i in range(len(run['t']))])
    t = IH.collapse_markers(run['t'], mr[:, 1])['t_x10']
    i0 = int(np.argmin(run['rho'][:, 3]))
    kn = float(1.0 / (np.sqrt(run['p'][i0, 3]) * u['sigma_hat']))
    return dict(alpha=a, C=C, t_x10_t0=t, t_x10_tdyn=t * u['t0_myr'] / u['tdyn_myr'], Kn_c=kn)


def main():
    fitp = os.path.join(EXT, 'learned_closures', 'closure_fit5.json')
    fit = json.load(open(fitp))
    frozen = (round(fit['all']['alpha'], 4), round(fit['all']['C'], 4))
    pre = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               frozen_closure=dict(alpha=frozen[0], C=frozen[1],
                                   source_sha256=hashlib.sha256(open(fitp, 'rb').read()).hexdigest()),
               rule=('The closure is frozen at the five-halo fit; nothing is refitted on E. Observable: t_x10 (first '
                     'time the mean density inside 0.3% of the mass exceeds 10x its pre-collapse minimum), in t_dyn, '
                     'seed mean of three 1e5-particle runs (seeds 31-33), same measurement pipeline as B, C, D. '
                     'Score: z = ln(T_obs / T_pred) / sigma_y, sigma_y = hypot(max(sd_ln, 0.03) / sqrt(n), 0.03). '
                     'Criterion: the frozen closure passes on a halo if |z| <= 2. E1 lies inside the fitted Kn_c range '
                     'and is a validation; E2 lies below it and is an extrapolation test, reported either way.'),
               haloes={})
    extra = json.load(open(os.path.join(HERE, 'nsphere_runs_extra.json')))
    for name, (sig, c) in PICK.items():
        IH.HALO[name] = dict(mass=MASS, rs=RS, falloff=c, kappa=1.0, tfinal_tdyn=1.0)
        kappa = round(sig / IH.halo_units(name)['sigma_hat'], 4)
        IH.HALO[name]['kappa'] = kappa
        preds = {'frozen_fit': predict(name, *frozen), 'harmonic': predict(name, 1.0, 0.712),
                 'literature': predict(name, 0.5, 0.9)}
        tmax = max(p['t_x10_tdyn'] for p in preds.values())
        tfinal = int(np.ceil(1.4 * tmax / 10.0) * 10)
        dtw = int(round(STEPS_PER_TDYN * tfinal / 500))
        extra['haloes'][name] = dict(mass=MASS, rs=RS, falloff=c, kappa=kappa)
        for s in SEEDS:
            extra['runs'][f'{name}_s{s}'] = dict(group='closure_design', halo=name, n=100_000, seed=s,
                                                tfinal=tfinal, dtwrite=dtw, steps=500 * dtw + 1)
        pre['haloes'][name] = dict(sigma_hat=IH.halo_units(name)['sigma_hat'], falloff=c, mass=MASS, rs=RS,
                                   kappa=kappa, tfinal_tdyn=tfinal, predictions=preds)
    # halo B half-time-step control: repeat of B1 (seed 1, 1e5) at half the step, to the B5/B6 horizon
    extra['runs']['B_dt2_s1'] = dict(group='multifidelity', halo='B', n=100_000, seed=1, tfinal=850,
                                     dtwrite=38636, steps=500 * 38636 + 1)
    s = json.dumps(pre, indent=1, sort_keys=True)
    pre['sha256_of_content_above'] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(pre, open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_E.json'), 'w'), indent=1)
    json.dump(extra, open(os.path.join(HERE, 'nsphere_runs_extra.json'), 'w'), indent=1)
    for k, v in pre['haloes'].items():
        print(k, v['kappa'], v['tfinal_tdyn'], {c: (round(p['t_x10_tdyn'], 2), round(p['Kn_c'], 3))
                                                for c, p in v['predictions'].items()})


if __name__ == '__main__':
    main()
