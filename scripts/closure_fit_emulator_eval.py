#!/usr/bin/env python3
"""The emulator retrained under a fitted closure (default: five-halo fit alpha 0.58, C 0.85;
the seven-halo fit: --dir results/closure_fit7_20261113 --prefix gravop_fit7_s --alpha 0.535 --C 0.886).

Five members results/closure_fit_20261112/runs/gravop_fit_s0..4, trained by
scripts/train.py with the production recipe, unchanged, on
data/d1/train_alpha_fit.h5 (scripts/gen_dataset_closure.py: the production
draws and seeds under alpha = 0.58, C = 0.85). Reported, as for the literature
retraining in scripts/closure_emulator_eval.py:

  * test-split accuracy (mean over seeds of each seed's test median), next to
    the production (harmonic) and literature-closure ensembles;
  * halo A: the ensemble against the N-body profiles with its own clock and no
    time rescaling;
  * haloes B, C, D1, D2, E1, E2: the ensemble's terminal time on the nearest
    in-family profile (untruncated NFW at the halo's sigma-hat) against the
    solver under the same closure, i.e. a consistent emulator-solver pair.

Writes <--dir>/emulator.json (default results/closure_fit_20261112/emulator.json).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(ROOT, 'src'), HERE,
                os.path.join(ROOT, 'experiments', 'research_extensions', 'scripts')]
import closure_emulator_eval as CE  # noqa: E402
import independent_halo_analysis as IH  # noqa: E402
from gravop import closure, solver  # noqa: E402

import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument('--dir', default='results/closure_fit_20261112')
_ap.add_argument('--prefix', default='gravop_fit_s')
_ap.add_argument('--alpha', type=float, default=0.58)
_ap.add_argument('--C', type=float, default=0.85)
_A = _ap.parse_args()
OUT = os.path.join(ROOT, _A.dir)
ALPHA, C = _A.alpha, _A.C
EXT = os.path.join(ROOT, 'experiments', 'research_extensions', 'closure_design')


def main():
    out = {'note': __doc__.strip(), 'closure': dict(alpha=ALPHA, C=C)}
    acc = {}
    for lab, d, pre in (('fitted', os.path.join(OUT, 'runs'), _A.prefix),
                        ('literature', os.path.join(ROOT, 'results', 'closure_20261108', 'runs'), 'gravop_lit_s'),
                        ('production', os.path.join(ROOT, 'results', 'runs'), 'gravop_s')):
        med = [json.load(open(os.path.join(d, f'{pre}{s}', 'metrics.json')))['test'] for s in range(5)]
        acc[lab] = {'median_rel_rho_mean_over_seeds': float(np.mean([m['median_rel_rho'] for m in med])),
                    'clock_rel_err_median_mean_over_seeds': float(np.mean([m['tc_rel_err_median'] for m in med]))}
    out['test_split'] = acc
    fit, nf = CE.ensemble(os.path.join(OUT, 'runs'), _A.prefix)
    out['halo_A'] = {'fitted_no_rescaling': CE.vs_nbody(fit, nf, 1.0)}
    for f in ('PREREGISTRATION_D.json', 'PREREGISTRATION_E.json'):
        for h, v in json.load(open(os.path.join(EXT, f)))['haloes'].items():
            IH.HALO[h] = dict(mass=v['mass'], rs=v['rs'], falloff=v['falloff'], kappa=v['kappa'], tfinal_tdyn=1.0)
    out['haloes'] = {}
    for h in ('B', 'C', 'D1', 'D2', 'E1', 'E2'):
        sh = IH.halo_units(h)['sigma_hat']
        r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
        prof, par = CE.inputs(nf, r, rho, p, sh)
        tn = CE.t_norm(fit, nf, prof, par)
        sol = closure.run_halo_g(sigma_m=sh, C=C, alpha=ALPHA, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                                 timestep_mode='implicit', rho_c_factor=1e3, kn_min=0.02, t_end=1200.0, max_snaps=3000)
        out['haloes'][h] = {'sigma_hat': sh, 'emulator_t_norm': tn, 'solver_t_final': sol['t_final'],
                            'rel': tn / sol['t_final'] - 1, 'solver_status': sol['status']}
    json.dump(out, open(os.path.join(OUT, 'emulator.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1, default=float)[:3000])


if __name__ == '__main__':
    main()
