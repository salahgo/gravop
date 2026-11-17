#!/usr/bin/env python3
"""The emulator re-aligned to the literature closure, against its solver and
the three reference haloes.

The five members results/closure_20261108/runs/gravop_lit_s0..4 are trained by
scripts/train.py with the production recipe, unchanged, on
data/d1/train_alpha_lit.h5 (scripts/gen_dataset_closure.py: the production
draws and seeds under alpha = 1/2, C = 0.9). Reported:

  * test-split accuracy against the production members (the mean over seeds of
    each seed's test median, as in the manuscript);
  * halo A: the ensemble against the N-body run of Fischer, Yu & Dolag, with
    the emulator's own clock and NO time rescaling, since the closure carries
    no calibration to this halo (scripts/e11_emulator_vs_nbody.py's scorer and
    window); the production ensemble at its calibrated rescaling for reference;
  * haloes B and C: collapse time of the ensemble on the nearest in-family
    (untruncated NFW) profile against the solver under the same closure.

Writes results/closure_20261108/emulator.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
from gravop import closure, solver, dataset as ds  # noqa: E402
from evaluate_all import load_model, NormObj  # noqa: E402
import e11_emulator_vs_nbody as E11  # noqa: E402
import independent_halo_analysis as IH  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'closure_20261108')
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def ensemble(prefix_dir, prefix):
    ms, n = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(prefix_dir, f'{prefix}{s}'))
        ms.append(m.to(DEV).eval())
        n = NormObj(nrm)
    return ms, n


def inputs(n, r, rho, p, sigma_hat):
    lr0, lv0 = ds.regrid_snapshot(r, rho, p)
    prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd, (lv0 - n.v_mu) / n.v_sd]),
                        dtype=torch.float32, device=DEV)[None]
    par = torch.tensor([[(np.log10(sigma_hat) - n.sig_mu) / n.sig_sd, (0.0 - n.w_mu) / n.w_sd, 0.0]],
                       dtype=torch.float32, device=DEV)
    return prof, par


def t_norm(models, n, prof, par):
    with torch.no_grad():
        h = [m(prof, torch.cat([par, torch.zeros(1, 1, device=DEV)], 1))[1][0, 0].item() for m in models]
    return float(10 ** (np.mean(h) * n.tc_sd + n.tc_mu))


def vs_nbody(models, n, s_rescale):
    r, _, rho, p = solver.nfw_profile(n_shells=300)
    prof, par = inputs(n, r, rho, p, E11.SIGMA_HAT)
    tn = t_norm(models, n, prof, par)
    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    keys = sorted({k.rsplit('_', 1)[0] for k in d.files})
    lg = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
    sq, cnt, rows = 0.0, 0, []
    for k in keys:
        t_nb = float(d[k + '_t']) / E11.T0G * s_rescale
        if t_nb == 0.0:
            continue
        tau = t_nb / tn
        if tau > 1.0:
            rows.append({'snapshot': k, 'tau': tau, 'note': 'beyond the emulator horizon'})
            continue
        cond = torch.cat([par, torch.tensor([[tau]], dtype=torch.float32, device=DEV)], 1)
        with torch.no_grad():
            mu = np.mean([m(prof, cond)[0][0, 0].cpu().numpy() for m in models], axis=0) * n.rho_sd + n.rho_mu
        rr = d[k + '_r'] / E11.R_S
        rh = d[k + '_rho'] / (E11.RHO_S * 1e-10)
        sel = (rr > E11.FIT_RANGE[0]) & (rr < E11.FIT_RANGE[1]) & (rh > 0)
        dif = np.interp(np.log10(rr[sel]), lg, mu) - np.log10(rh[sel])
        sq += np.sum(dif ** 2)
        cnt += len(dif)
        rows.append({'snapshot': k, 'tau': tau, 'rms_dex': float(np.sqrt(np.mean(dif ** 2)))})
    return {'t_norm': tn, 'time_rescaling': s_rescale, 'global_rms_dex': float(np.sqrt(sq / cnt)),
            'per_snapshot': rows}


def main():
    out = {'note': __doc__.strip()}
    acc = {}
    for lab, d, pre in (('literature', os.path.join(OUT, 'runs'), 'gravop_lit_s'),
                        ('production', os.path.join(ROOT, 'results', 'runs'), 'gravop_s')):
        med = [json.load(open(os.path.join(d, f'{pre}{s}', 'metrics.json')))['test'] for s in range(5)]
        acc[lab] = {'median_rel_rho_mean_over_seeds': float(np.mean([m['median_rel_rho'] for m in med])),
                    'clock_rel_err_median_mean_over_seeds': float(np.mean([m['tc_rel_err_median'] for m in med]))}
    out['test_split'] = acc
    lit, nl = ensemble(os.path.join(OUT, 'runs'), 'gravop_lit_s')
    prod, npr = ensemble(os.path.join(ROOT, 'results', 'runs'), 'gravop_s')
    out['halo_A'] = {'literature_no_rescaling': vs_nbody(lit, nl, 1.0),
                     'production_calibrated': vs_nbody(prod, npr, E11.S_RESCALE)}
    out['haloes_BC'] = {}
    for h in ('B', 'C'):
        u = IH.halo_units(h)
        r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
        prof, par = inputs(nl, r, rho, p, u['sigma_hat'])
        tn = t_norm(lit, nl, prof, par)
        sol = closure.run_halo_g(sigma_m=u['sigma_hat'], C=0.9, alpha=0.5, r=r, m=m, rho=rho, p=p,
                                 eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e3,
                                 kn_min=0.02, t_end=1200.0, max_snaps=3000)
        out['haloes_BC'][h] = {'emulator_t_norm': tn, 'solver_t_final': sol['t_final'],
                               'rel': tn / sol['t_final'] - 1, 'solver_status': sol['status']}
    json.dump(out, open(os.path.join(OUT, 'emulator.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1, default=float)[:2500])


if __name__ == '__main__':
    main()
