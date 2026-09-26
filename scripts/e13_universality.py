"""E13 — universality of gravothermal collapse across velocity dependence.

This is the experiment that justifies transferring a conductivity calibration
obtained from ONE constant-cross-section N-body run to the velocity-dependent
models the emulator spans.

Outmezguine et al. (2023, MNRAS 523, 4786) showed that the gravothermal
evolution of isolated SIDM haloes is approximately universal: velocity
-dependent models map onto velocity-independent ones once time is measured in
units of the scattering timescale evaluated at the epoch of MINIMUM central
density, and the collapse time is then "fully defined by the scattering
timescale, with negligible variation due to the velocity dependence".

If our solver independently reproduces that universality, then

  (i) the velocity-dependent branch is validated against a nontrivial published
      result, not merely against our own constant-cross-section case; and
  (ii) the calibration constant C, fitted to one constant-cross-section N-body
      run, transfers to velocity-dependent models by the model's own scaling
      rather than by assumption.

Definition used (matching the reference convention):
    t_c0 = 2 / (3 a C sigma_eff rho_c v_c)
evaluated at the epoch of minimum central density, with
    sigma_eff = sigma_0 F(v_c / w)
the same viscosity-averaged Yukawa-Born F used in the conductivity. The
universality statistic is t_coll / t_c0, which is independent of C because both
numerator and denominator scale as 1/C in the long-mean-free-path regime.

Usage: e13_universality.py [--n_runs 60] [--workers 20]
Output: results/e13/universality.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e13')
os.makedirs(OUT, exist_ok=True)

C_CAL = solver.C_DEFAULT
COLLAPSE_FACTOR = 1e3


def analyse(res, sigma_hat, w_hat, model):
    """Return t_coll, t_c0 (Outmezguine convention) and diagnostics."""
    rhoc = res['rho'][:, 3]
    pc = res['p'][:, 3]
    vc = np.sqrt(pc / rhoc)

    imin = int(np.argmin(rhoc))
    if imin == 0 or imin >= len(rhoc) - 1:
        return None                      # never formed a core: not applicable

    mid, mp_ = solver.model_params(model, yb_index=5)
    tgt = COLLAPSE_FACTOR * rhoc[0]
    if rhoc.max() < tgt:
        return None                      # did not collapse within the horizon
    i = int(np.argmax(rhoc >= tgt))
    t0, t1 = res['t'][i - 1], res['t'][i]
    l0, l1 = np.log(rhoc[i - 1]), np.log(rhoc[i])
    t_coll = float(t0 + (t1 - t0) * (np.log(tgt) - l0) / (l1 - l0))

    # scattering timescale at the epoch of minimum central density
    F = solver._F_elastic(vc[imin] / w_hat, mid, mp_)
    sigma_eff = sigma_hat * F
    t_c0 = 2.0 / (3.0 * solver.A_COEF * C_CAL * sigma_eff
                  * rhoc[imin] * vc[imin])
    # Knudsen number at the core, Kn = 1/(sqrt(p) sigma F): the fluid model is
    # long-mean-free-path (where universality is claimed) for Kn >> 1.
    kn_min_epoch = 1.0 / (np.sqrt(pc[imin]) * sigma_eff)
    # and at the moment the collapse threshold is crossed
    F_i = solver._F_elastic(vc[i] / w_hat, mid, mp_)
    kn_collapse = 1.0 / (np.sqrt(pc[i]) * sigma_hat * F_i)
    return {'t_coll': t_coll, 't_c0': float(t_c0),
            'ratio': float(t_coll / t_c0),
            'ratio_from_min': float((t_coll - res['t'][imin]) / t_c0),
            't_min': float(res['t'][imin]),
            'rho_c_min_over_init': float(rhoc[imin] / rhoc[0]),
            'v_c_min': float(vc[imin]), 'F_at_min': float(F),
            'sigma_eff': float(sigma_eff),
            'kn_at_min': float(kn_min_epoch),
            'kn_at_collapse': float(kn_collapse),
            'knudsen_limited': int(res['status'] == 2)}


def one(args):
    idx, seed = args
    rng = np.random.default_rng(seed)
    # constant-cross-section control (30%) and Yukawa-Born (70%),
    # spanning the emulator's training range
    if rng.random() < 0.3:
        model = 'constant'
        sigma_hat = 10 ** rng.uniform(-1.5, 0.3)
        w_hat = 1.0
    else:
        model = 'yukawa_tu'
        sigma_hat = 10 ** rng.uniform(-1.0, 0.7)
        w_hat = 10 ** rng.uniform(-1.0, 1.5)
    try:
        res = solver.run_halo(sigma_m=sigma_hat, w=w_hat,
                              model_lmfp=model, model_smfp=model,
                              n_shells=300, eps_t=3e-4,
                              rho_c_factor=COLLAPSE_FACTOR,
                              t_end=4000.0, kn_min=0.0,
                              max_steps=100_000_000,
                              timestep_mode='implicit')
        a = analyse(res, sigma_hat, w_hat, model)
        if a is None:
            return None
        a.update({'sigma_hat': sigma_hat, 'w_hat': w_hat, 'model': model,
                  'seed': seed, 'status': res['status']})
        return a
    except Exception as exc:                       # noqa: BLE001
        return {'error': str(exc), 'sigma_hat': sigma_hat, 'w_hat': w_hat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_runs', type=int, default=60)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--seed', type=int, default=515151)
    args = ap.parse_args()

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        raw = pool.map(one, [(i, args.seed + i) for i in range(args.n_runs)])
    rows = [r for r in raw if r and 'error' not in r]
    errs = [r for r in raw if r and 'error' in r]
    print(f'{len(rows)}/{args.n_runs} usable ({len(errs)} errors, '
          f'{sum(1 for r in raw if r is None)} no-collapse) '
          f'in {time.time() - t0:.0f}s', flush=True)

    ratios = np.array([r['ratio'] for r in rows])
    yuk = np.array([r['model'] == 'yukawa_tu' for r in rows])
    kn = np.array([r['kn_at_min'] for r in rows])
    lmfp = kn > 10.0        # long-mean-free-path core at the reference epoch

    def stats(mask, label):
        v = ratios[mask]
        if len(v) < 2:
            return None
        return {'label': label, 'n': int(len(v)), 'mean': float(v.mean()),
                'median': float(np.median(v)), 'std': float(v.std()),
                'frac_scatter': float(v.std() / v.mean()),
                'min': float(v.min()), 'max': float(v.max())}

    out = {'definition': 't_c0 = 2/(3 a C sigma_eff rho_c v_c) at minimum '
                         'central density; ratio = t_coll(1000x)/t_c0',
           'C_used': C_CAL,
           'note': 'ratio is independent of C in the LMFP regime',
           'reference': 'Outmezguine et al. 2023 (arXiv:2204.06568) report '
                        't_coll ~= 335 t_c0 with negligible variation across '
                        'velocity dependence',
           'all': stats(np.ones_like(yuk, dtype=bool), 'all models'),
           'constant_only': stats(~yuk, 'constant cross-section'),
           'yukawa_only': stats(yuk, 'Yukawa-Born'),
           'lmfp_only': stats(lmfp, 'LMFP core (Kn>10)'),
           'lmfp_yukawa': stats(lmfp & yuk, 'LMFP + Yukawa-Born'),
           'lmfp_constant': stats(lmfp & ~yuk, 'LMFP + constant'),
           'kn_at_min_range': [float(kn.min()), float(kn.max())],
           'runs': rows}
    for k in ('all', 'constant_only', 'yukawa_only', 'lmfp_only',
              'lmfp_yukawa', 'lmfp_constant'):
        if out[k]:
            s = out[k]
            print(f"  {s['label']:26s} n={s['n']:3d}  "
                  f"t_coll/t_c0 = {s['mean']:6.1f} +- {s['std']:5.1f}  "
                  f"({100 * s['frac_scatter']:.1f}% scatter)", flush=True)
    # does the ratio depend on the strength of velocity dependence?
    if yuk.sum() > 3:
        sel = yuk & lmfp
        lw = np.array([np.log10(r['w_hat']) for r in rows])[sel]
        rr = ratios[sel]
        out['yukawa_ratio_vs_logw_pearson'] = float(
            np.corrcoef(lw, rr)[0, 1])
        print(f"  correlation of ratio with log10 w-hat (LMFP only): "
              f"{out['yukawa_ratio_vs_logw_pearson']:+.3f}", flush=True)
    json.dump(out, open(os.path.join(OUT, 'universality.json'), 'w'), indent=1)
    print('saved results/e13/universality.json')


if __name__ == '__main__':
    main()
