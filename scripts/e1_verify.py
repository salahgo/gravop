"""E1 — solver verification.

Stages (select via argv):
  smoke      : one constant-sigma NFW run, sigma_hat=0.5, report t_c, energy drift
  conv       : convergence in n_shells and eps_t (constant sigma_hat=0.5)
  crosscheck : run reference GravothermalSIDM at matched config, compare rho(r,t), t_c
  vdep       : one Yukawa-Born run (sigma_hat0=2, w_hat=0.3) smoke test

Outputs: results/e1/*.json, logs to stdout (tee to logs/).
Collapse time here := t at central density = 1e3 x initial (operational definition;
also reports t at 10x, 100x, and final).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'e1')
os.makedirs(OUT, exist_ok=True)


def t_at_factor(res, factor):
    """Time when central density (shell 3) first reaches factor x initial."""
    rhoc = res['rho'][:, 3]
    target = factor * rhoc[0]
    idx = np.argmax(rhoc >= target)
    if rhoc[idx] < target:
        return None
    if idx == 0:
        return 0.0
    # log-interpolate between snapshots
    t0, t1 = res['t'][idx - 1], res['t'][idx]
    l0, l1 = np.log(rhoc[idx - 1]), np.log(rhoc[idx])
    lt = np.log(target)
    return float(t0 + (t1 - t0) * (lt - l0) / (l1 - l0))


def summarize(res, wall, label):
    E = res['E']
    drift = float(abs(E[-1] - E[0]) / abs(E[0]))
    t_relax = 2.0 / (3.0 * solver.A_COEF * res['params']['C']
                     * res['params']['sigma_m'])
    out = {
        'label': label, 'params': res['params'], 'status': res['status'],
        'n_steps': res['n_steps'], 'wall_seconds': wall,
        'n_snaps': len(res['t']), 't_final': res['t_final'],
        'rho_c_growth': float(res['rho'][-1, 3] / res['rho'][0, 3]),
        'energy_drift_rel': drift,
        't_relax_central_units': t_relax,
        't10': t_at_factor(res, 10.), 't100': t_at_factor(res, 100.),
        't1000': t_at_factor(res, 1000.),
    }
    for k in ('t10', 't100', 't1000'):
        if out[k] is not None:
            out[k + '_over_trelax'] = out[k] / t_relax
    return out


def run(label, **kw):
    t0 = time.time()
    res = solver.run_halo(**kw)
    wall = time.time() - t0
    s = summarize(res, wall, label)
    print(json.dumps(s, indent=1))
    with open(os.path.join(OUT, label + '.json'), 'w') as f:
        json.dump(s, f, indent=1)
    np.savez_compressed(os.path.join(OUT, label + '.npz'),
                        t=res['t'], r=res['r'], rho=res['rho'], p=res['p'],
                        E=res['E'], m=res['m'])
    return res, s


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'smoke'

    if stage == 'smoke':
        run('smoke_s0.5_N150_et1e-3', sigma_m=0.5, n_shells=150, eps_t=1e-3,
            rho_c_factor=1e3)

    elif stage == 'conv':
        for n in (100, 150, 200, 300):
            run(f'conv_N{n}_et1e-3', sigma_m=0.5, n_shells=n, eps_t=1e-3,
                rho_c_factor=1e3)
        for et in (3e-3, 3e-4, 1e-4):
            run(f'conv_N150_et{et:g}', sigma_m=0.5, n_shells=150, eps_t=et,
                rho_c_factor=1e3)

    elif stage == 'vdep':
        run('vdep_s2_w0.3_N150', sigma_m=2.0, w=0.3, model_lmfp='yukawa_tu',
            model_smfp='yukawa_tu', n_shells=150, eps_t=1e-3, rho_c_factor=1e3)

    elif stage == 'longsigma':
        # literature comparison: LMFP regime, expect t_c ~ 455 t_relax
        run('lit_s0.1_N200_et1e-3', sigma_m=0.1, n_shells=200, eps_t=1e-3,
            rho_c_factor=1e3)
    else:
        raise SystemExit(f'unknown stage {stage}')
