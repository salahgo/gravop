#!/usr/bin/env python3
"""Controlled baryonic experiments on the validated halo (halo A).

Halo A is the isolated N-body halo of Fischer, Yu & Dolag (2025): NFW,
r_s = 1.28 kpc, rho_s = 4.42e7 Msun/kpc^3, constant sigma/m = 80 cm^2/g
(sigma-hat = 0.9455), evolved at the conductivity calibrated on it,
C_eff = 0.712. The baryonic configurations are the dimensionless baryon
structures of Zhong, Yang & Yu (2023), applied to this halo:

  A  dark matter only (the validated evolution)
  B  extended: M_b-hat = 0.0153, a-hat = 0.132   (their 'baryonD')
  C  compact:  M_b-hat = 0.0306, a-hat = 0.0934  (their 'baryonC')

For halo A, M0 = 4 pi rho_s r_s^3 = 1.17e9 Msun, so B and C hold 1.8e7 and
3.6e7 Msun of baryons with Hernquist radii of 0.17 and 0.12 kpc.

A 3 x 3 sensitivity grid in (M_b-hat, a-hat) follows, with the ratio of the
terminal time to the dark-matter-only one compared with the analytic estimate
of Zhong et al. (their Eqs. 16-17, alpha = 0.4).

Measured on every trajectory, from the solver's shells: the central density
rho_c (shell 3, the production observable), its minimum and the time of the
minimum (core expansion), the core radius (where rho falls to rho_c / 2), the
central one-dimensional velocity dispersion, and the terminal time (Knudsen
floor 0.02 or rho_c = 1e3 rho_c(0)). All at production settings, r_min = 0.003.

Writes results/baryons_20261107/cases.json (and cases_traj.npz for the figure).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import baryons as B  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'baryons_20261107')
SIGMA_HAT, C_EFF = 0.94553884672, 0.712
R_S, RHO_S = 1.28, 4.42e7
T0_GYR = 1.0 / np.sqrt(4 * np.pi * 4.30091e-6 * RHO_S) * 0.97779
M0 = 4 * np.pi * RHO_S * R_S ** 3
KW = dict(n_shells=300, r_min=0.003, eps_t=3e-4, rho_c_factor=1e3, t_end=1200.0,
          max_snaps=3000, timestep_mode='implicit', kn_min=0.02)
CASES = {'A': (0.0, 0.1), 'B': (0.0153, 0.132), 'C': (0.0306, 0.0934)}
GRID_M = (0.003, 0.01, 0.03)
GRID_A = (0.03, 0.1, 0.3)
ALPHA = 0.4


def zhong_ratio(Mb, ab):
    """t_*(baryons) / t_*(dark matter only) from Zhong et al. Eqs. 16-17, in
    units of r_s, rho_s, M0 (|Phi_chi(0)| = 1, rho_h = 2 Mb / a^3)."""
    rho_eff = (1.0 + ALPHA * (2 * Mb / ab ** 3) * ab) / (1.0 + ALPHA * ab)
    return 1.0 / (rho_eff * np.sqrt(1.0 + Mb / ab))


def measures(res):
    t = res['t']
    rho, p, r = res['rho'], res['p'], res['r']
    rc = rho[:, 3]
    vc = np.sqrt(p[:, 3] / rho[:, 3])
    rcore = np.empty(len(t))
    for i in range(len(t)):
        rm = np.concatenate([[0.5 * r[i, 0]], 0.5 * (r[i, 1:] + r[i, :-1])])
        j = np.argmax(rho[i] < 0.5 * rc[i])
        rcore[i] = rm[j]
    imin = int(np.argmin(rc))
    imax = int(np.argmax(rcore))
    return dict(t=t, rho_c=rc / rc[0], v_c=vc, r_core=rcore), {
        't_final': res['t_final'], 't_final_Gyr': res['t_final'] * T0_GYR,
        'status': res['status'],
        'rho_c_min_over_initial': float(rc[imin] / rc[0]),
        't_min_over_t_final': float(t[imin] / res['t_final']),
        'r_core_max': float(rcore[imax]), 'r_core_max_kpc': float(rcore[imax] * R_S),
        'v_c_at_min_over_initial': float(vc[imin] / vc[0]),
        'v_c_initial_kms': float(vc[0] * np.sqrt(4 * np.pi * 4.30091e-6 * RHO_S) * R_S),
        'rel_energy_drift': float(np.max(np.abs(res['E'] - res['E'][0])) / abs(res['E'][0])),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'note': __doc__.strip(), 'halo_A': {'sigma_hat': SIGMA_HAT, 'C_eff': C_EFF,
                                               't0_Gyr': T0_GYR, 'M0_Msun': M0}}
    traj, cases = {}, {}
    for k, (Mb, ab) in CASES.items():
        res = B.run_halo_b(SIGMA_HAT, C=C_EFF, Mb=Mb, ab=ab, **KW)
        tr, m = measures(res)
        m.update(Mb_hat=Mb, a_hat=ab, Mb_Msun=Mb * M0, a_kpc=ab * R_S)
        cases[k] = m
        for q, v in tr.items():
            traj[f'{k}_{q}'] = v
    tA = cases['A']['t_final']
    for k in cases:
        cases[k]['t_final_over_dmo'] = cases[k]['t_final'] / tA
        if k != 'A':
            cases[k]['zhong_estimate_ratio'] = zhong_ratio(cases[k]['Mb_hat'], cases[k]['a_hat'])
    out['cases'] = cases
    grid = []
    for Mb in GRID_M:
        for ab in GRID_A:
            res = B.run_halo_b(SIGMA_HAT, C=C_EFF, Mb=Mb, ab=ab, **KW)
            _, m = measures(res)
            grid.append(dict(Mb_hat=Mb, a_hat=ab, phi_b0=Mb / ab, **m,
                             t_final_over_dmo=res['t_final'] / tA,
                             zhong_estimate_ratio=zhong_ratio(Mb, ab)))
    out['grid'] = grid
    json.dump(out, open(os.path.join(OUT, 'cases.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'cases_traj.npz'), **traj)
    for k, v in cases.items():
        print(k, {q: round(x, 4) if isinstance(x, float) else x for q, x in v.items()})
    for g in grid:
        print(f"Mb {g['Mb_hat']:.3f} a {g['a_hat']:.2f}: ratio {g['t_final_over_dmo']:.3f} "
              f"Zhong {g['zhong_estimate_ratio']:.3f} min {g['rho_c_min_over_initial']:.3f} "
              f"status {g['status']}")


if __name__ == '__main__':
    main()
