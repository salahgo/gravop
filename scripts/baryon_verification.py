#!/usr/bin/env python3
"""Verification of gravop.baryons (static Hernquist baryons) before any use.

  1. M_b = 0 reproduces gravop.solver.run_halo bit for bit (constant and
     Yukawa-Born cross-sections, production settings).
  2. The initial state with baryons is in hydrostatic equilibrium: the first
     Newton step of the baryonic hydrostatic solve moves no shell by more than
     the integration error of the Jeans quadrature.
  3. Energy, including the dark matter's energy in the static baryonic
     potential, is conserved through the evolution.
  4. Published benchmark: the collapse times t_* of Zhong, Yang & Yu (2023,
     MNRAS 526, 758, Table 3), for their constant-cross-section runs, each at
     their own calibration constant beta and their grid (152 shells,
     r-hat in [1e-2, 1e3]); t_* is the time at which the central Knudsen
     number reaches 0.1.
  5. Grid convergence for the most compact configuration.

Writes results/baryons_20261107/verification.json.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import baryons as B, solver  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'baryons_20261107')
G = 4.30091e-6                     # kpc (km/s)^2 / Msun
MSUN_KPC2 = 2.08909e-10            # Msun/kpc^2 -> g/cm^2
KPC_KMS_GYR = 3.0857e16 / 3.156e16  # kpc/(km/s) in Gyr

# Zhong, Yang & Yu (2023): halo and Table 3 (constant cross-sections)
ZHONG_HALO = dict(rho_s=6.9e6, r_s=9.1)
ZHONG_T3 = [  # name, sigma/m [cm^2/g], M_b [Msun], r_h [kpc], beta, t_* [Gyr]
    ('SIDM10-only', 10.0, 0.0, None, 0.84, 69.0),
    ('SIDM10+baryonM', 10.0, 1.0e9, 0.77, 0.89, 29.0),
    ('SIDM10+baryonD', 10.0, 1.0e9, 1.2, 0.91, 36.0),
    ('SIDM10+baryonC', 10.0, 2.0e9, 0.85, 0.83, 16.0),
    ('SIDM100+baryonM', 100.0, 1.0e9, 0.77, 0.58, 5.5),
]


def zhong_units():
    rho_s, r_s = ZHONG_HALO['rho_s'], ZHONG_HALO['r_s']
    M0 = 4 * np.pi * rho_s * r_s ** 3
    t0 = KPC_KMS_GYR / np.sqrt(4 * np.pi * G * rho_s)
    return M0, t0, rho_s * r_s * MSUN_KPC2


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'note': __doc__.strip()}
    prod = dict(n_shells=300, eps_t=3e-4, rho_c_factor=1e3, t_end=1200.0,
                timestep_mode='implicit', kn_min=0.02)

    # 1. bit identity at M_b = 0
    ident = {}
    for sig, w, model in [(0.5, 1.0, 'constant'), (2.0, 1.5, 'yukawa_tu')]:
        a = solver.run_halo(sig, w=w, model_lmfp=model, model_smfp=model, **prod)
        b = B.run_halo_b(sig, w=w, model_lmfp=model, model_smfp=model, Mb=0.0, ab=0.1, **prod)
        ident[model] = bool(all(np.array_equal(a[k], b[k]) for k in ('t', 'r', 'rho', 'p', 'E'))
                            and a['t_final'] == b['t_final'] and a['status'] == b['status'])
    out['bit_identical_at_Mb_0'] = ident

    # 2. initial hydrostatic equilibrium with baryons
    hs = {}
    for Mb, ab in [(0.0153, 0.132), (0.0306, 0.0934), (0.1, 0.03)]:
        r, m, rho, p = B.profile_with_baryons(Mb, ab, r_min=min(0.01, ab / 10), n_shells=300)
        r1, rho1, p1 = r.copy(), rho.copy(), p.copy()
        step = B._hydro_step_b(r1, m, rho1, p1, Mb, ab)
        r2, rho2, p2 = r.copy(), rho.copy(), p.copy()
        step_dmo = B._hydro_step_b(r2, m, rho2, p2, 0.0, ab)   # same profile, baryons ignored
        hs[f'{Mb}_{ab}'] = {'max_rel_shell_move_with_baryons': float(step),
                            'max_rel_shell_move_if_baryons_ignored': float(step_dmo)}
    r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
    hs['dmo_production_baseline'] = {'max_rel_shell_move': float(
        solver._hydro_step(r.copy(), m, rho.copy(), p.copy()))}
    out['initial_hydrostatic_equilibrium'] = hs

    # 3. energy conservation with baryons (diagnostic includes the external term)
    en = {}
    for Mb, ab in [(0.0153, 0.132), (0.0306, 0.0934)]:
        res = B.run_halo_b(0.5, Mb=Mb, ab=ab, **prod)
        E = res['E']
        en[f'{Mb}_{ab}'] = {'max_rel_energy_drift': float(np.max(np.abs(E - E[0])) / abs(E[0])),
                            't_final': res['t_final'], 'status': res['status']}
    out['energy_conservation'] = en

    # 4. Zhong, Yang & Yu (2023) Table 3
    M0, t0, rsig = zhong_units()
    out['zhong_units'] = {'M0_Msun': M0, 't0_Gyr': t0, 'rho_s_r_s_g_cm2': rsig}
    rows = []
    for name, sig, Mb, rh, beta, t_pub in ZHONG_T3:
        Mbh = Mb / M0
        ab = (rh / ZHONG_HALO['r_s']) if rh else 1.0
        t1 = time.time()
        res = B.run_halo_b(sig * rsig, C=beta, Mb=Mbh, ab=ab, n_shells=152, r_min=1e-2,
                           r_max=1e3, eps_t=3e-4, timestep_mode='implicit',
                           rho_c_factor=1e12, kn_min=0.1, t_end=5000.0)
        t_star = res['t_final'] * t0
        rows.append(dict(name=name, sigma_m=sig, sigma_hat=sig * rsig, Mb_hat=Mbh, a_hat=ab,
                         beta=beta, status=res['status'], t_star_Gyr=t_star,
                         t_star_published_Gyr=t_pub, rel_diff=t_star / t_pub - 1,
                         wall_s=time.time() - t1))
    out['zhong_table3'] = rows

    # 5. grid convergence, most compact Zhong benchmark at production settings
    conv = {}
    Mbh, ab = 2.0e9 / M0, 0.85 / ZHONG_HALO['r_s']
    for ns, rmin in [(300, 0.01), (600, 0.01), (300, 0.003)]:
        res = B.run_halo_b(10 * rsig, C=0.753, Mb=Mbh, ab=ab, n_shells=ns, r_min=rmin,
                           **{k: v for k, v in prod.items() if k != 'n_shells'})
        conv[f'{ns}_{rmin}'] = {'t_final': res['t_final'], 'status': res['status']}
    ref = conv['600_0.01']['t_final']
    out['grid_convergence_compact'] = {'runs': conv,
                                       'rel_change_300_to_600': conv['300_0.01']['t_final'] / ref - 1,
                                       'rel_change_rmin_0.01_to_0.003':
                                       conv['300_0.003']['t_final'] / conv['300_0.01']['t_final'] - 1}

    json.dump(out, open(os.path.join(OUT, 'verification.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1, default=float))


if __name__ == '__main__':
    main()
