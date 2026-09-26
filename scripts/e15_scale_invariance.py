"""E15 — exact halo-mass generalisation.

Both external reviews asked whether the emulator generalises across halo
masses, and proposed training on small haloes and testing on large ones. In the
dimensionless formulation of Section 2 that question has an exact answer rather
than an empirical one: the fluid equations contain no residual mass scale, so
two physically different haloes that map to the same dimensionless cross-section
follow *identical* trajectories. Halo-mass generalisation is therefore exact by
construction and does not have to be learned.

This script demonstrates it end to end for a 10^9 M_sun dwarf and a
10^15 M_sun cluster - six orders of magnitude apart in mass - and checks that
the physical read-outs differ exactly as the unit conversions demand.

Output: results/e15/scale_invariance.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver                                   # noqa: E402
from gravop.units import HaloUnits, matched_cross_section    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e15')
os.makedirs(OUT, exist_ok=True)


def main():
    dwarf = HaloUnits.from_nfw(m200=1e9, c200=15)
    cluster = HaloUnits.from_nfw(m200=1e15, c200=5)
    print('dwarf  :', dwarf)
    print('cluster:', cluster)

    sig_dwarf = 5.0                                   # cm^2/g
    sig_clust = matched_cross_section(dwarf, cluster, sig_dwarf)
    sh_a, sh_b = dwarf.sigma_hat(sig_dwarf), cluster.sigma_hat(sig_clust)
    print(f'\nmatched sigma/m: dwarf {sig_dwarf:.3f} cm^2/g  <->  '
          f'cluster {sig_clust:.4g} cm^2/g')
    print(f'  common sigma_hat = {sh_a:.8f} (cluster {sh_b:.8f})')

    kw = dict(n_shells=150, eps_t=3e-4, rho_c_factor=1e3,
              timestep_mode='implicit')
    a = solver.run_halo(sigma_m=sh_a, **kw)
    b = solver.run_halo(sigma_m=sh_b, **kw)

    # The two sigma_hat values differ by one floating-point ulp because they
    # are reached by different arithmetic paths; over ~10^4 stiff steps that
    # round-off amplifies slightly. The invariance is exact in exact
    # arithmetic, so we report the measured agreement rather than assert
    # bitwise equality.
    d_sigma = abs(sh_a - sh_b) / sh_a
    d_rho = float(np.max(np.abs(a['rho'][-1] / b['rho'][-1] - 1.0)))
    d_t = abs(a['t_final'] - b['t_final']) / a['t_final']
    print(f'\ndimensionless trajectories agree to:')
    print(f'  sigma_hat input differs by {d_sigma:.2e} (one ulp)')
    print(f'  final density  {d_rho:.2e}   final time {d_t:.2e}')
    identical = d_rho < 1e-9 and d_t < 1e-11

    tc_d = float(dwarf.time_to_gyr(a['t_final']))
    tc_c = float(cluster.time_to_gyr(b['t_final']))
    rc_d = float(dwarf.density_to_msun_kpc3(a['rho'][-1, 3]))
    rc_c = float(cluster.density_to_msun_kpc3(b['rho'][-1, 3]))
    print('\nsame dimensionless evolution, different physical predictions:')
    print(f'  collapse time      dwarf {tc_d:9.3f} Gyr | '
          f'cluster {tc_c:9.3f} Gyr  (ratio {tc_c / tc_d:.3f})')
    print(f'  central density    dwarf {rc_d:.4e}     | '
          f'cluster {rc_c:.4e} Msun/kpc^3')
    # the ratio of physical collapse times must equal the ratio of time units
    ratio_pred = cluster.t_0 / dwarf.t_0
    print(f'  time-unit ratio predicted {ratio_pred:.6f}, '
          f'measured {tc_c / tc_d:.6f}')

    # a cross-check of the unit machinery against the Fischer+2025 halo, whose
    # dimensionless cross-section is quoted independently in Section 3.5
    fischer = HaloUnits(r_s=1.28, rho_s=4.42e7)
    sh_fischer = fischer.sigma_hat(80.0)
    print(f'\nFischer+2025 halo cross-check: sigma_hat = {sh_fischer:.4f} '
          f'(Section 3.5 quotes 0.9455)')

    out = {'dwarf': {'m200': 1e9, 'c200': 15, 'r_s_kpc': dwarf.r_s,
                     'rho_s': dwarf.rho_s, 'v_0_kms': dwarf.v_0,
                     't_0_gyr': dwarf.t_0, 'sigma_over_m': sig_dwarf,
                     't_collapse_gyr': tc_d},
           'cluster': {'m200': 1e15, 'c200': 5, 'r_s_kpc': cluster.r_s,
                       'rho_s': cluster.rho_s, 'v_0_kms': cluster.v_0,
                       't_0_gyr': cluster.t_0,
                       'sigma_over_m': float(sig_clust),
                       't_collapse_gyr': tc_c},
           'common_sigma_hat': float(sh_a),
           'dimensionless_trajectories_identical': bool(identical),
           'sigma_hat_input_rel_diff_ulp': float(d_sigma),
           'final_density_rel_diff': float(d_rho),
           'final_time_rel_diff': float(d_t),
           't_final_dimensionless': [float(a['t_final']), float(b['t_final'])],
           'time_unit_ratio_predicted': float(ratio_pred),
           'time_unit_ratio_measured': float(tc_c / tc_d),
           'fischer_sigma_hat_check': float(sh_fischer),
           'fischer_sigma_hat_reference': 0.9455}
    json.dump(out, open(os.path.join(OUT, 'scale_invariance.json'), 'w'),
              indent=1)
    print('\nsaved results/e15/scale_invariance.json')


if __name__ == '__main__':
    main()
