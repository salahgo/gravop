"""Acceptance gates for gravop.baryons (static Hernquist baryons).

Verification gates recompute from the solver; regression gates check the
stored verification record (results/baryons_20261107/verification.json,
written by scripts/baryon_verification.py).
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import baryons as B, solver  # noqa: E402
REC = os.path.join(ROOT, 'results', 'baryons_20261107', 'verification.json')
PROD = dict(n_shells=300, eps_t=3e-4, rho_c_factor=1e3, t_end=1200.0,
            timestep_mode='implicit', kn_min=0.02)


@pytest.mark.verification
def test_no_baryons_reproduces_the_production_solver_bit_for_bit():
    a = solver.run_halo(0.5, **PROD)
    b = B.run_halo_b(0.5, Mb=0.0, ab=0.1, **PROD)
    for k in ('t', 'r', 'rho', 'p', 'E'):
        assert np.array_equal(a[k], b[k]), k
    assert a['t_final'] == b['t_final'] and a['status'] == b['status']


@pytest.mark.verification
def test_initial_state_is_in_equilibrium_in_the_total_potential():
    Mb, ab = 0.0306, 0.0934
    r, m, rho, p = B.profile_with_baryons(Mb, ab, r_min=0.003, n_shells=300)
    with_b = B._hydro_step_b(r.copy(), m, rho.copy(), p.copy(), Mb, ab)
    without = B._hydro_step_b(r.copy(), m, rho.copy(), p.copy(), 0.0, ab)
    assert with_b < 0.02 and without > 10 * with_b


@pytest.mark.verification
def test_energy_including_the_baryonic_potential_is_conserved():
    res = B.run_halo_b(0.5, Mb=0.0153, ab=0.132, **PROD)
    E = res['E']
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 5e-4


@pytest.mark.verification
def test_baryons_shorten_collapse_monotonically_with_compactness():
    t = [B.run_halo_b(0.9455, C=0.712, Mb=0.01, ab=a, r_min=0.003, **PROD)['t_final']
         for a in (0.3, 0.1, 0.03)]
    assert t[0] > t[1] > t[2]


@pytest.mark.regression
def test_published_baryonic_collapse_times_are_reproduced():
    if not os.path.exists(REC):
        pytest.skip('verification record not present')
    rows = json.load(open(REC))['zhong_table3']
    for r in rows:
        tol = 0.15 if r['sigma_m'] >= 100 else 0.06
        assert abs(r['rel_diff']) < tol, (r['name'], r['rel_diff'])
