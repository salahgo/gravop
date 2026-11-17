"""Acceptance gates for gravop.closure (selectable conductivity interpolation).

Verification gates recompute from the solver; the regression gate checks the
stored transfer record (results/closure_20261108/transfer.json, written by
scripts/closure_transfer.py).
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import baryons, closure, solver  # noqa: E402

REC = os.path.join(ROOT, 'results', 'closure_20261108', 'transfer.json')
PROD = dict(n_shells=300, eps_t=3e-4, rho_c_factor=1e3, t_end=1200.0,
            timestep_mode='implicit', kn_min=0.02)


def _same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ('t', 'r', 'rho', 'p', 'E')) \
        and a['t_final'] == b['t_final'] and a['status'] == b['status']


@pytest.mark.verification
def test_harmonic_closure_reproduces_the_production_solver_bit_for_bit():
    kw = dict(w=1.5, model_lmfp='yukawa_tu', model_smfp='yukawa_tu', **PROD)
    assert _same(solver.run_halo(2.0, **kw), closure.run_halo_g(2.0, alpha=1.0, **kw))


@pytest.mark.verification
def test_harmonic_closure_reproduces_the_baryonic_module_bit_for_bit():
    kw = dict(Mb=0.0306, ab=0.0934, r_min=0.003, **PROD)
    assert _same(baryons.run_halo_b(0.5, **kw), closure.run_halo_g(0.5, alpha=1.0, **kw))


@pytest.mark.verification
def test_gentler_interpolation_conducts_less_in_the_transition():
    # for alpha < 1, (k_s^a + k_l^a)^(1/a) >= k_s + k_l for the inverse
    # conductivities, so K is below the harmonic value across the transition and,
    # at equal C, the core collapses later; deep in either limit the two agree
    ks, kl = 2.0, 3.0
    assert (ks ** 0.5 + kl ** 0.5) ** -2 < 1.0 / (ks + kl)
    h = closure.run_halo_g(0.5, C=0.753, alpha=1.0, **PROD)['t_final']
    g = closure.run_halo_g(0.5, C=0.753, alpha=0.5, **PROD)['t_final']
    assert g > h


@pytest.mark.regression
def test_each_closure_misses_one_halo_by_little_more_than_the_reference_resolves():
    # 2026-11-11 record (two NSphere seeds at 2e5 particles per halo): the
    # harmonic closure is late in halo B and the literature closure in halo C,
    # each beyond the reference resolution but by at most ~1.6 times it, so the
    # kinetic test does not decide between them
    if not os.path.exists(REC):
        pytest.skip('transfer record not present')
    T = json.load(open(REC))['haloes']

    def unc(h):
        H = T[h]
        sp = H['N2e5_halfspread_rel'] or H['N1e5_seed_halfspread_rel']
        return float(np.hypot(sp, H['references']['N2e5_mean'] / H['references']['N1e5_seed_mean'] - 1))

    harm_B = T['B']['harmonic']['clock_error_vs_N2e5_mean']
    lit_C = T['C']['alpha_lit']['clock_error_vs_N2e5_mean']
    assert unc('B') < harm_B < 1.6 * unc('B')
    assert unc('C') < lit_C < 1.6 * unc('C')
    assert abs(T['B']['alpha_lit']['clock_error_vs_N2e5_mean']) < unc('B')
    assert abs(T['C']['harmonic']['clock_error_vs_N2e5_mean']) < unc('C')
    assert abs(T['A']['alpha_lit']['clock_error']) < 0.05
