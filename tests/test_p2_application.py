"""Gates for the state-conditioned application path (P2 / E29).

E29 turned on a constraint the rest of the project had missed: the state
coordinate is a density contrast *at a radius*, and that radius must be one the
observation resolves. Getting it wrong does not raise — the arrays broadcast and
the numbers come out meaningless — so the guards that prevent it are gated here
rather than trusted.

Two real defects found while building that path are each pinned by a test below:
reading the contrast's numerator and denominator at different radii, and
reporting a `t_norm` from a scalar head that was never trained.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from gravop import dataset as ds                    # noqa: E402
from gravop import phase as gphase                  # noqa: E402

# the two radii that actually constrain this project
NBODY_R_MIN = 0.00844      # innermost radius the Fischer+2025 profiles resolve
E9_R_MIN = 0.02            # innermost radius the E9 mock observation samples


@pytest.mark.verification
def test_phase_radius_matches_the_grid():
    lg = gphase.grid_log_radii()
    assert lg.shape == (ds.N_R,)
    assert gphase.phase_radius(0) == pytest.approx(10 ** ds.LOGR_MIN)
    assert gphase.phase_radius(ds.N_R - 1) == pytest.approx(10 ** ds.LOGR_MAX)


@pytest.mark.verification
@pytest.mark.parametrize('r_min', [NBODY_R_MIN, E9_R_MIN, 0.1, 1.0])
def test_required_phi_index_is_outside_and_minimal(r_min):
    """The suggested index must be at or outside the observation, and must be
    the smallest such — a larger one would throw away resolution for nothing."""
    idx = gphase.required_phi_index(r_min)
    assert gphase.phase_radius(idx) >= r_min
    assert gphase.phase_radius(idx - 1) < r_min


@pytest.mark.verification
def test_guard_rejects_the_cross_domain_index_for_both_applications():
    """This is the actual bug the guard exists for.

    Index 0 is what E20 and the cross-domain protocol use. It is 0.0032 r_s,
    inside everything that observes a halo, so conditioning either application
    on such an ensemble extrapolates below the data.
    """
    for r_min, want in ((NBODY_R_MIN, 13), (E9_R_MIN, 23)):
        with pytest.raises(ValueError) as exc:
            gphase.check_phase_radius(0, r_min)
        # the message must name the index to retrain with, not just complain
        assert f'--phi-index {want}' in str(exc.value)
        assert gphase.required_phi_index(r_min) == want


@pytest.mark.verification
@pytest.mark.parametrize('idx,r_min', [(13, NBODY_R_MIN), (23, E9_R_MIN)])
def test_guard_accepts_the_indices_the_ensembles_were_trained_at(idx, r_min):
    assert gphase.check_phase_radius(idx, r_min) == pytest.approx(
        gphase.phase_radius(idx))


@pytest.mark.verification
def test_contrast_uses_one_radius_for_both_terms():
    """Regression: the first version of the N-body contrast read the numerator
    at the innermost *N-body* radius and the denominator at the innermost
    *grid* radius, folding the profile slope into a quantity that is supposed
    to be a pure ratio. On a profile that is a pure power law the contrast of a
    field against itself must be exactly zero at every radius.
    """
    lg = gphase.grid_log_radii()
    lrho = -2.0 * lg + 1.0                      # rho ~ r^-2
    for r_phi in (0.01, 0.05, 0.5):
        assert gphase.contrast_at(lg, lrho, lrho, r_phi) == pytest.approx(0.0,
                                                                         abs=1e-9)
    # and a uniform 0.5 dex increase reads as exactly 0.5 dex anywhere
    assert gphase.contrast_at(lg, lrho + 0.5, lrho, 0.02) == pytest.approx(0.5)


@pytest.mark.verification
def test_phase_variables_radius_is_selectable_and_default_unchanged():
    """`phase_variables` gained an `r_index`; the default must still be the
    innermost point, or E20's committed numbers would silently change."""
    from e20_phase_conditioned import phase_variables
    rng = np.random.default_rng(0)
    lrho = rng.standard_normal((6, 12, ds.N_R)).cumsum(1)

    p0, f0 = phase_variables(lrho)
    assert np.allclose(p0, phase_variables(lrho, r_index=0)[0])
    # the contrast is measured against tau = 0, so it starts at zero
    assert np.allclose(p0[:, 0], 0.0)
    # a different radius gives a genuinely different coordinate
    p23, f23 = phase_variables(lrho, r_index=23)
    assert np.allclose(p23[:, 0], 0.0)
    assert not np.allclose(p0, p23)
    # the flag is a step: once set it never unsets
    for f in (f0, f23):
        assert np.all(np.diff(f, axis=1) >= 0)


@pytest.mark.verification
def test_phi_marginalisation_quadrature_is_normalised():
    """E9's phase arm integrates phi out with Gauss-Hermite. If the weights
    were mis-normalised the marginal likelihood would be silently biased, and
    the arm's null result could be blamed on the quadrature rather than on the
    physics. Check the rule reproduces the moments of a unit normal."""
    for n in (5, 9, 15):
        nodes, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        assert w.sum() == pytest.approx(1.0)
        assert (w * nodes).sum() == pytest.approx(0.0, abs=1e-12)
        assert (w * nodes ** 2).sum() == pytest.approx(1.0)


@pytest.mark.verification
def test_phase_ensemble_checkpoints_declare_what_they_are():
    """A phase member must be distinguishable from a tau member by its
    checkpoint alone — `cond_dim`, `cond_kind` and the phi normalisation — or
    a downstream script cannot tell which conditioning it just loaded.

    Skipped when the ensembles have not been trained on this machine; they are
    ~4.5 MB each and are not in the repository.
    """
    import torch
    root = os.path.join(os.path.dirname(__file__), '..', 'results', 'runs')
    for prefix, want_index in (('gravop_phase_nbody_s', 13),
                               ('gravop_phase_e9_s', 23)):
        p = os.path.join(root, f'{prefix}0', 'model.pt')
        if not os.path.exists(p):
            pytest.skip(f'{prefix}0 not trained here')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert a['cond_dim'] == 5 and a['cond_kind'] == 'phase'
        assert a['phi_index'] == want_index
        assert {'p_mu', 'p_sd'} <= set(ck['norm'])
        # and the radius it declares must be legal for its application
        r_min = NBODY_R_MIN if 'nbody' in prefix else E9_R_MIN
        gphase.check_phase_radius(a['phi_index'], r_min)


@pytest.mark.verification
def test_phase_members_have_no_usable_time_head():
    """Regression: e28 optimises the field loss only, so the scalar head of a
    phase member is never trained. E11 originally printed a `t_norm` from it,
    which would have put a fabricated number in the record. The gate is that
    the checkpoint says `cond_kind == 'phase'`, which is what E11 now branches
    on before reading that head at all."""
    import torch
    root = os.path.join(os.path.dirname(__file__), '..', 'results', 'runs')
    p = os.path.join(root, 'gravop_phase_e9_s0', 'model.pt')
    if not os.path.exists(p):
        pytest.skip('phase ensemble not trained here')
    assert torch.load(p, map_location='cpu',
                      weights_only=False)['args']['cond_kind'] == 'phase'

    tau_p = os.path.join(root, 'gravop_s0', 'model.pt')
    tau_args = torch.load(tau_p, map_location='cpu',
                          weights_only=False)['args']
    # the committed tau ensemble predates the key and must keep defaulting
    assert tau_args.get('cond_kind', 'tau') == 'tau'
    assert tau_args.get('cond_dim', 4) == 4


@pytest.mark.regression
def test_parametric_baseline_is_reproduced():
    """P9: the Yang et al. (2024) analytic baseline, pinned.

    This is a claim about someone else's published model, so the risk is not
    that our code drifts but that we quietly mis-implement theirs and then
    report a flattering comparison. The fitting functions are pinned at two
    points with known values, and the headline test numbers are pinned to the
    stored artefact.
    """
    import json
    root = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, os.path.join(root, 'scripts'))
    import baselines_parametric as bp

    # eq. (2.3) must reduce to the initial NFW at tau = 0: the log term is
    # constructed so that rho_s/rho_s0 = r_s/r_s0 = 1 and the core vanishes.
    rho_s, r_s, r_c = bp.profile_params(0.0)
    assert abs(rho_s - 1.0) < 1e-9, rho_s
    assert abs(r_s - 1.0) < 1e-9, r_s
    assert abs(r_c) < 1e-12, r_c

    # and eq. (2.1) must reduce to NFW when the core is absent
    r = 2.0
    assert abs(bp.density(r, 1.0, 1.0, 0.0) - 1.0 / (r * (1 + r) ** 2)) < 1e-12

    # The baseline hands the model rho_s0 = r_s0 = 1, i.e. it assumes our
    # dimensionless NFW initial conditions are exactly that. If they were not,
    # the parametric model would be scored against a profile it was never
    # given, and its error would be our artefact rather than its accuracy.
    import h5py
    data = os.path.join(root, 'data', 'd1', 'train.h5')
    if os.path.exists(data):
        from train import load_data
        d = load_data(data)
        with h5py.File(data) as f:
            ic = f['ic_type'][:]
        r = ds.r_grid()
        m = (r > 0.05) & (r < 20.0)
        analytic = 1.0 / (r * (1.0 + r) ** 2)
        for i in np.nonzero(ic == bp.IC_NFW)[0][:5]:
            ratio = (10.0 ** d['lrho0'][i])[m] / analytic[m]
            assert abs(np.median(ratio) - 1.0) < 1e-3, np.median(ratio)

    p = os.path.join(root, 'results', 'benchmark', 'parametric.json')
    if not os.path.exists(p):
        return                      # artefact not regenerated in this checkout
    res = json.load(open(p))['results']
    assert res['nfw_constant']['n_test'] == 11
    assert res['nfw_all']['n_test'] == 40
    # the emulator's advantage is large; pin it loosely so a rerun that changes
    # it by a few percent passes but a sign flip or an order of magnitude fails
    for pop, lo, hi in (('nfw_constant', 20.0, 35.0), ('nfw_all', 10.0, 22.0)):
        v = res[pop]['calibrated']['median_rel_rho_pct']
        assert lo < v < hi, (pop, v)


@pytest.mark.regression
def test_nbody_error_budget_is_measured_not_subtracted():
    """P17: the emulator's own N-body error must be a direct measurement.

    Guards two things at once: that the residual identity
    e_emul = e_fluid + e_es holds on the stored budget, and that the quadrature
    shortcut the manuscript used to make is genuinely wrong here -- if the
    cross term were negligible the old text would have been fine and this gate
    should stop claiming otherwise.
    """
    import json
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'results', 'e2',
                     'error_budget.json')
    if not os.path.exists(p):
        import pytest
        pytest.skip('run scripts/e2_error_budget.py first')
    b = json.load(open(p))
    ident = b['identity_check']
    assert abs(ident['lhs_rms_emul_sq'] - ident['rhs_sum']) < 1e-12, (
        'e_emul = e_fluid + e_es no longer holds')
    direct = b['rms_emul_vs_fluid_DIRECT']
    claim = b['quadrature_subtraction_claims']
    assert direct > 0
    assert abs(claim - direct) / direct > 0.2, (
        f'quadrature ({claim:.4f}) and direct ({direct:.4f}) now agree; the '
        'manuscript text asserting a 54% discrepancy needs updating')
    assert abs(b['cross_term_correlation']) > 0.1, (
        'residuals now look orthogonal; re-examine the claim that they are not')
    # and the headline ordering must survive
    assert direct < b['rms_fluid_vs_nbody'], (
        'surrogate error is no longer subdominant to model error')
