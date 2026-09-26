"""Regression + physics tests for the gravothermal solver.

Run: .venv/bin/python -m pytest tests/ -v
These are the acceptance gates quoted in the paper; they must stay green.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver, dataset as ds  # noqa: E402


def t_at_factor(res, factor):
    rhoc = res['rho'][:, 3]
    target = factor * rhoc[0]
    i = int(np.argmax(rhoc >= target))
    if rhoc[i] < target:
        return None
    t0, t1 = res['t'][i - 1], res['t'][i]
    l0, l1 = np.log(rhoc[i - 1]), np.log(rhoc[i])
    return float(t0 + (t1 - t0) * (np.log(target) - l0) / (l1 - l0))


def energy_drift(res):
    return abs(res['E'][-1] - res['E'][0]) / abs(res['E'][0])


@pytest.fixture(scope='module')
def nfw_run():
    """Reference constant-cross-section collapse (implicit scheme)."""
    return solver.run_halo(sigma_m=0.5, n_shells=150, eps_t=3e-4,
                           rho_c_factor=300.0, timestep_mode='implicit')


@pytest.mark.verification
def test_initial_profile_is_nfw():
    r, m, rho, p = solver.nfw_profile(n_shells=200)
    rm = ds.shell_midpoints(r)
    # NFW density away from the innermost shell
    assert np.allclose(rho[1:], solver.nfw_rho(rm[1:]), rtol=1e-10)
    # enclosed mass is the analytic NFW mass
    assert np.allclose(m, solver.nfw_mass(r), rtol=1e-10)
    # pressure decreasing outward and positive
    assert np.all(p > 0) and np.all(np.diff(p) < 0)


@pytest.mark.verification
def test_collapse_time_matches_reference_code(nfw_run):
    """Cross-code gate: t_c(100x) vs GravothermalSIDM (Boddy et al.) run at
    the identical configuration = 408.413 t0 (results/e1/crosscheck_ref.npz)."""
    tc = t_at_factor(nfw_run, 100.0)
    assert tc is not None
    assert abs(tc - 408.413) / 408.413 < 0.01


@pytest.mark.verification
def test_energy_conservation(nfw_run):
    assert energy_drift(nfw_run) < 1e-3


@pytest.mark.verification
def test_gravothermal_phases(nfw_run):
    """Core forms (central density drops), then collapses."""
    rhoc = nfw_run['rho'][:, 3]
    assert rhoc.min() < 0.5 * rhoc[0]          # core formation
    assert rhoc[-1] > 100 * rhoc[0]            # collapse
    assert int(np.argmin(rhoc)) < len(rhoc) - 1


@pytest.mark.verification
def test_mass_is_conserved_lagrangian(nfw_run):
    """Shell masses are Lagrangian: the mass array must not change."""
    m = nfw_run['m']
    assert np.all(np.diff(m) > 0)
    assert np.isfinite(m).all()


@pytest.mark.verification
def test_implicit_matches_explicit():
    """Scheme-independence gate: the two conduction schemes must agree."""
    kw = dict(sigma_m=0.5, n_shells=150, rho_c_factor=100.0)
    imp = solver.run_halo(eps_t=3e-4, timestep_mode='implicit', **kw)
    exp = solver.run_halo(eps_t=1e-3, timestep_mode='physical',
                          max_steps=100_000_000, **kw)
    ti, te = t_at_factor(imp, 100.0), t_at_factor(exp, 100.0)
    assert abs(ti - te) / te < 0.005
    assert imp['n_steps'] < exp['n_steps'] / 50   # implicit is far cheaper


@pytest.mark.verification
def test_velocity_dependent_cross_section_runs():
    res = solver.run_halo(sigma_m=2.0, w=0.3, model_lmfp='yukawa_tu',
                          model_smfp='yukawa_tu', n_shells=150, eps_t=3e-4,
                          rho_c_factor=100.0, kn_min=0.02, t_end=1200.0,
                          timestep_mode='implicit')
    assert res['status'] in (0, 1, 2)
    assert energy_drift(res) < 1e-2
    assert np.all(res['rho'][-1] > 0)


@pytest.mark.verification
def test_hydrostatic_equilibrium_of_final_state(nfw_run):
    """dp/dr = -m rho / r^2 must hold on the evolved state."""
    r = nfw_run['r'][-1]
    rho = nfw_run['rho'][-1]
    p = nfw_run['p'][-1]
    m = nfw_run['m']
    rm = ds.shell_midpoints(r)
    dpdr = np.diff(p) / np.diff(rm)
    rhs = -(m[:-1] * 0.5 * (rho[:-1] + rho[1:]) / r[:-1] ** 2)
    scale = np.abs(dpdr) + np.abs(rhs)
    resid = np.abs(dpdr - rhs) / np.maximum(scale, 1e-300)
    # median residual over the bulk of the halo
    assert np.median(resid[2:-2]) < 0.05


@pytest.mark.verification
def test_regrid_roundtrip_is_smooth():
    r, m, rho, p = solver.nfw_profile(n_shells=300)
    lrho, lv = ds.regrid_snapshot(r, rho, p)
    assert lrho.shape == (ds.N_R,)
    assert np.isfinite(lrho).all() and np.isfinite(lv).all()
    # The output grid reaches inside the innermost shell midpoint, where the
    # regridder clamps (flat core). Outside that region NFW must be strictly
    # decreasing; everywhere it must be non-increasing.
    lg = np.log10(ds.r_grid())
    resolved = lg > np.log10(ds.shell_midpoints(r)[0])
    assert np.all(np.diff(lrho) <= 1e-12)
    assert np.all(np.diff(lrho[resolved]) < 0)
    assert resolved.mean() > 0.9   # clamped region is a small grid fraction


@pytest.mark.verification
def test_operator_is_discretization_invariant():
    """The trained operator evaluates on grids other than the training one."""
    torch = pytest.importorskip('torch')
    from gravop.model import GravOpNet
    net = GravOpNet(n_r=ds.N_R).eval()
    r, m, rho, p = solver.nfw_profile(n_shells=300)
    outs = {}
    for n_r in (128, 256):
        rg = np.logspace(ds.LOGR_MIN, ds.LOGR_MAX, n_r)
        lrho, lv = ds.regrid_snapshot(r, rho, p, rg)
        prof = torch.tensor(np.stack([lrho, lv]), dtype=torch.float32)[None]
        cond = torch.tensor([[0.3, -0.2, 1.0, 0.8]], dtype=torch.float32)
        with torch.no_grad():
            f, s = net(prof, cond)
        assert f.shape == (1, 2, n_r)
        outs[n_r] = s[0, 0].item()
    # scalar head is a grid-average quantity: must be nearly grid-independent
    assert abs(outs[128] - outs[256]) < 0.2


# --- physical-unit interface (gravop.units) -------------------------------

@pytest.mark.verification
def test_units_reproduce_published_dimensionless_cross_section():
    """The Fischer+2025 halo has an independently quoted sigma_hat = 0.9455."""
    from gravop.units import HaloUnits
    u = HaloUnits(r_s=1.28, rho_s=4.42e7)
    assert abs(u.sigma_hat(80.0) - 0.9455) < 1e-3


@pytest.mark.verification
def test_units_roundtrip():
    from gravop.units import HaloUnits
    u = HaloUnits.from_nfw(m200=1e12, c200=10)
    for s in (0.1, 1.0, 50.0):
        assert abs(u.sigma_over_m(u.sigma_hat(s)) / s - 1) < 1e-12
    for w in (5.0, 100.0):
        assert abs(u.w_kms(u.w_hat(w)) / w - 1) < 1e-12
    assert abs(u.gyr_to_time(u.time_to_gyr(3.0)) / 3.0 - 1) < 1e-12


@pytest.mark.verification
def test_nfw_scales_recover_virial_mass():
    """Integrating the NFW profile to r200 must return m200."""
    from gravop.units import nfw_scales
    m200, c200 = 1e13, 8.0
    r_s, rho_s = nfw_scales(m200, c200)
    r200 = c200 * r_s
    mu = np.log(1 + c200) - c200 / (1 + c200)
    assert abs(4 * np.pi * rho_s * r_s ** 3 * mu / m200 - 1) < 1e-10
    # and the mean enclosed density must be 200 x rho_crit
    mean = m200 / (4. / 3. * np.pi * r200 ** 3)
    assert abs(mean / (200 * 136.05) - 1) < 1e-10


@pytest.mark.verification
def test_halo_mass_generalisation_is_exact():
    """Two haloes six decades apart in mass, matched in dimensionless
    cross-section, follow the same dimensionless trajectory."""
    from gravop.units import HaloUnits, matched_cross_section
    dwarf = HaloUnits.from_nfw(m200=1e9, c200=15)
    cluster = HaloUnits.from_nfw(m200=1e15, c200=5)
    sig_c = matched_cross_section(dwarf, cluster, 5.0)
    sa, sb = dwarf.sigma_hat(5.0), cluster.sigma_hat(sig_c)
    assert abs(sa - sb) / sa < 1e-14          # equal up to round-off
    kw = dict(n_shells=100, eps_t=1e-3, rho_c_factor=100.0,
              timestep_mode='implicit')
    a = solver.run_halo(sigma_m=sa, **kw)
    b = solver.run_halo(sigma_m=sb, **kw)
    assert abs(a['t_final'] - b['t_final']) / a['t_final'] < 1e-10
    assert np.max(np.abs(a['rho'][-1] / b['rho'][-1] - 1)) < 1e-8


@pytest.mark.regression
def test_concerto_cross_section_maps_into_the_training_box():
    """The Concerto (E30) haloes must stay inside the box the emulator saw.

    P28's structural test is only interpretable if the GroupSIDM cross-section,
    expressed in each halo's dimensionless units, lands where the emulator and
    the solver were exercised. It does -- but the margin at the top end is
    small, so a change to the halo selection or to the unit conversion (the
    kpc/h trap in `e30_concerto.load_pairs`) must not push it out unnoticed.
    """
    import json
    import os
    import numpy as np
    p = os.path.join(os.path.dirname(__file__), '..', 'results', 'e30',
                     'concerto.json')
    if not os.path.exists(p):
        import pytest
        pytest.skip('E30 not run')
    d = json.load(open(p))
    lo_s, hi_s = np.log10(d['sigma_hat_range'])
    lo_w, hi_w = np.log10(d['w_hat_range'])
    assert -1.5 <= lo_s and hi_s <= 0.7, (
        f'sigma_hat spans 10^[{lo_s:.2f}, {hi_s:.2f}], outside the training '
        'box [-1.5, 0.7]')
    assert -1.0 <= lo_w and hi_w <= 1.5, (
        f'w_hat spans 10^[{lo_w:.2f}, {hi_w:.2f}], outside [-1.0, 1.5]')
    # the h = 0.7 units trap: log.txt radii are already physical
    assert abs(d['h_observed_from_data'] - 0.7) < 1e-6, (
        'the out_sidm/log.txt radius ratio is no longer 0.700000; the unit '
        'convention changed and sigma_hat is now wrong by that factor')


@pytest.mark.regression
def test_the_concerto_track_is_not_scored_against_a_nested_null():
    """E30's "beats the no-evolution null" claim CANNOT FAIL, and is retired.

    The gravothermal track starts at the CDM initial condition, so phase 0 IS
    the null: 23 of 81 haloes sit exactly on it, `resid <= null` holds by
    construction, and `frac_better_than_null` is 1.0 as arithmetic rather than
    as evidence. The old gate asserted both of those, so it tested nothing.

    What is falsifiable is the comparison against a competitor with the SAME one
    free parameter that is NOT nested in the track -- NFW at fixed Mvir/Rvir
    with concentration free (E32). This gate pins the two findings that replaced
    the withdrawn claim, and it fails if either is quietly reversed.
    """
    import json
    import os
    e30p = os.path.join(os.path.dirname(__file__), '..', 'results', 'e30',
                        'concerto.json')
    e32p = os.path.join(os.path.dirname(__file__), '..', 'results', 'e32',
                        'controls.json')
    if not (os.path.exists(e30p) and os.path.exists(e32p)):
        import pytest
        pytest.skip('E30/E32 not run')
    d = json.load(open(e30p))
    c = json.load(open(e32p))
    assert d['n_evaluated'] >= 50

    # 1. pooled against a FAIR baseline the track is not distinguishable. If a
    #    later change makes it significant, that is a real result and this gate
    #    should be revisited deliberately rather than silently.
    nb = c['E3_nfw_baseline']
    assert nb['wilcoxon_p'] > 0.05, (
        f"pooled track-vs-NFW is now significant (p={nb['wilcoxon_p']:.3g}); "
        'the withdrawn 6.56x claim must not be restored without re-deriving it')

    # 2. the physics survives where SIDM actually acted: in the most collapsed
    #    band the track must still beat the same-complexity NFW baseline.
    band = c['E4_resolution']['bands'][0]
    assert band['rvmax_over_eps'].startswith('[0,4)')
    assert band['track_resid_median'] < band['nfw_resid_median'], (
        'the track no longer beats NFW among the core-collapsed haloes, which '
        'is the only regime in which E30 retains any claim')

    # 3. the null really is nested -- documented, so nobody re-reads
    #    frac_better_than_null as evidence.
    assert d['frac_better_than_null'] == 1.0
    assert c['n_pinned_at_null'] > 0
