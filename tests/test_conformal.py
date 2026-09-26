"""P37: the conformal machinery, and the claims it supports.

The coverage theorem is the control. If split conformal does not attain its
nominal level on exchangeable synthetic data, the implementation is wrong and
nothing downstream of it means anything.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import conformal as C  # noqa: E402

ART = os.path.join(ROOT, 'results', 'statecond', 'conformal.json')


@pytest.mark.verification
def test_coverage_theorem_holds_on_exchangeable_data():
    """The control. Exchangeable scores must be covered at nominal."""
    rng = np.random.default_rng(0)
    for alpha in (0.10, 0.05, 0.20):
        x = np.abs(rng.standard_normal(40000))
        cal, tst = x[:20000], x[20000:]
        q = C.conformal_quantile(cal, alpha)
        cov = float(np.mean(tst <= q))
        lo, hi = C.finite_sample_band(cal.size, alpha)
        assert lo - 0.01 <= cov <= hi + 0.01, (alpha, cov, lo, hi)


@pytest.mark.verification
def test_finite_sample_correction_is_present():
    """Without the (n+1) correction the quantile undercovers by O(1/n).

    A plain numpy percentile is the bug this guards against.
    """
    s = np.arange(1, 11, dtype=float)          # n = 10
    q = C.conformal_quantile(s, 0.10)
    assert q == 10.0, q                        # ceil(11*0.9) = 10 -> s[9]
    assert q > np.percentile(s, 90)


@pytest.mark.verification
def test_alpha_too_small_for_the_calibration_set_is_infinite():
    """With n=10 no finite threshold can certify 99%; say so, do not guess."""
    assert C.conformal_quantile(np.arange(1, 11, dtype=float), 0.01) == np.inf


@pytest.mark.verification
def test_normalisation_preserves_the_guarantee():
    rng = np.random.default_rng(1)
    sig = rng.uniform(0.5, 3.0, 40000)
    r = np.abs(rng.standard_normal(40000)) * sig
    cal = C.calibrate(r[:20000], sig[:20000], 0.10)
    cov = float(C.covers(r[20000:], sig[20000:], cal).mean())
    assert 0.88 <= cov <= 0.92, cov


@pytest.mark.verification
def test_break_even_deferral_matches_the_reference_formula():
    # arXiv:2608.23075 reports r = 0.17; with a free gate d_break = (1-r)/(1-r)
    assert C.break_even_deferral(0.17, 0.0) == pytest.approx(1.0)
    # a gate costing 39.1% of a solver call, their upper g
    assert C.break_even_deferral(0.17, 0.391) == pytest.approx(
        (1 - 0.391 - 0.17) / (1 - 0.17))
    # a surrogate no cheaper than the solver can never pay
    assert C.break_even_deferral(1.0, 0.0) == 0.0


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(ART), reason='run sc_conformal.py')
def test_measured_coverage_is_consistent_with_nominal():
    """The trajectory-level claim, against its own artefact."""
    d = json.load(open(ART))
    for k, v in d['trajectory_level'].items():
        lo, hi = v['coverage_test_ci95']
        band_lo, band_hi = v['finite_sample_band']
        assert lo <= band_hi and hi >= band_lo, (k, v)


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(ART), reason='run sc_conformal.py')
def test_the_ensemble_alone_undercovers():
    """The finding that motivates the work; it must not silently reverse."""
    d = json.load(open(ART))['in_distribution_test']
    for k in ('0.90', '0.95'):
        assert d[k]['gaussian_ensemble'] < float(k), (k, d[k])


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(ART), reason='run sc_conformal.py')
def test_gating_pays_at_this_cost_ratio():
    """Every required deferral must sit below break-even, or the claim dies."""
    g = json.load(open(ART))['gating_economics']
    db = g['break_even_deferral']
    for k, v in g['required_deferral'].items():
        for where, d in v.items():
            assert d < db, (k, where, d, db)


GATE = os.path.join(ROOT, 'results', 'statecond', 'gated_deployment.json')


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_the_gate_never_serves_a_violation_out_of_distribution():
    """The safety claim. One leak and the gate is not a gate."""
    d = json.load(open(GATE))
    for tag in ('ood_ic', 'ood_param'):
        for s in d['sets'][tag]['rms']['sweep']:
            assert s['violations'] == 0, (tag, s)


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_the_gate_defers_everything_on_out_of_family_initial_conditions():
    """Out-of-family data must be refused, not served fast and wrong."""
    d = json.load(open(GATE))
    for s in d['sets']['ood_ic']['rms']['sweep']:
        assert s['deferral'] == 1.0, s


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_the_gate_pays_in_distribution():
    """It must buy something, or safety is being had for free by refusing."""
    sw = json.load(open(GATE))['sets']['test']['rms']['sweep']
    assert max(s['speedup_vs_solver'] for s in sw) > 4.0, sw


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_full_deferral_costs_only_the_gate():
    """The worst case is a small overhead, not a collapse."""
    c = json.load(open(GATE))['costs_seconds']
    worst = c['solver'] / (c['solver'] + c['gate'])
    assert 0.95 < worst < 1.0, worst


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_the_budget_ladder_is_shared_across_sets():
    """Re-deriving budgets per set would hand the gate its own answer."""
    d = json.load(open(GATE))
    lad = d['budget_ladder']['rms']
    for tag in ('test', 'ood_ic', 'ood_param'):
        got = [s['budget'] for s in d['sets'][tag]['rms']['sweep']]
        assert got == pytest.approx(lad), (tag, got, lad)


DECOMP = os.path.join(ROOT, 'results', 'statecond', 'decomposition.json')


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(DECOMP), reason='run sc_decomposition.py')
def test_every_decomposition_row_closes_under_the_published_convention():
    """Reconstruct each row from its own columns, not from a stored summary.

    The identity E_dep^2 = F^2 + |e|^2 + cross is quadratic, so it survives a
    mean over seeds of the SQUARED norms and dies under a mean of the norms.
    A published version averaged each column independently and left rows closing
    to 0.974. This gate rebuilds the closure from the printed quantities.
    """
    d = json.load(open(DECOMP))
    assert len(d) == 15, len(d)
    for k, v in d.items():
        closure = (v['F'] ** 2 + v['e_dep'] ** 2 + v['cross']) / v['E_dep'] ** 2
        assert abs(closure - 1.0) < 1e-6, (k, closure)


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(DECOMP), reason='run sc_decomposition.py')
def test_the_aggregate_is_the_rms_over_seeds_not_the_mean():
    """The convention itself, checked against the per-seed rows it came from."""
    d = json.load(open(DECOMP))
    for k, v in d.items():
        S = v['per_seed']
        for key in ('F', 'e_dep', 'E_A', 'E_dep'):
            rms = float(np.sqrt(np.mean([r[key] ** 2 for r in S])))
            assert abs(v[key] - rms) < 1e-9, (k, key, v[key], rms)
        assert abs(v['cross'] - float(np.mean([r['cross'] for r in S]))) < 1e-12


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_the_emulator_cost_is_not_a_cpu_gpu_average():
    """The emulator is a GPU code, so its cost must be a GPU cost.

    A published version took the median over every trajectory benchmark entry,
    a set holding two CUDA and two CPU timings; the median of that set is the
    midpoint between a GPU number and a CPU number and overstated the emulator
    by 21x. Recompute the cost from the benchmark and require the GPU value.
    """
    bench = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    gpu = [e['seconds_per_item'] for e in bench['emulator']
           if e['kind'] == 'trajectory' and e['batch'] >= 64
           and str(e['device']).startswith('cuda')]
    assert gpu, 'no GPU emulator timings'
    got = json.load(open(GATE))['costs_seconds']['emulator']
    assert abs(got - float(np.median(gpu))) < 1e-12, (got, np.median(gpu))


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(GATE), reason='run sc_gated_deployment')
def test_every_speedup_follows_the_stated_cost_model():
    """speed-up = 1/(g+d), the model the manuscript prints. No row may drift."""
    d = json.load(open(GATE))
    c = d['costs_seconds']
    g = c['gate'] / c['solver']
    assert abs(d['break_even_deferral'] - (1.0 - g)) < 1e-12
    for tag, byset in d['sets'].items():
        for stat, v in byset.items():
            for row in v['sweep']:
                want = 1.0 / (g + row['deferral'])
                assert abs(row['speedup_vs_solver'] - want) < 1e-6, (
                    tag, stat, row['deferral'], row['speedup_vs_solver'], want)


PRED = os.path.join(ROOT, 'results', 'statecond', 'predictor.json')


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(PRED), reason='run sc_predictor.py')
def test_the_displacement_rules_beat_the_free_no_penalty_baseline():
    """If predicting "no penalty" wins, the rules carry no information.

    Aggregated, that must hold at both clocks. Per system it holds everywhere
    at the published clock and in four of five at the converged one: the
    exception is `burgers`, whose converged penalty is 0.96-1.01x, so there is
    almost no penalty there to predict and "predict none" is hard to beat. The
    test asserts the exception rather than tolerating any exception, so a
    SECOND system slipping below the free baseline fails here.
    """
    d = json.load(open(PRED))
    losses = {}
    for clock in ('published_clock', 'converged_clock'):
        base = d[clock]['R_unit_no_penalty']
        best = min(d[clock][r]['mean_of_system_rms']
                   for r in ('R1_quadrature', 'R2_bare'))
        assert best < base['mean_of_system_rms'], (clock, best, base)
        losses[clock] = set()
        for sysname, v in base['per_system'].items():
            better = min(d[clock][r]['per_system'][sysname]['rms_log_residual']
                         for r in ('R1_quadrature', 'R2_bare'))
            if better >= v['rms_log_residual']:
                losses[clock].add(sysname)
    assert losses['published_clock'] == set(), losses
    assert losses['converged_clock'] == {'burgers'}, losses


@pytest.mark.regression
def test_the_two_conformal_rows_are_different_bands():
    """The elementwise and per-trajectory rows are SEPARATELY calibrated.

    A referee derived the nesting inequality -- whole-object coverage cannot
    exceed elementwise coverage -- and observed that Table 19 violates it
    (0.9615 > 0.9418 at the 0.95 level). The inequality holds for ONE band. It
    does not apply here because the two rows are two calibrations: the
    elementwise quantile is taken over individual elements, the trajectory
    quantile over per-trajectory maxima, and the latter is an order of magnitude
    larger. This gate pins that they remain different, so the table can never be
    read as one construction, and pins the direction of the gap.
    """
    import json
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond',
                     'conformal.json')
    if not os.path.exists(p):
        import pytest
        pytest.skip('conformal not run')
    d = json.load(open(p))
    for lvl in ('0.90', '0.95'):
        q_elem = d['thresholds'][lvl]['q_normalised']
        q_traj = d['trajectory_level'][lvl]['q']
        assert q_traj > 5.0 * q_elem, (
            f'at {lvl} the trajectory quantile {q_traj:.3f} is no longer far '
            f'above the elementwise {q_elem:.3f}; if these ever become the same '
            'band the coverage table must satisfy the nesting inequality')
    # and the reported coverages must still be the ones the manuscript quotes
    assert abs(d['trajectory_level']['0.95']['coverage_test'] - 125 / 130) < 1e-9
    assert abs(d['in_distribution_test']['0.95']['conformal_normalised']
               - 0.9418) < 5e-5


@pytest.mark.regression
def test_the_wilson_interval_is_reported_at_the_precision_it_has():
    """125/130 gives an upper endpoint of 0.98346, which rounds to 0.983.

    The manuscript printed 0.984. A rounding slip in a coverage interval is
    small but it is in the table carrying the only guarantee the paper claims.
    """
    import json
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond',
                     'conformal.json')
    if not os.path.exists(p):
        import pytest
        pytest.skip('conformal not run')
    d = json.load(open(p))
    lo, hi = d['trajectory_level']['0.95']['coverage_test_ci95']
    assert f'{hi:.3f}' == '0.983', f'Wilson upper now rounds to {hi:.3f}'
    assert f'{lo:.3f}' == '0.913'
