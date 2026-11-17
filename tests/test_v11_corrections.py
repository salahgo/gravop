"""V5 referee: the corrections that must not silently come back.

Two of these are VERIFICATION gates -- they recompute a quantity from the code
or from first principles and assert a property -- and the rest are REGRESSION
gates on artefacts. The distinction matters here more than usual, because the
defect these tests exist to catch was invisible to every regression gate in the
suite: a stored number stayed consistent with a stored number while both were
computed at the wrong clock error.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')


def _need(*rel):
    """Skip when the archival record this gate reads is absent.

    The source release ships without `results/`; the suite is meant to run
    once the archival tree is unpacked beside it (REPRODUCING.md). A missing
    record is then a skip that says so, not a FileNotFoundError.
    """
    missing = [p for p in rel if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        pytest.skip('archival record absent: ' + ', '.join(missing))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')


def _load(*parts):
    p = os.path.join(RES, *parts)
    if not os.path.exists(p):
        pytest.skip(f'{os.path.join(*parts)} not present')
    return json.load(open(p))


# --------------------------------------------------------------- the defect

@pytest.mark.verification
def test_a_normalised_log_time_difference_is_not_a_time_ratio():
    """The arithmetic behind the defect, with no artefacts involved.

    `lts` and the clock head's output are both (log10 t - mu) / sd. The
    physical ratio is 10 ** ((lts - lts_hat) * sd). Dropping `sd` does not
    cancel: it exponentiates the WRONG number, and for sd < 1 it inflates the
    error. This test states the identity so that a future reader who deletes
    the `* n.t_sd` has to delete a failing test as well.
    """
    rng = np.random.default_rng(11)
    for sd in (0.15, 0.40, 0.86):
        mu = 2.5
        true_log = rng.normal(mu, sd, 500)
        pred_log = true_log + rng.normal(0.0, 0.01, 500)
        z_true, z_pred = (true_log - mu) / sd, (pred_log - mu) / sd
        right = 10.0 ** ((z_true - z_pred) * sd)
        wrong = 10.0 ** (z_true - z_pred)
        assert np.allclose(right, 10.0 ** (true_log - pred_log))
        # and the omission inflates the error by 1/sd in the exponent
        assert np.median(np.abs(wrong - 1.0)) > np.median(np.abs(right - 1.0))


@pytest.mark.regression
def test_two_evaluation_paths_agree_on_the_converged_clock():
    """The gate that caught it.

    `metric_bridge.json` and `timehead.json` compute E_A and E_dep from the
    same frozen weights by different code. Under the slot-mean convention they
    must agree to round-off. They did not before the fix, by up to 6.7x.
    """
    B, T = _load('statecond', 'metric_bridge.json'), _load('statecond',
                                                           'timehead.json')
    for name in SYSTEMS:
        r = B.get(f'{name}_fno')
        if r is None:
            continue
        for key, ref in (('E_A', 'armA'), ('E_dep', 'armB_converged')):
            got, want = r[key]['mean_of_slot_rms'], T[name][ref]
            assert abs(got / want - 1.0) < 1e-4, (name, key, got, want)


@pytest.mark.regression
def test_the_displacement_is_the_deployed_floor_under_one_convention():
    """F and floor_deployed are the same quantity, summarised the same way."""
    B, F = _load('statecond', 'metric_bridge.json'), _load('statecond',
                                                           'floor_conv.json')
    dep = {x['system']: x for x in F['deployed_comparison']
           if x['backbone'] == 'fno'}
    for name, row in dep.items():
        r = B.get(f'{name}_fno')
        if r is None:
            continue
        assert abs(r['F']['mean_of_slot_rms'] / row['floor_deployed'] - 1.0) < 1e-4


@pytest.mark.regression
def test_the_withdrawn_reversal_has_not_come_back():
    """At the converged clock, quadrature beats the bare rule. The opposite
    ordering was an artefact of the defect above and is withdrawn."""
    pr = _load('statecond', 'predictor.json')
    for regime in ('published_clock', 'converged_clock'):
        q = pr[regime]['R1_quadrature']['mean_of_system_rms']
        b = pr[regime]['R2_bare']['mean_of_system_rms']
        assert q < b, (regime, q, b)


# ------------------------------------------------- MC2: the deployed path

@pytest.mark.regression
def test_the_deployed_input_path_is_calibrated_and_costs_something():
    """Two claims at once, because either alone would be misleading: the
    guarantee survives recalibration on the path the application uses, and it
    is materially more expensive there."""
    D = _load('statecond', 'conformal_deployed.json')
    lo, hi = D['paths']['deployed']['levels']['0.90']['coverage_whole_ci95']
    assert lo <= 0.90 <= hi, (lo, hi)
    c = D['comparison']['0.90']
    assert c['q_ratio'] > 2.0 and c['halfwidth_ratio_p50'] > 2.0, c


@pytest.mark.regression
def test_the_simultaneous_band_covers_almost_every_element():
    """Same band, both readings. A trajectory-max band scored elementwise must
    nest above the whole-trajectory coverage -- unlike the two SEPARATELY
    calibrated rows of the coverage table, which need not."""
    D = _load('statecond', 'conformal_deployed.json')
    for path in ('oracle', 'deployed'):
        for lvl in ('0.90', '0.95'):
            b = D['paths'][path]['levels'][lvl]
            assert b['coverage_elementwise'] >= b['coverage_whole_trajectory']
            assert b['coverage_elementwise'] > 0.999


# ------------------------------------------------- MC5: the fallback cost

@pytest.mark.regression
def test_the_constant_cost_model_is_optimistic_and_says_so():
    C_ = _load('statecond', 'deferred_cost.json')['statistics']['rms']
    assert C_['spearman_halfwidth_vs_solver_cost'] > 0
    assert C_['spearman_p'] < 0.05
    worst = max(r['optimism_of_constant_model'] for r in C_['sweep'])
    assert worst > 1.0, 'the constant-cost model is no longer optimistic'


# ------------------------------------------- E37: the closed-form check

@pytest.mark.regression
def test_the_event_time_bias_is_set_by_the_threshold_not_the_grid():
    """u_t = u^2 exactly: t_sing - t_event = 1/U, independent of u0."""
    V = _load('statecond', 'exact_verification.json')
    for r in V['event_vs_singular_time']['rows']:
        assert abs(r['bias'] * r['threshold_U'] - 1.0) < 1e-6, r


@pytest.mark.regression
def test_the_displacement_is_recovered_when_the_shift_clears_a_grid_cell():
    """And is not, when it does not. Both halves are the finding."""
    rows = _load('statecond', 'exact_verification.json')[
        'grid_resolution_of_the_displacement']['rows']
    coarse = [r for r in rows if r['shift_over_near_cell'] > 10]
    fine = [r for r in rows if r['shift_over_near_cell'] < 0.5]
    assert coarse and fine, 'the sweep no longer brackets one grid cell'
    assert max(r['rel_error'] for r in coarse) < 5e-3
    assert max(r['rel_error'] for r in fine) > 2e-2


# ------------------------------------------ MC6: the cosmological subgroup

@pytest.mark.regression
def test_the_cosmological_advantage_survives_a_resolution_cut():
    S = _load('e32', 'cut_sensitivity.json')
    b = S['contracted_and_resolved']['block']
    assert b['n'] >= 10
    assert b['median_paired_log_ratio_ci95'][1] < 0.0
    assert b['wilcoxon_p'] < 0.05


@pytest.mark.regression
def test_every_halo_passes_the_suites_own_convergence_cut():
    """The criterion is the Concerto authors' own: R_Vmax is well measured
    above ~2000 particles, and a SMALL R_Vmax is not itself a warning. The
    binding count is the smaller of the two runs, since the comparison is
    between them. If a halo ever falls below the cut, the withdrawn resolution
    caveat has to come back and this test says so."""
    S = _load('e32', 'cut_sensitivity.json')['suite_convergence_criterion']
    assert S['n_passing'] == S['n_total'], S
    assert S['min_N_p'] >= 2000.0, S
    assert S['min_N_p_among_contracted'] >= 2000.0, S


@pytest.mark.regression
def test_the_cosmological_advantage_is_not_a_particle_count_artefact():
    """N_p does not contain R_Vmax, so if the advantage tracked N_p the
    resolution objection would stand. It does not."""
    S = _load('e32', 'cut_sensitivity.json')
    rho, p = S['particle_count']['spearman_Np_vs_paired_diff']
    assert abs(rho) < 0.2 and p > 0.05, (rho, p)


@pytest.mark.regression
def test_the_cut_sweep_decays_rather_than_collapsing():
    """If the advantage were carried by a handful of the worst-resolved
    haloes it would vanish as soon as they left the subgroup."""
    sweep = [r for r in _load('e32', 'cut_sensitivity.json')
             ['rvmax_eps_cut_sweep'] if 'median_paired_log_ratio' in r]
    assert len(sweep) >= 6
    for r in sweep:
        assert r['median_paired_log_ratio'] < 0.0, r['cut']
    tight = min(sweep, key=lambda r: r['cut'])
    loose = max(sweep, key=lambda r: r['cut'])
    assert loose['median_paired_log_ratio'] > tight['median_paired_log_ratio']


# --------------------------------------------- MC4: the work-precision record

@pytest.mark.regression
def test_the_matched_accuracy_speedup_follows_from_the_printed_curve():
    """928x must be the ratio of the two walls the table prints, not a number
    carried alongside it."""
    W = _load('e8', 'workprecision.json')
    m = W['matched_accuracy_scheme_speedup']['3e-04']
    assert abs(m['explicit_wall'] / m['implicit_wall']
               - m['scheme_speedup']) < 1e-6
    assert abs(m['scheme_speedup'] - 928.0) < 1.0


@pytest.mark.regression
def test_the_censored_explicit_runs_are_still_in_the_record():
    W = _load('e8', 'workprecision.json')
    cens = [r for r in W['curves']['classic'] if r['wall_median'] is None]
    assert len(cens) == 2, 'censored rows dropped from the work-precision record'
    for r in cens:
        assert r['exceeded_cap_s'] == 1500.0


# ------------------------------------------- E40: the rate, and its limits

@pytest.mark.regression
def test_the_collapse_time_table_factorises_as_claimed():
    """The table exists because the dimensionless solution depends on the halo
    only through (sigma_hat, w_hat). If interpolation error were large the
    factorisation would be doing no work, so the measured error is the gate."""
    T = _load('e40', 'table_check.json')
    assert T['median_rel_error'] <= 0.02, T['median_rel_error']
    assert T['n_test'] >= 12


@pytest.mark.regression
def test_the_fitted_phase_does_not_depend_on_the_parameter_being_fitted():
    """If phi_obs moved with C_eff the rate scan would be circular."""
    P = _load('e40', 'phase_c_independence.json')
    assert P['all_reproduce_e32_at_0p797']
    assert P['spread_below_phase_0p6'] < 0.01, P['spread_below_phase_0p6']
    assert P['max_spread'] < 0.05, P['max_spread']


@pytest.mark.regression
def test_the_rate_scan_excludes_the_isolated_run_conductivity():
    R = _load('e40', 'rate.json')
    assert R['result']['interior']
    bs = R['C_eff_bootstrap']
    assert bs['frac_at_or_above_isolated_value'] == 0.0
    assert bs['ci95'][1] < R['result']['isolated_run_value']
    assert R['censoring_bias']['best_C_eff_uncensored_worst_case'] < R['result']['isolated_run_value']


@pytest.mark.regression
def test_the_rate_result_is_not_quietly_upgraded():
    """Gated in the NEGATIVE direction, which is the unusual one and the point.

    The paper says this is not a precision calibration because the objective is
    barely better than a constant-phase predictor. If that ever stops being
    true the claim must be rewritten upward -- and this test fails, rather than
    the paper silently keeping an understated description of a better result.
    """
    vc = _load('e40', 'rate.json')['vs_constant_null']
    gain = (vc['rms_constant'] - vc['rms_model']) / vc['rms_constant']
    assert gain < 0.05, f'the rate model now gains {gain:.1%} over a constant'
    assert gain > 0.0, 'the rate model no longer beats a constant at all'


@pytest.mark.regression
def test_the_rate_model_orders_the_haloes_even_though_it_misses_the_amount():
    """The split the paper reports: rank information without scale."""
    rows = _load('e40', 'rate.json')['rank_agreement']['rows']
    assert all(r['spearman'] > 0.2 for r in rows)
    assert all(r['p'] < 0.05 for r in rows)


# ------------------------- E41: the reference's own discretisation error

@pytest.mark.regression
def test_the_reference_converges_cleanly_in_shell_count():
    """Second order is what makes a one-line error statement defensible.

    If the radial convergence stopped being clean, the paper could no longer
    quote a single percentage for the bias the displacement inherits, and the
    E40 error budget would need a refit rather than a correction.
    """
    S = _load('e40', 'reference_convergence.json')['summary']
    assert 1.8 < S['median_observed_order'] < 2.2, S['median_observed_order']


@pytest.mark.regression
def test_the_shell_count_bias_cannot_explain_the_conductivity_gap():
    """E40 fits C_eff = 0.20 against an isolated-run 0.712. The reference's
    spatial error biases that fit LOW, so it pushes in the direction that
    would manufacture the gap -- which is exactly why its size is gated."""
    S = _load('e40', 'reference_convergence.json')['summary']
    assert S['max_rel_error_at_150_shells'] < 0.05
    R = _load('e40', 'rate.json')
    gap = R['result']['isolated_run_value'] / R['result']['best_C_eff']
    assert gap > 2.0, gap
    assert S['max_rel_error_at_150_shells'] * 100 < gap


@pytest.mark.regression
def test_the_field_converges_more_slowly_than_the_clock():
    """The finding E42 exists to record. If the field ever converged as fast as
    the clock, the paper's separate treatment of the two would be unnecessary
    and the text would need rewriting -- so both halves are asserted."""
    F = _load('e40', 'field_convergence.json')['summary']
    C = _load('e40', 'reference_convergence.json')['summary']
    assert F['median_observed_order'] < C['median_observed_order']
    assert 1.0 < F['median_observed_order'] < 1.6, F['median_observed_order']


@pytest.mark.regression
def test_no_field_error_is_reported_across_a_wide_snapshot_gap():
    """The trap that produced a spurious 1.16 dex result.

    Snapshots stop being written once the budget binds, and a linear
    interpolation across the resulting gap in tau manufactured a convergence
    failure that was non-monotone in resolution. Any phase whose snapshot gap
    is too wide must be SKIPPED, not interpolated.
    """
    d = _load('e40', 'field_convergence.json')
    for c in d['cases']:
        for ph, v in c['by_phase'].items():
            if 'skipped' in v:
                continue
            assert v['max_tau_gap'] <= 0.02 + 1e-12, (c['label'], ph, v)
            # and the error must rise monotonically with coarseness
            assert v['150'] > v['300'] > v['600'] > 0, (c['label'], ph, v)


# ----------------------------------- E43: the predictor, out of sample

@pytest.mark.regression
def test_the_predictor_transfers_to_the_reserved_draws():
    """The pre-registered reading. If this fails the manuscript is claiming a
    transfer the artefact no longer supports, and the retrospective caveat
    lifted on the strength of it has to go back."""
    P = _load('e43', 'prospective.json')
    q = P['rules']['R1_quadrature']['mean_of_system_rms']
    r = P['retrospective_for_comparison']['R1_quadrature']
    assert q < 1.5 * r, (q, r)
    assert q < P['rules']['R0_constant']['mean_of_system_rms']
    assert q < P['rules']['R_unit_no_penalty']['mean_of_system_rms']


@pytest.mark.regression
def test_the_prospective_score_is_the_frozen_rule_applied():
    """Recompute the rule from the stored components. If the reported score
    ever stops being sqrt(E_A^2+F^2)/E_A, something has been fitted."""
    P = _load('e43', 'prospective.json')
    per = {}
    for k, p in P['points'].items():
        sysname = k.rsplit('_', 1)[0]
        pred = np.sqrt(p['E_A'] ** 2 + p['F'] ** 2) / p['E_A']
        meas = p['E_dep'] / p['E_A']
        per.setdefault(sysname, []).append(np.log10(pred / meas))
    mean = float(np.mean([np.sqrt(np.mean(np.square(v))) for v in per.values()]))
    assert abs(mean - P['rules']['R1_quadrature']['mean_of_system_rms']) < 1e-9


@pytest.mark.regression
def test_the_reserved_clock_heads_are_at_the_same_operating_point():
    """A predictor scored at an easier clock would not be a transfer test."""
    P = _load('e43', 'prospective.json')
    T = _load('statecond', 'timehead.json')
    for k, p in P['points'].items():
        sysname = k.rsplit('_', 1)[0]
        ref = T[sysname]['converged_head_err_pct']
        assert 0.3 * ref <= p['clock_err_pct'] <= 3.0 * ref, (k, p, ref)


@pytest.mark.regression
def test_the_prospective_result_is_not_described_as_an_improvement():
    """The mean is lower out of sample; per system it is worse in three of five.

    The paper says "comparable, not better" and names the split. If the split
    ever changes, that sentence is wrong and this fails -- which is the point,
    because a mean that drifts favourable is exactly the kind of number a paper
    quietly starts calling an improvement.
    """
    P = _load('e43', 'prospective.json')
    R = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                    'predictor.json')))
    ret = R['converged_clock']['R1_quadrature']['per_system']
    pro = P['rules']['R1_quadrature']['per_system']
    worse = [s for s in pro if pro[s] > ret[s]['rms_log_residual']]
    assert len(worse) == 3, sorted(worse)
    assert set(worse) == {'rd', 'ks', 'burgers'}, sorted(worse)


@pytest.mark.regression
def test_the_displacement_without_clock_truth_is_scored_as_stated():
    """E45: F from the validation split's clock errors, scored by the frozen rule.

    The realised construction must reproduce the stored F that the in-sample
    and reserved scores used, each score must be the frozen rule applied to the
    stored E_A and E_dep, and the reading the text rests on -- looser than the
    realised score, still ahead of a fitted constant on the mean, behind it in
    at least one system -- must hold for every construction.
    """
    D = _load('statecond', 'deployable_displacement.json')
    pts = {'in_sample': _load('statecond', 'decomposition.json'),
           'reserved': _load('e43', 'prospective.json')['points']}
    for reg, points in pts.items():
        blk = D[reg]
        for sysname, f in blk['F'].items():
            assert abs(f['realised'] / blk['F_stored'][sysname] - 1) < 1e-4
        for kind in ('realised', 'val_errors', 'val_only', 'val_median'):
            per = {}
            for k, p in points.items():
                sysname, bb = k.rsplit('_', 1)
                if bb not in ('fno', 'cno', 'transformer'):
                    continue
                pred = np.hypot(p['E_A'], blk['F'][sysname][kind]) / p['E_A']
                per.setdefault(sysname, []).append(
                    np.log10(pred / (p['E_dep'] / p['E_A'])))
            mean = float(np.mean([np.sqrt(np.mean(np.square(v)))
                                  for v in per.values()]))
            assert abs(mean - blk[kind]['mean_of_system_rms']) < 1e-9, (reg, kind)
            vs = blk['against_constant'][kind]
            assert vs['factor_on_mean'] > 1, (reg, kind)
            if kind != 'realised':
                assert blk[kind]['mean_of_system_rms'] > blk['realised']['mean_of_system_rms']
                assert vs['systems_behind_constant'], (reg, kind)


# ------------------- E44: the hand inspection the data's authors asked for

@pytest.mark.regression
def test_the_smallest_halo_is_collapsed_not_broken():
    """A resolution-destroyed centre flattens inside a few softening lengths.
    This one steepens, which is the opposite signature, and the claim in the
    paper rests on that distinction rather than on the catalogue summary."""
    P = _load('e44', 'profile_inspection.json')
    assert P['inner_log_slope_0p5_to_4_eps'] < -2.0
    assert P['n_within_R_Vmax'] >= 100, P['n_within_R_Vmax']
    rows = [r for r in P['profile'] if r['n_particles'] >= 50
            and r['r_over_eps'] < 20]
    rho = [r['rho_Msun_kpc3'] for r in rows][::-1]
    assert all(b > a for a, b in zip(rho, rho[1:])), 'density not monotone'


@pytest.mark.regression
def test_the_inspected_halo_is_one_that_carries_the_result():
    """Inspecting a halo outside the contracted subgroup would prove nothing
    about the subgroup the structural claim rests on."""
    import numpy as _np
    P = _load('e44', 'profile_inspection.json')
    z = _np.load(os.path.join(ROOT, 'results', 'e32', 'controls.npz'))
    k = int(_np.argmin(z['rvmax_eps']))
    assert int(z['sidm_id'][k]) == P['halo']['sidm_id']
    assert z['rvmax_eps'][k] < 4.0
    assert z['collapse_dex'][k] < -0.3


@pytest.mark.regression
def test_the_quadrature_residual_is_an_identity_in_three_measured_quantities():
    """The predictor's error is exactly determined, not empirically small.

    This is the instrument the disjoint-system experiment depends on, so it is checked on data already in hand rather than
    assumed on data that is not. Writing c for the cross-term share, k for the
    inflation of the error at the corrupted coordinate over the oracle error,
    and g for the displacement in units of the oracle error,

        r = log10(quadrature / E_dep) = 0.5*log10[(1+g^2)(1-c)/(g^2+k^2)]

    is algebra, not a fit, and it is zero exactly when c = 0 and k = 1. A
    discrepancy would mean one of the four measured quantities in the
    decomposition is wrong, so the bar is machine epsilon rather than a
    tolerance chosen to pass.
    """
    _need('results/statecond/decomposition.json')
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from d0_identity_check import (diagnostics, residual_from_identity,
                                   residual_measured)
    src = os.path.join(os.path.dirname(__file__), '..', 'results',
                       'statecond', 'decomposition.json')
    rows = {k: v for k, v in json.load(open(src)).items()
            if isinstance(v, dict) and 'E_A' in v}
    assert len(rows) == 15, f'{len(rows)} decomposition points, expected 15'
    worst = max(abs(residual_from_identity(*diagnostics(v)) - residual_measured(v))
                for v in rows.values())
    assert worst < 1e-12, (
        f'the closed form departs from the measurement by {worst:.2e} dex; '
        'one of E_A, F, E_dep, e_dep is not what it claims to be')

    # and it reproduces the two summary statements the article makes about the
    # cross term, which is what ties the algebra to the printed claim
    shares = [diagnostics(v)[0] for v in rows.values()]
    assert sum(s < 0 for s in shares) == 12, 'the cross term changed sign somewhere'
    assert abs(float(np.median(shares)) + 0.22) < 0.01, (
        f'median cross share is {np.median(shares):+.4f}; the article says -0.22')


@pytest.mark.regression
def test_the_matched_accuracy_speedup_is_the_stated_interpolation():
    r"""Recompute the 928x from the sweep, by log-log interpolation at 3e-4.

    This is a verification gate, not a regression one: it does not compare the
    stored speed-up with a remembered number, it rebuilds it from the two
    work-precision curves and the interpolation rule written in the text. A
    referee did the same arithmetic at the target the text *used* to quote,
    2e-4, got 798x, and reported the headline as irreproducible. The number was
    right and the stated target was wrong -- the artefact interpolates at 3e-4.
    Nothing caught that, because every gate on this claim compared 928 with 928.
    """
    _need('results/e8/workprecision.json')
    import json, math
    rec = json.load(open(os.path.join(ROOT, 'results', 'e8', 'workprecision.json')))

    def interpolate(curve, target):
        pts = sorted(((r['rel_err'], r['wall_median']) for r in curve
                      if r.get('rel_err') and r.get('wall_median')), reverse=True)
        for (e1, t1), (e2, t2) in zip(pts, pts[1:]):
            if e2 <= target <= e1:
                f = ((math.log10(target) - math.log10(e1))
                     / (math.log10(e2) - math.log10(e1)))
                return 10 ** (math.log10(t1) + f * (math.log10(t2) - math.log10(t1)))
        raise AssertionError(f'{target:g} is not bracketed by the measured points')

    target = 3e-4
    imp = interpolate(rec['curves']['implicit'], target)
    exp = interpolate(rec['curves']['classic'], target)

    stored = rec['matched_accuracy_scheme_speedup']['3e-04']
    assert abs(imp - stored['implicit_wall']) < 1e-9, (imp, stored['implicit_wall'])
    assert abs(exp - stored['explicit_wall']) < 1e-6, (exp, stored['explicit_wall'])
    assert abs(exp / imp - stored['scheme_speedup']) < 1e-6

    assert f'{exp / imp:.0f}' == '928'

    # and the alternative target the referee read it at, so the sensitivity the
    # text now reports is checked rather than asserted
    alt = interpolate(rec['curves']['classic'], 2e-4) / interpolate(
        rec['curves']['implicit'], 2e-4)
    assert f'{alt:.0f}' == '798', alt


@pytest.mark.regression
def test_the_serving_regime_of_each_input_path_is_recomputed_from_the_records():
    r"""Rebuild the gate's serving regime on both paths from the per-trajectory
    records, rather than comparing a stored summary with itself.

    The gate returns an answer when the RMS conformal half-width over a
    trajectory's field fits the user's budget, so the budget a path can serve
    at is a property of that statistic's distribution. The ladder in the gate
    table is fixed on the oracle path; this recomputes where the deployed path
    sits against it, and pins the regime it finds.
    """
    _need('results/statecond/conformal_deployed_records.npz')
    import json
    import numpy as np

    rec = np.load(os.path.join(ROOT, 'results', 'statecond',
                               'conformal_deployed_records.npz'))
    gated = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                        'gated_deployment.json')))
    # Normalised field units, matching the ladder: the budget ladder is fixed
    # on residuals that were never rescaled by the channel standard
    # deviations, so a serving decision must be taken in the same units.
    stat = rec['gate_statistic_rms']
    assert stat.shape == (130,)

    oracle_median = gated['sets']['test']['rms']['median_statistic']
    ladder = sorted(float(k.rsplit('_', 1)[1]) for k in rec.files
                    if k.startswith('served_at_'))
    assert len(ladder) == 5
    # the ladder is fixed on the oracle path, whose median is its first rung;
    # the rung is carried in the record key, rounded to six decimals
    assert abs(ladder[0] - oracle_median) < 1e-6

    # every rung of the oracle ladder lies below the whole deployed
    # distribution, so the deployed path serves in a different budget regime
    assert stat.min() > ladder[-1], (stat.min(), ladder[-1])
    for rung in ladder:
        served = rec[f'served_at_{rung:.6f}']
        assert served.sum() == 0, (rung, int(served.sum()))

    # median gate statistic, narrowest band, the budget that serves nine in ten
    assert (f'{np.median(stat):.3f}', f'{stat.min():.3f}',
            f'{np.percentile(stat, 90):.3f}') == ('0.458', '0.226', '0.854')

    # coverage is what makes the wide band a valid one, so it travels with it
    assert f"{rec['covered'].mean():.3f}" == '0.877'


@pytest.mark.regression
def test_the_gate_statistic_and_its_budget_ladder_share_one_unit():
    r"""The serving statistic and the budget it is compared against must be
    measured in the same units.

    The budget ladder is fixed on residuals taken straight from the normalised
    field, never rescaled by the per-channel standard deviations. A statistic
    carrying that rescaling would be compared against a ladder that does not,
    and the two channels convert to dex through scales that differ by an order
    of magnitude, so the mixture would read as neither. This rebuilds the
    identity between the normalised statistic and the two per-channel physical
    ones through the recorded scales, which holds only if all three come from
    the same half-widths.
    """
    _need('results/statecond/conformal_deployed_records.npz')
    import json
    import numpy as np

    rec = np.load(os.path.join(ROOT, 'results', 'statecond',
                               'conformal_deployed_records.npz'))
    dep = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                      'conformal_deployed.json')))
    scales = (dep['paths']['deployed']['levels']['0.90']
              ['channel_scales_dex_per_normalised_unit'])
    s_rho, s_v = scales['density'], scales['velocity']
    assert s_rho / s_v > 10, (s_rho, s_v)

    norm = rec['gate_statistic_rms']
    rho = rec['gate_statistic_rms_dex_density'] / s_rho
    vel = rec['gate_statistic_rms_dex_velocity'] / s_v
    # RMS over both channels is the quadrature mean of the two per-channel RMS
    rebuilt = np.sqrt((rho ** 2 + vel ** 2) / 2.0)
    assert np.allclose(rebuilt, norm, rtol=1e-6), (
        float(np.abs(rebuilt - norm).max()))


@pytest.mark.verification
def test_the_reserved_sidm_retention_is_recomputed_from_the_solved_draws():
    r"""Rebuild the event-completion rate of the reserved SIDM prior.

    The predictor's SIDM score is taken over the draws that terminate at the
    density criterion or the Knudsen floor inside the integration horizon, so
    the size, the shape and the make-up of that population are part of the
    claim. All three are recomputed here from the solved draws and pinned.
    """
    import h5py
    import numpy as np

    path = os.path.join(ROOT, 'data', 'sc', 'reserved', 'sidm_full.h5')
    if not os.path.exists(path):
        pytest.skip('reserved SIDM draws are not in this tree')
    with h5py.File(path, 'r') as h:
        status = h['status'][:]
        lsig = h['log_sigma0'][:]
    keep = status != 1
    assert keep.size == 64
    assert int(keep.sum()) == 31
    # which of the two criteria ended each retained run
    n_density, n_knudsen = int((status == 0).sum()), int((status == 2).sum())
    assert (n_density, n_knudsen) == (13, 18), (n_density, n_knudsen)

    # retention is governed by the cross-section normalisation
    edges = np.quantile(lsig, [1 / 3, 2 / 3])
    bands = [lsig <= edges[0],
             (lsig > edges[0]) & (lsig <= edges[1]),
             lsig > edges[1]]
    counts = [(int(keep[b].sum()), int(b.sum())) for b in bands]
    assert counts == [(1, 22), (14, 21), (16, 21)], counts
    assert counts[0][0] / counts[0][1] < counts[-1][0] / counts[-1][1]


@pytest.mark.regression
def test_the_matched_accuracy_speedup_survives_its_timing_uncertainty():
    r"""Re-derive the interval quoted for the matched-accuracy speed-up.

    The explicit points are single measurements, so the speed-up is quoted with
    an assumed timing uncertainty rather than a measured one. This rebuilds the
    interval from the sweep by moving each explicit bracket independently and
    carrying the measured semi-implicit spreads, and pins what that
    construction gives.
    """
    _need('results/e8/workprecision.json')
    import itertools
    import json

    import numpy as np

    wp = json.load(open(os.path.join(ROOT, 'results', 'e8',
                                     'workprecision.json')))
    def brackets(curve, target):
        rows = [r for r in wp['curves'][curve] if r.get('rel_err')]
        rows.sort(key=lambda r: r['rel_err'])
        below = [r for r in rows if r['rel_err'] <= target]
        above = [r for r in rows if r['rel_err'] > target]
        assert below and above, (curve, target)
        return [below[-1], above[0]]

    def interp(rows, target, scale):
        (a, b) = rows
        x = np.log10([a['rel_err'], b['rel_err']])
        y = np.log10([a['wall_median'] * scale[0], b['wall_median'] * scale[1]])
        return 10 ** (y[0] + (y[1] - y[0])
                      * (np.log10(target) - x[0]) / (x[1] - x[0]))

    for target, quoted, lo_hi, head in ((3e-4, 928, (880, 980), 930),
                                        (2e-4, 798, (750, 840), 800)):
        semi, expl = brackets('implicit', target), brackets('classic', target)
        base = interp(expl, target, (1, 1)) / interp(semi, target, (1, 1))
        assert abs(base - quoted) / quoted < 0.01, (target, base)
        spread = [r['wall_rel_spread'] for r in semi]
        vals = [interp(expl, target, de) / interp(semi, target, ds)
                for de in itertools.product((0.95, 1.0, 1.05), repeat=2)
                for ds in itertools.product(*[(1 - s, 1.0, 1 + s)
                                              for s in spread])]
        # The explicit points are single runs, so the factor is quoted to
        # the two significant figures the spread supports rather than to the
        # full precision of the interpolation.
        assert (round(min(vals), -1), round(max(vals), -1)) == lo_hi, (
            target, min(vals), max(vals))
        assert round(base, -1) == head, (target, base)


@pytest.mark.regression
def test_the_speedup_printed_on_the_figure_is_the_one_the_sweep_gives():
    """A figure label is not gated by any number check, and drifted.

    The matched-accuracy factor is quoted to two significant figures, while
    `fig20_work_precision` once kept rendering the full-precision value, 928
    against 930. This reads the text back out of the figure and requires it to
    match the sweep at the quoted precision.
    """
    import json
    import re

    pypdf = pytest.importorskip('pypdf')
    fig = os.path.join(ROOT, 'results', 'figures', 'fig20_work_precision.pdf')
    if not os.path.exists(fig):
        pytest.skip('figure not built')
    drawn = pypdf.PdfReader(fig).pages[0].extract_text() or ''
    labels = re.findall(r'(\d{3,4})\s*[x×]', drawn)
    assert labels, f'no speed-up label found in the figure: {drawn[:200]!r}'

    w = json.load(open(os.path.join(ROOT, 'results', 'e8',
                                    'workprecision.json')))
    want = round(w['matched_accuracy_scheme_speedup']['3e-04']['scheme_speedup'], -1)
    assert int(labels[0]) == int(want), (
        f'figure prints {labels[0]}x, the sweep gives {want:.0f}x')
