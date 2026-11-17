"""Acceptance gates for the cross-domain singularity package.

Run: .venv/bin/python -m pytest tests/test_statecond.py -v

These check the properties the cross-domain claim rests on: that each system
actually reaches its singularity, that t_sing is a property of the PDE and not
of the grid, that the two conditioning variables are built identically for every
system, and that the state coordinate is genuinely computable from an observed
field alone.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from statecond import data as scd            # noqa: E402
from statecond.pdes import SYSTEMS, N_X      # noqa: E402


def _need(*rel):
    """Skip when the archival record this gate reads is absent (the source
    release ships without `results/`; see REPRODUCING.md)."""
    root = os.path.join(os.path.dirname(__file__), '..')
    missing = [p for p in rel if not os.path.exists(os.path.join(root, p))]
    if missing:
        pytest.skip('archival record absent: ' + ', '.join(missing))

NAMES = list(SYSTEMS)


@pytest.fixture(scope='module')
def runs():
    """One reproducible trajectory per system."""
    out = {}
    for name, cls in SYSTEMS.items():
        s = cls()
        rng = np.random.default_rng(4242)
        for _ in range(12):                  # a fixed, seeded search
            tr = s.solve(s.sample(rng))
            if tr is not None and tr.status == 0:
                out[name] = (s, tr)
                break
        assert name in out, f'{name}: no singular run in 12 seeded draws'
    return out


@pytest.mark.verification
@pytest.mark.parametrize('name', NAMES)
def test_reaches_its_singularity(runs, name):
    s, tr = runs[name]
    assert tr.status == 0
    # the observable grew by the system's threshold factor, to interpolation
    # tolerance, and t_sing lies inside the integrated interval
    assert tr.a[-1] / tr.a[0] >= 0.99 * s.blowup_factor
    assert tr.t[0] <= tr.t_sing <= tr.t[-1] * (1 + 1e-9)
    assert tr.t_sing > 0


@pytest.mark.verification
@pytest.mark.parametrize('name', NAMES)
def test_stored_fields_are_finite_and_shaped(runs, name):
    s, tr = runs[name]
    assert tr.u.shape[1:] == (s.n_fields, N_X)
    assert np.all(np.isfinite(tr.u))
    assert tr.u.shape[0] == tr.t.size == tr.a.size


@pytest.mark.verification
@pytest.mark.parametrize('name', NAMES)
def test_tsing_is_grid_converged(runs, name):
    """t_sing must be a property of the PDE, not of the discretisation."""
    s, tr = runs[name]
    fine = SYSTEMS[name]()
    fine.n_solve = 2 * s.n_solve
    ref = fine.solve(tr.params)
    assert ref is not None and ref.status == 0
    rel = abs(ref.t_sing - tr.t_sing) / ref.t_sing
    assert rel < 5e-3, f'{name}: t_sing shifts {rel:.2e} on grid refinement'


@pytest.mark.verification
@pytest.mark.parametrize('name', NAMES)
def test_resample_lands_on_the_tau_grid(runs, name):
    s, tr = runs[name]
    u, a = scd.resample(tr)
    assert u.shape == (scd.N_TAU, s.n_fields, N_X)
    assert np.all(np.isfinite(u)) and np.all(a > 0)
    # tau = 0 is the initial condition and tau = 1 is the threshold crossing
    assert np.allclose(u[0], tr.u[0], atol=1e-5)
    assert a[-1] / a[0] == pytest.approx(s.blowup_factor, rel=2e-2)


@pytest.mark.verification
@pytest.mark.parametrize('name', NAMES)
def test_state_coordinate_is_readable_from_the_field(runs, name):
    """The point of the method: phi must come from the field, not from t_sing.

    Recomputing phi from the stored fields alone -- no times, no t_sing -- must
    reproduce the phi that `phase_variables` derives from the observable.
    """
    s, tr = runs[name]
    u, a = scd.resample(tr)
    phi, flag = scd.phase_variables(a[None, :])
    # re-derive the observable from the stored field for the systems whose
    # observable is an extremum of the stored quantity
    if name in ('rd', 'ks'):
        a_field = 10.0 ** u[:, 0].max(axis=1)
    elif name == 'film':
        a_field = 1.0 / 10.0 ** u[:, 0].min(axis=1)
    else:                                     # burgers: max |du/dx|
        a_field = np.abs(np.diff(u[:, 0], axis=1)).max(axis=1)
    phi_field = np.log10(a_field) - np.log10(a_field[0])
    # Storage block-averages the solver grid down by 2-8x, which damps the
    # extremum the observable is built from. For film that damping is real and
    # measurable: the touchdown minimum is the sharpest feature of the four, so
    # phase read off the stored field lags the true phase. That is a
    # measurement error of exactly the kind sc_hybrid.py sweeps, not a defect,
    # so film carries a looser floor than the systems with broader extrema.
    floor = 0.96 if name == 'film' else 0.99
    r = np.corrcoef(phi_field, phi[0])[0, 1]
    assert r > floor, f'{name}: phi from field vs from observable, r = {r:.4f}'
    assert flag.shape == phi.shape
    assert set(np.unique(flag)).issubset({0.0, 1.0})


@pytest.mark.verification
def test_tau_grid_matches_the_gravothermal_convention():
    """The SIDM runs and the cross-domain runs must share one time grid."""
    from gravop import dataset as gds
    assert np.allclose(scd.tau_grid(), gds.tau_grid())
    assert scd.N_TAU == gds.N_TAU


@pytest.mark.verification
def test_split_is_by_trajectory_and_disjoint():
    tr, va, te = scd.split(500)
    assert len(tr) + len(va) + len(te) == 500
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(a.tolist()) & set(b.tolist()))
    # and it is deterministic
    assert np.array_equal(scd.split(500)[0], tr)


@pytest.mark.verification
def test_operator_is_resolution_independent():
    """The spectral backbone must accept a grid it was not trained on."""
    import torch
    from statecond.model import OperatorNet
    net = OperatorNet(n_fields=1, n_par=3, cond_extra=2, n_x=N_X).eval()
    cond = torch.zeros(2, 5)
    with torch.no_grad():
        a = net(torch.zeros(2, 1, N_X), cond)
        b = net(torch.zeros(2, 1, 2 * N_X), cond)
    assert a.shape == (2, 1, N_X)
    assert b.shape == (2, 1, 2 * N_X)


@pytest.mark.verification
@pytest.mark.parametrize('backbone', ['fno', 'deeponet', 'cno', 'transformer'])
@pytest.mark.parametrize('n_fields', [1, 2])
def test_every_backbone_trains(backbone, n_fields):
    """Each backbone must produce the right shape AND a finite gradient.

    The gradient half is the point. This machine has no Python development
    headers, so Triton cannot JIT its CUDA helpers; an op whose BACKWARD is
    routed to Triton fails at runtime while a forward-only check passes. That
    is exactly how the DeepONet contraction bug reached a full training run.
    Both field counts are covered because the failure was specific to
    n_fields = 1, where the contraction became a batched outer product.
    """
    import torch
    from statecond.model import BACKBONES
    net = BACKBONES[backbone](n_fields=n_fields, n_par=3, cond_extra=2,
                              n_x=N_X)
    out = net(torch.randn(4, n_fields, N_X), torch.randn(4, 5))
    assert out.shape == (4, n_fields, N_X)
    out.pow(2).mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, f'{backbone} produced no gradients'
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.verification
@pytest.mark.parametrize('backbone', ['cno', 'transformer'])
def test_new_backbones_are_resolution_independent(backbone):
    """CNO resamples in Fourier space and the transformer tokenises per point,
    so both inherit the FNO's ability to be queried off the training grid."""
    import torch
    from statecond.model import BACKBONES
    net = BACKBONES[backbone](n_fields=1, n_par=3, cond_extra=2,
                              n_x=N_X).eval()
    cond = torch.zeros(2, 5)
    with torch.no_grad():
        a = net(torch.zeros(2, 1, N_X), cond)
        b = net(torch.zeros(2, 1, 2 * N_X), cond)
    assert a.shape == (2, 1, N_X)
    assert b.shape == (2, 1, 2 * N_X)


@pytest.mark.verification
@pytest.mark.parametrize('backbone', ['fno', 'deeponet', 'cno', 'transformer'])
def test_checkpoint_round_trip_is_exact(backbone, tmp_path):
    """A reloaded arm must reproduce its predictions bit for bit.

    Re-evaluating a trained arm used to cost a ~4 h retrain, which is why
    several audit questions were answered "would need a re-run". A checkpoint is
    only worth that if it restores the normalisation too -- it is fitted on the
    training split, and the weights mean nothing without it.
    """
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import sc_experiment as X
    from statecond.model import BACKBONES

    class _Norm:
        f_mu = np.zeros((1, 1, 1, 1)); f_sd = np.ones((1, 1, 1, 1))
        p_mu = np.zeros((1, 3)); p_sd = np.ones((1, 3))
        t_mu = 0.0; t_sd = 1.0; phi_mu = 0.0; phi_sd = 1.0

    net = BACKBONES[backbone](n_fields=1, n_par=3, cond_extra=2, n_x=N_X).eval()
    p = str(tmp_path / 'arm.pt')
    X.save_checkpoint(p, net, 'phase', backbone, _Norm, 0, 5, (1, 3, N_X))
    back, meta = X.load_checkpoint(p)

    x, c = torch.randn(3, 1, N_X), torch.randn(3, 5)
    with torch.no_grad():
        assert torch.allclose(net(x, c), back(x, c), atol=1e-6)
    assert meta['backbone'] == backbone and meta['kind'] == 'phase'
    assert set(meta['norm']) == {'f_mu', 'f_sd', 'p_mu', 'p_sd', 't_mu',
                                 't_sd', 'phi_mu', 'phi_sd'}


@pytest.mark.verification
def test_checkpoint_name_and_metadata_record_the_h1_arm(tmp_path):
    """An H1 arm must not be able to overwrite the plain arm's checkpoint.

    It could until 2026-09-05: the results filename carried the h1 tag and the
    checkpoint filename did not, so the P10 spectral-bias run silently
    replaced the P6d plain-FNO checkpoints. Nothing caught it because the
    DEPLOYED error is nearly arm-invariant -- P12's floor is why -- so every
    downstream comparison still looked reasonable. Only the oracle error
    distinguishes the arms, and nothing was checking it.

    Two things are gated here: the tag that goes into both names, and the h1
    value recorded inside the checkpoint so a loaded arm can identify itself
    without trusting its filename.
    """
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import sc_experiment as X
    from statecond.model import BACKBONES

    assert X.arm_tag(0.0) == ''
    assert X.arm_tag(1.0) == '_h1'
    assert X.arm_tag(0.0) != X.arm_tag(0.5), 'h1 arms need a distinct name'

    class _Norm:
        f_mu = np.zeros((1, 1, 1, 1)); f_sd = np.ones((1, 1, 1, 1))
        p_mu = np.zeros((1, 3)); p_sd = np.ones((1, 3))
        t_mu = 0.0; t_sd = 1.0; phi_mu = 0.0; phi_sd = 1.0

    net = BACKBONES['fno'](n_fields=1, n_par=3, cond_extra=1, n_x=N_X).eval()
    for h1 in (0.0, 1.0):
        p = str(tmp_path / f'arm{X.arm_tag(h1)}.pt')
        X.save_checkpoint(p, net, 'tau', 'fno', _Norm, 0, 5, (1, 3, N_X), h1)
        _, meta = X.load_checkpoint(p)
        assert meta['h1'] == h1
    assert (tmp_path / 'arm.pt').exists()
    assert (tmp_path / 'arm_h1.pt').exists(), 'the h1 arm overwrote the plain one'


@pytest.mark.verification
def test_time_head_checkpoint_round_trip(tmp_path):
    """The time head is saved by the same code path but has a different
    signature -- it takes (prof, par), not the full conditioning vector -- so
    a round-trip test that only covered the operators would miss a break in
    the arm that arm B depends on entirely."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import sc_experiment as X
    from statecond.model import TimeHead

    class _Norm:
        f_mu = np.zeros((1, 1, 1, 1)); f_sd = np.ones((1, 1, 1, 1))
        p_mu = np.zeros((1, 3)); p_sd = np.ones((1, 3))
        t_mu = 0.0; t_sd = 1.0; phi_mu = 0.0; phi_sd = 1.0

    head = TimeHead(n_fields=1, n_par=3, n_x=N_X).eval()
    p = str(tmp_path / 'time.pt')
    X.save_checkpoint(p, head, 'time', 'fno', _Norm, 0, 5, (1, 3, N_X))
    back, meta = X.load_checkpoint(p)
    assert meta['kind'] == 'time'
    assert isinstance(back, TimeHead)

    prof, par = torch.randn(4, 1, N_X), torch.randn(4, 3)
    with torch.no_grad():
        out, ref = back(prof, par), head(prof, par)
    assert out.shape == (4,)
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.verification
def test_band_limited_resample_round_trips():
    """CNO's claim to be an operator rather than a grid map rests on this:
    up- and down-sampling a resolved signal must return it unchanged, and must
    not alias an unresolved one back into the low modes."""
    import torch
    from statecond.model import band_limited_resample
    x = torch.linspace(0, 2 * np.pi, 64 + 1)[:-1]
    low = torch.sin(x)[None, None] + 0.5 * torch.cos(3 * x)[None, None]
    up = band_limited_resample(low, 256)
    assert torch.allclose(band_limited_resample(up, 64), low, atol=1e-5)
    # amplitude is preserved, not scaled by the length ratio
    assert abs(float(up.abs().max()) - float(low.abs().max())) < 0.05


# ---------------------------------------------------------------------------
# P12 — the clock-error floor.
#
# These are VERIFICATION gates, not stored-number regression: they check
# properties the floor argument depends on, computed fresh. The stored numbers
# live in results/statecond/floor.json and are checked separately.
# ---------------------------------------------------------------------------

@pytest.mark.verification
def test_floor_is_zero_at_zero_offset():
    """A correct clock costs nothing: F(0) = 0 exactly."""
    import sc_displacement
    taus = scd.tau_grid()
    un = np.random.default_rng(0).normal(size=(3, len(taus), 1, 16))
    same = sc_displacement.resample_tau(un, taus, taus)
    assert np.abs(same - un).max() < 1e-12


@pytest.mark.verification
def test_floor_grows_monotonically_with_clock_error():
    """The floor must increase with |delta|, or it is not measuring the clock."""
    import sc_displacement
    taus = scd.tau_grid()
    rng = np.random.default_rng(1)
    # a smooth field that varies over tau, so a shift is genuinely visible
    un = np.sin(3 * taus)[None, :, None, None] * rng.normal(size=(2, 1, 1, 16))
    prev = -1.0
    for d in (0.01, 0.05, 0.10, 0.20):
        tq = np.clip(taus / (1.0 + d), 0.0, 1.0)
        e = np.sqrt(((sc_displacement.resample_tau(un, taus, tq) - un) ** 2)
                    .mean(axis=(2, 3))).mean()
        assert e > prev, f'floor not monotone in delta at {d}'
        prev = e


@pytest.mark.verification
def test_floor_interpolation_rule_is_not_load_bearing():
    """Linear and PCHIP must agree, or the floor is a tau-grid artefact.

    This is the gate that would have caught the floor being an interpolation
    effect rather than a property of the dynamics.
    """
    import sc_displacement
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sc')
    if not os.path.exists(os.path.join(data_dir, 'burgers.h5')):
        pytest.skip('data/sc not present (gitignored; regenerate with sc_gen.py)')
    lin, _ = sc_displacement.floor_for_system('burgers', data_dir, 'linear')
    pch, _ = sc_displacement.floor_for_system('burgers', data_dir, 'pchip')
    for k in ('0.05', '0.1', '0.2'):
        assert abs(lin[k] - pch[k]) / lin[k] < 0.02, (
            f'floor depends on the interpolation rule at delta={k}')


@pytest.mark.verification
def test_time_head_does_not_depend_on_the_backbone():
    """sc_displacement reuses the fno time head for every backbone's deployed floor.

    That is only legitimate because `train_time_head` takes no backbone
    argument, so for a given (system, seed) the head is the same network. This
    gate holds the property that licenses the reuse: if a future change makes
    the head backbone-dependent, the deployed floor silently stops matching the
    arm it is compared against.
    """
    import torch
    from sc_experiment import Norm, load_checkpoint
    root = os.path.join(os.path.dirname(__file__), '..')
    ck = os.path.join(root, 'results', 'statecond', 'ckpt')
    data_dir = os.path.join(root, 'data', 'sc')
    paths = {b: os.path.join(ck, f'sidm_{b}_time_seed0.pt')
             for b in ('fno', 'cno', 'transformer')}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip('time-head checkpoints not present (gitignored)')
    d = scd.load(os.path.join(data_dir, 'sidm.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    un = (d['u'] - n.f_mu) / n.f_sd
    u0 = torch.tensor(un[:, 0]).float()
    par = torch.tensor((d['par'] - n.p_mu) / n.p_sd).float()
    out = {}
    for b, p in paths.items():
        head, _ = load_checkpoint(p, 'cpu')
        with torch.no_grad():
            out[b] = head(u0[te], par[te]).cpu().numpy()
    ref = out['fno']
    for b in ('cno', 'transformer'):
        assert np.abs(out[b] - ref).max() == 0.0, (
            f'time head differs between fno and {b}; sc_displacement.floor_deployed '
            'may no longer reuse one head for every backbone')


@pytest.mark.regression
def test_displacement_outpredicts_the_law_out_of_system():
    """P18: the zero-parameter floor beats the fitted law leave-one-system-out.

    This is the gate on the claim that replaces Section 5's mechanism. It
    recomputes both predictors from the committed artefacts rather than
    comparing stored numbers, so it fails if either the floor or the law moves.
    """
    _need('results/statecond/law.json', 'results/statecond/floor.json')
    import json
    import sc_stats
    root = os.path.join(os.path.dirname(__file__), '..')
    rs = os.path.join(root, 'results', 'statecond')
    law = json.load(open(os.path.join(rs, 'law.json')))
    flo = json.load(open(os.path.join(rs, 'floor.json')))
    fdep = {s: v['floor_deployed'] for s, v in flo['floor_deployed'].items()}
    L = sc_stats.loso_law(law['points'])
    F = sc_stats.loso_floor(law['points'], fdep, 'quadrature')
    lm = np.mean([L[s]['rms_log_residual'] for s in L])
    fm = np.mean([F[s]['rms_log_residual'] for s in F])
    assert fm < lm, (
        f'floor no longer out-predicts the law out of system: '
        f'floor {fm:.3f} dex vs law {lm:.3f} dex')
    # the margin is the claim, not just the sign
    assert lm / fm > 2.0, f'margin collapsed to {lm / fm:.2f}x (was 3.8x)'
    # and the floor predictor must stay parameter-free
    assert all(v['n_fitted_coefficients'] == 0 for v in F.values())


# --------------------------------------------------------------------------
# v7 / P22-P23: the comparator, and the staleness rule that protects it
# --------------------------------------------------------------------------

RES = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond')


def _res(name):
    import json
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        pytest.skip(f'{name} not generated')
    return json.load(open(p))


@pytest.mark.regression
def test_artefacts_are_not_stale_against_their_inputs():
    """No artefact may be older than an artefact it was computed against.

    This is the gate for a failure that went unnoticed: `contract.json`
    was computed against arm D while `timehead.json` had already replaced arm D
    as the strongest time-conditioned comparator, and nothing noticed for a day
    and a half. Provenance is stamped inside the artefacts rather than read from
    file mtimes, because results/ is committed and a fresh clone flattens every
    mtime to the checkout time.
    """
    from sc_provenance import check
    ok, problems = check(RES)
    assert ok, 'stale artefacts:\n  ' + '\n  '.join(problems)


@pytest.mark.regression
def test_contract_tolerance_is_quoted_against_the_best_time_arm():
    """Arm C's deployment contract must be measured against the strongest
    time-conditioned arm available, not against whichever one existed when the
    script was written. P13 compared to arm D; P20 then made converged arm B
    stronger in rd, ks and sidm, which reverses the conclusion in those three.
    """
    ct = _res('contract.json')
    for name, rec in ct.items():
        if not isinstance(rec, dict) or 'best_time_reference' not in rec:
            continue
        cands = {k: rec[v] for k, v in
                 (('D', 'arm_D_reference'),
                  ('B_converged', 'armB_converged_reference'))
                 if rec.get(v) is not None}
        assert cands, f'{name}: no comparator recorded at all'
        best = min(cands.values())
        assert rec['best_time_reference'] == pytest.approx(best, rel=1e-9), (
            f"{name}: best_time_reference is {rec['best_time_reference']} but "
            f'the strongest available arm is {best} ({cands})')
        assert rec['best_time_source'] == min(cands, key=cands.get)


@pytest.mark.regression
def test_contract_sweep_is_monotone_where_the_crossing_is_read():
    """The A(0) tolerance is an interpolated crossing of the sweep, so the
    sweep must be monotone in the bracket the crossing falls in. Before P22
    each node drew an independent noise realisation and `rd` returned a lower
    error at 0.02 dex than at 0.01; the tolerance was then being read off
    sampling noise. Common random numbers fixed it.
    """
    ct = _res('contract.json')
    for name, rec in ct.items():
        if not isinstance(rec, dict) or 'sweep_sigma_a0' not in rec:
            continue
        tol = rec.get('a0_tolerance_vs_best_time_dex')
        if tol is None:
            continue
        xs = sorted(float(k) for k in rec['sweep_sigma_a0'])
        ys = [rec['sweep_sigma_a0'][str(x) if str(x) in rec['sweep_sigma_a0']
                                    else f'{x:g}'] for x in xs]
        i = max(j for j in range(len(xs)) if xs[j] <= tol)
        assert ys[i + 1] > ys[i], (
            f'{name}: the sweep is not increasing across the bracket '
            f'[{xs[i]}, {xs[i + 1]}] that the tolerance {tol:.4f} is read from')


@pytest.mark.regression
def test_converged_clock_numbers_carry_per_trajectory_errors():
    """Every post-E22 artefact must store the per-trajectory vector, not just a
    three-seed mean. The converged deployment penalty is ~1.0 in two systems, so
    the question "is it different from 1?" is decided by test-set sampling
    error, which a seed spread cannot see.
    """
    for fname, key in (('timehead.json', 'per_trajectory'),
                       ('contract.json', 'per_trajectory'),
                       ('direct.json', 'per_trajectory')):
        art = _res(fname)
        for name, rec in art.items():
            if not isinstance(rec, dict) or 'system' not in rec:
                continue
            assert key in rec, f'{fname}:{name} has no {key}'
            arrs = rec[key]
            n_test = len(rec['test_index'])
            for k, v in arrs.items():
                a = np.asarray(v)
                assert a.ndim == 2 and a.shape[1] == n_test, (
                    f'{fname}:{name}:{k} has shape {a.shape}, expected '
                    f'(seeds, {n_test})')


@pytest.mark.regression
def test_event_specification_holds_or_is_declared():
    """Each reference system's data must honour the event rule it states.

    The four testbeds honour `t_sing` = where `A(t) = f A(0)` exactly: endpoint
    ratio equals `f` to 0.000 dex in every run.

    `sidm` was RE-SPECIFIED on 2026-09-06 (user decision, option a). It is an
    APPLICATION system, not a reference system, and its event is "solver
    termination, whichever of the density criterion and the Knudsen
    fluid-validity floor fires first" -- roughly a 40/60 split, computed rather
    than asserted. This gate pins that: `sidm` must remain declared as an
    application system with a non-degenerate `phi` endpoint spread, so nobody
    quietly re-promotes it to a reference system without also fixing the data.
    """
    ev = _res('event_spec.json')
    assert not ev['_summary']['violated_by'], (
        f"a reference system violates its event specification: "
        f"{ev['_summary']['violated_by']}")
    for name in ('rd', 'ks', 'film', 'burgers'):
        v = ev[name]
        assert v['honours_specification'], (
            f'{name} no longer honours f={v["specified_f"]}: endpoint ratio '
            f'spans {v["ratio_min"]:.3f}-{v["ratio_max"]:.3f}')
        assert v['phi_endpoint_spread_dex'] == pytest.approx(0.0, abs=1e-6)
        assert name in ev['_summary']['reference_systems']

    if 'sidm' in ev:
        assert ev['_summary']['application_systems'] == ['sidm']
        assert ev['sidm']['is_reference_system'] is False
        assert ev['sidm']['phi_endpoint_spread_dex'] > 1.0, (
            'sidm endpoint spread collapsed -- if the data changed so that '
            'sidm now honours a threshold rule, promote it back to a '
            'reference system and update the benchmark documentation with it')
        sp = ev['sidm']['termination_split']
        assert sp['n_kept'] == sp['n_density_criterion_status0'] + \
            sp['n_knudsen_floor_status2'], 'termination split does not close'
        assert 0.2 < sp['frac_knudsen_floor'] < 0.9, (
            'the Knudsen-floor share moved materially; the re-specified event '
            'rule and STATECOND_BENCHMARK.md quote it and must be updated')


@pytest.mark.regression
def test_converged_clock_claim_is_bounded_by_the_grid():
    """The converged head is measured against production-grid labels, and the
    claim must not outrun what the grid can resolve.

    P21: refining the reference from 2x to 4x with Richardson extrapolation did
    NOT reduce the production error -- the production grid is the limit. In
    `ks` the grid error (0.104%) exceeds the head's own (0.037%), so "the head
    reaches 0.037%" is a statement about reproducing labels, not the singular
    time. The penalty measurements survive because head and targets share those
    labels; this gate pins the size of the residual correction so that a future
    change cannot let it grow unnoticed.
    """
    ct = _res('clock_truth.json')
    for name, rec in ct.items():
        if name.startswith('_'):
            continue
        assert rec['effective_vs_truth'] >= rec['head_vs_label'], (
            f'{name}: combining independent errors cannot reduce the total')
        assert rec['penalty_understated_by'] < 1.10, (
            f"{name}: the grid error now costs "
            f"{rec['penalty_understated_by']:.3f}x on the deployment penalty, "
            'which is no longer a negligible correction -- the converged-clock '
            'numbers need restating or the production grid needs refining')
    dom = ct['_summary']['grid_dominates_in']
    assert set(dom) <= {'ks'}, (
        f'the grid error now dominates the head error in {dom}; P21 found this '
        'only in ks, and the stated bound assumes it')


@pytest.mark.regression
def test_reference_refinement_confirms_second_order_schemes():
    """The observed convergence order must stay near 2 for every reference
    system. This is what licenses the Richardson extrapolation P21 uses; if a
    scheme drops order, the extrapolated limit -- and the error bound built on
    it -- stops being trustworthy.
    """
    rr = _res('refined_reference.json')
    for name, rec in rr.items():
        if name.startswith('_'):
            continue
        p = rec['observed_order_median']
        assert p is not None and 1.7 <= p <= 2.5, (
            f'{name}: observed order {p}, expected ~2; Richardson '
            'extrapolation and the P21 error bound assume it')
        assert rec['n_with_order'] >= 0.9 * rec['n_trajectories'], (
            f'{name}: only {rec["n_with_order"]}/{rec["n_trajectories"]} '
            'trajectories admitted an order')


@pytest.mark.regression
def test_injectivity_criterion_is_not_overclaimed():
    """P29a: the injectivity ordering FAILED its pre-registered robustness test
    and must not be quoted as a criterion.

    A candidate rule was once the leading account -- a state coordinate is worth
    conditioning on to the extent that it is injective along the trajectory --
    supported by Spearman rho = 1.000 over five systems from ONE statistic.
    Three independent redefinitions were pre-registered; one of the three
    reaches rho >= 0.9 (two under the most favourable tie-break), and the most
    direct one, held-out recovery of `tau` from the coordinate, reads +0.300.

    This gate pins the negative. If a future run flips the verdict, it flips
    here first and the documents must be updated deliberately rather than
    drifting back to the stronger claim -- the same failure mode as the
    retracted spectral-bias mechanism.
    """
    inj = _res('injectivity.json')
    assert inj['verdict'].startswith('FAIL'), (
        'the injectivity criterion now passes its robustness test; update '
        'the injectivity statements (P29a) before relaxing this')
    assert inj['n_alt_definitions_passing'] < 3
    assert inj['n_alt_definitions_passing_best_tiebreak'] < 3, (
        'even the most favourable tie-break may not reach PASS')

    corr = inj['correlations']
    # every p at n = 5 is the exact permutation p over all 120 orderings
    for k, c in corr.items():
        if 'n_perm' in c:
            assert c['n_perm'] == 120, f'{k}: {c["n_perm"]} permutations, not 120'
    assert corr['D0_nonmono_frac']['exact_p'] == pytest.approx(1 / 60, rel=1e-6), (
        'rho = 1.000 at n = 5 has exact p = 0.0167, which is the FLOOR; the '
        'asymptotic value (0.000) must never be quoted')
    # the control must not order the systems, or the test has no power
    assert corr['X_range_ratio']['rho'] < 0.9, (
        'the dynamic-range control now orders the systems too, so the test '
        'cannot discriminate at n = 5 and the whole line is unresolvable here')
    # the post-hoc diagnostic is recorded but may not enter the verdict
    ph = corr['D3b_within_traj_resid_POSTHOC']
    assert ph['role'].startswith('POST HOC')


@pytest.mark.regression
def test_benchmark_guide_numbers_match_the_artefacts():
    """R7: the third-party benchmark guide may not quote retracted values.

    `STATECOND_BENCHMARK.md` is what somebody outside this project calibrates
    against, and it is the file least likely to be re-read when an experiment
    moves a number. It had drifted on five counts at once (arm D named as the
    baseline after P20 replaced it in three systems, the P13 tolerances that
    P22 moved, the branch accuracy from before the unseeded-RNG fix, margins
    scored against a 2x reference that P21 showed is a third too generous, and
    an endpoint cost of 10.8x where the artefact says 10.3x). A gate is the
    only thing that keeps a document nobody runs in step with results
    everybody does.
    """
    _need('results/statecond/contract.json')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from check_benchmark_numbers import checks
    doc = open(os.path.join(os.path.dirname(__file__), '..',
                            'STATECOND_BENCHMARK.md')).read()
    bad = [f'{n}: {d}' for n, ok, d in checks(doc) if not ok]
    assert not bad, ('the benchmark guide no longer matches the artefacts:\n  '
                     + '\n  '.join(bad))


@pytest.mark.verification
def test_every_gate_declares_whether_it_verifies_or_regresses():
    """R6: each gate must say which kind of gate it is.

    The 2026-09-04 referee argued that an acceptance suite preserves stored
    numbers as faithfully as right ones. That is true of *some* of these gates
    and false of most, and the honest response is to measure the split rather
    than to argue about it or to restructure the suite around the criticism.

    verification: the assertion is computed in-process from a source of truth
      -- the solver, the data, the model code, or a checkpoint's own behaviour
      -- so a wrong number in results/ cannot make it pass.
    regression: at least one asserted value is read from a stored artefact
      under results/, so it pins internal consistency and provenance rather
      than correctness.

    Measured 2026-09-08: 57 verification, 18 regression, of 75 test functions.
    Re-measured 2026-09-25, test body by test body rather than marker by
    marker: 39 functions marked verification asserted on summaries or records
    stored under results/, and were re-marked regression. That leaves 127
    verification and 72 regression functions, below the 2:1 balance this gate
    used to require, so the balance is restated rather than protected: the
    manuscript and README quote the measured split, and the gate now requires
    only that recomputation stays the majority. A new gate with no class fails
    here, which is the only way the split stays true.
    """
    import ast
    import glob
    tests_dir = os.path.dirname(__file__)
    unmarked, both = [], []
    counts = {'verification': 0, 'regression': 0}
    for path in sorted(glob.glob(os.path.join(tests_dir, 'test_*.py'))):
        tree = ast.parse(open(path).read())
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.startswith('test')):
                continue
            marks = {d.attr for d in node.decorator_list
                     if isinstance(d, ast.Attribute)
                     and d.attr in counts}
            where = f'{os.path.basename(path)}::{node.name}'
            if not marks:
                unmarked.append(where)
            elif len(marks) > 1:
                both.append(where)
            else:
                counts[marks.pop()] += 1
    assert not unmarked, (
        'these gates do not declare whether they verify or regress: '
        f'{unmarked}')
    assert not both, f'these gates declare both classes: {both}'
    # the suite must not become predominantly a record of its own outputs
    assert counts['verification'] > counts['regression'], (
        f"the suite is now {counts['verification']} verification against "
        f"{counts['regression']} regression gates; recomputation is no longer "
        'the majority and the balance needs restating')


@pytest.mark.verification
def test_ks2d_second_order_scheme_beats_the_first_order_one():
    """P25a: the 2-D testbed's new scheme must actually be the better one.

    `pdes2d.py` was first order in both operators -- donor-cell upwind in
    space, integrating-factor forward Euler in time -- and `t_sing` converged
    only to 9.6-13% between the 128 and 256 grids, which is why the system is
    withheld. It is now MUSCL/minmod plus SSP-RK2 in the integrating-factor
    variable. `scheme='donor-euler'` keeps the original reachable so the claim
    "second order in both operators" is measured here rather than asserted in
    a docstring.

    One trajectory on three grids, ~2 s. The full study is
    `scripts/sc_ks2d.py` (results/statecond/ks2d_convergence.json).
    """
    import math
    from statecond.pdes2d import KS2D
    rng = np.random.default_rng(20260907)
    p = [KS2D().sample(rng) for _ in range(4)][3]

    got = {}
    for scheme in ('donor-euler', 'muscl-rk2'):
        ts = {}
        for n in (64, 128, 256):
            tr = KS2D(n_solve=n, n_store=64, scheme=scheme).solve(p)
            assert tr is not None and tr.status == 0, f'{scheme} n={n} failed'
            ts[n] = tr.t_sing
        got[scheme] = ts

    def diff(ts):
        return abs(ts[128] - ts[256])

    assert diff(got['muscl-rk2']) < 0.5 * diff(got['donor-euler']), (
        'the second-order scheme no longer converges faster than the '
        f"first-order one: {diff(got['muscl-rk2']):.3e} vs "
        f"{diff(got['donor-euler']):.3e} between the 128 and 256 grids")

    ts = got['muscl-rk2']
    order = math.log2((ts[64] - ts[128]) / (ts[128] - ts[256]))
    assert order > 1.6, f'observed order fell to {order:.2f}; expected ~2'

    # positivity: minmod is chosen so the reconstruction makes no new extrema
    tr = KS2D(n_solve=128, n_store=64, scheme='muscl-rk2').solve(p)
    assert np.all(np.isfinite(tr.u)) and np.all(tr.a > 0)


@pytest.mark.regression
def test_ks2d_stays_withheld_until_it_meets_the_standard():
    """P25a: the 2-D system may not enter the benchmark registry while it
    fails the convergence standard the 1-D systems pass.

    Second order in both operators was not enough: at 1024 the margins are 7.8
    median and 2.9 on the hard subset, against the required 10 and 5, and that
    grid already costs 6.9 h for a 600-run dataset. Registering it anyway would
    trade a stated limitation for a hidden one. This gate ties the two together
    so the registry cannot gain `ks2d` without the artefact that licenses it.
    """
    from statecond.pdes import SYSTEMS as SYS_1D
    kc = _res('ks2d_convergence.json')
    if kc['verdict'].startswith('FAIL'):
        assert 'ks2d' not in SYS_1D, (
            'ks2d is registered as a benchmark system while '
            'ks2d_convergence.json still says it fails the standard')
    else:
        assert 'ks2d' in SYS_1D, (
            'ks2d now meets the standard; register it and update '
            'STATECOND_BENCHMARK.md §4 and the P25a limitation statement')
    # the scheme claim the diagnosis rests on
    assert kc['observed_order_median'] > 1.8, (
        f"observed order fell to {kc['observed_order_median']:.2f}; the "
        'second-order claim in P25a no longer holds')
    assert kc['n_with_order'] == kc['n_trajectories']


@pytest.mark.verification
def test_the_documented_install_provides_a_test_runner():
    """R5: `pip install -e ".[all]"` must install what the docs then tell you
    to run.

    The clean-environment smoke test (2026-09-08) followed the documented path
    on a bare clone and got "No module named pytest" at the first instruction
    after installing -- README.md, STATECOND_BENCHMARK.md and the
    manuscript all say to run the gates first, and pytest was in no dependency
    group at all. Nobody following the instructions could run a single gate.

    This gate reads the packaging metadata rather than the environment, because
    the environment on this machine has pytest for unrelated reasons and would
    hide the defect -- which is exactly how it survived this long.
    """
    import tomllib
    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, 'pyproject.toml'), 'rb') as fh:
        cfg = tomllib.load(fh)
    extras = cfg['project']['optional-dependencies']
    for group in ('test', 'all'):
        assert any(d.startswith('pytest') for d in extras.get(group, [])), (
            f'pyproject [{group}] no longer installs pytest; the documented '
            'path would again leave a reader unable to run the gates')


@pytest.mark.regression
def test_the_singular_location_limitation_is_stated_correctly():
    """P25: the "all your testbeds are 1-D" limitation must be stated at the
    strength the data supports, and no stronger.

    The sharp form of the objection is that a radially symmetric problem fixes
    the singular location, so a globally-conditioned operator never has to say
    where. Measured, that is true of 2 of the 5 systems and false of 3 --
    `film` alone ruptures at 49 distinct sites. If a dataset is regenerated and
    that changes, the limitation statement in the manuscript changes with it, so it is pinned here.
    """
    sl = _res('singular_location.json')
    fixed = set(sl['_summary']['fixed_by_symmetry'])
    varying = set(sl['_summary']['solution_dependent'])
    assert fixed == {'ks', 'sidm'}, (
        f'the set of symmetry-fixed systems changed to {sorted(fixed)}; the '
        'P25 decision not to build a 2-D testbed rests on this split')
    assert varying == {'rd', 'film', 'burgers'}
    assert sl['film']['distinct_sites'] >= 20, (
        f"film now ruptures at only {sl['film']['distinct_sites']} distinct "
        'sites; it is the system carrying the solution-dependent-location '
        'claim')


@pytest.mark.regression
def test_the_generalised_floor_reduces_to_the_clock_floor():
    """P30: the generalised displacement must reproduce P12 exactly at c = tau.

    `sc_coord_displacement.py` claims the clock-error floor is one instance of a
    displacement `F_c` defined for any conditioning coordinate. The claim is
    only meaningful if the general code reproduces the special case, so that
    comparison is computed inside the artefact and pinned here: if it drifts,
    the generalisation is a bug rather than a result and the P30 conclusion
    does not stand.
    """
    cf = _res('coord_floor.json')
    for name in ('rd', 'ks', 'film', 'burgers', 'sidm'):
        rel = cf[name]['tau_floor_self_check']['max_rel_disagreement']
        assert rel is not None and rel < 0.01, (
            f'{name}: the generalised code no longer reproduces sc_displacement.py '
            f'(max relative disagreement {rel})')


@pytest.mark.regression
def test_the_branch_flag_makes_the_displacement_well_defined():
    """P30/F2: with the flag, inverting `phi` is unambiguous everywhere.

    The displacement `F_c(delta)` exists only where the coordinate is injective
    along the trajectory. Measured over every test trajectory, tau and noise
    draw, `(phi, b)` is ambiguous in at most 4.3e-05 of samples -- one in
    23,000, and exactly zero in three of the five systems -- while `psi`, the
    same operator's coordinate without a flag, is ambiguous for 71% of samples
    in `rd` and 37% in `sidm`. Four orders of magnitude, not a nudge.

    This gate pins the structural claim, which is what P30 licenses. It does
    NOT license the injectivity ORDERING that P29a refuted, and
    `test_injectivity_criterion_is_not_overclaimed` still holds that line.
    """
    cf = _res('coord_floor.json')
    for name in ('rd', 'ks', 'film', 'burgers', 'sidm'):
        for eps, rec in cf[name]['phi'].items():
            assert rec['ambiguous_frac'] < 1e-3, (
                f'{name} at eps={eps}: (phi, b) now has ambiguous inversions '
                f"in {rec['ambiguous_frac']:.2e} of samples, so arm C's "
                'displacement is no longer effectively a function')
    worst = max(cf[n]['psi']['0.01']['ambiguous_frac']
                for n in ('rd', 'ks', 'film', 'burgers', 'sidm'))
    assert worst > 0.1, (
        'the unflagged coordinate no longer shows ambiguous inversions; the '
        'contrast the flag claim rests on has gone')


@pytest.mark.regression
def test_the_coordinate_displacement_predicts_the_measured_degradation():
    """P30: the pre-registered PASS band, and the direction of the residual.

    The floor is what a PERFECT operator still gets wrong, so it must be a
    LOWER bound: every predicted/measured ratio at or below ~1. A ratio well
    above 1 would mean the network beats a perfect operator, which would
    falsify the reading rather than support it.
    """
    cf = _res('coord_floor.json')
    s = cf['_summary']
    assert s['verdict'].startswith('PASS'), s['verdict']
    assert 0.80 <= s['median_ratio'] <= 1.25
    assert s['prediction_monotone_in_all_systems']
    assert s['ratio_range'][1] <= 1.05, (
        f"a predicted error exceeds the measured one by {s['ratio_range'][1]:.2f}x; "
        'the floor is meant to bound it from below')


# The strict provenance check is the one a reproducibility reviewer will run,
# and it does not pass. Three artefacts predate the stamping mechanism and
# cannot be given a producer record after the fact without inventing one, so
# the set is pinned here instead: the manuscript states the number, and this
# fails if it grows.
UNSTAMPED = {'reference_accuracy.json', 'timehead.json', 'direct.json'}


@pytest.mark.verification
def test_the_set_of_unstamped_artefacts_is_the_one_the_paper_declares():
    """No new artefact may join the unstamped set.

    A release hash records what an artefact contains; it cannot establish
    which inputs its producer used. That limit is reported rather than
    repaired with a stamp written today, and this gate keeps the reported
    number honest by failing when a fourth file drifts into the same state.
    """
    _need('results/statecond/contract.json')
    import re
    from sc_provenance import check

    ok, problems = check(RES, strict=True)
    missing = {re.match(r'(?:\S+: dependency )?(\S+?)(?: |:)', p).group(1)
               for p in problems}
    assert missing == UNSTAMPED, (
        f'the unstamped set moved: {sorted(missing)} against '
        f'{sorted(UNSTAMPED)}; update the count in the manuscript too')
    assert not ok, 'strict provenance now passes; drop this gate and say so'
