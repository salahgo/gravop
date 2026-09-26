"""Gates for the public benchmark API.

The benchmark is released so that the comparison can be rerun and extended, and
`STATECOND_BENCHMARK.md` documents how. That claim is only
true if a third party can actually register their own architecture and their own
equation and have them work. These tests do exactly what the guide tells a
reader to do, so the guide cannot silently stop being correct.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import statecond as sc                                   # noqa: E402


@pytest.mark.verification
def test_public_api_is_exported():
    """The names the guide tells people to import must exist."""
    for name in ('System', 'SYSTEMS', 'BACKBONES', 'register_system',
                 'register_backbone', 'tau_grid', 'split', 'build'):
        assert name in sc.__all__, f'{name} missing from statecond.__all__'
        assert hasattr(sc, name), f'statecond.{name} not importable'
    assert set(sc.BACKBONES) >= {'fno', 'deeponet', 'cno', 'transformer'}
    assert set(sc.SYSTEMS) >= {'rd', 'ks', 'film', 'burgers'}


# ---------------------------------------------------------------- backbones


class _ToyOperator:
    """The minimal architecture the guide describes, built lazily so torch is
    only imported inside the tests that need it."""


def _toy_backbone_cls():
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, **kw):
            super().__init__()
            self.lift = nn.Conv1d(n_fields, 16, 1)
            self.cond = nn.Linear(n_par + cond_extra, 16)
            self.proj = nn.Conv1d(16, n_fields, 1)

        def forward(self, prof, cond):
            h = self.lift(prof) + self.cond(cond)[:, :, None]
            return self.proj(h.tanh())
    return Toy


@pytest.mark.verification
def test_register_backbone_accepts_a_conforming_architecture():
    """This is the guide's example, executed."""
    Toy = _toy_backbone_cls()
    try:
        sc.register_backbone('toy_ok', Toy)
        assert sc.BACKBONES['toy_ok'] is Toy
        # and it works for both conditioning widths the protocol uses
        import torch
        for cond_extra in (1, 2):
            net = Toy(n_fields=2, n_par=3, cond_extra=cond_extra, n_x=64)
            out = net(torch.randn(2, 2, 64), torch.randn(2, 3 + cond_extra))
            assert out.shape == (2, 2, 64)
    finally:
        sc.BACKBONES.pop('toy_ok', None)


@pytest.mark.verification
def test_register_backbone_rejects_a_wrong_signature():
    import torch.nn as nn

    class Bad(nn.Module):
        def __init__(self, **kw):
            super().__init__()

        def forward(self, prof, cond):
            return prof

    with pytest.raises(TypeError, match='n_fields'):
        sc.register_backbone('toy_badsig', Bad)
    assert 'toy_badsig' not in sc.BACKBONES


@pytest.mark.verification
def test_register_backbone_rejects_a_wrong_output_shape():
    import torch.nn as nn

    class Bad(nn.Module):
        def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, **kw):
            super().__init__()
            self.lin = nn.Linear(n_x, n_x // 2)

        def forward(self, prof, cond):
            return self.lin(prof)

    with pytest.raises(ValueError, match='expected'):
        sc.register_backbone('toy_badshape', Bad)
    assert 'toy_badshape' not in sc.BACKBONES


@pytest.mark.verification
def test_register_backbone_rejects_a_detached_output():
    """The gradient check is the point of validating at all: this project has a
    machine where some ops fail only in backward."""
    import torch
    import torch.nn as nn

    class Bad(nn.Module):
        def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, **kw):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, prof, cond):
            return prof.detach()

    with pytest.raises(ValueError, match='BACKWARD|gradient'):
        sc.register_backbone('toy_nograd', Bad)
    assert 'toy_nograd' not in sc.BACKBONES


# ------------------------------------------------------------------ systems


def _toy_system_cls():
    class ToyPDE(sc.System):
        """u_t = u^2 on a periodic grid: blows up in finite time, and the
        blow-up time depends on the initial amplitude, which is the structure
        the benchmark is about."""
        name = 'toypde'
        n_fields = 1
        field_names = ('u',)
        param_names = ('amp',)
        n_solve = 64
        blowup_factor = 1e2
        t_end = 10.0

        def solve_grid(self):
            return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

        def store_grid(self):
            return np.linspace(0.0, 1.0, 32, endpoint=False)

        def sample(self, rng):
            return {'amp': float(rng.uniform(0.5, 1.5))}

        def initial(self, p):
            x = self.solve_grid()
            return p['amp'] * (1.0 + 0.3 * np.sin(2 * np.pi * x))

        def rhs(self, t, y, p):
            return y ** 2

        def observable(self, y):
            return float(np.max(y))

        def store(self, y):
            return np.interp(self.store_grid(), self.solve_grid(),
                             y)[None, :]
    return ToyPDE


@pytest.mark.verification
def test_register_system_accepts_a_conforming_system():
    """The guide's system example, executed end to end through the contract."""
    ToyPDE = _toy_system_cls()
    try:
        sc.register_system(ToyPDE)
        assert sc.SYSTEMS['toypde'] is ToyPDE
        inst = ToyPDE()
        p = inst.sample(np.random.default_rng(0))
        y0 = inst.initial(p)
        assert inst.observable(y0) > 0
        assert inst.store(y0).shape == (1, 32)
        assert np.asarray(inst.rhs(0.0, y0, p)).shape == y0.shape
    finally:
        sc.SYSTEMS.pop('toypde', None)


@pytest.mark.verification
def test_register_system_rejects_a_non_system():
    class NotASystem:
        name = 'nope'

    with pytest.raises(TypeError, match='subclass'):
        sc.register_system(NotASystem)
    assert 'nope' not in sc.SYSTEMS


@pytest.mark.verification
def test_register_system_rejects_a_non_positive_observable():
    """`phi` is log10(A(t)/A(0)), so a non-positive observable produces a
    conditioning variable that silently means nothing. It must not register."""
    ToyPDE = _toy_system_cls()

    class SignError(ToyPDE):
        name = 'toy_signerr'

        def observable(self, y):
            return -float(np.max(y))          # the direction mistake

    with pytest.raises(ValueError, match='positive'):
        sc.register_system(SignError)
    assert 'toy_signerr' not in sc.SYSTEMS


@pytest.mark.verification
def test_register_system_rejects_a_bad_store_shape():
    ToyPDE = _toy_system_cls()

    class BadStore(ToyPDE):
        name = 'toy_badstore'

        def store(self, y):
            return np.interp(self.store_grid(), self.solve_grid(), y)

    with pytest.raises(ValueError, match='n_fields'):
        sc.register_system(BadStore)
    assert 'toy_badstore' not in sc.SYSTEMS


# ---------------------------------------------------------------- reference


@pytest.mark.regression
def test_reference_results_are_complete_and_self_describing():
    """`reference.json` is what a third party compares against, so it must
    carry the protocol with the numbers rather than the numbers alone."""
    import json
    p = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond',
                     'reference.json')
    if not os.path.exists(p):
        pytest.skip('run scripts/sc_reference.py')
    ref = json.load(open(p))
    assert {'protocol', 'systems', 'law', 'points'} <= set(ref)
    assert {'metric', 'arms', 'seeds', 'reproduce'} <= set(ref['protocol'])
    assert len(ref['points']) == 20
    assert len({q['backbone'] for q in ref['points']}) == 4
    for q in ref['points']:
        assert q['deployment_penalty_B_over_A'] == pytest.approx(
            q['B_tau_deployed'] / q['A_tau_oracle'], rel=1e-6)
        assert q['state_vs_deployed_B_over_C'] == pytest.approx(
            q['B_tau_deployed'] / q['C_state'], rel=1e-6)
    assert ref['law']['n_points'] == len(ref['points'])


@pytest.mark.verification
def test_register_system_rejects_an_array_from_sample():
    """Regression, and the reason the validator was strengthened.

    `data.build` assembles the parameter matrix as `[p[k] for k in
    param_names]`, so `sample` must return a mapping. Returning an array passed
    every earlier check and then failed inside a `ProcessPoolExecutor` worker
    with "cannot convert dictionary update sequence element #0 to a sequence" —
    an error naming neither the system nor the cause. The guide documented the
    wrong contract for exactly as long as the validator failed to enforce it.
    """
    ToyPDE = _toy_system_cls()

    class ArraySample(ToyPDE):
        name = 'toy_arraysample'

        def sample(self, rng):
            return np.array([rng.uniform(0.5, 1.5)])

        def initial(self, p):
            return p[0] * (1.0 + 0.3 * np.sin(2 * np.pi * self.solve_grid()))

    with pytest.raises(TypeError, match='dict keyed by param_names'):
        sc.register_system(ArraySample)
    assert 'toy_arraysample' not in sc.SYSTEMS


@pytest.mark.verification
def test_register_system_rejects_mismatched_param_names():
    ToyPDE = _toy_system_cls()

    class WrongKeys(ToyPDE):
        name = 'toy_wrongkeys'

        def sample(self, rng):
            return {'amplitude': float(rng.uniform(0.5, 1.5))}

        def initial(self, p):
            return p['amplitude'] * np.ones(self.n_solve)

    with pytest.raises(ValueError, match='param_names'):
        sc.register_system(WrongKeys)
    assert 'toy_wrongkeys' not in sc.SYSTEMS


@pytest.mark.verification
def test_plugin_loading_makes_registrations_visible():
    """The guide tells a reader to register in their own module and then run
    `sc_experiment.py`. Registration mutates a dict in the *calling* process, so
    without a plugin hook the runner rejects the name at argument parsing —
    which is exactly what happened when this workflow was first tried.
    """
    import tempfile
    src = '''
import numpy as np, torch.nn as nn
from statecond import System, register_backbone, register_system

class P(nn.Module):
    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, **kw):
        super().__init__()
        self.c = nn.Conv1d(n_fields, n_fields, 1)
        self.k = nn.Linear(n_par + cond_extra, n_fields)
    def forward(self, prof, cond):
        return self.c(prof) + self.k(cond)[:, :, None]
register_backbone('plugin_bk', P)

class S(System):
    name = 'plugin_sys'
    n_fields, field_names, param_names = 1, ('u',), ('amp',)
    n_solve, blowup_factor, t_end = 32, 1e2, 10.0
    def solve_grid(self): return np.linspace(0, 1, self.n_solve, endpoint=False)
    def store_grid(self): return np.linspace(0, 1, 16, endpoint=False)
    def sample(self, rng): return {'amp': float(rng.uniform(0.5, 1.5))}
    def initial(self, p): return p['amp'] * np.ones(self.n_solve)
    def rhs(self, t, y, p): return y ** 2
    def observable(self, y): return float(np.max(y))
    def store(self, y): return np.interp(self.store_grid(), self.solve_grid(), y)[None, :]
register_system(S)
'''
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'plug.py')
        open(path, 'w').write(src)
        try:
            assert sc.plugin_args_from_argv(['x', '--plugin', path]) == [path]
            assert sc.plugin_args_from_argv([f'--plugin={path}']) == [path]
            sc.load_plugins([path])
            assert 'plugin_bk' in sc.BACKBONES
            assert 'plugin_sys' in sc.SYSTEMS
        finally:
            sc.BACKBONES.pop('plugin_bk', None)
            sc.SYSTEMS.pop('plugin_sys', None)
