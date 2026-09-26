"""statecond — a benchmark for surrogates of PDEs with solution-dependent
finite-time singularities.

The problem. A PDE develops a singularity at a time `t_sing` that depends on the
initial data and parameters. A surrogate is naturally trained in normalised time
`tau = t / t_sing`, but at deployment `t_sing` is unknown and must be predicted,
and near the singularity a small error in it is amplified into a large field
error. Conditioning instead on an observable coordinate of the solution's
*state* removes the dependence on `t_sing` entirely.

This package isolates that phenomenon from any particular physics so the same
measurement can be repeated across domains and architectures.

  `pdes`   four 1-D reference systems with finite-time singularities
  `data`   uniform trajectory -> (fields, conditioning) pipeline
  `model`  four operator architectures behind one interface

WHAT IT IS FOR. Two extension points are supported and tested, so that a third
party can put their own architecture or their own equation through the identical
protocol and compare against the reference results in `results/statecond/`:

    from statecond import register_backbone, register_system

See `STATECOND_BENCHMARK.md` for the guide, the contracts both must satisfy,
and the reference numbers to compare against.

Quick start:

    python scripts/sc_gen.py --all --n 600          # generate the systems
    python scripts/sc_experiment.py --all           # the three-arm comparison
    python scripts/sc_table.py                      # results as markdown
    python scripts/sc_law.py                        # the sensitivity law
"""
import inspect

from . import data, model, pdes
from .data import (build, load, phase_variables, resample, save, split,
                   tau_grid)
from .model import BACKBONES, OperatorNet, PhaseHead, TimeHead
from .pdes import SYSTEMS, System, Trajectory

__all__ = [
    'data', 'model', 'pdes',
    'System', 'SYSTEMS', 'Trajectory', 'register_system',
    'BACKBONES', 'OperatorNet', 'TimeHead', 'PhaseHead', 'register_backbone',
    'build', 'load', 'save', 'split', 'tau_grid', 'resample',
    'phase_variables', 'load_plugins',
]


def load_plugins(specs):
    """Import modules that register systems or backbones, before use.

    Registration mutates a dict in this process, so a class registered in a
    user's own session is invisible to `sc_experiment.py`, which runs as a
    separate process. Without a hook the documented extension path silently
    does not work — the runner rejects the name at argument parsing.

    Each spec is either an importable dotted module name or a path to a `.py`
    file. The scripts scan `sys.argv` for `--plugin` and call this *before*
    building their argument parser, so newly registered names are already valid
    choices.

    Returns the list of imported modules.
    """
    import importlib
    import importlib.util
    import os

    mods = []
    for spec in specs or ():
        if spec.endswith('.py') or os.path.sep in spec:
            path = os.path.abspath(spec)
            name = os.path.splitext(os.path.basename(path))[0]
            s = importlib.util.spec_from_file_location(name, path)
            if s is None or s.loader is None:
                raise ImportError(f'cannot load plugin from {spec!r}')
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
        else:
            mod = importlib.import_module(spec)
        mods.append(mod)
    return mods


def plugin_args_from_argv(argv=None):
    """Pull `--plugin X` / `--plugin=X` out of an argv list.

    Used by the scripts before argparse runs. Returns the specs; the caller
    passes them to `load_plugins`.
    """
    import sys as _sys
    argv = list(_sys.argv if argv is None else argv)
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == '--plugin' and i + 1 < len(argv):
            out.append(argv[i + 1])
            i += 2
        elif a.startswith('--plugin='):
            out.append(a.split('=', 1)[1])
            i += 1
        else:
            i += 1
    return out

# the contract a backbone must satisfy: __init__ takes these, forward takes
# (prof, cond) and returns fields shaped like prof
_BACKBONE_ARGS = ('n_fields', 'n_par', 'cond_extra', 'n_x')


def register_backbone(name, cls, validate=True, device='cpu'):
    """Add an operator architecture to `BACKBONES` under `name`.

    The architecture must accept `(n_fields, n_par, cond_extra, n_x)` as
    keyword arguments and implement `forward(prof, cond)` mapping
    `(B, n_fields, n_x)` and `(B, n_par + cond_extra)` to `(B, n_fields, n_x)`.

    With `validate=True` the contract is *executed*, not merely inspected, and
    the check includes a **backward pass**. That is deliberate: this project
    has a machine on which Triton cannot JIT, so an op whose backward routes to
    Triton fails at runtime while every forward-only smoke test passes. It cost
    a full training run to find once. Both `n_fields = 1` and `2` are exercised,
    because the failure was specific to the singleton case.

    Returns the class, so it can be used as a decorator.
    """
    if validate:
        _validate_backbone(name, cls, device)
    BACKBONES[name] = cls
    return cls


def _validate_backbone(name, cls, device='cpu'):
    import torch

    sig = inspect.signature(cls.__init__)
    missing = [a for a in _BACKBONE_ARGS if a not in sig.parameters]
    if missing:
        raise TypeError(
            f'backbone {name!r} must accept {missing} as keyword arguments; '
            f'its __init__ signature is {sig}')

    for n_fields in (1, 2):
        net = cls(n_fields=n_fields, n_par=3, cond_extra=2, n_x=64).to(device)
        prof = torch.randn(4, n_fields, 64, device=device)
        cond = torch.randn(4, 5, device=device)
        out = net(prof, cond)
        if tuple(out.shape) != (4, n_fields, 64):
            raise ValueError(
                f'backbone {name!r} returned {tuple(out.shape)}, expected '
                f'{(4, n_fields, 64)} for n_fields={n_fields}')
        try:
            out.pow(2).mean().backward()
        except Exception as exc:                       # noqa: BLE001
            raise ValueError(
                f'backbone {name!r} failed in the BACKWARD pass with '
                f'{type(exc).__name__}: {exc}. If this is a gcc '
                f'CalledProcessError, the op has been routed to a Triton '
                f'kernel that cannot be compiled here — express it with '
                f'elementwise operations and a reduction, or reshape to avoid '
                f'a singleton dimension.') from exc
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads:
            raise ValueError(
                f'backbone {name!r} produced no gradients: its output does '
                f'not depend on its parameters')
        if not all(torch.isfinite(g).all() for g in grads):
            raise ValueError(
                f'backbone {name!r} produced non-finite gradients')
    return True


def register_system(cls, name=None, validate=True):
    """Add a PDE system to `SYSTEMS`.

    `cls` must subclass `System` and implement the contract documented there:
    `solve_grid`, `store_grid`, `sample`, `initial`, `rhs`, `observable` and
    `store`. `observable` must be positive and diverge at the singularity —
    the state coordinate is its logarithmic contrast, so a sign error there
    produces a conditioning variable that silently means nothing.

    Returns the class, so it can be used as a decorator.
    """
    if not (inspect.isclass(cls) and issubclass(cls, System)):
        raise TypeError(f'{cls!r} must be a subclass of statecond.System')
    key = name or getattr(cls, 'name', None)
    if not key or key == 'base':
        raise ValueError(
            'give the system a `name` class attribute, or pass name=')
    if validate:
        _validate_system(cls, key)
    SYSTEMS[key] = cls
    return cls


def _validate_system(cls, key):
    import numpy as np

    inst = cls()
    for meth in ('solve_grid', 'store_grid', 'sample', 'initial', 'rhs',
                 'observable', 'store'):
        if getattr(type(inst), meth) is getattr(System, meth, None):
            raise NotImplementedError(
                f'system {key!r} does not implement {meth}()')
    p = inst.sample(np.random.default_rng(0))
    # `sample` must return a MAPPING keyed by `param_names`: `data.build`
    # assembles the parameter matrix as [p[k] for k in param_names], so an
    # array here fails later, inside a worker process, with an error that names
    # neither the system nor the cause.
    if not hasattr(p, 'keys'):
        raise TypeError(
            f'system {key!r}: sample() must return a dict keyed by '
            f'param_names {tuple(inst.param_names)}, got {type(p).__name__}')
    missing = [k for k in inst.param_names if k not in p]
    extra = [k for k in p if k not in inst.param_names]
    if missing or extra:
        raise ValueError(
            f'system {key!r}: sample() keys do not match param_names; '
            f'missing {missing}, unexpected {extra}')
    y0 = np.asarray(inst.initial(p))
    a0 = float(inst.observable(y0))
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError(
            f'system {key!r}: observable(initial) must be finite and positive '
            f'(got {a0}); it is used as log10(A(t)/A(0))')
    stored = np.asarray(inst.store(y0))
    if stored.ndim != 2 or stored.shape[0] != inst.n_fields:
        raise ValueError(
            f'system {key!r}: store() must return (n_fields, N_X); got '
            f'{stored.shape} with n_fields={inst.n_fields}')
    dy = np.asarray(inst.rhs(0.0, y0, p))
    if dy.shape != y0.shape:
        raise ValueError(
            f'system {key!r}: rhs() returned {dy.shape}, expected {y0.shape}')
    return True
