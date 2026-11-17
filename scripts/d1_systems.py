"""Track B (disjoint-system transfer): six new systems, registered as a plugin.

Protocol: docs/DISJOINT_PREREGISTRATION.md. Loaded with `--plugin scripts/d1_systems.py`
by `sc_gen.py` and `sc_experiment.py`; nothing in `src/statecond` is modified (those files
are pinned by F9).

Tier 1, mechanism-disjoint (expected to hold):
  quench   u_t = u_xx + lam (1-u)^-beta, Dirichlet: the source becomes singular while u stays
           bounded (A = 1/(1 - max u)); first non-periodic member.
  pme      u_t = (u^m)_xx + u^p, periodic, compactly supported data: degenerate diffusion,
           finite propagation speed, free boundary (A = max u).
  cgl      A_t = (1 + i alpha) A_xx + (1 + i beta) |A|^2 A, periodic: two coupled fields with a
           phase (stored as asinh Re A, asinh Im A); A = max |A|.
  hj       u_t = u_xx + |u_x|^q + lam, Dirichlet: gradient blow-up at the boundary while u
           stays bounded (A = max wall |u_x|).
Tier 2, adversarial (pre-registered to fail in a stated direction):
  corr     rd plus a drift a(p) u_x whose speed is tied to the amplitude that sets t_sing, so
           the clock displacement and the operator error become co-directed.
  bimodal  film with a domain-length box straddling the critical length 2 pi, giving a
           heavy-tailed t_sing (runs that do not rupture by t_end are dropped, as in the suite).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from statecond import register_system  # noqa: E402
from statecond.pdes import N_X, RD, Film, System, _banded_periodic  # noqa: E402


def _banded(n, half):
    s = np.zeros((n, n), dtype=np.int8)
    for d in range(-half, half + 1):
        i = np.arange(max(0, -d), min(n, n - d))
        s[i, i + d] = 1
    return s


class Quench(System):
    name = 'quench'
    field_names = ('u',)
    param_names = ('log_lam', 'beta', 'a', 'skew')
    n_solve = 512
    blowup_factor = 1e2
    t_end = 20.0
    rtol, atol = 1e-8, 1e-11

    def solve_grid(self):            # interior nodes of [0, 1], u = 0 at both walls
        return (np.arange(self.n_solve) + 1.0) / (self.n_solve + 1)

    def store_grid(self):
        return (np.arange(N_X) + 0.5) / N_X

    def sample(self, rng):
        return {'log_lam': float(rng.uniform(0.6, 1.3)), 'beta': float(rng.uniform(0.5, 2.0)),
                'a': float(rng.uniform(0.0, 0.5)), 'skew': float(rng.uniform(-0.3, 0.3))}

    def initial(self, p):
        x = self.solve_grid()
        return p['a'] * np.sin(np.pi * x) * (1 + p['skew'] * np.sin(2 * np.pi * x)) / (1 + abs(p['skew']))

    def rhs(self, t, y, p):
        dx = 1.0 / (self.n_solve + 1)
        yp = np.concatenate([[0.0], y, [0.0]])
        lap = (yp[2:] - 2 * y + yp[:-2]) / dx ** 2
        gap = np.maximum(1.0 - y, 1e-12)
        return lap + 10.0 ** p['log_lam'] * gap ** -p['beta']

    def jac_sparsity(self):
        return _banded(self.n_solve, 1)

    def observable(self, y):
        return float(1.0 / max(1.0 - np.max(y), 1e-12))

    def store(self, y):
        x = np.concatenate([[0.0], self.solve_grid(), [1.0]])
        return np.interp(self.store_grid(), x, np.concatenate([[0.0], y, [0.0]]))[None, :]


class PME(System):
    name = 'pme'
    field_names = ('log10_1pu',)
    param_names = ('m', 'dp', 'log_a', 'w')
    n_solve = 512
    blowup_factor = 1e2
    t_end = 50.0
    rtol, atol = 1e-7, 1e-10

    def solve_grid(self):
        return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

    def store_grid(self):
        return np.linspace(0.0, 1.0, N_X, endpoint=False)

    def sample(self, rng):
        return {'m': float(rng.uniform(1.5, 3.0)), 'dp': float(rng.uniform(0.5, 1.5)),
                'log_a': float(rng.uniform(0.0, 0.5)), 'w': float(rng.uniform(0.12, 0.30))}

    def initial(self, p):
        z = (self.solve_grid() - 0.5) / p['w']
        return 10.0 ** p['log_a'] * np.where(np.abs(z) < 1, np.cos(0.5 * np.pi * z) ** 2, 0.0)

    def rhs(self, t, y, p):
        dx = 1.0 / self.n_solve
        um = np.maximum(y, 0.0) ** p['m']
        return 0.01 * (np.roll(um, -1) - 2 * um + np.roll(um, 1)) / dx ** 2 + np.maximum(y, 0.0) ** (p['m'] + p['dp'])

    def jac_sparsity(self):
        return _banded_periodic(self.n_solve, 1)

    def observable(self, y):
        return float(np.max(y))

    def store(self, y):
        s = self.n_solve // N_X
        return np.log10(1.0 + np.maximum(y, 0.0).reshape(N_X, s).mean(1))[None, :]


class CGL(System):
    name = 'cgl'
    n_fields = 2
    field_names = ('asinh_re', 'asinh_im')
    param_names = ('log_a', 'alpha', 'beta', 'k')
    n_solve = 256
    blowup_factor = 1e2
    t_end = 20.0
    rtol, atol = 1e-8, 1e-10

    def solve_grid(self):
        return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

    def store_grid(self):
        return np.linspace(0.0, 1.0, N_X, endpoint=False)

    def sample(self, rng):
        return {'log_a': float(rng.uniform(-0.1, 0.3)), 'alpha': float(rng.uniform(-0.5, 0.5)),
                'beta': float(rng.uniform(-0.5, 0.5)), 'k': float(rng.uniform(0.5, 2.5))}

    def initial(self, p):
        x = self.solve_grid()
        A = 10.0 ** p['log_a'] * np.exp(p['k'] * (np.cos(2 * np.pi * x) - 1)) * np.exp(1j * 0.5 * np.sin(2 * np.pi * x))
        return np.concatenate([A.real, A.imag])

    def rhs(self, t, y, p):
        n = self.n_solve; dx = 1.0 / n; D = 1e-2
        A = y[:n] + 1j * y[n:]
        lap = (np.roll(A, -1) - 2 * A + np.roll(A, 1)) / dx ** 2
        dA = D * (1 + 1j * p['alpha']) * lap + (1 + 1j * p['beta']) * np.abs(A) ** 2 * A
        return np.concatenate([dA.real, dA.imag])

    def jac_sparsity(self):
        b = _banded_periodic(self.n_solve, 1)
        return np.block([[b, b], [b, b]])

    def observable(self, y):
        n = self.n_solve
        return float(np.max(np.hypot(y[:n], y[n:])))

    def store(self, y):
        n = self.n_solve; s = n // N_X
        return np.stack([np.arcsinh(y[:n].reshape(N_X, s).mean(1)), np.arcsinh(y[n:].reshape(N_X, s).mean(1))])


class HJ(System):
    """Forced viscous Hamilton-Jacobi equation u_t = u_xx + |u_x|^q + lam, u = 0 at both walls.
    Above a critical forcing no steady state exists and |u_x| blows up at the wall while u stays
    bounded. (The unforced form of the plan, lam = 0, either decays or blows up within
    1e-4..1e-2 of the diffusion time in a prototype scan, so the forcing sets the clock.)"""
    name = 'hj'
    field_names = ('u',)
    param_names = ('log_lam', 'q', 'log_a', 'skew')
    n_solve = 768
    blowup_factor = 1e2
    t_end = 5.0
    rtol, atol = 1e-8, 1e-11

    def __init__(self):
        # nodes clustered at both walls, where the gradient blows up
        s = np.linspace(0.0, 1.0, self.n_solve + 2)
        self._x = 0.5 * (1 - np.cos(np.pi * s))

    def solve_grid(self):
        return self._x[1:-1]

    def store_grid(self):
        return (np.arange(N_X) + 0.5) / N_X

    def sample(self, rng):
        return {'log_lam': float(rng.uniform(0.8, 1.4)), 'q': float(rng.uniform(2.5, 3.5)),
                'log_a': float(rng.uniform(-1.0, -0.4)), 'skew': float(rng.uniform(-0.3, 0.3))}

    def initial(self, p):
        x = self.solve_grid()
        return 10.0 ** p['log_a'] * np.sin(np.pi * x) * (1 + p['skew'] * np.sin(2 * np.pi * x)) / (1 + abs(p['skew']))

    def _grad(self, y):
        x = self._x; u = np.concatenate([[0.0], y, [0.0]]); h = np.diff(x)
        hl, hr = h[:-1], h[1:]
        ux = (hl ** 2 * u[2:] - hr ** 2 * u[:-2] + (hr ** 2 - hl ** 2) * u[1:-1]) / (hl * hr * (hl + hr))
        uxx = 2 * (hl * u[2:] - (hl + hr) * u[1:-1] + hr * u[:-2]) / (hl * hr * (hl + hr))
        return u, h, ux, uxx

    def rhs(self, t, y, p):
        _, _, ux, uxx = self._grad(y)
        return uxx + np.abs(ux) ** p['q'] + 10.0 ** p['log_lam']

    def jac_sparsity(self):
        return _banded(self.n_solve, 1)

    def observable(self, y):
        u, h, _, _ = self._grad(y)
        # one-sided second-order wall gradients
        g0 = (-(2 * h[0] + h[1]) / (h[0] * (h[0] + h[1])) * u[0] + (h[0] + h[1]) / (h[0] * h[1]) * u[1] - h[0] / (h[1] * (h[0] + h[1])) * u[2])
        g1 = ((2 * h[-1] + h[-2]) / (h[-1] * (h[-1] + h[-2])) * u[-1] - (h[-1] + h[-2]) / (h[-1] * h[-2]) * u[-2] + h[-1] / (h[-2] * (h[-1] + h[-2])) * u[-3])
        return float(max(abs(g0), abs(g1)))

    def store(self, y):
        return np.interp(self.store_grid(), self._x, np.concatenate([[0.0], y, [0.0]]))[None, :]


class Corr(RD):
    """rd with a drift whose speed is set by the amplitude log_a, the parameter that sets
    t_sing. Speed v(log_a) = 3 + 6 (log_a + 0.3) / 0.9 domain lengths per unit time, so a run
    that blows up later also travels further; the field is translated by v t."""
    name = 'corr'
    param_names = ('log_a', 'p', 'log_D', 'k')

    def rhs(self, t, y, p):
        dx = 1.0 / self.n_solve
        v = 3.0 + 6.0 * (p['log_a'] + 0.3) / 0.9
        adv = -v * (np.roll(y, -1) - np.roll(y, 1)) / (2 * dx)
        return RD.rhs(self, t, y, p) + adv

    def jac_sparsity(self):
        return _banded_periodic(self.n_solve, 1)


class Bimodal(Film):
    name = 'bimodal'

    def sample(self, rng):
        return {'L': float(rng.uniform(5.6, 7.0)), 'a1': float(rng.uniform(0.05, 0.30)),
                'a2': float(rng.uniform(-0.15, 0.15)), 'phi': float(rng.uniform(0.0, 2 * np.pi))}


TIER1 = ('quench', 'pme', 'cgl', 'hj')
TIER2 = ('corr', 'bimodal')
for _cls in (Quench, PME, CGL, HJ, Corr, Bimodal):
    register_system(_cls)
