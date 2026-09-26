"""statecond.pdes — PDE systems with finite-time singularities, one spatial
coordinate each (`ks` is 2-D reduced to radial symmetry, not a 1-D model).

Four systems from unrelated fields, each with a solution-dependent singular
time t_sing and a scalar observable A(t) that diverges as t -> t_sing:

  rd    reaction-diffusion blow-up   u_t = D u_xx + u^p            (combustion)
  ks    Keller-Segel aggregation     rho_t = div(grad rho - chi rho grad c),
                                     -Lap c = rho, 2-D radial      (biology)
  film  thin-film rupture            h_t = -(h^3 h_xxx)_x - (h^-1 h_x)_x
                                                                   (interfaces)
  burg  Burgers shock formation      u_t + u u_x = nu u_xx        (gas dynamics)

They differ in order (2nd local, 2nd nonlocal, 4th, 2nd advective), in the
mechanism (thermal runaway, nonlocal attraction, van der Waals touchdown,
nonlinear steepening), in whether the singular quantity grows or vanishes, and
in whether the singularity is genuine (rd, ks, film) or regularised (burgers).
What they share is the structure this package studies: t_sing is not known in
advance, it depends on the initial data and the parameters, and it must be
predicted before a tau-conditioned surrogate can be queried at a physical time.

Design, following the gravothermal solver in `gravop`:

* The integration grid (`n_solve`) is finer than the storage grid (N_X = 128).
  A collapsing solution concentrates, so the solver must resolve structure the
  surrogate is never asked to represent. `store()` maps the solver state onto
  the storage grid, which is log-stretched toward the singular point for the
  system whose singularity is a point concentration (ks) and uniform for the
  systems whose singular region stays O(domain) (rd, film, burgers).
* t_sing is defined post hoc by interpolating A(t) to `blowup_factor * A(0)`,
  exactly as `gravop.dataset.t_collapse` interpolates to rho_c = 10^3 rho_c(0).
  Integration is allowed to fail past that point -- a vanishing step size is
  the expected signature of a genuine blow-up, not an error -- and everything
  after the crossing is discarded.

Interface:

    sys = RD()                       # or KS(), Film(), Burgers()
    p   = sys.sample(rng)            # dict of physical parameters
    tr  = sys.solve(p)               # Trajectory, or None if unusable

    Trajectory.u       (n_snap, n_fields, N_X)  stored fields
    Trajectory.a       (n_snap,)                singular observable, A -> inf
    Trajectory.status  0 singular termination, 1 no singularity by t_end
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.integrate import solve_ivp

N_X = 128                # storage resolution, matches gravop.dataset.N_R


@dataclass
class Trajectory:
    t: np.ndarray
    u: np.ndarray
    a: np.ndarray
    x: np.ndarray
    t_sing: float
    status: int
    params: dict = field(default_factory=dict)
    n_steps: int = 0
    wall: float = 0.0


class System:
    """Base class. Subclasses define sample/initial/rhs/observable/store."""
    name = 'base'
    n_fields = 1
    field_names = ('u',)
    param_names = ()
    n_solve = 512
    blowup_factor = 1e3
    t_end = 1e3
    method = 'BDF'
    rtol, atol = 1e-6, 1e-9

    # -- geometry ----------------------------------------------------------
    def solve_grid(self):
        raise NotImplementedError

    def store_grid(self):
        raise NotImplementedError

    # -- model -------------------------------------------------------------
    def sample(self, rng):
        raise NotImplementedError

    def initial(self, p):
        raise NotImplementedError

    def rhs(self, t, y, p):
        raise NotImplementedError

    def observable(self, y):
        """Scalar A(y) > 0, defined so that A -> +inf at the singularity."""
        raise NotImplementedError

    def store(self, y):
        """Map the solver state to stored fields, shape (n_fields, N_X)."""
        raise NotImplementedError

    def jac_sparsity(self):
        return None

    # -- driver ------------------------------------------------------------
    def solve(self, p):
        y0 = self.initial(p)
        a0 = self.observable(y0)
        thresh = a0 * self.blowup_factor

        def event(t, y, *_):
            return self.observable(y) - thresh
        event.terminal = True
        event.direction = 1.0

        t0 = time.time()
        try:
            sol = solve_ivp(self.rhs, (0.0, self.t_end), y0, args=(p,),
                            method=self.method, rtol=self.rtol, atol=self.atol,
                            events=event, jac_sparsity=self.jac_sparsity())
        except Exception:
            return None
        wall = time.time() - t0
        if sol.t.size < 12:
            return None
        return self._finish(sol.t, sol.y.T, a0, thresh, p, wall, sol.t.size)

    def _finish(self, ts, ys, a0, thresh, p, wall, n_steps):
        """Truncate at the threshold crossing and package a Trajectory."""
        a = np.array([self.observable(y) for y in ys])
        good = np.isfinite(a) & (a > 0)
        if good.sum() < 12 or not good[0]:
            return None
        cut = int(np.argmin(good)) if not good.all() else len(a)
        ts, ys, a = ts[:cut], ys[:cut], a[:cut]

        # the terminal event lands on the threshold to within root-finding
        # tolerance, so compare against a slightly relaxed value
        reached = np.nonzero(a >= thresh * (1.0 - 1e-6))[0]
        if reached.size:
            j = int(reached[0])
            if j == 0:
                return None
            l0, l1 = np.log(a[j - 1]), np.log(a[j])
            frac = (np.log(thresh) - l0) / max(l1 - l0, 1e-300)
            t_sing = float(ts[j - 1] + frac * (ts[j] - ts[j - 1]))
            ts, ys, a = ts[:j + 1], ys[:j + 1], a[:j + 1]
            status = 0
        else:
            t_sing, status = float(ts[-1]), 1

        u = np.stack([self.store(y) for y in ys]).astype(np.float32)
        if not np.all(np.isfinite(u)) or ts.size < 12:
            return None
        return Trajectory(t=np.asarray(ts, float), u=u, a=a,
                          x=self.store_grid(), t_sing=t_sing, status=status,
                          params=dict(p), n_steps=n_steps, wall=wall)


# ---------------------------------------------------------------------------
# 1. reaction-diffusion blow-up:  u_t = D u_xx + u^p, periodic, u > 0
#    Positive periodic data stays positive, so log u is defined everywhere and
#    the solution blows up at the initial maximum in finite time for p > 1.
#    Small k gives near-homogeneous (ODE-like) blow-up, large k a localised
#    peak, so the parameter box spans both regimes.

class RD(System):
    name = 'rd'
    field_names = ('log10_u',)
    param_names = ('log_a', 'p', 'log_D', 'k')
    n_solve = 512
    blowup_factor = 1e3
    t_end = 50.0

    def solve_grid(self):
        return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

    def store_grid(self):
        return np.linspace(0.0, 1.0, N_X, endpoint=False)

    def sample(self, rng):
        return {'log_a': float(rng.uniform(-0.3, 0.6)),
                'p': float(rng.uniform(2.0, 4.0)),
                'log_D': float(rng.uniform(-1.5, -0.5)),
                'k': float(rng.uniform(0.5, 3.0))}

    def initial(self, p):
        x = self.solve_grid()
        return 10.0 ** p['log_a'] * np.exp(p['k'] * (np.cos(2 * np.pi * x) - 1))

    def rhs(self, t, y, p):
        dx = 1.0 / self.n_solve
        lap = (np.roll(y, -1) - 2.0 * y + np.roll(y, 1)) / dx ** 2
        return 10.0 ** p['log_D'] * lap + np.abs(y) ** p['p']

    def jac_sparsity(self):
        return _banded_periodic(self.n_solve, 1)

    def observable(self, y):
        return float(np.max(y))

    def store(self, y):
        s = self.n_solve // N_X
        return np.log10(np.maximum(y.reshape(N_X, s).mean(1), 1e-12))[None, :]


# ---------------------------------------------------------------------------
# 2. Keller-Segel aggregation, 2-D radial parabolic-elliptic:
#       rho_t = (1/r) d_r [ r rho_r + chi rho M(r) / (2 pi) ],
#       M(r)  = int_0^r 2 pi s rho(s) ds
#    Finite-time concentration at the origin above the critical mass 8 pi / chi;
#    below it the profile relaxes. Finite volume on a uniform radial grid with
#    no-flux at r = R; stored on a log-radial grid, as for the SIDM haloes.

class KS(System):
    name = 'ks'
    field_names = ('log10_rho',)
    param_names = ('log_mass_ratio', 'w', 'shape')
    n_solve = 384
    blowup_factor = 1e2
    t_end = 20.0
    rtol, atol = 1e-6, 1e-10

    R = 1.0
    chi = 1.0
    log_r_min = -2.0

    def solve_grid(self):
        dr = self.R / self.n_solve
        return (np.arange(self.n_solve) + 0.5) * dr

    def store_grid(self):
        return np.logspace(self.log_r_min, np.log10(self.R), N_X)

    def sample(self, rng):
        # mass is set relative to the critical mass 8 pi / chi; widths are kept
        # broad so that the profile first relaxes and only then concentrates,
        # which is what makes the observable non-monotone
        return {'log_mass_ratio': float(rng.uniform(0.02, 0.65)),
                'w': float(rng.uniform(0.15, 0.45)),
                'shape': float(rng.uniform(1.0, 3.0))}

    def initial(self, p):
        r = self.solve_grid()
        # a 2% pedestal keeps the Gaussian tail from underflowing: without it
        # the stored log density spans twelve decades of numerically irrelevant
        # tail and the field normalisation is dominated by it
        prof = np.exp(-(r / p['w']) ** p['shape']) + 0.02
        dr = self.R / self.n_solve
        mass = np.sum(2 * np.pi * r * prof * dr)
        target = (8 * np.pi / self.chi) * 10.0 ** p['log_mass_ratio']
        return prof * (target / mass)

    def rhs(self, t, y, p):
        r = self.solve_grid()
        dr = self.R / self.n_solve
        rho = np.maximum(y, 0.0)
        M = np.cumsum(2 * np.pi * r * rho * dr)       # mass inside right face
        rf = r + 0.5 * dr
        rho_r = (rho[1:] - rho[:-1]) / dr
        rho_f = 0.5 * (rho[1:] + rho[:-1])
        F = -(rho_r + self.chi * rho_f * M[:-1] / (2 * np.pi * rf[:-1]))
        rF = np.zeros(self.n_solve + 1)
        rF[1:-1] = rf[:-1] * F                        # no flux at r = 0 and R
        return -(rF[1:] - rF[:-1]) / (r * dr)

    def jac_sparsity(self):
        # the drift term is nonlocal inward-looking: cell i couples to all j<=i
        return np.tril(np.ones((self.n_solve, self.n_solve), dtype=np.int8), 1)

    def observable(self, y):
        return float(np.max(y))

    def store(self, y):
        lr = np.log10(self.solve_grid())
        val = np.log10(np.maximum(y, 1e-8))
        return np.interp(np.log10(self.store_grid()), lr, val)[None, :]


# ---------------------------------------------------------------------------
# 3. thin-film rupture with van der Waals attraction:
#       h_t = -(h^3 h_xxx)_x - (h^-1 h_x)_x,  periodic
#    The film touches down (min h -> 0) in finite time. The singular observable
#    is 1 / min h so that it diverges like the others; the touchdown region
#    stays O(domain) wide, so a uniform storage grid resolves it.

class Film(System):
    name = 'film'
    field_names = ('log10_h',)
    param_names = ('L', 'a1', 'a2', 'phi')
    n_solve = 256
    blowup_factor = 1e2
    t_end = 500.0
    rtol, atol = 1e-8, 1e-10

    def solve_grid(self):
        return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

    def store_grid(self):
        return np.linspace(0.0, 1.0, N_X, endpoint=False)

    def sample(self, rng):
        # the flat film is linearly unstable only for domains longer than the
        # critical wavelength 2 pi, so the box starts above it
        return {'L': float(rng.uniform(7.0, 20.0)),
                'a1': float(rng.uniform(0.10, 0.40)),
                'a2': float(rng.uniform(-0.15, 0.15)),
                'phi': float(rng.uniform(0.0, 2 * np.pi))}

    def initial(self, p):
        s = self.solve_grid()
        return (1.0 + p['a1'] * np.cos(2 * np.pi * s)
                + p['a2'] * np.cos(4 * np.pi * s + p['phi']))

    def rhs(self, t, y, p):
        dx = p['L'] / self.n_solve
        h = np.maximum(y, 1e-8)
        hx = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)
        hxx = (np.roll(h, -1) - 2 * h + np.roll(h, 1)) / dx ** 2
        hxxx = (np.roll(hxx, -1) - np.roll(hxx, 1)) / (2 * dx)
        q = h ** 3 * hxxx + hx / h
        return -(np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)

    def jac_sparsity(self):
        return _banded_periodic(self.n_solve, 3)

    def observable(self, y):
        return float(1.0 / max(np.min(y), 1e-12))

    def store(self, y):
        s = self.n_solve // N_X
        return np.log10(np.maximum(y.reshape(N_X, s).mean(1), 1e-8))[None, :]


# ---------------------------------------------------------------------------
# 4. viscous Burgers shock formation:  u_t + u u_x = nu u_xx, periodic
#    The steepening is arrested by viscosity, so max |u_x| does not diverge --
#    it rises by an order of magnitude and then relaxes. That makes this the
#    hardest test of the two-valued state coordinate: the observable takes the
#    same value on the way up and on the way down, so the post-minimum flag is
#    doing real work here, unlike in the monotone systems.
#
#    The shock width is O(nu), so nu is bounded below by the requirement that
#    the shock be resolved on the 128-point *storage* grid, not merely on the
#    integration grid.

class Burgers(System):
    name = 'burgers'
    field_names = ('u',)
    param_names = ('a1', 'a2', 'phi', 'log_nu')
    n_solve = 1024
    # viscosity arrests the steepening, so the achievable amplification of
    # max|u_x| is a factor of ~3-10 over this parameter box rather than the
    # 10^2-10^3 of the genuinely singular systems; the threshold is set below
    # the smallest achievable growth so that every run reaches it
    blowup_factor = 3.0
    t_end = 3.0
    rtol, atol = 1e-8, 1e-10

    def solve_grid(self):
        return np.linspace(0.0, 1.0, self.n_solve, endpoint=False)

    def store_grid(self):
        return np.linspace(0.0, 1.0, N_X, endpoint=False)

    def sample(self, rng):
        return {'a1': float(rng.uniform(0.85, 1.50)),
                'a2': float(rng.uniform(-0.4, 0.4)),
                'phi': float(rng.uniform(0.0, 2 * np.pi)),
                'log_nu': float(rng.uniform(-2.20, -1.85))}

    def initial(self, p):
        x = self.solve_grid()
        return (p['a1'] * np.sin(2 * np.pi * x)
                + p['a2'] * np.sin(4 * np.pi * x + p['phi']))

    def rhs(self, t, y, p):
        dx = 1.0 / self.n_solve
        nu = 10.0 ** p['log_nu']
        flux = 0.5 * y ** 2
        adv = -(np.roll(flux, -1) - np.roll(flux, 1)) / (2 * dx)
        lap = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / dx ** 2
        return adv + nu * lap

    def jac_sparsity(self):
        return _banded_periodic(self.n_solve, 1)

    def observable(self, y):
        dx = 1.0 / self.n_solve
        return float(np.max(np.abs(np.roll(y, -1) - np.roll(y, 1))) / (2 * dx))

    def store(self, y):
        s = self.n_solve // N_X
        return y.reshape(N_X, s).mean(1)[None, :]


# ---------------------------------------------------------------------------

def _banded_periodic(n, half):
    """Sparsity of a periodic banded Jacobian with the given half-width."""
    s = np.zeros((n, n), dtype=np.int8)
    idx = np.arange(n)
    for d in range(-half, half + 1):
        s[idx, (idx + d) % n] = 1
    return s


SYSTEMS = {'rd': RD, 'ks': KS, 'film': Film, 'burgers': Burgers}
