"""statecond.pdes2d — a 2-D system with a finite-time singularity.

`ASSESSMENT.md` §3.6 and the standing limitation in `CROSS_DOMAIN.md` §7: all
four benchmark systems are 1-D. Their job was to vary the *domain* while holding
the protocol fixed, not to be hard, but "all your testbeds are one-dimensional"
is a fair objection and this closes it.

Keller-Segel without radial symmetry, on a periodic square:

    rho_t = div( grad rho - chi rho grad c ),      -Lap c = rho - <rho>

The radial version already in `pdes.py` imposes the singularity's location by
construction; here the aggregate forms wherever the initial data sends it, and
several peaks may compete and merge before one runs away. That is the point of
the 2-D case: the singular *location* is now part of what a surrogate has to
get right, not a coordinate choice.

The Poisson solve is spectral (exact on the periodic grid, and the zero mode is
set to zero, which is the solvability condition). Transport uses a conservative
flux form with an upwinded drift term, integrated with an adaptive explicit
step, since the parabolic limit here is mild compared with the 4th-order 1-D
film problem and an implicit solve on 128^2 unknowns is not worth its cost.

**Second order in both operators (P25a, 2026-09-07).** The original scheme was
first order in space (donor-cell upwind) and first order in time
(integrating-factor forward Euler), and `t_sing` converged only to 9.6-13%
between the 128 and 256 grids -- two orders of magnitude short of what a study
of a few-percent effect needs, which is why the system was withheld. It is now
MUSCL with a **minmod** limiter in space and **SSP-RK2 (Heun) in the
integrating-factor variable** in time. Minmod because the field is a density
that must stay positive and the limiter must not create new extrema near the
aggregate. `scheme='donor-euler'` restores the original, so the order of the
two schemes is measurable rather than asserted (`scripts/sc_ks2d.py`).

Interface matches `statecond.pdes.System`, except that the stored field is
(n_fields, N_Y, N_X) rather than (n_fields, N_X).
"""
from dataclasses import dataclass, field
import time

import numpy as np

N_SOLVE = 128        # integration grid
N_STORE = 64         # storage grid handed to the operator


@dataclass
class Trajectory2D:
    t: np.ndarray
    u: np.ndarray                 # (n_snap, n_fields, N_STORE, N_STORE)
    a: np.ndarray
    x: np.ndarray
    t_sing: float
    status: int
    params: dict = field(default_factory=dict)
    n_steps: int = 0
    wall: float = 0.0


class KS2D:
    name = 'ks2d'
    n_fields = 1
    field_names = ('log10_rho',)
    param_names = ('log_mass_ratio', 'w', 'n_blobs', 'asym')
    blowup_factor = 30.0
    t_end = 5.0
    chi = 1.0

    def __init__(self, n_solve=N_SOLVE, n_store=N_STORE,
                 scheme='muscl-rk2'):
        if scheme not in ('muscl-rk2', 'donor-euler'):
            raise ValueError(f'unknown scheme {scheme!r}')
        self.n_solve, self.n_store = n_solve, n_store
        self.scheme = scheme

    def store_grid(self):
        return np.linspace(0.0, 1.0, self.n_store, endpoint=False)

    def sample(self, rng):
        return {'log_mass_ratio': float(rng.uniform(0.20, 0.75)),
                'w': float(rng.uniform(0.06, 0.16)),
                'n_blobs': int(rng.integers(1, 4)),
                'asym': float(rng.uniform(0.0, 0.6))}

    def initial(self, p, rng=None):
        """Blobs at reproducible positions derived from the parameters.

        Positions come from a generator seeded by the parameter values, so a
        trajectory is a deterministic function of `p` alone and regenerates
        exactly.
        """
        n = self.n_solve
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing='ij')
        seed = int(abs(hash((round(p['log_mass_ratio'], 6), round(p['w'], 6),
                             p['n_blobs'], round(p['asym'], 6)))) % (2 ** 31))
        r = np.random.default_rng(seed)
        rho = np.full((n, n), 0.02)
        for i in range(p['n_blobs']):
            cx, cy = r.uniform(0.2, 0.8, 2)
            amp = 1.0 - p['asym'] * r.uniform(0.0, 1.0)
            # periodic distance
            dx = np.abs(X - cx)
            dy = np.abs(Y - cy)
            dx = np.minimum(dx, 1 - dx)
            dy = np.minimum(dy, 1 - dy)
            rho += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * p['w'] ** 2))
        mass = rho.mean()
        target = (8 * np.pi / self.chi) * 10.0 ** p['log_mass_ratio']
        return rho * (target / mass)

    def _poisson(self, rho, k2inv):
        """Solve -Lap c = rho - <rho> spectrally."""
        rh = np.fft.rfft2(rho - rho.mean())
        return np.fft.irfft2(rh * k2inv, s=rho.shape)

    @staticmethod
    def _minmod(a, b):
        """The limiter. Zero wherever the two slopes disagree in sign, so the
        reconstruction introduces no new extrema -- which for a density that
        must stay positive is the property that matters, and the reason this
        is minmod rather than a sharper limiter."""
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a),
                                                            np.abs(b))

    def _face_states(self, rho, axis):
        """Left and right MUSCL states at the face between cell i and i+1."""
        d = np.roll(rho, -1, axis) - rho                    # forward difference
        slope = self._minmod(np.roll(d, 1, axis), d)
        left = rho + 0.5 * slope
        right = np.roll(rho, -1, axis) - 0.5 * np.roll(slope, -1, axis)
        return left, right

    def drift_divergence(self, rho, k2inv, dx):
        """-div(chi rho grad c), the nonlinear part, in conservative flux form.

        Only the advective term is treated explicitly. Diffusion is handled
        exactly in Fourier space by the integrating factor in `solve`, which is
        what makes the step size advection-limited (dt ~ dx/|v|) rather than
        diffusion-limited (dt ~ dx^2/4). On a 128^2 grid that is the difference
        between ~10^6 steps and ~10^4.

        The face value of `rho` is the upwind MUSCL reconstruction
        (`scheme='muscl-rk2'`, second order away from extrema) or the cell value
        itself (`scheme='donor-euler'`, first order). The velocity is a centred
        difference of `c` averaged to the face, which is second order in both
        cases, so the reconstruction is the only thing that changes the order.
        """
        c = self._poisson(rho, k2inv)
        vx = self.chi * (np.roll(c, -1, 0) - np.roll(c, 1, 0)) / (2 * dx)
        vy = self.chi * (np.roll(c, -1, 1) - np.roll(c, 1, 1)) / (2 * dx)
        vxf = 0.5 * (vx + np.roll(vx, -1, 0))
        vyf = 0.5 * (vy + np.roll(vy, -1, 1))
        if self.scheme == 'muscl-rk2':
            xl, xr = self._face_states(rho, 0)
            yl, yr = self._face_states(rho, 1)
        else:
            xl, xr = rho, np.roll(rho, -1, 0)
            yl, yr = rho, np.roll(rho, -1, 1)
        fx = -vxf * np.where(vxf > 0, xl, xr)
        fy = -vyf * np.where(vyf > 0, yl, yr)
        div = ((fx - np.roll(fx, 1, 0)) + (fy - np.roll(fy, 1, 1))) / dx
        return div, max(float(np.abs(vxf).max()), float(np.abs(vyf).max()))

    def store(self, rho):
        assert self.n_solve % self.n_store == 0, (
            'block-averaging needs n_solve to be a multiple of n_store; '
            f'got {self.n_solve} / {self.n_store}')
        s = self.n_solve // self.n_store
        blk = rho.reshape(self.n_store, s, self.n_store, s).mean(axis=(1, 3))
        return np.log10(np.maximum(blk, 1e-6))[None, :, :]

    def solve(self, p):
        n, dx = self.n_solve, 1.0 / self.n_solve
        kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        ky = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
        K2 = kx[:, None] ** 2 + ky[None, :] ** 2
        k2inv = np.zeros_like(K2)
        k2inv[K2 > 0] = 1.0 / K2[K2 > 0]

        rho = self.initial(p)
        a0 = float(rho.max())
        thresh = a0 * self.blowup_factor
        ts, us, av = [0.0], [self.store(rho)], [a0]
        t, steps = 0.0, 0
        last_a = a0
        t0 = time.time()
        K2full = kx[:, None] ** 2 + ky[None, :] ** 2
        while t < self.t_end and steps < 200_000:
            amax = float(rho.max())
            if amax >= thresh or not np.isfinite(amax):
                break
            adv, vmax = self.drift_divergence(rho, k2inv, dx)
            # advective CFL only; diffusion is exact below
            dt = min(0.25 * dx / (vmax + 1e-12), 0.02)
            E = np.exp(-K2full * dt)
            if self.scheme == 'muscl-rk2':
                # SSP-RK2 (Heun) in the integrating-factor variable
                # v = exp(K2 t) rho_hat, so that
                #   rho1     = E (rho_hat + dt N0)
                #   rho^{n+1}= E rho_hat + (dt/2)(E N0 + N1)
                # which is exact on the diffusion and second order on the
                # advection. One extra Poisson solve and one extra FFT pair per
                # step, for one order.
                r0 = np.fft.rfft2(rho)
                n0 = np.fft.rfft2(adv)
                rho1 = np.fft.irfft2((r0 + dt * n0) * E, s=rho.shape)
                rho1 = np.maximum(rho1, 1e-12)
                adv1, vmax1 = self.drift_divergence(rho1, k2inv, dx)
                n1 = np.fft.rfft2(adv1)
                rho = np.fft.irfft2(E * r0 + 0.5 * dt * (E * n0 + n1),
                                    s=rho.shape)
                vmax = max(vmax, vmax1)
            else:
                rh = np.fft.rfft2(rho) + dt * np.fft.rfft2(adv)
                rho = np.fft.irfft2(rh * E, s=rho.shape)
            rho = np.maximum(rho, 1e-12)
            t += dt
            steps += 1
            amax = float(rho.max())
            if abs(np.log10(amax) - np.log10(last_a)) > 0.013:
                ts.append(t)
                us.append(self.store(rho))
                av.append(amax)
                last_a = amax
        wall = time.time() - t0
        ts.append(t)
        us.append(self.store(rho))
        av.append(float(rho.max()))
        a = np.array(av)
        if len(ts) < 12 or not np.all(np.isfinite(a)):
            return None

        reached = np.nonzero(a >= thresh * (1 - 1e-6))[0]
        if reached.size:
            j = int(reached[0])
            if j == 0:
                return None
            l0, l1 = np.log(a[j - 1]), np.log(a[j])
            frac = (np.log(thresh) - l0) / max(l1 - l0, 1e-300)
            t_sing = float(ts[j - 1] + frac * (ts[j] - ts[j - 1]))
            ts, us, a = ts[:j + 1], us[:j + 1], a[:j + 1]
            status = 0
        else:
            t_sing, status = float(ts[-1]), 1

        u = np.stack(us).astype(np.float32)
        if not np.all(np.isfinite(u)):
            return None
        return Trajectory2D(t=np.array(ts, float), u=u, a=a,
                            x=self.store_grid(), t_sing=t_sing, status=status,
                            params=dict(p), n_steps=steps, wall=wall)


SYSTEMS_2D = {'ks2d': KS2D}
