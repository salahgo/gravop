"""Generalised conductivity interpolation, selectable at run time.

The production solver (gravop.solver) combines the short- and long-mean-free-
path conductivities harmonically, K = (K_s^-1 + K_l^-1)^-1. van den Bosch &
Tokayer (2026, MNRAS, doi:10.1093/mnras/stag2015) calibrate the gravothermal
fluid model against spherical direct-simulation Monte Carlo and find that the
harmonic form is inaccurate across the transition; the generalised form

    K = (K_s^-alpha + K_l^-alpha)^(-1/alpha)

with alpha = 1/2 and a single long-mean-free-path constant beta = 0.9 fits NFW
haloes over a range of concentrations and truncations. Their K_LMFP and
K_SMFP are those of gravop.solver with beta in place of C (same a = 4/sqrt(pi)
and b = 25 sqrt(pi)/32), so (alpha, C) = (1/2, 0.9) is their calibration.

run_halo_g is gravop.baryons.run_halo_b with alpha as an argument: the
conduction coefficient is the only change, made in the two functions of the
time loop that form it. With alpha = 1 the arithmetic is the production one,
and the module reproduces gravop.solver.run_halo (no baryons) and
gravop.baryons.run_halo_b (baryons) bit for bit (tests/test_closure.py).
"""
import numpy as np
from numba import njit

from . import solver as S
from .solver import A_COEF, B_COEF, _F_elastic, _timestep, _conduct
from .baryons import _hydro_step_b, _total_energy_b, profile_with_baryons

# Named closures. LEGACY is the production solver's harmonic form with the
# conductivity calibrated on the N-body halo of Fischer, Yu & Dolag (2025);
# LITERATURE is van den Bosch & Tokayer's (2026) calibration. FIVE_HALO is the
# fit to haloes A, B, C, D1, D2; it was frozen and tested on two pre-registered
# haloes E1 (passed) and E2 (below the fitted Knudsen range: 11% early).
# CALIBRATED is the fit to all seven haloes (docs/CLOSURE_TEST_RESULTS_2026-11-12.md):
# it reproduces them to about 5% for 0.45 < Kn_c < 6.7, with a residual
# dependence on the halo profile at that level. The harmonic form fails below a
# central Knudsen number of about one.
LEGACY = dict(alpha=1.0, C=0.712)
LITERATURE = dict(alpha=0.5, C=0.9)
FIVE_HALO = dict(alpha=0.58, C=0.85)
CALIBRATED = dict(alpha=0.535, C=0.886)


@njit(cache=True)
def _keff(kinv_s, kinv_l, alpha):
    if alpha == 1.0:
        return 1.0 / (kinv_s + kinv_l)
    return (kinv_s ** alpha + kinv_l ** alpha) ** (-1.0 / alpha)


@njit(cache=True)
def _derived_a(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn, alpha):
    """gravop.solver._derived with the generalised interpolation."""
    n = r.shape[0]
    Keff = np.empty(n)
    for i in range(n):
        u[i] = 1.5 * p[i] / rho[i]
        v[i] = np.sqrt(p[i] / rho[i])
        Fl = _F_elastic(v[i] / w, ml, mpl)
        Fs = _F_elastic(v[i] / w, ms, mps)
        kinv_s = sigma_m * Fs / (B_COEF * v[i])
        kinv_l = 1.0 / (A_COEF * C * v[i] * p[i] * sigma_m * Fl)
        Keff[i] = _keff(kinv_s, kinv_l, alpha)
        Kn[i] = 1.0 / (np.sqrt(p[i]) * sigma_m * Fl)
    L[0] = -r[0] * r[0] * 0.5 * (Keff[0] + Keff[1]) * (u[1] - u[0]) / (r[1] * 0.5)
    for i in range(1, n - 1):
        L[i] = (-r[i] * r[i] * 0.5 * (Keff[i] + Keff[i + 1])
                * (u[i + 1] - u[i]) / ((r[i + 1] - r[i - 1]) * 0.5))
    L[n - 1] = 0.0


@njit(cache=True)
def _conduct_implicit_try_a(r, m, rho, p, u, sigma_m, w, C, ml, mpl, ms, mps,
                            dt, unew, alpha):
    """gravop.solver._conduct_implicit_try with the generalised interpolation."""
    n = r.shape[0]
    Keff = np.empty(n)
    for i in range(n):
        vi = np.sqrt(p[i] / rho[i])
        Fl = _F_elastic(vi / w, ml, mpl)
        Fs = _F_elastic(vi / w, ms, mps)
        kinv_s = sigma_m * Fs / (B_COEF * vi)
        kinv_l = 1.0 / (A_COEF * C * vi * p[i] * sigma_m * Fl)
        Keff[i] = _keff(kinv_s, kinv_l, alpha)
    a = np.zeros(n)
    a[0] = r[0] * r[0] * 0.5 * (Keff[0] + Keff[1]) / (r[1] * 0.5)
    for i in range(1, n - 1):
        a[i] = (r[i] * r[i] * 0.5 * (Keff[i] + Keff[i + 1])
                / ((r[i + 1] - r[i - 1]) * 0.5))
    lo = np.empty(n - 1)
    di = np.empty(n)
    up = np.empty(n - 1)
    rhs = np.empty(n)
    for i in range(n):
        dm = m[i] - (m[i - 1] if i > 0 else 0.0)
        c1 = dt / dm
        am1 = a[i - 1] if i > 0 else 0.0
        di[i] = 1.0 + c1 * (a[i] + am1)
        if i < n - 1:
            up[i] = -c1 * a[i]
        if i > 0:
            lo[i - 1] = -c1 * am1
        rhs[i] = u[i]
    for i in range(1, n):
        wfac = lo[i - 1] / di[i - 1]
        di[i] -= wfac * up[i - 1]
        rhs[i] -= wfac * rhs[i - 1]
    unew[n - 1] = rhs[n - 1] / di[n - 1]
    for i in range(n - 2, -1, -1):
        unew[i] = (rhs[i] - up[i] * unew[i + 1]) / di[i]
    mx = 0.0
    for i in range(n):
        rel = abs(unew[i] - u[i]) / u[i]
        if rel > mx:
            mx = rel
    return mx


@njit(cache=True)
def _evolve_g(r, m, rho, p, sigma_m, w, C, ml, mpl, ms, mps,
              eps_t, eps_r, t_end, rho_c_factor, kn_min, max_steps,
              snap_dlnrho, snap_dlnt, max_snaps,
              snap_t, snap_r, snap_rho, snap_p, snap_E,
              n_hydro_max, relax_prefac, implicit_flag, Mb, ab, alpha):
    """gravop.baryons._evolve_b with the generalised interpolation."""
    n = r.shape[0]
    u = np.empty(n)
    v = np.empty(n)
    L = np.empty(n)
    Kn = np.empty(n)
    _derived_a(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn, alpha)

    rho_c0 = rho[3]
    t = 0.0
    isnap = 0
    E, _, _ = _total_energy_b(r, m, rho, p, u, Mb, ab)
    snap_t[isnap] = 0.0
    snap_r[isnap] = r
    snap_rho[isnap] = rho
    snap_p[isnap] = p
    snap_E[isnap] = E
    isnap += 1
    last_lnrho = np.log(rho[3])
    last_t = 0.0

    status = 3
    step = 0
    dt_prev = -1.0
    u_try = np.empty(n)
    while step < max_steps:
        step += 1
        if implicit_flag == 1:
            dt = _timestep(u, v, rho, m, L, eps_t, 1e300)
            if dt_prev > 0.0 and dt > 1.5 * dt_prev:
                dt = 1.5 * dt_prev
            if not np.isfinite(dt) or dt <= 0.0:
                status = 4
                break
            ok = False
            for _try in range(25):
                mx = _conduct_implicit_try_a(r, m, rho, p, u, sigma_m, w, C,
                                             ml, mpl, ms, mps, dt, u_try, alpha)
                if mx <= 5.0 * eps_t:
                    ok = True
                    break
                dt *= 0.5
            if not ok:
                status = 4
                break
            for i in range(n):
                du = u_try[i] - u[i]
                p[i] += p[i] * du / u[i]
                u[i] = u_try[i]
            dt_prev = dt
        else:
            dt = _timestep(u, v, rho, m, L, eps_t, relax_prefac)
            if not np.isfinite(dt) or dt <= 0.0:
                status = 4
                break
            _conduct(r, m, rho, p, u, L, dt)
        t += dt
        nh = 0
        while True:
            max_rel = _hydro_step_b(r, m, rho, p, Mb, ab)
            nh += 1
            if max_rel < eps_r or nh >= n_hydro_max:
                break
        _derived_a(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn, alpha)

        bad = False
        for i in range(n):
            if not (rho[i] > 0.0 and p[i] > 0.0 and np.isfinite(rho[i])
                    and np.isfinite(p[i])):
                bad = True
                break
        if bad:
            status = 4
            break

        rho_c = rho[3]
        lnrho = np.log(rho_c)
        want_snap = False
        if abs(lnrho - last_lnrho) >= snap_dlnrho:
            want_snap = True
        if last_t > 0.0 and (t - last_t) / last_t >= snap_dlnt:
            want_snap = True
        if last_t == 0.0 and t > 0.0:
            want_snap = True
        if want_snap and isnap < max_snaps - 1:
            E, _, _ = _total_energy_b(r, m, rho, p, u, Mb, ab)
            snap_t[isnap] = t
            snap_r[isnap] = r
            snap_rho[isnap] = rho
            snap_p[isnap] = p
            snap_E[isnap] = E
            isnap += 1
            last_lnrho = lnrho
            last_t = t

        if rho_c >= rho_c_factor * rho_c0:
            status = 0
            break
        if t >= t_end:
            status = 1
            break
        kmin = Kn[0]
        for i in range(n):
            if Kn[i] < kmin:
                kmin = Kn[i]
        if kmin <= kn_min:
            status = 2
            break

    E, _, _ = _total_energy_b(r, m, rho, p, u, Mb, ab)
    snap_t[isnap] = t
    snap_r[isnap] = r
    snap_rho[isnap] = rho
    snap_p[isnap] = p
    snap_E[isnap] = E
    isnap += 1
    return isnap, status, t, step


def run_halo_g(sigma_m, w=1.0, model_lmfp='constant', model_smfp='constant',
               C=S.C_DEFAULT, alpha=1.0, Mb=0.0, ab=1.0, rho_func=None, r=None,
               m=None, rho=None, p=None, n_shells=200, r_min=0.01, r_max=100.0,
               eps_t=1e-4, eps_r=1e-14, t_end=np.inf, rho_c_factor=1e4,
               kn_min=0.0, max_steps=200_000_000, snap_dlnrho=0.02,
               snap_dlnt=0.05, max_snaps=2000, n_hydro_max=50,
               powerlaw_n=0.0, yb_index=5, timestep_mode='classic'):
    """gravop.solver.run_halo with a selectable conductivity interpolation
    (alpha) and an optional static Hernquist baryon component (Mb, ab)."""
    if r is None:
        r, m, rho, p = profile_with_baryons(Mb, ab, rho_func, r_min, r_max, n_shells)
    r, m, rho, p = (np.ascontiguousarray(x, dtype=np.float64).copy()
                    for x in (r, m, rho, p))
    n = r.shape[0]
    ml, mpl = S.model_params(model_lmfp, powerlaw_n, yb_index)
    ms, mps = S.model_params(model_smfp, powerlaw_n, yb_index)
    implicit_flag = 0
    if timestep_mode == 'implicit':
        implicit_flag, relax_prefac = 1, 1.0
    elif timestep_mode == 'classic':
        relax_prefac = 1.0
    elif timestep_mode == 'physical':
        F1 = S._F_elastic(1.0 / w, ml, mpl)
        relax_prefac = min(2.0 / (3.0 * S.A_COEF * C * sigma_m * max(F1, 1e-12)), 1e4)
    else:
        raise ValueError(timestep_mode)
    snap_t = np.zeros(max_snaps)
    snap_r = np.zeros((max_snaps, n))
    snap_rho = np.zeros((max_snaps, n))
    snap_p = np.zeros((max_snaps, n))
    snap_E = np.zeros(max_snaps)
    n_snaps, status, t, steps = _evolve_g(
        r, m, rho, p, float(sigma_m), float(w), float(C), ml, mpl, ms, mps,
        float(eps_t), float(eps_r), float(t_end), float(rho_c_factor),
        float(kn_min), int(max_steps), float(snap_dlnrho), float(snap_dlnt),
        int(max_snaps), snap_t, snap_r, snap_rho, snap_p, snap_E,
        int(n_hydro_max), float(relax_prefac), implicit_flag, float(Mb), float(ab),
        float(alpha))
    return {
        't': snap_t[:n_snaps], 'r': snap_r[:n_snaps], 'rho': snap_rho[:n_snaps],
        'p': snap_p[:n_snaps], 'E': snap_E[:n_snaps], 'm': m,
        'status': int(status), 't_final': float(t), 'n_steps': int(steps),
        'params': {'sigma_m': float(sigma_m), 'w': float(w),
                   'model_lmfp': model_lmfp, 'model_smfp': model_smfp,
                   'C': float(C), 'alpha': float(alpha), 'n_shells': int(n),
                   'r_min': float(r_min), 'r_max': float(r_max), 'eps_t': float(eps_t),
                   'eps_r': float(eps_r), 'rho_c_factor': float(rho_c_factor),
                   'kn_min': float(kn_min), 'yb_index': int(yb_index),
                   'powerlaw_n': float(powerlaw_n), 'timestep_mode': timestep_mode,
                   'Mb': float(Mb), 'ab': float(ab)},
    }
