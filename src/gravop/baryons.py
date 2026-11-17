"""Gravothermal evolution in a static baryonic potential (Hernquist).

Following Zhong, Yang & Yu (2023, MNRAS 526, 758), the baryons enter the
gravothermal fluid equations only through the gravitating mass in hydrostatic
equilibrium,

    d(rho v^2)/dr = -(M(r) + M_b(r)) rho / r^2         (G = 1, code units),

with mass continuity, the conduction law and the conductivity unchanged. The
baryons are a static Hernquist (1990) sphere,

    M_b(r) = M_b r^2 / (r + a)^2,      Phi_b(r) = -M_b / (r + a),

with M_b in units of M0 = 4 pi rho_s r_s^3 and a in units of r_s, so the
problem keeps the scale-free form of gravop.solver: a dwarf and a cluster with
the same (sigma-hat, w-hat, M_b-hat, a-hat) evolve identically.

The initial dark-matter profile is the dark-matter-only one (NFW unless a
profile function is given), with its pressure set by Jeans equilibrium in the
total potential; as in Zhong et al., no adiabatic contraction is applied. The
Jeans integral is done in ln r, which stays accurate for compact baryons.

Implementation. The conduction step, time-step control, cross-section models
and every other numerical choice are imported unchanged from gravop.solver,
which is not modified. Only the hydrostatic Newton step (the enclosed mass is
M + M_b, with M_b re-evaluated at the current shell radii on every iteration
and held fixed inside the linearisation, as in Zhong et al.'s code), the
evolution loop that calls it, and the energy diagnostic (which adds the energy
of the dark matter in the static external potential) are re-implemented. With
M_b = 0 the module reproduces gravop.solver.run_halo bit for bit
(tests/test_baryons.py).
"""
import numpy as np
from numba import njit
from scipy import integrate

from . import solver as S
from .solver import GAMMA, _tdma, _derived, _timestep, _conduct_implicit_try, _conduct


def hernquist_mass(r, Mb, a):
    return Mb * r * r / (r + a) ** 2


@njit(cache=True)
def _hydro_step_b(r, m, rho, p, Mb, ab):
    """gravop.solver._hydro_step with the enclosed mass M + M_b(r)."""
    n = r.shape[0]
    re = np.empty(n + 1)
    me = np.empty(n + 1)
    re[0] = 0.0
    me[0] = 0.0
    re[1:] = r
    for k in range(n):
        me[k + 1] = m[k] + Mb * r[k] * r[k] / ((r[k] + ab) * (r[k] + ab))
    r2 = re * re
    r3 = r2 * re

    a = np.empty(n - 2)
    b = np.empty(n - 1)
    c = np.empty(n - 2)
    d = np.empty(n - 1)

    for i in range(n - 2):
        a[i] = ((12. * GAMMA * p[i + 1] * r2[i + 1] * r2[i + 2]
                 + me[i + 2] * (-3. * r2[i + 1] * re[i + 3] * rho[i + 1]
                                + r3[i + 1] * (2. * rho[i + 1] - rho[i + 2])
                                + r3[i + 2] * (rho[i + 1] + rho[i + 2])))
                / (4. * (r3[i + 1] - r3[i + 2])))
        c[i] = ((12. * GAMMA * p[i + 1] * r2[i + 1] * r2[i + 2]
                 + me[i + 1] * (r3[i + 1] * (rho[i] + rho[i + 1])
                                - r2[i + 2] * (re[i + 2] * rho[i]
                                               + 3. * re[i] * rho[i + 1]
                                               - 2. * re[i + 2] * rho[i + 1])))
                / (4. * (r3[i + 1] - r3[i + 2])))
    for i in range(n - 1):
        b[i] = (re[i + 1]
                * (-4. * p[i] * (2. * r3[i] + (-2. + 3. * GAMMA) * r3[i + 1])
                   * (r3[i + 1] - r3[i + 2])
                   - 4. * p[i + 1] * (r3[i] - r3[i + 1])
                   * ((-2. + 3. * GAMMA) * r3[i + 1] + 2. * r3[i + 2])
                   + 3. * me[i + 1] * re[i + 1] * (re[i] - re[i + 2])
                   * (r3[i + 2] * rho[i] + r3[i] * rho[i + 1]
                      - r3[i + 1] * (rho[i] + rho[i + 1])))
                / (4. * (r3[i] - r3[i + 1]) * (r3[i + 1] - r3[i + 2])))
        d[i] = (-p[i] * r2[i + 1] + p[i + 1] * r2[i + 1]
                + 0.25 * me[i + 1] * (re[i + 2] - re[i])
                * (rho[i] + rho[i + 1]))

    dr = _tdma(a, b, c, d, n - 1)

    alpha = 1.0
    max_rel = 0.0
    for i in range(n - 1):
        rel = abs(dr[i] / r[i])
        if rel > max_rel:
            max_rel = rel
        gap_in = r[i] - (r[i - 1] if i > 0 else 0.0)
        gap_out = r[i + 1] - r[i]
        gap = gap_in if gap_in < gap_out else gap_out
        adr = abs(dr[i])
        if adr > 0.25 * gap:
            cand = 0.25 * gap / adr
            if cand < alpha:
                alpha = cand
    for i in range(n):
        r2dr_i = r[i] * r[i] * dr[i] if i < n - 1 else 0.0
        r2dr_im = r[i - 1] * r[i - 1] * dr[i - 1] if i > 0 else 0.0
        voli = (r[i] ** 3 - (r[i - 1] ** 3 if i > 0 else 0.0)) / 3.
        dp_rel = GAMMA * abs(r2dr_i - r2dr_im) / voli
        if dp_rel > 0.25:
            cand = 0.25 / dp_rel
            if cand < alpha:
                alpha = cand
    if alpha < 1.0:
        for i in range(n - 1):
            dr[i] *= alpha

    r2dr = np.empty(n)
    for i in range(n - 1):
        r2dr[i] = r[i] * r[i] * dr[i]
    r2dr[n - 1] = 0.0

    vol0 = r[0] ** 3 / 3.
    drho0 = -rho[0] * r2dr[0] / vol0
    dp0 = -GAMMA * p[0] * r2dr[0] / vol0
    for i in range(n - 1, 0, -1):
        voli = (r[i] ** 3 - r[i - 1] ** 3) / 3.
        rho[i] += -rho[i] * (r2dr[i] - r2dr[i - 1]) / voli
        p[i] += -GAMMA * p[i] * (r2dr[i] - r2dr[i - 1]) / voli
    rho[0] += drho0
    p[0] += dp0
    for i in range(n - 1):
        r[i] += dr[i]
    return max_rel


@njit(cache=True)
def _total_energy_b(r, m, rho, p, u, Mb, ab):
    """gravop.solver._total_energy plus the energy of the dark matter in the
    static baryonic potential, sum of Phi_b(r_mid) dm (conserved with it)."""
    n = r.shape[0]
    E_int = u[0] * m[0]
    E_pot = -0.5 * m[0] * m[0] / r[0]
    E_ext = -m[0] * Mb / (0.5 * r[0] + ab)
    for i in range(1, n):
        dm = m[i] - m[i - 1]
        E_int += u[i] * dm
        rmid = 0.5 * (r[i] + r[i - 1])
        mmid = 0.5 * (m[i] + m[i - 1])
        E_pot += -mmid * dm / rmid
        E_ext += -dm * Mb / (rmid + ab)
    return E_int + E_pot + E_ext, E_int, E_pot + E_ext


@njit(cache=True)
def _evolve_b(r, m, rho, p, sigma_m, w, C, ml, mpl, ms, mps,
              eps_t, eps_r, t_end, rho_c_factor, kn_min, max_steps,
              snap_dlnrho, snap_dlnt, max_snaps,
              snap_t, snap_r, snap_rho, snap_p, snap_E,
              n_hydro_max, relax_prefac, implicit_flag, Mb, ab):
    """gravop.solver._evolve with the baryonic hydrostatic step and energy."""
    n = r.shape[0]
    u = np.empty(n)
    v = np.empty(n)
    L = np.empty(n)
    Kn = np.empty(n)
    _derived(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn)

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
                mx = _conduct_implicit_try(r, m, rho, p, u, sigma_m, w, C,
                                           ml, mpl, ms, mps, dt, u_try)
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
        _derived(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn)

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


def profile_with_baryons(Mb, ab, rho_func=None, r_min=0.01, r_max=100.0,
                         n_shells=200, p_rmax_factor=10.0):
    """(r, m, rho, p) for a dark-matter profile in Jeans equilibrium in the
    total (dark matter + Hernquist) potential. Mb = 0 with rho_func None returns
    gravop.solver.nfw_profile exactly."""
    if Mb == 0.0:
        if rho_func is None:
            return S.nfw_profile(r_min, r_max, n_shells)
        return S.profile_from_rho(rho_func, r_min, r_max, n_shells, p_rmax_factor)
    if rho_func is None:
        r, m, rho, _ = S.nfw_profile(r_min, r_max, n_shells)
        rho_func, m_of = S.nfw_rho, S.nfw_mass
    else:
        r, m, rho, _ = S.profile_from_rho(rho_func, r_min, r_max, n_shells, p_rmax_factor)
        mi = lambda x: x * x * rho_func(x)
        m_of = np.vectorize(lambda x: integrate.quad(mi, 0.0, x, limit=200)[0])
    r_mid = np.empty(n_shells)
    r_mid[0] = r[0] / 2.0
    r_mid[1:] = 0.5 * (r[:-1] + r[1:])
    xmax = p_rmax_factor * r_max
    # Jeans integral p(x) = int_x^xmax (M + M_b) rho / s^2 ds, in ln s
    f = lambda ls: ((m_of(np.exp(ls)) + hernquist_mass(np.exp(ls), Mb, ab))
                    * rho_func(np.exp(ls)) / np.exp(ls))
    p = np.empty(n_shells)
    for i in range(n_shells):
        p[i] = integrate.quad(f, np.log(r_mid[i]), np.log(xmax), limit=400,
                              points=[np.log(ab)] if r_mid[i] < ab < xmax else None)[0]
    return r, m, rho, p


def run_halo_b(sigma_m, w=1.0, model_lmfp='constant', model_smfp='constant',
               C=S.C_DEFAULT, Mb=0.0, ab=1.0, rho_func=None, r=None, m=None,
               rho=None, p=None, n_shells=200, r_min=0.01, r_max=100.0,
               eps_t=1e-4, eps_r=1e-14, t_end=np.inf, rho_c_factor=1e4,
               kn_min=0.0, max_steps=200_000_000, snap_dlnrho=0.02,
               snap_dlnt=0.05, max_snaps=2000, n_hydro_max=50,
               powerlaw_n=0.0, yb_index=5, timestep_mode='classic'):
    """gravop.solver.run_halo with a static Hernquist baryon component
    (Mb = M_b / M0, ab = a / r_s). Returns the same dictionary, with the baryon
    parameters added to 'params'."""
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
    n_snaps, status, t, steps = _evolve_b(
        r, m, rho, p, float(sigma_m), float(w), float(C), ml, mpl, ms, mps,
        float(eps_t), float(eps_r), float(t_end), float(rho_c_factor),
        float(kn_min), int(max_steps), float(snap_dlnrho), float(snap_dlnt),
        int(max_snaps), snap_t, snap_r, snap_rho, snap_p, snap_E,
        int(n_hydro_max), float(relax_prefac), implicit_flag, float(Mb), float(ab))
    return {
        't': snap_t[:n_snaps], 'r': snap_r[:n_snaps], 'rho': snap_rho[:n_snaps],
        'p': snap_p[:n_snaps], 'E': snap_E[:n_snaps], 'm': m,
        'status': int(status), 't_final': float(t), 'n_steps': int(steps),
        'params': {'sigma_m': float(sigma_m), 'w': float(w),
                   'model_lmfp': model_lmfp, 'model_smfp': model_smfp,
                   'C': float(C), 'n_shells': int(n), 'r_min': float(r_min),
                   'r_max': float(r_max), 'eps_t': float(eps_t),
                   'eps_r': float(eps_r), 'rho_c_factor': float(rho_c_factor),
                   'kn_min': float(kn_min), 'yb_index': int(yb_index),
                   'powerlaw_n': float(powerlaw_n), 'timestep_mode': timestep_mode,
                   'Mb': float(Mb), 'ab': float(ab)},
    }
