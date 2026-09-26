"""
gravop.solver — gravothermal fluid solver for SIDM halo evolution.

Independent implementation of the standard conduction + quasi-static hydrostatic
adjustment scheme (Balberg & Shapiro 2002; Nishikawa, Boddy & Kaplinghat 2020;
Outmezguine et al. 2023; cf. the GPLv3 GravothermalSIDM code of Boddy et al.,
against which this implementation is cross-validated). Fully Numba-compiled
step loop for speed.

Dimensionless units: lengths in r_s, density in rho_s, mass in M0 = 4 pi rho_s r_s^3,
velocity in v0 = sqrt(G M0 / r_s), time in t0 = 1/sqrt(4 pi G rho_s),
pressure in rho_s v0^2, cross-section per mass in 1/(rho_s r_s).

State per Lagrangian shell i (outer edge radius r[i], enclosed mass m[i]):
rho[i], p[i] shell-averaged. Derived: u = 1.5 p/rho, v = sqrt(p/rho).

Conductivity: L/r^2 = -K_eff du/dr with
  1/K_eff = sigma F_s(v/w) / (b v)  +  1 / (a C v p sigma F_l(v/w))
a = 4/sqrt(pi), b = 25 sqrt(pi)/32, C = 0.753 (N-body calibrated), gamma = 5/3.

Cross-section velocity dependence F(v/w):
  model 0: constant, F = 1
  model 1: power law, F = (v/w)^n
  model 2: Yukawa-Born viscosity average (t+u channel), order-1 fit of
           Outmezguine et al. 2023 (arXiv:2204.06568, eq. 5), index K5
  model 3: same, t-channel only

License: GPLv3.
"""
import numpy as np
from numba import njit
from scipy import integrate

A_COEF = 4.0 / np.sqrt(np.pi)
B_COEF = 25.0 * np.sqrt(np.pi) / 32.0
C_DEFAULT = 0.753
GAMMA = 5.0 / 3.0

# Yukawa-Born viscosity fit coefficients [par0, par1, par2, par3]
# (Outmezguine et al. 2023 eq. 5 parametrization), index -> coefficients.
YB_TU = {3: [8., 0.0339848, 0.37, 0.63], 5: [24., 0.251115, 0.41, 0.71],
         7: [48., 0.682602, 0.42, 0.74], 9: [80., 1.32953, 0.43, 0.76]}
YB_T = {3: [8., 0.303941, 0.74, 0.68], 5: [24., 0.826198, 0.82, 0.76],
        7: [48., 1.36217, 0.83, 0.79], 9: [80., 1.90106, 0.84, 0.81]}


@njit(cache=True, fastmath=False)
def _F_elastic(s, model_id, mp):
    """Velocity-dependence factor F(s), s = v/w."""
    if model_id == 0:
        return 1.0
    elif model_id == 1:
        return s ** mp[0]
    else:
        s2 = s * s + 1e-4
        s4 = s * s * s * s + 1e-8
        sn = s2 if model_id == 3 else s4
        g = 1.5 * np.log(1.0 + (sn * mp[1]) ** mp[2]) / (mp[0] * mp[2] * s4)
        return (1.0 + 1.0 / g ** mp[3]) ** (-1.0 / mp[3])


@njit(cache=True)
def _tdma(a, b, c, d, n):
    """Solve tridiagonal system A x = -d in place; returns x (overwrites b, d).
    A zero pivot yields NaNs (caught by the caller's finite-state check)."""
    x = np.empty(n)
    dd = -d
    for i in range(1, n):
        piv = b[i - 1]
        if piv == 0.0:
            piv = 1e-300
        w = a[i - 1] / piv
        b[i] -= w * c[i - 1]
        dd[i] -= w * dd[i - 1]
    for i in range(n - 1, -1, -1):
        piv = b[i]
        if piv == 0.0:
            piv = 1e-300
        if i == n - 1:
            x[i] = dd[i] / piv
        else:
            x[i] = (dd[i] - c[i] * x[i + 1]) / piv
    return x


@njit(cache=True)
def _derived(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn):
    """Update u, v, L, Kn in place from (r, rho, p)."""
    n = r.shape[0]
    Keff = np.empty(n)
    for i in range(n):
        u[i] = 1.5 * p[i] / rho[i]
        v[i] = np.sqrt(p[i] / rho[i])
        Fl = _F_elastic(v[i] / w, ml, mpl)
        Fs = _F_elastic(v[i] / w, ms, mps)
        kinv_s = sigma_m * Fs / (B_COEF * v[i])
        kinv_l = 1.0 / (A_COEF * C * v[i] * p[i] * sigma_m * Fl)
        Keff[i] = 1.0 / (kinv_s + kinv_l)
        Kn[i] = 1.0 / (np.sqrt(p[i]) * sigma_m * Fl)
    L[0] = -r[0] * r[0] * 0.5 * (Keff[0] + Keff[1]) * (u[1] - u[0]) / (r[1] * 0.5)
    for i in range(1, n - 1):
        L[i] = (-r[i] * r[i] * 0.5 * (Keff[i] + Keff[i + 1])
                * (u[i + 1] - u[i]) / ((r[i + 1] - r[i - 1]) * 0.5))
    L[n - 1] = 0.0


@njit(cache=True)
def _timestep(u, v, rho, m, L, eps_t, relax_prefac):
    """dt = eps_t * min(specific-energy change time, local relaxation time).

    relax_prefac scales the relaxation criterion: the classic conservative
    choice is dt2 = min 1/(rho v) (relax_prefac = 1); the physical local
    relaxation time for cross-section sigma F is 2/(3 a C sigma F rho v) =
    relax_prefac/(rho v) with relax_prefac supplied by the caller. Capped at
    1e4 to avoid runaway steps in the weak-coupling limit."""
    n = u.shape[0]
    dt1 = 1e300
    dL0 = L[0] / m[0]
    if dL0 != 0.0:
        dt1 = abs(u[0] / dL0)
    for i in range(1, n):
        dLdm = (L[i] - L[i - 1]) / (m[i] - m[i - 1])
        if dLdm != 0.0:
            cand = abs(u[i] / dLdm)
            if cand < dt1:
                dt1 = cand
    dt2 = 1e300
    for i in range(n):
        cand = relax_prefac / (rho[i] * v[i])
        if cand < dt2:
            dt2 = cand
    dt = dt1 if dt1 < dt2 else dt2
    return eps_t * dt


@njit(cache=True)
def _conduct_implicit_try(r, m, rho, p, u, sigma_m, w, C, ml, mpl, ms, mps,
                          dt, unew):
    """One backward-Euler conduction solve into unew (state unchanged).
    Returns max |du/u|."""
    n = r.shape[0]
    Keff = np.empty(n)
    for i in range(n):
        vi = np.sqrt(p[i] / rho[i])
        Fl = _F_elastic(vi / w, ml, mpl)
        Fs = _F_elastic(vi / w, ms, mps)
        kinv_s = sigma_m * Fs / (B_COEF * vi)
        kinv_l = 1.0 / (A_COEF * C * vi * p[i] * sigma_m * Fl)
        Keff[i] = 1.0 / (kinv_s + kinv_l)
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
def _conduct_implicit(r, m, rho, p, u, sigma_m, w, C, ml, mpl, ms, mps, dt):
    """Semi-implicit (backward-Euler) conduction step: solves the linear
    tridiagonal system for u^{n+1} with conductivities and radii frozen at
    step n. Unconditionally stable (diagonally dominant M-matrix), so dt can
    track the physical relaxation time instead of the explicit stability
    limit. Pressure updated adiabatically in energy: dp = p du/u."""
    n = r.shape[0]
    # interface conduction coefficients a_i = r_i^2 Kbar_i / dr_i  (a_{n-1}=0)
    Keff = np.empty(n)
    for i in range(n):
        vi = np.sqrt(p[i] / rho[i])
        Fl = _F_elastic(vi / w, ml, mpl)
        Fs = _F_elastic(vi / w, ms, mps)
        kinv_s = sigma_m * Fs / (B_COEF * vi)
        kinv_l = 1.0 / (A_COEF * C * vi * p[i] * sigma_m * Fl)
        Keff[i] = 1.0 / (kinv_s + kinv_l)
    a = np.zeros(n)
    a[0] = r[0] * r[0] * 0.5 * (Keff[0] + Keff[1]) / (r[1] * 0.5)
    for i in range(1, n - 1):
        a[i] = (r[i] * r[i] * 0.5 * (Keff[i] + Keff[i + 1])
                / ((r[i + 1] - r[i - 1]) * 0.5))
    # tridiagonal system
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
    # Thomas algorithm (system is diagonally dominant; pivots stay positive)
    for i in range(1, n):
        wfac = lo[i - 1] / di[i - 1]
        di[i] -= wfac * up[i - 1]
        rhs[i] -= wfac * rhs[i - 1]
    unew = np.empty(n)
    unew[n - 1] = rhs[n - 1] / di[n - 1]
    for i in range(n - 2, -1, -1):
        unew[i] = (rhs[i] - up[i] * unew[i + 1]) / di[i]
    for i in range(n):
        du = unew[i] - u[i]
        p[i] += p[i] * du / u[i]
        u[i] = unew[i]


@njit(cache=True)
def _relax_timestep(u, v, rho, p, sigma_m, w, C, ml, mpl, eps_t):
    """dt = eps_t * min over shells of the local LMFP/SMFP-interpolated
    relaxation time t_r = 2/(3 a C sigma F rho v) (capped for weak coupling).
    Used with the implicit conduction step (no stability limit)."""
    n = u.shape[0]
    tmin = 1e300
    for i in range(n):
        Fl = _F_elastic(v[i] / w, ml, mpl)
        sig_eff = sigma_m * Fl
        if sig_eff < 1e-12:
            sig_eff = 1e-12
        tr = 2.0 / (3.0 * A_COEF * C * sig_eff * rho[i] * v[i])
        if tr < tmin:
            tmin = tr
    if tmin > 1e4:
        tmin = 1e4
    return eps_t * tmin


@njit(cache=True)
def _conduct(r, m, rho, p, u, L, dt):
    """Explicit conduction step at fixed shell density."""
    n = r.shape[0]
    du0 = -dt * L[0] / m[0]
    p[0] += p[0] * du0 / u[0]
    u[0] += du0
    for i in range(1, n):
        du = -dt * (L[i] - L[i - 1]) / (m[i] - m[i - 1])
        p[i] += p[i] * du / u[i]
        u[i] += du


@njit(cache=True)
def _hydro_step(r, m, rho, p):
    """One linearized (Newton) hydrostatic adjustment with adiabatic response.
    Returns max |delta_r / r|."""
    n = r.shape[0]
    # extended arrays with r=0, m=0 at center
    re = np.empty(n + 1)
    me = np.empty(n + 1)
    re[0] = 0.0
    me[0] = 0.0
    re[1:] = r
    me[1:] = m
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

    # Damp the Newton step so no shell moves more than 40% of the gap to a
    # neighbour (prevents shell crossing, which makes the linearization
    # singular). alpha = 1 for benign steps, so converged behavior is
    # unchanged; pathological steps are shortened and re-iterated.
    alpha = 1.0
    max_rel = 0.0  # of the UNDAMPED step, so damping cannot fake convergence
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
    # additionally keep the adiabatic response bounded: |delta p / p| <= 0.5
    # (gamma * volume change), which also keeps rho, p strictly positive
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

    # apply updates: delta_r for shells 0..n-2, outer shell fixed
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
def _total_energy(r, m, rho, p, u):
    """Total energy = internal + potential (shell approximation)."""
    n = r.shape[0]
    E_int = u[0] * m[0]
    E_pot = 0.0
    # potential: -G m(r) dm / r with G=1/(4 pi)-> in our units G M0/r_s = v0^2,
    # dimensionless: dE_pot = - m_mid dm / r_mid  (units of M0 v0^2)
    rmid0 = 0.5 * r[0]
    E_pot += -0.5 * m[0] * m[0] / r[0]  # inner sphere approx
    for i in range(1, n):
        dm = m[i] - m[i - 1]
        E_int += u[i] * dm
        rmid = 0.5 * (r[i] + r[i - 1])
        mmid = 0.5 * (m[i] + m[i - 1])
        E_pot += -mmid * dm / rmid
    return E_int + E_pot, E_int, E_pot


@njit(cache=True)
def _evolve(r, m, rho, p, sigma_m, w, C, ml, mpl, ms, mps,
            eps_t, eps_r, t_end, rho_c_factor, kn_min, max_steps,
            snap_dlnrho, snap_dlnt, max_snaps,
            snap_t, snap_r, snap_rho, snap_p, snap_E,
            n_hydro_max, relax_prefac, implicit_flag):
    """Main evolution loop. Records snapshots when central density changes by
    snap_dlnrho (log) or time by snap_dlnt (relative). Returns
    (n_snaps, status, t, n_steps). status: 0=rho_c reached, 1=t_end,
    2=kn_min, 3=max_steps, 4=numerical failure."""
    n = r.shape[0]
    u = np.empty(n)
    v = np.empty(n)
    L = np.empty(n)
    Kn = np.empty(n)
    _derived(r, rho, p, sigma_m, w, C, ml, mpl, ms, mps, u, v, L, Kn)

    rho_c0 = rho[3]
    t = 0.0
    isnap = 0
    # record initial state
    E, _, _ = _total_energy(r, m, rho, p, u)
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
            # unconditionally stable step: dt from the energy-accuracy
            # criterion (using the current smooth luminosity), with growth
            # cap and nonlinear step rejection (frozen-K consistency).
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
        # hydrostatic adjustment until converged
        nh = 0
        while True:
            max_rel = _hydro_step(r, m, rho, p)
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
            E, _, _ = _total_energy(r, m, rho, p, u)
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

    # final snapshot
    E, _, _ = _total_energy(r, m, rho, p, u)
    snap_t[isnap] = t
    snap_r[isnap] = r
    snap_rho[isnap] = rho
    snap_p[isnap] = p
    snap_E[isnap] = E
    isnap += 1
    return isnap, status, t, step


# ----------------------------------------------------------------------------
# initial profiles


def nfw_rho(x):
    return 1.0 / (x * (1.0 + x) ** 2)


def nfw_mass(x):
    return np.log(1.0 + x) - x / (1.0 + x)


def cored_nfw_rho(x, rc):
    """NFW with a core: rho = 1 / ((x^b + rc^b)^(1/b) (1+x)^2), b=4 smooth core."""
    b = 4.0
    return 1.0 / ((x ** b + rc ** b) ** (1.0 / b) * (1.0 + x) ** 2)


def profile_from_rho(rho_func, r_min=0.01, r_max=100.0, n_shells=200,
                     p_rmax_factor=10.0):
    """Build (r, m, rho, p) arrays for an arbitrary density profile function.
    Pressure from hydrostatic integration p(x) = int_x^xmax m rho / r^2 dr."""
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_shells)
    r_mid = np.empty(n_shells)
    r_mid[0] = r[0] / 2.0
    r_mid[1:] = 0.5 * (r[:-1] + r[1:])

    mass_int = lambda x: x * x * rho_func(x)
    m = np.empty(n_shells)
    acc = integrate.quad(mass_int, 0.0, r[0])[0]
    m[0] = acc
    for i in range(1, n_shells):
        acc += integrate.quad(mass_int, r[i - 1], r[i])[0]
        m[i] = acc

    rho = np.empty(n_shells)
    rho[0] = 3.0 * m[0] / r[0] ** 3
    rho[1:] = rho_func(r_mid[1:])

    def m_of(x):
        return integrate.quad(mass_int, 0.0, x)[0]

    p_int = lambda x: m_of(x) * rho_func(x) / (x * x)
    xmax = p_rmax_factor * r_max
    p = np.empty(n_shells)
    for i in range(n_shells):
        p[i] = integrate.quad(p_int, r_mid[i], xmax, limit=200)[0]
    return r, m, rho, p


def nfw_profile(r_min=0.01, r_max=100.0, n_shells=200):
    """Analytic-mass NFW initialization (pressure numeric)."""
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_shells)
    r_mid = np.empty(n_shells)
    r_mid[0] = r[0] / 2.0
    r_mid[1:] = 0.5 * (r[:-1] + r[1:])
    m = nfw_mass(r)
    rho = np.empty(n_shells)
    rho[0] = 3.0 * m[0] / r[0] ** 3
    rho[1:] = nfw_rho(r_mid[1:])
    p_int = lambda x: nfw_mass(x) * nfw_rho(x) / (x * x)
    xmax = 10.0 * r_max
    p = np.empty(n_shells)
    for i in range(n_shells):
        p[i] = integrate.quad(p_int, r_mid[i], xmax, limit=200)[0]
    return r, m, rho, p


# ----------------------------------------------------------------------------
# driver


MODEL_IDS = {'constant': 0, 'powerlaw': 1, 'yukawa_tu': 2, 'yukawa_t': 3}


def model_params(model, n=0.0, yb_index=5):
    mid = MODEL_IDS[model]
    if mid == 0:
        return mid, np.zeros(4)
    if mid == 1:
        return mid, np.array([n, 0., 0., 0.])
    tab = YB_TU if mid == 2 else YB_T
    return mid, np.array(tab[yb_index])


def run_halo(sigma_m, w=1.0, model_lmfp='constant', model_smfp='constant',
             C=C_DEFAULT, r=None, m=None, rho=None, p=None,
             n_shells=200, r_min=0.01, r_max=100.0,
             eps_t=1e-4, eps_r=1e-14, t_end=np.inf, rho_c_factor=1e4,
             kn_min=0.0, max_steps=200_000_000, snap_dlnrho=0.02,
             snap_dlnt=0.05, max_snaps=2000, n_hydro_max=50,
             powerlaw_n=0.0, yb_index=5, timestep_mode='classic'):
    """Evolve a halo; returns dict of snapshots and metadata."""
    if r is None:
        r, m, rho, p = nfw_profile(r_min, r_max, n_shells)
    r, m, rho, p = (np.ascontiguousarray(x, dtype=np.float64).copy()
                    for x in (r, m, rho, p))
    n = r.shape[0]
    ml, mpl = model_params(model_lmfp, powerlaw_n, yb_index)
    ms, mps = model_params(model_smfp, powerlaw_n, yb_index)

    implicit_flag = 0
    if timestep_mode == 'implicit':
        implicit_flag = 1
        relax_prefac = 1.0
    elif timestep_mode == 'classic':
        relax_prefac = 1.0
    elif timestep_mode == 'physical':
        # local LMFP relaxation time 2/(3 a C sigma rho v); F(v/w) <= its
        # maximum over the halo is absorbed conservatively by using F at
        # v = 1 (scale velocity). Capped for very weak coupling.
        F1 = _F_elastic(1.0 / w, ml, mpl)
        relax_prefac = min(2.0 / (3.0 * A_COEF * C * sigma_m * max(F1, 1e-12)),
                           1e4)
    else:
        raise ValueError(timestep_mode)

    snap_t = np.zeros(max_snaps)
    snap_r = np.zeros((max_snaps, n))
    snap_rho = np.zeros((max_snaps, n))
    snap_p = np.zeros((max_snaps, n))
    snap_E = np.zeros(max_snaps)

    n_snaps, status, t, steps = _evolve(
        r, m, rho, p, float(sigma_m), float(w), float(C), ml, mpl, ms, mps,
        float(eps_t), float(eps_r), float(t_end), float(rho_c_factor),
        float(kn_min), int(max_steps), float(snap_dlnrho), float(snap_dlnt),
        int(max_snaps), snap_t, snap_r, snap_rho, snap_p, snap_E,
        int(n_hydro_max), float(relax_prefac), implicit_flag)

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
                   'powerlaw_n': float(powerlaw_n),
                   'timestep_mode': timestep_mode},
    }
