"""gravop.dataset — turn solver trajectories into fixed-grid training samples.

A sample = one solver run resampled onto:
  radial grid: N_R points, log10 r in [LOGR_MIN, LOGR_MAX]
  time grid:   N_TAU points in normalized time tau = t / t_c, where
               t_c := t(rho_c = 10^3 rho_c(0)) for collapsing runs,
               else t_end (flag no_collapse=1).
Fields stored: log10 rho(r, tau), log10 v(r, tau); plus initial profile
log10 rho_0(r), log10 p_0(r); params (log10 sigma0, log10 w, model);
scalar targets log10 t_c and no_collapse flag.

Lagrangian -> Eulerian: interpolate shell-midpoint (log r, log field) linearly;
inside the innermost midpoint, fields are held constant (flat core);
outside the outermost midpoint, log-log linear extrapolation from last two shells.
Time interpolation: linear in t on log-fields (snapshots are dense in
d ln rho_c <= 0.02, so inter-snapshot changes are ~ percent-level).
"""
import numpy as np

LOGR_MIN, LOGR_MAX, N_R = -2.5, 2.0, 128
N_TAU = 48
RHO_COLLAPSE_FACTOR = 1e3


def r_grid():
    return np.logspace(LOGR_MIN, LOGR_MAX, N_R)


def tau_grid():
    """24 uniform points in [0, 0.92], 24 in (0.92, 1.0]."""
    a = np.linspace(0.0, 0.92, 24, endpoint=False)
    b = np.linspace(0.92, 1.0, 24)
    return np.concatenate([a, b])


def shell_midpoints(r):
    rm = np.empty_like(r)
    rm[0] = 0.5 * r[0]
    rm[1:] = 0.5 * (r[:-1] + r[1:])
    return rm


def regrid_snapshot(r_shells, rho, p, rg=None):
    """Interpolate one snapshot onto the fixed Eulerian grid.
    Returns (log10 rho, log10 v) arrays of length N_R."""
    if rg is None:
        rg = r_grid()
    rm = shell_midpoints(r_shells)
    lr = np.log10(rm)
    lrho = np.log10(rho)
    lv = 0.5 * (np.log10(p) - np.log10(rho))
    lg = np.log10(rg)
    out_rho = np.interp(lg, lr, lrho)  # np.interp clamps at ends = flat core
    out_v = np.interp(lg, lr, lv)
    # log-log linear extrapolation outward (np.interp clamps; replace tail)
    mask = lg > lr[-1]
    if mask.any():
        s_rho = (lrho[-1] - lrho[-2]) / (lr[-1] - lr[-2])
        s_v = (lv[-1] - lv[-2]) / (lr[-1] - lr[-2])
        out_rho[mask] = lrho[-1] + s_rho * (lg[mask] - lr[-1])
        out_v[mask] = lv[-1] + s_v * (lg[mask] - lr[-1])
    return out_rho, out_v


def t_collapse(res, factor=RHO_COLLAPSE_FACTOR):
    rhoc = res['rho'][:, 3]
    target = factor * rhoc[0]
    if rhoc.max() < target:
        return None
    idx = int(np.argmax(rhoc >= target))
    if idx == 0:
        return 0.0
    t0, t1 = res['t'][idx - 1], res['t'][idx]
    l0, l1 = np.log(rhoc[idx - 1]), np.log(rhoc[idx])
    return float(t0 + (t1 - t0) * (np.log(target) - l0) / (l1 - l0))


def sample_from_run(res):
    """Build one training sample dict from a solver result.

    Time normalization: t_norm = the run's termination time t_final, which is
    t(rho_c = 10^3 rho_c,0) for collapse-terminated runs (status 0),
    t(Kn_min = floor) for runs cut at the fluid-validity boundary (status 2),
    or t_end for runs that do neither (status 1, flagged no_collapse)."""
    rg = r_grid()
    taus = tau_grid()
    no_collapse = res['status'] == 1
    t_norm = res['t_final']

    # regrid every snapshot once
    n_snap = len(res['t'])
    lrho_s = np.empty((n_snap, N_R), dtype=np.float32)
    lv_s = np.empty((n_snap, N_R), dtype=np.float32)
    for i in range(n_snap):
        a, b = regrid_snapshot(res['r'][i], res['rho'][i], res['p'][i], rg)
        lrho_s[i] = a
        lv_s[i] = b

    # time interpolation onto tau grid
    ts = res['t']
    t_targets = np.clip(taus * t_norm, 0.0, ts[-1])
    lrho = np.empty((N_TAU, N_R), dtype=np.float32)
    lv = np.empty((N_TAU, N_R), dtype=np.float32)
    idx = np.searchsorted(ts, t_targets, side='right') - 1
    idx = np.clip(idx, 0, n_snap - 2)
    w = (t_targets - ts[idx]) / np.maximum(ts[idx + 1] - ts[idx], 1e-300)
    w = np.clip(w, 0.0, 1.0)[:, None].astype(np.float32)
    lrho = (1 - w) * lrho_s[idx] + w * lrho_s[idx + 1]
    lv = (1 - w) * lv_s[idx] + w * lv_s[idx + 1]

    p = res['params']
    return {
        'lrho': lrho, 'lv': lv,
        'lrho0': lrho_s[0], 'lv0': lv_s[0],
        'log_sigma0': np.float32(np.log10(p['sigma_m'])),
        'log_w': np.float32(np.log10(p['w'])),
        'model_id': np.int8(0 if p['model_lmfp'] == 'constant' else 1),
        'log_tc': np.float32(np.log10(t_norm)),
        'no_collapse': np.int8(no_collapse),
        'status': np.int8(res['status']),
    }
