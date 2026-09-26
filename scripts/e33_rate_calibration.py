"""E6/E7 -- a SECOND calibration of C_eff, from halo104 trajectories.

WHY THIS IS THE EXPERIMENT THAT MATTERS
---------------------------------------
E32 showed the z = 0 structural test cannot settle anything on its own: against
a fair one-parameter competitor (NFW, concentration free) the gravothermal track
is indistinguishable on the pooled 81 haloes, Wilcoxon p = 0.41. It wins by 4x
only among the 17 haloes that genuinely collapsed -- and those sit within four
softening lengths. One epoch is simply not enough information.

A trajectory is. `C_eff` sets the gravothermal RATE, and a rate is exactly what
a 236-epoch history constrains and a single snapshot cannot. This is the
measurement P28 declared impossible from the public release and that the
authors' trees now permit.

THE DESIGN, AND WHAT EACH CHOICE DEFENDS AGAINST
------------------------------------------------
* Quasi-static window. The fluid model describes an ISOLATED halo, so it is
  only entitled to epochs where the halo is not being rebuilt by accretion. We
  take the longest run of late epochs with |Mvir/Mvir(z=0) - 1| < 0.2, pid == -1
  and >= 1e4 particles. Median window is ~39 epochs.

* NFW initial condition, and the test that it is allowed. Integration starts
  from an NFW fit to the SIDM halo's own (vmax, RVmax) at the window start. That
  is only legitimate if SIDM has not yet driven the halo away from NFW, so we
  REQUIRE the SIDM and CDM counterparts to still agree there. Haloes already
  collapsed at window start are dropped rather than fitted with a profile that
  does not describe them.

* Fit the rate, not the phase. E30 fitted a free phase per halo, which is what
  made its null nested. Here C_eff is ONE number shared by every halo and every
  epoch; nothing is fitted per halo. A halo can therefore genuinely disagree.

* Metric follows project convention: RMS over epochs within a halo, then the
  unweighted median over haloes, so a long-lived halo does not outvote a short
  one.

PRE-REGISTERED INTERPRETATION (written before the scan was run)
---------------------------------------------------------------
If the profile likelihood has a clear interior minimum and the implied C_eff is
consistent across haloes, that is a second calibration. If the minimum is at a
grid edge, or C_eff trends with sigma_hat/w_hat, that is a NEGATIVE result: it
means one constant conductivity does not describe a cosmological population, and
it will be reported as such rather than tuned away. Both outcomes are publishable
and the choice between them is not ours to make after the fact.

Output: results/e33/rate_calibration.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import e30_concerto as e30                                        # noqa: E402
from gravop import solver                                         # noqa: E402
from gravop.units import HaloUnits                                # noqa: E402

H, MPART = 0.7, 6.3e3
OM, OL, H0_GYR = 0.286, 0.714, 13.968      # 1/H0 in Gyr for h = 0.7


def cosmic_time_gyr(a):
    """Flat LCDM age at scale factor a."""
    return (2.0 / (3.0 * np.sqrt(OL))) * H0_GYR * np.arcsinh(
        np.sqrt(OL / OM) * a ** 1.5)


def quasi_static_window(M, pid, n, tol=0.2, min_len=20):
    """Longest run of late epochs at near-constant mass, isolated, resolved."""
    M, pid = M[:n], pid[:n]
    ok = (np.abs(M / M[-1] - 1.0) < tol) & (pid == -1) & (M / (MPART * H) >= 1e4)
    i = n - 1
    while i >= 0 and ok[i]:
        i -= 1
    start = i + 1
    return (start, n) if (n - start) >= min_len else None


def halo_residual(traj, i, c_eff, max_snaps=1200):
    """RMS residual in (log vmax, log rmax) over a halo's window, or None."""
    n = int(traj['sidm_n'][i])
    win = quasi_static_window(traj['sidm_Mvir'][i], traj['sidm_pid'][i], n)
    if win is None:
        return None
    s, e = win
    a = traj['sidm_scale'][i, s:e]
    vm_obs = traj['sidm_vmax'][i, s:e]
    rm_obs = traj['sidm_RVmax'][i, s:e] / H          # comoving kpc/h -> physical
    rm_obs = rm_obs * a                              # comoving -> physical
    if not np.all(np.isfinite(vm_obs)) or np.any(rm_obs <= 0):
        return None

    # the NFW initial condition must still be a fair description at window start
    ac = traj['cdm_scale'][i, :int(traj['cdm_n'][i])]
    j = int(np.argmin(np.abs(ac - a[0])))
    vc = traj['cdm_vmax'][i, j]
    rc = traj['cdm_RVmax'][i, j] / H * ac[j]
    if not (np.isfinite(vc) and rc > 0):
        return None
    if abs(np.log10(vm_obs[0] / vc)) > 0.10 or abs(np.log10(rm_obs[0] / rc)) > 0.15:
        return None                                   # already collapsed: drop

    r_s, rho_s = e30.nfw_from_vmax_rmax(vm_obs[0], rm_obs[0])
    u = HaloUnits(r_s, rho_s)
    t_gyr = cosmic_time_gyr(a) - cosmic_time_gyr(a[0])
    t_hat = t_gyr / u.t_0
    try:
        res = solver.run_halo(
            sigma_m=u.sigma_hat(e30.SIGMA0_CM2G), w=u.w_hat(e30.W_KMS),
            model_lmfp='yukawa_tu', model_smfp='yukawa_tu', C=c_eff,
            n_shells=150, eps_t=3e-4, rho_c_factor=300.0,
            t_end=float(t_hat[-1]) * 1.05, kn_min=0.02,
            max_steps=20_000_000, max_snaps=max_snaps,
            timestep_mode='implicit')
    except Exception:
        return None
    if res['status'] in (3, 4) or len(res['t']) < 5:
        return None
    vm_hat, rm_hat = e30.track_vmax_rmax(res)
    tt = np.asarray(res['t'])
    if tt[-1] < 0.9 * t_hat[-1]:
        return None                                   # did not cover the window
    vm_p = np.interp(t_hat, tt, u.v_0 * vm_hat)
    rm_p = np.interp(t_hat, tt, r_s * rm_hat)
    d = np.sqrt(np.mean(np.log10(vm_p / vm_obs) ** 2
                        + np.log10(rm_p / rm_obs) ** 2))
    return {'resid': float(d), 'n_epoch': int(e - s),
            'a_start': float(a[0]), 'sigma_hat': float(u.sigma_hat(e30.SIGMA0_CM2G)),
            'w_hat': float(u.w_hat(e30.W_KMS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='0.2,2.0,13')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'e33'))
    args = ap.parse_args()
    lo, hi, k = args.grid.split(',')
    grid = np.linspace(float(lo), float(hi), int(k))

    traj = np.load(os.path.join(ROOT, 'results', 'e31', 'trajectories.npz'))
    nh = len(traj['sidm_n'])
    usable = [i for i in range(nh)
              if quasi_static_window(traj['sidm_Mvir'][i], traj['sidm_pid'][i],
                                     int(traj['sidm_n'][i])) is not None]
    print(f'haloes with a usable window: {len(usable)}/{nh}', flush=True)

    per_c, rows = {}, {}
    for c in grid:
        vals = []
        for i in usable:
            r = halo_residual(traj, i, float(c))
            if r is not None:
                vals.append(r['resid'])
                rows.setdefault(i, {})[float(c)] = r['resid']
        per_c[float(c)] = {'n': len(vals),
                           'median': float(np.median(vals)) if vals else None}
        print(f'  C_eff={c:.3f}  n={len(vals):3d}  median resid='
              f'{per_c[float(c)]["median"]}', flush=True)

    # A halo that the solver collapses before the window ends yields no residual,
    # and it collapses SOONER at larger C_eff. Taking the median over whatever
    # survived at each grid point would therefore compare different samples and
    # penalise large C_eff for a selection effect. Score only the haloes that
    # produced a residual at EVERY grid point.
    common = [i for i, v in rows.items() if len(v) == len(grid)]
    for c in grid:
        vals = [rows[i][float(c)] for i in common]
        per_c[float(c)]['median_common'] = (float(np.median(vals)) if vals else None)
        per_c[float(c)]['n_common'] = len(vals)
    ok = [(v['median_common'], c) for c, v in per_c.items()
          if v.get('median_common') is not None]
    best = min(ok)[1] if ok else None
    out = {'question': 'a second calibration of C_eff, from trajectories',
           'grid': [float(g) for g in grid],
           'per_c_eff': per_c,
           'n_haloes_with_window': len(usable),
           'n_haloes_scored_at_every_c_eff': len(common),
           'scoring_note': ('median taken over the haloes admissible at EVERY '
                            'grid point, so the sample does not change with C_eff'),
           'fischer_c_eff': 0.797,
           'best_c_eff': best,
           'at_grid_edge': bool(best in (grid[0], grid[-1])) if best else None,
           'per_halo': {str(i): v for i, v in rows.items()}}
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, 'rate_calibration.json'), 'w'),
              indent=2)
    print('best C_eff:', best, '| Fischer 0.797 |',
          'AT GRID EDGE - treat as a bound, not a calibration' if out['at_grid_edge']
          else 'interior minimum')


if __name__ == '__main__':
    sys.exit(main())
