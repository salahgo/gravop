"""P28 — SIDM Concerto as a second N-body check, and what it can and cannot do.

THE ANSWER TO THE QUESTION ASKED, FIRST
---------------------------------------
P28 asked for a **second calibration of the conductivity constant** `C_eff`,
which is currently fitted to a single isolated N-body run (Fischer, Yu & Dolag;
`C_eff = 0.797`). **The public SIDM Concerto release cannot provide one.**
`C_eff` sets the gravothermal *rate*, so calibrating it requires the halo's
evolution in time, and the release contains:

  * particle snapshots for **snapshot_235 only** (z = 0) -- verified by listing
    the archive, not assumed;
  * Rockstar `hlist` catalogues for **a = 1.00000 only**;
  * parametric-model outputs carrying a phase `tr1 = t/t_c` but **no absolute
    time** (the column named `mint` is a mass -- it equals `mcdm` -- not an
    integration time);
  * and the merger trees are withheld: "The RCT tree*.dat files are too large
    to upload here; please contact the authors."

No trees means no halo ages, and one snapshot means no evolution. The
single-calibration limitation therefore **stands**, and closing it needs either
the Concerto trees from the authors or a suite released with time series.

WHAT THE RELEASE *CAN* DO, AND THIS SCRIPT DOES
-----------------------------------------------
A **structural** test, which is independent of `C_eff` by construction and is
worth having on its own. For each matched CDM->SIDM halo pair, the gravothermal
fluid model predicts a one-parameter family of profiles indexed by phase. We
ask whether the *observed* z = 0 SIDM structure lies on that family:

  1. take the CDM halo's NFW from (vmax, rmax);
  2. convert GroupSIDM (sigma_0/m = 147.1 cm^2/g, w = 120 km/s) into this
     halo's dimensionless (sigma_hat, w_hat);
  3. evolve with our solver and read (vmax, rmax) along the trajectory;
  4. find the phase minimising the distance in (log vmax, log rmax) to the
     measured SIDM values, and report the residual.

Because the phase is fitted per halo, this tests the **track shape** -- the
locus the fluid model says an SIDM halo must move along -- and NOT the rate. It
is a genuine independent check: a different N-body suite, a different
(velocity-dependent) cross-section from the constant one Fischer used, and 485
haloes over four decades in mass against Fischer's one.

Controls, both necessary:
  * the **CDM null**: the residual if the halo had not evolved at all. If our
    track is no better than "nothing happened", it has explained nothing.
  * **isolated haloes only** (`pid == -1`, cross-matched into the SIDM hlist),
    because subhaloes are tidally stripped and our model has no tides.

Output: results/e30/concerto.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from gravop import solver                                         # noqa: E402
from gravop.units import HaloUnits, G_KPC                         # noqa: E402
from sc_provenance import stamp                                   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e30')
BASE = os.path.join(ROOT, 'data', 'd3_concerto', 'x', 'central', 'groups',
                    'carnegie_poc', 'enadler', 'zoomins')
PARAM = os.path.join(BASE, 'sidm_concerto', 'parametric', 'concertoSIDM',
                     'LMC_Halo104_GroupSIDM', 'log.txt')
OUT_SIDM = os.path.join(BASE, 'sidm_concerto', 'parametric', 'concertoSIDM',
                        'LMC_Halo104_GroupSIDM', 'out_sidm.txt')

# Nadler et al. 2025, table of cross-section models
SIGMA0_CM2G, W_KMS = 147.1, 120.0
X_VMAX = 2.16258            # argmax of mu(x)/x for NFW
NFW_VMAX_COEF = None        # filled below


def _nfw_vmax_coef():
    mu = np.log(1 + X_VMAX) - X_VMAX / (1 + X_VMAX)
    return np.sqrt(mu / X_VMAX)


NFW_VMAX_COEF = _nfw_vmax_coef()


def nfw_from_vmax_rmax(vmax, rmax):
    """(r_s, rho_s) from (vmax, rmax), both physical (kpc, km/s)."""
    r_s = rmax / X_VMAX
    v_0 = vmax / NFW_VMAX_COEF
    rho_s = v_0 ** 2 / (4.0 * np.pi * G_KPC * r_s ** 2)
    return r_s, rho_s


def track_vmax_rmax(res):
    """(vmax_hat, rmax_hat) per snapshot, in solver units.

    v_circ(x) = v_0 sqrt(m_hat / x), so the maximum over shells is taken on
    m_hat / x and converted once.
    """
    r, m = res['r'], res['m']
    vc2 = m[None, :] / r                       # (n_snap, n_shell)
    j = np.argmax(vc2, axis=1)
    i = np.arange(r.shape[0])
    return np.sqrt(vc2[i, j]), r[i, j]


# out_sidm.txt columns (its header names `order` twice; positions are stable)
OUT_SIDM_ORDER, OUT_SIDM_VMAX, OUT_SIDM_RMAX, OUT_SIDM_PID = 0, 9, 10, 15


def load_pairs():
    """Matched CDM->SIDM pairs, with the isolation flag, in PHYSICAL units.

    UNITS, established from the data rather than assumed. `out_sidm.txt` and
    `log.txt` list the same haloes -- `vd100id` indexes `out_sidm`'s `order`,
    and their vmax columns agree to 0.0 relative error across all 485, which
    confirms the row matching. Their rmax columns differ by a ratio of
    **exactly 0.700000** for every halo, so `out_sidm` is in kpc/h with
    h = 0.7 and `log.txt` is already PHYSICAL kpc. An earlier version of this
    function divided `log.txt` by h as well, which double-corrected and would
    have scaled every sigma_hat by 1/0.7 = 1.43.

    The isolation flag comes from `out_sidm`'s `pid` column (-1 = not inside a
    larger halo). The Rockstar `hlist` cannot supply it: its `id` column is a
    different numbering from `vd100id`, and cross-matching on it returns zero
    haloes -- checked, not assumed.
    """
    names = [c.split('(')[0] for c in
             open(PARAM).readline().strip('#').split()]
    d = np.loadtxt(PARAM, comments='#')
    idx = {n: i for i, n in enumerate(names)}

    o = np.genfromtxt(OUT_SIDM, comments='#')
    row_of = {int(k): j for j, k in enumerate(o[:, OUT_SIDM_ORDER])}
    rows = [row_of[int(v)] for v in d[:, idx['vd100id']]]
    vmax_check = o[rows, OUT_SIDM_VMAX]
    assert np.nanmax(np.abs(vmax_check / d[:, idx['vd100Vmaxz0']] - 1)) < 1e-6, (
        'log.txt and out_sidm.txt disagree on vmax: the row matching by '
        'vd100id -> order is wrong and the isolation flag cannot be trusted')
    h_obs = float(np.nanmedian(o[rows, OUT_SIDM_RMAX]
                               / d[:, idx['vd100Rmaxz0']]))
    return {
        'cdm_vmax': d[:, idx['cdmVmaxz0']],
        'cdm_rmax': d[:, idx['cdmRmaxz0']],          # physical kpc
        'sidm_vmax': d[:, idx['vd100Vmaxz0']],
        'sidm_rmax': d[:, idx['vd100Rmaxz0']],       # physical kpc
        'pid': o[rows, OUT_SIDM_PID],
        'tr1': d[:, idx['tr1']],
        'mcdm': d[:, idx['mcdm']],
        'h_observed': h_obs,
    }


def one_halo(vmax_c, rmax_c, vmax_s, rmax_s, c_eff, n_shells=150,
             rho_c_factor=300.0):
    """Best-fit phase on our track, and the residual there."""
    r_s, rho_s = nfw_from_vmax_rmax(vmax_c, rmax_c)
    u = HaloUnits(r_s, rho_s)
    sh, wh = u.sigma_hat(SIGMA0_CM2G), u.w_hat(W_KMS)
    try:
        res = solver.run_halo(
            sigma_m=sh, w=wh, model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
            C=c_eff, n_shells=n_shells, eps_t=3e-4,
            rho_c_factor=rho_c_factor, t_end=1e4, kn_min=0.02,
            max_steps=20_000_000, max_snaps=1200,
            timestep_mode='implicit')
    except Exception:
        return None
    if res['status'] in (3, 4) or len(res['t']) < 5:
        return None
    vm_hat, rm_hat = track_vmax_rmax(res)
    vm = u.v_0 * vm_hat
    rm = r_s * rm_hat
    # distance in log-log; both axes matter and both are multiplicative
    d2 = (np.log10(vm / vmax_s) ** 2 + np.log10(rm / rmax_s) ** 2)
    k = int(np.argmin(d2))
    null = np.sqrt(np.log10(vmax_c / vmax_s) ** 2
                   + np.log10(rmax_c / rmax_s) ** 2)
    return {
        'sigma_hat': float(sh), 'w_hat': float(wh),
        'r_s_kpc': float(r_s), 'rho_s': float(rho_s),
        'best_phase': float(res['t'][k] / res['t'][-1]),
        'resid_dex': float(np.sqrt(d2[k])),
        'resid_vmax_dex': float(np.log10(vm[k] / vmax_s)),
        'resid_rmax_dex': float(np.log10(rm[k] / rmax_s)),
        'null_resid_dex': float(null),
        'improvement': float(null / np.sqrt(d2[k])) if d2[k] > 0 else None,
        'status': int(res['status']),
        'n_snap': int(len(res['t'])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120,
                    help='haloes to evaluate (0 = all)')
    ap.add_argument('--c-eff', type=float, default=0.797)
    ap.add_argument('--min-vmax', type=float, default=8.0,
                    help='drop the poorly resolved tail; the suite resolves '
                         'haloes down to vmax ~3 km/s but an NFW fit there '
                         'rests on few particles')
    ap.add_argument('--isolated-only', action='store_true', default=True)
    ap.add_argument('--include-subhaloes', dest='isolated_only',
                    action='store_false')
    ap.add_argument('--seed', type=int, default=20260906)
    args = ap.parse_args()

    p = load_pairs()
    keep = np.isfinite(p['cdm_vmax']) & np.isfinite(p['sidm_vmax'])
    keep &= (p['cdm_vmax'] > args.min_vmax) & (p['cdm_rmax'] > 0)
    keep &= (p['sidm_vmax'] > 0) & (p['sidm_rmax'] > 0)
    n_before_iso = int(keep.sum())
    if args.isolated_only:
        keep &= (p['pid'] == -1)
    sel = np.nonzero(keep)[0]
    rng = np.random.default_rng(args.seed)
    if args.n and len(sel) > args.n:
        sel = np.sort(rng.choice(sel, args.n, replace=False))

    rows = []
    for k, i in enumerate(sel):
        r = one_halo(p['cdm_vmax'][i], p['cdm_rmax'][i],
                     p['sidm_vmax'][i], p['sidm_rmax'][i], args.c_eff)
        if r is None:
            continue
        r.update(cdm_vmax=float(p['cdm_vmax'][i]),
                 sidm_vmax=float(p['sidm_vmax'][i]),
                 mcdm=float(p['mcdm'][i]), tr1_theirs=float(p['tr1'][i]))
        rows.append(r)
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(sel)} done', flush=True)

    res_d = np.array([r['resid_dex'] for r in rows])
    null_d = np.array([r['null_resid_dex'] for r in rows])
    rec = {
        'question_asked': 'a second calibration of C_eff',
        'answer': ('NOT POSSIBLE from the public release: particle snapshots '
                   'are snapshot_235 (z=0) only, hlists are a=1.00000 only, '
                   'the parametric outputs carry a phase but no absolute time, '
                   'and the merger trees are withheld by the authors. C_eff '
                   'sets a rate and a rate needs time evolution. The '
                   'single-calibration limitation STANDS.'),
        'what_is_tested_instead': ('the structural track: does the observed '
                                   'z=0 SIDM (vmax, rmax) lie on the family '
                                   'our fluid model predicts? Phase is fitted '
                                   'per halo, so this is independent of C_eff '
                                   'and tests track SHAPE, not rate.'),
        'cross_section': {'model': 'GroupSIDM', 'sigma0_cm2_per_g': SIGMA0_CM2G,
                          'w_km_s': W_KMS,
                          'form': 'Yukawa-Born, our yukawa_tu (Outmezguine+23)'},
        'h_observed_from_data': p['h_observed'],
        'c_eff': args.c_eff,
        'n_pairs_total': int(len(p['cdm_vmax'])),
        'n_after_cuts': n_before_iso,
        'isolated_only': bool(args.isolated_only),
        'n_evaluated': len(rows),
        'sigma_hat_range': [min(r['sigma_hat'] for r in rows),
                            max(r['sigma_hat'] for r in rows)],
        'w_hat_range': [min(r['w_hat'] for r in rows),
                        max(r['w_hat'] for r in rows)],
        'mass_range_msun': [min(r['mcdm'] for r in rows),
                            max(r['mcdm'] for r in rows)],
        'resid_dex_median': float(np.median(res_d)),
        'resid_dex_p90': float(np.quantile(res_d, 0.9)),
        'null_resid_dex_median': float(np.median(null_d)),
        'improvement_median': float(np.median(null_d / res_d)),
        'frac_better_than_null': float((res_d < null_d).mean()),
        'rows': rows,
        '_provenance': stamp('e30_concerto.py'),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'concerto.json')
    json.dump(rec, open(path, 'w'), indent=1)
    print(f"\nn={rows and len(rows)}  sigma_hat "
          f"{rec['sigma_hat_range'][0]:.2f}-{rec['sigma_hat_range'][1]:.2f}  "
          f"w_hat {rec['w_hat_range'][0]:.2f}-{rec['w_hat_range'][1]:.2f}")
    print(f"  our track residual  {rec['resid_dex_median']:.4f} dex (median), "
          f"p90 {rec['resid_dex_p90']:.4f}")
    print(f"  CDM null residual   {rec['null_resid_dex_median']:.4f} dex")
    print(f"  improvement         {rec['improvement_median']:.2f}x median, "
          f"better than null in {100*rec['frac_better_than_null']:.0f}%")
    print('wrote', path)


if __name__ == '__main__':
    main()
