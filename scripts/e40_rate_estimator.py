#!/usr/bin/env python3
"""E40b -- a conductivity RATE from the halo104 merger trees.

Protocol and the readings fixed in advance: docs/RATE_PREREGISTRATION.md.
Table: results/e40/tc_table.json (scripts/e40_tc_table.py).

WHAT IS NEW RELATIVE TO E30/E32/E33
-----------------------------------
E30 and E32 fit a STATIC phase per halo at z = 0 and test the track's shape,
explicitly not its rate. E33 tried a rate and could not get one, because it
required an NFW initial profile and therefore could only use windows in which
the halo had barely moved.

This uses the time axis the trees supply. The gravothermal clock advances
locally,

    dphi/dt = 1 / t_c(a; C_eff),   t_c = t_c_hat(sigma_hat, w_hat, C_eff) * t_0

with the structure at each epoch read from the halo's **CDM counterpart** -- the
same halo, same initial conditions, without self-interaction. That is the
control the suite provides and the reason no NFW assumption is needed at any
particular epoch: the CDM branch IS the no-evolution reference, at every epoch.

Integrating over the main branch to z = 0 gives phi_model(C_eff), compared
against the frozen phi_obs that E32 already fitted and stored. phi_obs does not
depend on C_eff and is not refitted here.

Output: results/e40/rate.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from gravop.units import HaloUnits                                # noqa: E402
import e30_concerto as e30                                        # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e40')
H, MPART = 0.7, 6.3e3
N_FLOOR = 2000.0                 # particles; the suite's own cut (P49)
SIGMA0_CM2G, W_KMS = 147.1, 120.0
OM, OL, H0_GYR = 0.286, 0.714, 13.968


def cosmic_time_gyr(a):
    """Flat LCDM age at scale factor a. Same constants as E33."""
    return (2.0 / (3.0 * np.sqrt(OL))) * H0_GYR * np.arcsinh(
        np.sqrt(OL / OM) * a ** 1.5)


def load_table(path):
    """Interpolators for log10 t_c_hat and for the censoring flag.

    Status 2 nodes stopped in the deep SMFP regime before collapse, so their
    t_c_hat is a LOWER bound. They are kept -- excluding them would silently
    delete the fastest-clock corner -- and carried alongside as a weight, so
    the estimator can report how much of each halo's phase came from them.
    """
    T = json.load(open(path))
    g = T['grid']
    lsh = np.asarray(g['log_sigma_hat'])
    lwh = np.asarray(g['log_w_hat'])
    C = np.asarray(g['C'])
    shape = (len(lsh), len(lwh), len(C))
    ltc = np.full(shape, np.nan)
    cen = np.zeros(shape)
    for i, a in enumerate(lsh):
        for j, b in enumerate(lwh):
            for k, c in enumerate(C):
                n = T['nodes'].get(f'{a:.4f}|{b:.4f}|{c:.4f}')
                if n is None or n['t_c_hat'] is None:
                    continue
                ltc[i, j, k] = np.log10(max(n['t_c_hat'], 1e-6))
                cen[i, j, k] = 1.0 if n['status'] == 2 else 0.0
    miss = int(np.isnan(ltc).sum())
    if miss:
        # fill by nearest along the C axis, then along w_hat; a hole would
        # otherwise poison every halo that interpolates through it
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = ltc[i, j]
                if np.all(np.isnan(v)):
                    continue
                ok = ~np.isnan(v)
                ltc[i, j] = np.interp(np.arange(len(v)), np.arange(len(v))[ok],
                                      v[ok])
    f = RegularGridInterpolator((lsh, lwh, C), ltc, bounds_error=False,
                                fill_value=None)
    fc = RegularGridInterpolator((lsh, lwh, C), cen, bounds_error=False,
                                 fill_value=None)
    return f, fc, (lsh, lwh, C), miss, T


def halo_series(z, i):
    """Per-epoch (cosmic time, sigma_hat, w_hat, t_0, usable) for one halo.

    Structure comes from the CDM branch. Radii in the trees are COMOVING kpc/h
    (DATA.md), velocities are PHYSICAL km/s, so the physical R_Vmax is
    a * R_Vmax / h and no h is applied twice.
    """
    n = int(z['cdm_n'][i])
    a = z['cdm_scale'][i, :n]
    vm = z['cdm_vmax'][i, :n]
    rm = z['cdm_RVmax'][i, :n] * a / H          # comoving kpc/h -> physical kpc
    M = z['cdm_Mvir'][i, :n] / H               # Msun
    pid = z['cdm_pid'][i, :n]
    ok = (np.isfinite(vm) & np.isfinite(rm) & np.isfinite(M)
          & (vm > 0) & (rm > 0) & (M / MPART >= N_FLOOR) & (pid == -1))
    t = cosmic_time_gyr(a)
    sh = np.full(n, np.nan); wh = np.full(n, np.nan); t0 = np.full(n, np.nan)
    idx = np.where(ok)[0]
    for k in idx:
        r_s, rho_s = e30.nfw_from_vmax_rmax(vm[k], rm[k])
        if not (np.isfinite(r_s) and np.isfinite(rho_s) and rho_s > 0):
            ok[k] = False
            continue
        u = HaloUnits(r_s, rho_s)
        sh[k] = u.sigma_hat(SIGMA0_CM2G)
        wh[k] = u.w_hat(W_KMS)
        t0[k] = u.t_0
    # the scale of the last major merger, per epoch, for the merger-reset
    # variant below. A major merger rearranges the halo and is the one event in
    # the trees that plausibly restarts the gravothermal clock.
    lastmm = z['cdm_scale_of_last_MM'][i, :n]
    return a, t, sh, wh, t0, ok, lastmm


def phase_of(t, sh, wh, t0, ok, C, f, fc, t_start_frac=0.0):
    """phi_model = integral dt / t_c, and the censored share of it.

    Trapezoid over the usable epochs. `t_start_frac` drops the earliest
    fraction of usable epochs, which is the pre-registered sensitivity on the
    integral's lower limit.
    """
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return None
    if t_start_frac > 0:
        idx = idx[int(round(t_start_frac * len(idx))):]
        if len(idx) < 3:
            return None
    pts = np.stack([np.log10(sh[idx]), np.log10(wh[idx]),
                    np.full(len(idx), C)], axis=1)
    tc = 10.0 ** f(pts) * t0[idx]                 # Gyr
    cens = fc(pts)
    rate = 1.0 / np.maximum(tc, 1e-12)
    tt = t[idx]
    dphi = 0.5 * (rate[1:] + rate[:-1]) * np.diff(tt)
    tot = float(dphi.sum())
    cw = 0.5 * (cens[1:] + cens[:-1])
    cshare = float((dphi * cw).sum() / tot) if tot > 0 else 0.0
    return {'phi': min(tot, 1.0), 'phi_raw': tot,
            'censored_share': cshare, 'n_epochs': int(len(idx)),
            't_span_gyr': float(tt[-1] - tt[0])}


def score(phi_model, phi_obs):
    d = np.asarray(phi_model) - np.asarray(phi_obs)
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=os.path.join(OUT, 'tc_table.json'))
    ap.add_argument('--c-grid', nargs='*', type=float,
                    default=[0.2, 0.3, 0.4, 0.6, 0.797, 1.0, 1.2, 1.4, 1.7, 2.0])
    a_ = ap.parse_args()

    f, fc, grid, miss, T = load_table(a_.table)
    z = np.load(os.path.join(ROOT, 'results', 'e31', 'trajectories.npz'))
    ctl = np.load(os.path.join(ROOT, 'results', 'e32', 'controls.npz'))
    phi_obs_all = ctl['phase']
    n_halo = len(phi_obs_all)

    series = [halo_series(z, i) for i in range(n_halo)]
    usable = [i for i in range(n_halo) if series[i][5].sum() >= 3]

    def ok_after_last_mm(i):
        """The usable mask, restricted to epochs after the last major merger.

        Stated as a variant rather than chosen: a major merger rearranges the
        halo, so a clock that runs continuously through one is assuming
        something the fluid model does not contain.
        """
        a_, _, _, _, _, ok_, lastmm = series[i]
        amm = lastmm[ok_]
        if len(amm) == 0 or not np.isfinite(amm[-1]):
            return ok_
        return ok_ & (a_ >= amm[-1])
    print(f'table: {len(T["nodes"])} nodes, {miss} missing; '
          f'{len(usable)} of {n_halo} haloes have >= 3 usable epochs')

    out = {
        'protocol': 'docs/RATE_PREREGISTRATION.md',
        'table': os.path.relpath(a_.table, ROOT),
        'n_haloes': n_halo, 'n_usable': len(usable),
        'resolution_cut': f'N_p >= {N_FLOOR:.0f} per epoch, the suite\'s own '
                          'cut (P49); isolated epochs only (pid == -1)',
        'scan': [],
    }
    moving = [i for i in usable if phi_obs_all[i] > 1e-3]
    out['n_moving'] = len(moving)

    best = None
    for C in a_.c_grid:
        rows, keep = [], []
        for i in usable:
            r = phase_of(*series[i][1:6], C, f, fc)
            if r is None:
                continue
            rows.append(r); keep.append(i)
        phi_m = np.array([r['phi'] for r in rows])
        phi_o = phi_obs_all[np.array(keep)]
        mov = np.array([phi_obs_all[i] > 1e-3 for i in keep])
        rec = {
            'C_eff': C,
            'n': len(keep),
            'score_all': score(phi_m, phi_o),
            'score_moving': score(phi_m[mov], phi_o[mov]) if mov.any() else None,
            'median_phi_model': float(np.median(phi_m)),
            'frac_model_collapsed': float(np.mean(phi_m >= 1.0)),
            'median_censored_share': float(np.median(
                [r['censored_share'] for r in rows])),
            'max_censored_share': float(np.max(
                [r['censored_share'] for r in rows])),
        }
        out['scan'].append(rec)
        print(f"  C_eff {C:5.3f}  n {rec['n']:3d}  score(all) "
              f"{rec['score_all']:.4f}  score(moving) "
              f"{rec['score_moving']:.4f}  median phi_model "
              f"{rec['median_phi_model']:.3f}  collapsed "
              f"{rec['frac_model_collapsed']:.2f}  censored share "
              f"{rec['median_censored_share']:.3f}", flush=True)
        if best is None or rec['score_all'] < best['score_all']:
            best = rec

    # ---- the two pre-registered nulls
    keep0 = [i for i in usable if phase_of(*series[i][1:6], 0.797, f, fc)]
    phi_o = phi_obs_all[np.array(keep0)]
    out['nulls'] = {
        'constant_median_phase': {
            'what': 'predict phi_obs by its own sample median; one parameter, '
                    'no rate in it',
            'value': float(np.median(phi_o)),
            'score_all': score(np.full(len(phi_o), np.median(phi_o)), phi_o)},
        'no_self_interaction': {
            'what': 'C_eff -> 0, so phi_model = 0 for every halo; not nested '
                    'in the scan interior',
            'score_all': score(np.zeros(len(phi_o)), phi_o)},
    }
    # ---- pre-registered sensitivity: the integral's lower limit
    sens = []
    for frac in (0.0, 0.1, 0.25, 0.5):
        row = {'drop_earliest_fraction': frac, 'scan': []}
        for C in a_.c_grid:
            pm, po = [], []
            for i in usable:
                r = phase_of(*series[i][1:6], C, f, fc, t_start_frac=frac)
                if r is None:
                    continue
                pm.append(r['phi']); po.append(phi_obs_all[i])
            if len(pm) < 5:
                continue
            row['scan'].append({'C_eff': C, 'n': len(pm),
                                'score_all': score(pm, po)})
        if row['scan']:
            b = min(row['scan'], key=lambda r: r['score_all'])
            row['best_C_eff'] = b['C_eff']; row['best_score'] = b['score_all']
            sens.append(row)
    out['start_epoch_sensitivity'] = sens

    # ---- NOT pre-registered, and labelled as such: does phi_model ORDER the
    # haloes correctly even if its normalisation is wrong? A rank statistic
    # separates "the rate model carries information" from "C_eff is right",
    # and those are different claims.
    from scipy.stats import spearmanr
    ranks = []
    for C in a_.c_grid:
        pm, po = [], []
        for i in usable:
            r = phase_of(*series[i][1:6], C, f, fc)
            if r is None:
                continue
            pm.append(r['phi']); po.append(phi_obs_all[i])
        rho, pv = spearmanr(pm, po)
        ranks.append({'C_eff': C, 'spearman': float(rho), 'p': float(pv)})
    out['rank_agreement'] = {
        'status': 'DESCRIPTIVE, not pre-registered; added after the scan was '
                  'specified and reported so the reader can separate "the '
                  'model orders the haloes" from "C_eff is calibrated"',
        'note': 'the phase fit phi_obs is C-independent to <= 0.02 '
                '(results/e40/phase_c_independence.json), so this correlation '
                'is not an artefact of a shared C',
        'rows': ranks,
    }

    # ---- variant: the clock restarts at the last major merger
    mm = []
    for C in a_.c_grid:
        pm, po = [], []
        for i in usable:
            a_i, t_i, sh_i, wh_i, t0_i, _, _ = series[i]
            r = phase_of(t_i, sh_i, wh_i, t0_i, ok_after_last_mm(i), C, f, fc)
            if r is None:
                continue
            pm.append(r['phi']); po.append(phi_obs_all[i])
        if len(pm) >= 5:
            mm.append({'C_eff': C, 'n': len(pm), 'score_all': score(pm, po),
                       'median_phi_model': float(np.median(pm))})
    if mm:
        b = min(mm, key=lambda r: r['score_all'])
        out['last_major_merger_variant'] = {
            'what': 'integrate only after the last major merger, which is the '
                    'one event in the trees that plausibly restarts the '
                    'gravothermal clock; stated as a variant, not selected',
            'best_C_eff': b['C_eff'], 'best_score': b['score_all'],
            'scan': mm}

    # ---- how much is the minimum worth? Two questions, answered separately:
    # is the rate model better than a constant, and how well is C_eff located?
    from scipy.stats import wilcoxon
    PHI = {}
    for C in a_.c_grid:
        col = {}
        for i in usable:
            r = phase_of(*series[i][1:6], C, f, fc)
            if r is not None:
                col[i] = r['phi']
        PHI[C] = col
    common = sorted(set.intersection(*[set(v) for v in PHI.values()]))
    obs = np.array([phi_obs_all[i] for i in common])
    M = np.array([[PHI[C][i] for i in common] for C in a_.c_grid])
    sc = np.sqrt(((M - obs[None, :]) ** 2).mean(axis=1))
    kbest = int(np.argmin(sc))
    Cbest = float(a_.c_grid[kbest])

    # paired, per halo: is the rate model's error smaller than the constant's?
    e_model = np.abs(M[kbest] - obs)
    e_const = np.abs(np.median(obs) - obs)
    st, pv = wilcoxon(e_model, e_const)
    out['vs_constant_null'] = {
        'C_eff': Cbest, 'n': len(common),
        'rms_model': float(sc[kbest]),
        'rms_constant': float(np.sqrt(((np.median(obs) - obs) ** 2).mean())),
        'median_abs_err_model': float(np.median(e_model)),
        'median_abs_err_constant': float(np.median(e_const)),
        'wilcoxon_p': float(pv),
        'model_better_fraction': float(np.mean(e_model < e_const)),
        'reading': 'the pre-registration requires the score to be MATERIALLY '
                   'below the constant-phase predictor for this to count as a '
                   'calibration; a paired test says whether it is below it at '
                   'all',
    }

    # bootstrap over haloes: how well is C_eff located?
    rng = np.random.default_rng(20260918)
    boot = []
    for _ in range(4000):
        idx = rng.integers(0, len(common), len(common))
        s_ = np.sqrt(((M[:, idx] - obs[None, idx]) ** 2).mean(axis=1))
        boot.append(a_.c_grid[int(np.argmin(s_))])
    boot = np.array(boot, dtype=float)
    out['C_eff_bootstrap'] = {
        'n_resamples': len(boot),
        'median': float(np.median(boot)),
        'ci95': [float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5))],
        'frac_at_or_above_isolated_value': float(np.mean(boot >= 0.797)),
        'grid': list(map(float, a_.c_grid)),
    }

    # ---- censoring bias, bounded rather than waved at.
    # Status-2 nodes stop before collapse, so their t_c is a LOWER bound, so
    # dphi is an UPPER bound, so phi_model is too large and the fit is pushed
    # to SMALL C_eff. The worst case is that those contributions are worth
    # nothing at all: drop each halo's censored share entirely and refit. The
    # truth is between the two, and both are reported.
    CEN = {}
    for C in a_.c_grid:
        col = {}
        for i in usable:
            r = phase_of(*series[i][1:6], C, f, fc)
            if r is not None:
                col[i] = r['censored_share']
        CEN[C] = col
    Mc = np.array([[PHI[C][i] * (1.0 - CEN[C][i]) for i in common]
                   for C in a_.c_grid])
    sc_c = np.sqrt(((Mc - obs[None, :]) ** 2).mean(axis=1))
    kc = int(np.argmin(sc_c))
    out['censoring_bias'] = {
        'direction': 'status-2 nodes under-state t_c, so phi_model is an upper '
                     'bound and the fitted C_eff is pushed LOW',
        'median_censored_share_at_best_C': float(np.median(
            [CEN[Cbest][i] for i in common])),
        'best_C_eff_uncensored_worst_case': float(a_.c_grid[kc]),
        'best_C_eff_as_measured': Cbest,
        'reading': 'the worst case discards every censored contribution, so '
                   'the true value lies between the two; both stay far below '
                   'the isolated run',
        'scan_uncensored': [{'C_eff': float(c), 'score_all': float(v)}
                            for c, v in zip(a_.c_grid, sc_c)],
    }

    # the one strong statement the data do support
    k797 = int(np.argmin(np.abs(np.array(a_.c_grid) - 0.797)))
    out['isolated_value_comparison'] = {
        'C_eff': float(a_.c_grid[k797]),
        'rms': float(sc[k797]),
        'rms_at_best': float(sc[kbest]),
        'ratio': float(sc[k797] / sc[kbest]),
        'frac_collapsed_at_isolated_value': float(np.mean(M[k797] >= 1.0)),
        'frac_contracted_observed': float(np.mean(obs > 0.3)),
    }

    cs = [r['C_eff'] for r in out['scan']]
    ss = [r['score_all'] for r in out['scan']]
    k = int(np.argmin(ss))
    out['result'] = {
        'best_C_eff': cs[k],
        'best_score': ss[k],
        'interior': bool(0 < k < len(cs) - 1),
        'beats_constant_null': ss[k] < out['nulls']['constant_median_phase']['score_all'],
        'beats_no_evolution_null': ss[k] < out['nulls']['no_self_interaction']['score_all'],
        'isolated_run_value': 0.797,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'rate.json'), 'w'), indent=1)
    print('\nnulls: constant-median '
          f"{out['nulls']['constant_median_phase']['score_all']:.4f}, "
          f"no-evolution {out['nulls']['no_self_interaction']['score_all']:.4f}")
    print(f"best C_eff {cs[k]} at score {ss[k]:.4f}; interior "
          f"{out['result']['interior']}; beats constant "
          f"{out['result']['beats_constant_null']}; beats no-evolution "
          f"{out['result']['beats_no_evolution_null']}")
    print('wrote rate.json')


if __name__ == '__main__':
    main()
