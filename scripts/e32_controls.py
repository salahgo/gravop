"""E2-E4 -- controls the E30 structural test never had.

E2  Environment. Does the structural residual track merger history, accretion
    rate or tidal force? DATA.md sets this suite aside for calibration
    because cosmological zoom-ins mix conductivity with environment. That was a
    stated worry, never a measurement. The z=0 Rockstar catalogue carries
    `scale_of_last_MM`, four `Acc_Rate_*` columns, `Tidal_Force_Tdyn`,
    `Mpeak`, `Vpeak` and `Halfmass_Scale`, so it can now be measured.

E3  A NON-NESTED baseline. E30 compared its fitted track against "the halo did
    not evolve". That null is the phase-0 member of the track itself, so
    `resid <= null` holds by construction and `frac_better_than_null = 1.0` is
    arithmetic rather than evidence -- 23 of 81 haloes sit exactly on it.
    The honest competitor has the SAME number of free parameters: hold Mvir and
    Rvir at their measured CDM values and let the halo stay NFW while its
    CONCENTRATION changes. One parameter, like the phase, and it is not nested
    in the gravothermal track. To claim the track captures SIDM-specific
    physics it must beat generic structural change, not just standing still.

E4  Resolution. SIDM RVmax reaches the softening: 13% of resolved isolated
    haloes sit below 4*eps and 26% below 10*eps (eps = 40 pc/h), against 1% in
    CDM. The most core-collapsed haloes are the least trustworthy, so every
    result is stratified by RVmax/eps rather than averaged over it.

Output: results/e32/controls.json
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import e30_concerto as e30                                        # noqa: E402
from gravop.units import G_KPC                                    # noqa: E402

H, EPS = 0.7, 0.04            # h, softening in comoving kpc/h
M_PART = 6.3e3                # Msun per particle (DATA.md)
X_VMAX = e30.X_VMAX
COEF = e30.NFW_VMAX_COEF
# z=0 hlist columns beyond the tree's 61
C = {'id': 1, 'pid': 5, 'Mvir': 10, 'Rvir': 11, 'lastMM': 15, 'vmax': 16,
     'RVmax': 60, 'Mpeak': 62, 'Vpeak': 64, 'Halfmass_Scale': 65,
     'Acc_Rate_Inst': 66, 'Acc_Rate_100Myr': 67, 'Acc_Rate_1Tdyn': 68,
     'Acc_Rate_2Tdyn': 69, 'Tidal_Force_Tdyn': 79}


def hlist(tar):
    r = subprocess.run(['tar', 'xOf', os.path.join(ROOT, 'data/d3_concerto', tar)],
                       capture_output=True, text=True)
    rows = [l.split() for l in r.stdout.splitlines() if not l.startswith('#')]
    return [f for f in rows if len(f) >= 80]


def nfw_concentration_family(mvir_phys, rvir_phys, c_grid):
    """(vmax, rmax) along the one-parameter NFW concentration family.

    Mvir and Rvir are held at their measured CDM values; only c varies. This is
    the E3 competitor: one free parameter, not nested in the gravothermal track.
    """
    r_s = rvir_phys / c_grid
    mu = np.log(1.0 + c_grid) - c_grid / (1.0 + c_grid)
    rho_s = mvir_phys / (4.0 * np.pi * r_s ** 3 * mu)
    v_0 = np.sqrt(4.0 * np.pi * G_KPC * rho_s * r_s ** 2)
    return COEF * v_0, X_VMAX * r_s


def main():
    p = e30.load_pairs()
    keep = np.isfinite(p['cdm_vmax']) & np.isfinite(p['sidm_vmax'])
    keep &= (p['cdm_vmax'] > 8.0) & (p['cdm_rmax'] > 0)
    keep &= (p['sidm_vmax'] > 0) & (p['sidm_rmax'] > 0) & (p['pid'] == -1)
    sel = np.nonzero(keep)[0]

    e30res = json.load(open(os.path.join(ROOT, 'results', 'e30', 'concerto.json')))
    R = e30res['rows']
    assert len(R) == len(sel), 'E30 rows and reproduced cuts disagree'
    resid = np.array([r['resid_dex'] for r in R])
    null = np.array([r['null_resid_dex'] for r in R])
    phase = np.array([r['best_phase'] for r in R])
    sighat = np.array([r['sigma_hat'] for r in R])

    # --- exact physical match into both z=0 catalogues (see e31 docstring)
    cdm, sidm = hlist('LMC_Halo104_CDM_RCT.tar'), hlist('LMC_Halo104_GroupSIDM_RCT.tar')
    cv = np.array([float(f[C['vmax']]) for f in cdm])
    cm = np.array([float(f[C['Mvir']]) for f in cdm])
    sv = np.array([float(f[C['vmax']]) for f in sidm])
    sr = np.array([float(f[C['RVmax']]) for f in sidm])

    def match(x1, y1, x2, y2, lab):
        d = np.sqrt((np.log10(x1[:, None]) - np.log10(x2[None, :])) ** 2
                    + (np.log10(y1[:, None]) - np.log10(y2[None, :])) ** 2)
        j = d.argmin(1)
        assert d[np.arange(len(x1)), j].max() < 1e-9, f'{lab} match not exact'
        assert len(set(j.tolist())) == len(j), f'{lab} match not unique'
        return j

    jc = match(p['cdm_vmax'][sel], p['mcdm'][sel] * H, cv, cm, 'cdm')
    js = match(p['sidm_vmax'][sel], p['sidm_rmax'][sel] * H, sv, sr, 'sidm')

    env = {k: np.array([float(cdm[i][C[k]]) for i in jc])
           for k in ('lastMM', 'Mpeak', 'Vpeak', 'Halfmass_Scale',
                     'Acc_Rate_Inst', 'Acc_Rate_100Myr', 'Acc_Rate_1Tdyn',
                     'Acc_Rate_2Tdyn', 'Tidal_Force_Tdyn')}
    env['Mpeak_over_Mvir'] = env['Mpeak'] / np.array([float(cdm[i][C['Mvir']]) for i in jc])
    rvmax_eps = np.array([float(sidm[i][C['RVmax']]) for i in js]) / EPS
    # Saved so the figure and the cut sensitivity of E39 are reproducible from
    # this script alone. `collapse_dex` is the measured shrinkage of R_Vmax
    # between the two runs -- the physical contraction the bands are meant to
    # stand for -- and `mvir_cdm` supports an independent resolution criterion
    # (particle count) that does not contain R_Vmax at all.
    collapse_dex = np.log10(
        np.array([float(sidm[i][C['RVmax']]) for i in js])
        / np.array([float(cdm[i][C['RVmax']]) for i in jc]))
    mvir_cdm = np.array([float(cdm[i][C['Mvir']]) for i in jc]) / H   # Msun
    mvir_sidm = np.array([float(sidm[i][C['Mvir']]) for i in js]) / H  # Msun
    # Particle count, for the convergence criterion the suite's own authors
    # use: R_Vmax is well measured above ~2000 particles (E. Nadler, priv.
    # comm., 15 Sep 2026), the same cut applied in the SIDM Concerto paper.
    # Both runs are counted; the binding number is the smaller.
    npart_cdm = mvir_cdm / M_PART
    npart_sidm = mvir_sidm / M_PART
    # Halo identities, so the sample can be reconstructed rather than described
    # (V5 minor 15). These are the Rockstar catalogue IDs in each run's own
    # z=0 hlist; the two runs number their haloes independently, so both are
    # needed to name one matched pair.
    cdm_id = np.array([int(float(cdm[i][C['id']])) for i in jc], dtype=np.int64)
    sidm_id = np.array([int(float(sidm[i][C['id']])) for i in js], dtype=np.int64)

    fitted = phase >= 1e-3
    out = {'n': len(sel), 'n_fitted': int(fitted.sum()),
           'n_pinned_at_null': int((~fitted).sum())}

    # ---- E2
    e2 = {}
    for k, v in env.items():
        r_all, p_all = spearmanr(v, resid)
        r_fit, p_fit = spearmanr(v[fitted], resid[fitted])
        e2[k] = {'rho_all': float(r_all), 'p_all': float(p_all),
                 'rho_fitted': float(r_fit), 'p_fitted': float(p_fit)}
    # Holm-Bonferroni over the 10 environment tests, on the fitted subset
    ps = sorted((v['p_fitted'], k) for k, v in e2.items())
    m = len(ps)
    for rank, (pv, k) in enumerate(ps):
        e2[k]['p_holm'] = float(min(1.0, pv * (m - rank)))
    out['E2_environment'] = e2
    out['E2_any_significant_after_holm'] = bool(
        any(v['p_holm'] < 0.05 for v in e2.values()))

    # ---- E3
    c_grid = np.geomspace(1.0, 100.0, 4000)
    nfw_res = np.empty(len(sel)); nfw_c = np.empty(len(sel))
    for n, i in enumerate(sel):
        mvir = float(cdm[jc[n]][C['Mvir']]) / H          # physical Msun
        rvir = float(cdm[jc[n]][C['Rvir']]) / H          # physical kpc (a=1)
        vm, rm = nfw_concentration_family(mvir, rvir, c_grid)
        d2 = (np.log10(vm / p['sidm_vmax'][i]) ** 2
              + np.log10(rm / p['sidm_rmax'][i]) ** 2)
        k = int(np.argmin(d2))
        nfw_res[n] = np.sqrt(d2[k]); nfw_c[n] = c_grid[k]
    out['E3_nfw_baseline'] = {
        'what': 'NFW at fixed Mvir,Rvir with concentration free: 1 parameter, '
                'NOT nested in the gravothermal track',
        'resid_median_dex': float(np.median(nfw_res)),
        'track_resid_median_dex': float(np.median(resid)),
        'no_evolution_null_median_dex': float(np.median(null)),
        'frac_track_beats_nfw': float(np.mean(resid < nfw_res)),
        'frac_track_beats_nfw_fitted_only': float(np.mean(resid[fitted] < nfw_res[fitted])),
        'median_ratio_nfw_over_track': float(np.median(nfw_res / resid)),
        'best_c_median': float(np.median(nfw_c)),
    }
    from scipy.stats import wilcoxon
    w, pw = wilcoxon(resid, nfw_res)
    out['E3_nfw_baseline']['wilcoxon_p'] = float(pw)
    w2, pw2 = wilcoxon(resid[fitted], nfw_res[fitted])
    out['E3_nfw_baseline']['wilcoxon_p_fitted_only'] = float(pw2)

    # ---- E4
    bands = [(0, 4), (4, 10), (10, 1e9)]
    e4 = []
    for lo, hi in bands:
        m_ = (rvmax_eps >= lo) & (rvmax_eps < hi)
        if m_.sum() == 0:
            continue
        e4.append({'rvmax_over_eps': f'[{lo},{hi})', 'n': int(m_.sum()),
                   'track_resid_median': float(np.median(resid[m_])),
                   'nfw_resid_median': float(np.median(nfw_res[m_])),
                   'frac_pinned': float(np.mean(~fitted[m_])),
                   'sigma_hat_median': float(np.median(sighat[m_]))})
    out['E4_resolution'] = {'softening_kpc_h': EPS, 'bands': e4,
                            'frac_below_4eps': float(np.mean(rvmax_eps < 4)),
                            'frac_below_10eps': float(np.mean(rvmax_eps < 10))}

    dst = os.path.join(ROOT, 'results', 'e32')
    os.makedirs(dst, exist_ok=True)
    np.savez(os.path.join(dst, 'controls.npz'), resid=resid, null=null,
             nfw_resid=nfw_res, nfw_c=nfw_c, phase=phase, sigma_hat=sighat,
             rvmax_eps=rvmax_eps, collapse_dex=collapse_dex,
             mvir_cdm=mvir_cdm, mvir_sidm=mvir_sidm,
             npart_cdm=npart_cdm, npart_sidm=npart_sidm,
             cdm_id=cdm_id, sidm_id=sidm_id,
             **{f'env_{k}': v for k, v in env.items()})
    json.dump(out, open(os.path.join(dst, 'controls.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2)[:3000])


if __name__ == '__main__':
    sys.exit(main())
