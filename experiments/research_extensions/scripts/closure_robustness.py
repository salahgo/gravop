#!/usr/bin/env python3
"""Robustness of the five-halo (alpha, C) fit to the reference assumptions.

Uses the grids and references of closure_fit5.py, entirely in memory (the
canonical record learned_closures/closure_fit5.json is never written here), and
changes one assumption at a time. Also reports fixed-exponent baselines with C
re-optimised (harmonic alpha = 1 and alpha = 1/2), so that the gain from the
exponent is not confused with the gain from the normalisation.

Writes experiments/research_extensions/learned_closures/closure_robustness.json.

    python experiments/research_extensions/scripts/closure_robustness.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EXT = os.path.join(ROOT, 'experiments', 'research_extensions')
sys.path.insert(0, os.path.join(EXT, 'scripts'))
import closure_fit5 as F  # noqa: E402

LAD = json.load(open(os.path.join(EXT, 'multifidelity', 'ladder_analysis.json')))
SC = json.load(open(os.path.join(EXT, 'closure_design', 'score_D.json')))['haloes']
FA = np.linspace(0.3, 1.5, 241)
FC = np.linspace(np.log(min(F.CS)), np.log(max(F.CS)), 261)
A_, C_ = np.meshgrid(FA, FC, indexing='ij')
PTS = np.stack([A_.ravel(), C_.ravel()], 1)


def fit(refs):
    """Best (alpha, C), chi2, profile interval in alpha, leave-one-out; same likelihood as closure_fit5."""
    F.REF.clear(); F.REF.update(refs)
    G = {h: F.grid(h) for h in refs}
    lr = {h: G[h](PTS) for h in refs}                       # ln(t_fluid / t_ref)
    R = {h: lr[h] / refs[h][1] for h in refs}

    def best(hs, mask=None):
        chi = sum(R[h] ** 2 for h in hs)
        if mask is not None:
            chi = np.where(mask, chi, np.nan)
        k = int(np.nanargmin(chi))
        return k, chi

    k, chi = best(list(refs))
    ok = np.isfinite(chi) & (chi <= chi[k] + 1)
    out = dict(alpha=float(PTS[k, 0]), C=float(np.exp(PTS[k, 1])), chi2=float(chi[k]), dof=len(refs) - 2,
               alpha_range=[float(PTS[ok, 0].min()), float(PTS[ok, 0].max())],
               resid_pct={h: float(100 * np.expm1(lr[h][k])) for h in refs})
    out['loho_pct'] = {}
    for h in refs:
        kk, _ = best([x for x in refs if x != h])
        out['loho_pct'][h] = float(100 * np.expm1(lr[h][kk]))
    for name, a in (('harmonic_Cfit', 1.0), ('alpha_half_Cfit', 0.5)):
        kk, c2 = best(list(refs), mask=np.isclose(PTS[:, 0], a))
        out[name] = dict(C=float(np.exp(PTS[kk, 1])), chi2=float(c2[kk]))
    for name, (a, C) in (('harmonic_fixed', (1.0, 0.712)), ('literature_fixed', (0.5, 0.9))):
        p = np.array([[a, np.log(C)]])
        out[name] = dict(chi2=float(sum((G[h](p)[0] / refs[h][1]) ** 2 for h in refs)))
    return out


def main():
    base = {h: tuple(v) for h, v in F.REF.items()}
    res = {'baseline': fit(dict(base))}
    v = dict(base); v['A'] = (1.0, 0.06); res['sigma_A_6pct'] = fit(v)
    res['drop_A_nsphere_only'] = fit({h: x for h, x in base.items() if h != 'A'})
    res['extra_3pct_all'] = fit({h: (t, float(np.hypot(s, 0.03))) for h, (t, s) in base.items()})
    res['systematic_floor_1pct'] = fit({h: (t, float(np.hypot(np.sqrt(max(s ** 2 - 0.03 ** 2, 0)), 0.01)))
                                        if h != 'A' else (t, s) for h, (t, s) in base.items()})
    v = dict(base)
    for h in ('B', 'C'):
        v[h] = (LAD[h]['markers']['t_x10_s']['models']['none']['T_inf'], base[h][1])
    for h in ('D1', 'D2'):
        v[h] = (SC[h]['t_x10_s']['200000']['mean'], base[h][1])
    res['smoothed_particle_marker'] = fit(v)
    F.REF.clear(); F.REF.update(base)
    path = os.path.join(EXT, 'learned_closures', 'closure_robustness.json')
    json.dump(res, open(path, 'w'), indent=1)
    for k, r in res.items():
        print(f"{k:26s} alpha {r['alpha']:.3f} {np.round(r['alpha_range'], 3)} C {r['C']:.3f} "
              f"chi2 {r['chi2']:.2f}/{r['dof']} | harm fixed {r['harmonic_fixed']['chi2']:.1f} "
              f"Cfit {r['harmonic_Cfit']['chi2']:.1f}@{r['harmonic_Cfit']['C']:.3f} | a=1/2 fixed "
              f"{r['literature_fixed']['chi2']:.1f} Cfit {r['alpha_half_Cfit']['chi2']:.1f}@{r['alpha_half_Cfit']['C']:.3f}"
              f" | loho {({h: round(x, 1) for h, x in r['loho_pct'].items()})}")


if __name__ == '__main__':
    main()
