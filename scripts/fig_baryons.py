#!/usr/bin/env python3
"""Baryonic sensitivity of halo A: solver, baryon-aware emulator, and the
dark-matter-only emulator, for no, extended and compact Hernquist baryons.

  predict  emulator predictions for cases A, B, C of scripts/baryon_cases.py
           -> results/baryons_20261107/cases_emulator.npz
  plot     -> results/figures/fig24_baryons.{pdf,eps,png}

All central quantities are read on the emulator's radial grid (its innermost
point, r = 10^-2.5 r_s) for solver and emulator alike, so the two are compared
on the same footing; the core radius is where the density falls to half that
value. Physical units are those of halo A (r_s = 1.28 kpc,
rho_s = 4.42e7 Msun/kpc^3).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, 'results', 'baryons_20261107')
FIG = os.path.join(ROOT, 'results', 'figures')

R_S, RHO_S = 1.28, 4.42e7
V0_KMS = np.sqrt(4 * np.pi * 4.30091e-6 * RHO_S) * R_S
T0_GYR = 1.0 / np.sqrt(4 * np.pi * 4.30091e-6 * RHO_S) * 0.97779
SIGMA_HAT, C_EFF, S_RESCALE = 0.94553884672, 0.712, 0.945
CASES = {'A': (0.0, 0.1), 'B': (0.0153, 0.132), 'C': (0.0306, 0.0934)}
KW = dict(n_shells=300, r_min=0.003, eps_t=3e-4, rho_c_factor=1e3, t_end=1200.0,
          max_snaps=3000, timestep_mode='implicit', kn_min=0.02)


def grid_measures(lrho, lv, lg):
    """central density, core radius (half the central density), central v."""
    rc = lrho[..., 0]
    half = lrho - (rc[..., None] - np.log10(2.0))
    core = np.array([10 ** lg[np.argmax(h < 0)] for h in half.reshape(-1, half.shape[-1])])
    return 10 ** rc, core.reshape(rc.shape), 10 ** lv[..., 0]


def predict():
    import torch
    from gravop import baryons as B, dataset as ds
    from gravop.model import GravOpNet
    from evaluate_all import load_model, NormObj
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    lg = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
    bm, bn = [], None
    for s in range(3):
        ck = torch.load(os.path.join(RES, 'runs', f'seed{s}', 'model.pt'), map_location=dev,
                        weights_only=False)
        m = GravOpNet(n_r=ds.N_R, width=64, modes=32, n_layers=4, cond_dim=6).to(dev)
        m.load_state_dict(ck['model'])
        bm.append(m.eval())
        bn = type('N', (), ck['norm'])
    pm, pn = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs', f'gravop_s{s}'))
        pm.append(m.to(dev).eval())
        pn = NormObj(nrm)
    taus = np.linspace(0.0, 1.0, 97)
    out = {}
    for k, (Mb, ab) in CASES.items():
        res = B.run_halo_b(SIGMA_HAT, C=C_EFF, Mb=Mb, ab=ab, **KW)
        snaps = [ds.regrid_snapshot(res['r'][i], res['rho'][i], res['p'][i])
                 for i in range(len(res['t']))]
        sl = np.array([s[0] for s in snaps])
        sv = np.array([s[1] for s in snaps])
        out[f'{k}_solver_t'] = res['t']
        out[f'{k}_solver_tfinal'] = res['t_final']
        for q, v in zip(('rho_c', 'r_core', 'v_c'), grid_measures(sl, sv, lg)):
            out[f'{k}_solver_{q}'] = v
        lr0, lv0 = sl[0], sv[0]
        for tag, models, n, binputs in (('baryon', bm, bn, True), ('dmo', pm, pn, False)):
            prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd, (lv0 - n.v_mu) / n.v_sd]),
                                dtype=torch.float32, device=dev)[None]
            par = [(np.log10(SIGMA_HAT) - n.sig_mu) / n.sig_sd, (0.0 - n.w_mu) / n.w_sd, 0.0]
            if binputs:
                par += [Mb / 0.05, (np.log10(ab) - n.la_mu) / n.la_sd]
            par = torch.tensor([par], dtype=torch.float32, device=dev)
            with torch.no_grad():
                heads = [m(prof, torch.cat([par, torch.zeros(1, 1, device=dev)], 1))[1][0, 0].item()
                         for m in models]
                F = np.array([[m(prof, torch.cat([par, torch.tensor([[t]], dtype=torch.float32,
                                                                       device=dev)], 1))[0][0].cpu().numpy()
                               for m in models] for t in taus])
            lr = (F[:, :, 0] * n.rho_sd + n.rho_mu).mean(1)
            lv = (F[:, :, 1] * n.v_sd + n.v_mu).mean(1)
            # emulators run at C = 0.753; the clock at C_eff is t_norm / s
            t_norm = 10 ** (np.mean(heads) * n.tc_sd + n.tc_mu) / S_RESCALE
            out[f'{k}_{tag}_t'] = taus * t_norm
            out[f'{k}_{tag}_tnorm'] = t_norm
            for q, v in zip(('rho_c', 'r_core', 'v_c'), grid_measures(lr, lv, lg)):
                out[f'{k}_{tag}_{q}'] = v
        print(k, 'solver t_final', res['t_final'], 'baryon emulator', out[f'{k}_baryon_tnorm'],
              'dmo emulator', out[f'{k}_dmo_tnorm'])
    np.savez(os.path.join(RES, 'cases_emulator.npz'), **out)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle as FS
    FS.use('plot')
    e = np.load(os.path.join(RES, 'cases_emulator.npz'))
    cj = json.load(open(os.path.join(RES, 'cases.json')))
    col = {'A': FS.INK, 'B': FS.STATE_HUE, 'C': FS.TAU_HUE}
    lab = {'A': 'no baryons', 'B': 'extended', 'C': 'compact'}
    fig, ax = plt.subplots(2, 2, figsize=(4.9, 3.9))
    a1, a2, a3, a4 = ax.ravel()
    for k in 'ABC':
        ts = e[f'{k}_solver_t'] * T0_GYR
        a1.plot(ts, e[f'{k}_solver_rho_c'] * RHO_S, color=col[k], lw=1.1, label=lab[k])
        a2.plot(ts, e[f'{k}_solver_r_core'] * R_S, color=col[k], lw=1.1)
        a3.plot(ts, e[f'{k}_solver_v_c'] * V0_KMS, color=col[k], lw=1.1)
        sel = slice(0, None, 6)
        tb = e[f'{k}_baryon_t'] * T0_GYR
        a1.plot(tb[sel], (e[f'{k}_baryon_rho_c'] * RHO_S)[sel], 'o', ms=2.6, mfc='none',
                mec=col[k], mew=0.7)
        a2.plot(tb[sel], (e[f'{k}_baryon_r_core'] * R_S)[sel], 'o', ms=2.6, mfc='none',
                mec=col[k], mew=0.7)
        a3.plot(tb[sel], (e[f'{k}_baryon_v_c'] * V0_KMS)[sel], 'o', ms=2.6, mfc='none',
                mec=col[k], mew=0.7)
    td = e['C_dmo_t'] * T0_GYR
    a1.plot(td, e['C_dmo_rho_c'] * RHO_S, ls=':', color=col['C'], lw=0.9)
    for a, yl in ((a1, r'$\rho_c$ [M$_\odot$ kpc$^{-3}$]'), (a2, r'$r_{\rm core}$ [kpc]'),
                  (a3, r'$\nu_c$ [km s$^{-1}$]')):
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('t [Gyr]')
        a.set_ylabel(yl)
        a.set_xlim(0.03, 6)
    h = [plt.Line2D([], [], color=col[k], lw=1.1, label=lab[k]) for k in 'ABC']
    h += [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=FS.MUTED, ms=2.6, label='emulator, baryon-aware'),
          plt.Line2D([], [], ls=':', color=FS.MUTED, label='emulator, no baryon input')]
    a3.legend(handles=h, loc='lower right', fontsize=FS.MINPT - 0.5, handlelength=1.4, frameon=False)
    a3.set_yticks([5, 10, 20, 30])
    a3.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a3.set_ylim(4, 40)
    g = cj['grid']
    phi = np.array([x['phi_b0'] for x in g])
    rs_ = np.array([x['t_final_over_dmo'] for x in g])
    rz = np.array([x['zhong_estimate_ratio'] for x in g])
    a4.plot(phi, rs_, 'o', color=FS.INK, ms=3.2, label='solver')
    a4.plot(phi, rz, 'x', color=FS.MUTED, ms=3.6, label='Zhong et al. estimate')
    for k in 'BC':
        c = cj['cases'][k]
        a4.plot(c['Mb_hat'] / c['a_hat'], c['t_final_over_dmo'], 's', ms=4, mfc='none', mec=col[k])
    a4.set_xscale('log')
    a4.set_yscale('log')
    a4.set_xlabel(r'$|\Phi_b(0)|/v_0^2 = \hat M_b/\hat a$')
    a4.set_ylabel(r'$t_{\rm end}/t_{\rm end}^{\rm DM}$')
    a4.legend(loc='lower left', fontsize=FS.MINPT - 0.5, frameon=False)
    for a, t in zip((a1, a2, a3, a4), 'abcd'):
        right = t == 'd'
        a.text(0.97 if right else 0.03, 0.95, f'({t})', transform=a.transAxes,
               ha='right' if right else 'left', va='top', fontsize=FS.MINPT)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.8)
    FS.save(fig, 'fig24_baryons', [FIG])


if __name__ == '__main__':
    {'predict': predict, 'plot': plot}[sys.argv[1]]()
