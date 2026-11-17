"""Figures for the GravOp application results. Reads saved artifacts only
(plus light emulator evaluations). Outputs results/figures/*.{pdf,png}.

Style: validated categorical palette (dataviz skill, 2026-08-27):
GravOp #2a78d6, no-physics #eb6834, DeepONet #1baf7a, PCA+MLP #eda100,
reference/solver #e87ba4; sequential time ramp = single-hue blues.
"""
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIG = os.path.join(ROOT, 'results', 'figures')
OUTDIRS = [FIG]
os.makedirs(FIG, exist_ok=True)

C = {'gravop': '#2a78d6', 'nophys': '#eb6834', 'deeponet': '#1baf7a',
     'pcamlp': '#eda100', 'ref': '#e87ba4', 'nbody': '#333333'}
import figstyle as FS

FS.use('plot')
plt.rcParams.update({'axes.titlesize': 9.5})

# Application-figure identity, drawn from the shared categorical set so the
# solver figures agree with the method figures.
C = dict(C, gravop=FS.STATE_HUE, nophys=FS.TAU_HUE, deeponet=FS.DIRECT_HUE)


def save(fig, name):
    # EPS as well as PDF: CPC asks for EPS and this script wrote none.
    FS.save(fig, name, OUTDIRS)


def midpoints(r):
    rm = np.empty_like(r)
    rm[0] = 0.5 * r[0]
    rm[1:] = 0.5 * (r[:-1] + r[1:])
    return rm


def fig_solver_validation():
    """(a) ours vs GravothermalSIDM profiles at matched epochs;
    (b) central density vs time: implicit vs explicit + reference."""
    ref = np.load(os.path.join(ROOT, 'results', 'e1', 'crosscheck_ref.npz'))
    ours = np.load(os.path.join(ROOT, 'results', 'e1',
                                'smoke_s0.5_N150_et1e-3.npz'))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, 3))
    for k, fac in enumerate((2., 10., 100.)):
        io = np.argmax(ours['rho'][:, 3] >= fac * ours['rho'][0, 3])
        ir = np.argmax(ref['rho'][:, 3] >= fac * ref['rho'][0, 3])
        ax.loglog(midpoints(ours['r'][io]), ours['rho'][io],
                  color=blues[k], label=f'{fac:g}$\\times$ (this work)')
        ax.loglog(midpoints(ref['r'][ir]), ref['rho'][ir], ls='--',
                  color=C['ref'], lw=1.1,
                  label='reference' if k == 0 else None)
    ax.set(xlabel=r'$r/r_s$', ylabel=r'$\rho/\rho_s$',
           xlim=(1e-2, 30), ylim=(1e-5, 3e4))
    ax.legend(frameon=False, loc='lower left', title=r'$\rho_c/\rho_{c,0}$')
    ax.set_title('(a) profiles vs GravothermalSIDM', loc='left')

    ax = axes[1]
    t_e, rc_e = ours['t'], ours['rho'][:, 3]
    from gravop import solver
    res_i = solver.run_halo(sigma_m=0.5, n_shells=150, eps_t=3e-4,
                            rho_c_factor=1e3, timestep_mode='implicit')
    ax.semilogy(t_e, rc_e / rc_e[0], color=C['ref'], label='explicit')
    ax.semilogy(res_i['t'], res_i['rho'][:, 3] / res_i['rho'][0, 3],
                ls='--', color=C['gravop'],
                label='semi-implicit (this work)')
    ax.set(xlabel=r'$t/t_0$', ylabel=r'$\rho_c/\rho_{c,0}$')
    ax.legend(frameon=False, loc='upper left')
    ax.set_title(r'(b) $\hat{\sigma}=0.5$', loc='left')
    save(fig, 'fig1_solver_validation')


def fig_nbody():
    """N-body benchmark vs (a) the solver and (b) the emulator alone."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from evaluate_all import load_model, NormObj
    from gravop import solver as gsolver, dataset as gds

    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    f = np.load(os.path.join(ROOT, 'results', 'e2',
                             'fluid_run_production.npz'))
    R_S, rho_s, t0g, s = 1.28, 4.42e7, 0.02045991174237841, 0.945
    keys = ['snapshot_010', 'snapshot_030', 'snapshot_040', 'snapshot_045']
    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(keys)))

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models, norm = [], None
    for sd_ in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{sd_}'))
        models.append(m)
        norm = nrm
    n = NormObj(norm)
    rr, mm, rho_i, p_i = gsolver.nfw_profile(n_shells=300)
    lr0, lv0 = gds.regrid_snapshot(rr, rho_i, p_i)
    prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd,
                                  (lv0 - n.v_mu) / n.v_sd]),
                        dtype=torch.float32, device=dev)[None]
    par = torch.tensor([[(np.log10(0.94553884672) - n.sig_mu) / n.sig_sd,
                         (0.0 - n.w_mu) / n.w_sd, 0.0]],
                       dtype=torch.float32, device=dev)
    with torch.no_grad():
        c0 = torch.cat([par, torch.zeros(1, 1, device=dev)], 1)
        ltc = np.mean([m(prof, c0)[1][0, 0].item() for m in models])
    t_norm = 10 ** (ltc * n.tc_sd + n.tc_mu)
    rg = gds.r_grid()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for panel, ax in enumerate(axes):
        for k, key in enumerate(keys):
            # Snapshot times are stored in Gadget units, kpc/(km/s) = 0.978
            # Gyr; dividing by t0g puts them in t0, the unit the epoch legend
            # declares. s rescales the fluid clock only, so the legend omits it.
            t_nb = float(d[key + '_t']) / t0g * s
            r_nb = d[key + '_r'] / R_S
            rho_nb = d[key + '_rho'] / (rho_s * 1e-10)
            sel = (r_nb > 0.04) & (r_nb < 12) & (rho_nb > 0)
            ax.loglog(r_nb[sel], rho_nb[sel], 'o', ms=2.6, mfc='none',
                      mec=blues[k], mew=0.9)
            if panel == 0:
                i = np.clip(np.searchsorted(f['t'], t_nb) - 1,
                            0, len(f['t']) - 2)
                ax.loglog(midpoints(f['r'][i]), f['rho'][i], color=blues[k],
                          lw=1.3, label=f"$t={float(d[key + '_t']) / t0g:.0f}$")
            else:
                tau = min(t_nb / t_norm, 1.0)
                cond = torch.cat([par, torch.tensor(
                    [[tau]], dtype=torch.float32, device=dev)], 1)
                with torch.no_grad():
                    pr = np.mean([m(prof, cond)[0][0, 0].cpu().numpy()
                                  for m in models], axis=0)
                lrho = pr * n.rho_sd + n.rho_mu
                ax.loglog(rg, 10 ** lrho, color=blues[k], lw=1.3, ls='--',
                          label=f"$t={float(d[key + '_t']) / t0g:.0f}$")
        ax.set(xlabel=r'$r/r_s$', xlim=(4e-2, 12), ylim=(1e-3, 4e2))
    axes[0].set_ylabel(r'$\rho/\rho_s$')
    # Two keys, not one. The epoch legend said nothing about which mark was
    # the simulation and which was the model, and the times carried no unit.
    epochs = axes[0].legend(frameon=False, loc='lower left',
                            title=r'epoch $t\,[\,t_0\,]$')
    axes[0].add_artist(epochs)
    # Explicit handles: ax.legend() would re-collect the epoch lines and draw
    # that legend a second time.
    from matplotlib.lines import Line2D
    points = Line2D([], [], linestyle='none', marker='o', ms=2.6, mfc='none',
                    mec=FS.INK, mew=0.9, label='N-body (Fischer+ 2025)')
    for ax, style, name in ((axes[0], '-', 'solver'),
                            (axes[1], '--', 'emulator')):
        curve = Line2D([], [], linestyle=style, color=FS.INK, lw=1.3,
                       label=name)
        ax.legend(handles=[points, curve], frameon=False, loc='upper right',
                  fontsize=7, handlelength=1.6)
    # The conductivity rescaling is fitted once, against panel (a); panel (b)
    # inherits it. The caption says so and the text quotes both RMS values, so
    # the panels carry labels only.
    axes[0].set_title(r'(a) solver, $C_{\rm eff}=0.712$', loc='left', fontsize=8)
    axes[1].set_title('(b) emulator', loc='left', fontsize=8)
    save(fig, 'fig2_nbody_validation')


def eval_model_curves():
    """Per-tau error curves + example profiles for fig 3."""
    import torch
    from train import load_data, split, Norm, make_tensors
    from evaluate_all import load_model, NormObj
    from gravop import dataset as ds
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    curves, models = {}, {}
    for name in ('gravop_s0', 'fno_nophys', 'deeponet'):
        m, norm, a = load_model(os.path.join(ROOT, 'results', 'runs', name))
        models[name] = (m, norm)
        no = NormObj(norm)
        lrho, lv, prof, par, tc, nc = make_tensors(d, no, dev)
        taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)
        errs = np.zeros(ds.N_TAU)
        with torch.no_grad():
            for s0 in range(0, len(te), 128):
                ri = te[s0:s0 + 128]
                B = len(ri)
                pi = prof[ri].repeat_interleave(ds.N_TAU, dim=0)
                ci = torch.cat([par[ri].repeat_interleave(ds.N_TAU, dim=0),
                                taus.repeat(B)[:, None]], dim=1)
                f, _ = m(pi, ci)
                pr = f[:, 0].reshape(B, ds.N_TAU, ds.N_R)
                tgt = lrho[ri]
                rel = (10 ** ((pr - tgt) * no.rho_sd) - 1).abs()
                errs += rel.mean(dim=(0, 2)).cpu().numpy() * B
        curves[name] = errs / len(te)
    # example run: test run with largest rho_c growth (collapse showcase)
    growth = d['lrho'][te][:, -1, :].max(axis=1)
    ex = te[np.argsort(growth)[-3]]
    m, norm = models['gravop_s0']
    no = NormObj(norm)
    lrho, lv, prof, par, tc, nc = make_tensors(d, no, dev)
    taus_t = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)
    with torch.no_grad():
        pi = prof[[ex]].repeat_interleave(ds.N_TAU, dim=0)
        ci = torch.cat([par[[ex]].repeat_interleave(ds.N_TAU, dim=0),
                        taus_t[:, None]], dim=1)
        f, _ = m(pi, ci)
    pred = (f[:, 0].cpu().numpy() * no.rho_sd + no.rho_mu)
    true = d['lrho'][ex]
    return curves, pred, true, d, ex


def fig_emulator():
    curves, pred, true, d, ex = eval_model_curves()
    from gravop import dataset as ds
    rg = ds.r_grid()
    taus = ds.tau_grid()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    idxs = [0, 12, 24, 47]
    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(idxs)))
    for k, i in enumerate(idxs):
        ax.loglog(rg, 10 ** true[i], color=blues[k], lw=1.3,
                  label=f'$\\tau={taus[i]:.2f}$')
        ax.loglog(rg, 10 ** pred[i], ls='--', color=C['nophys'], lw=1.1)
    ax.set(xlabel=r'$r/r_s$', ylabel=r'$\rho/\rho_s$', xlim=(3e-3, 100),
           ylim=(1e-7, 1e4))
    ax.legend(frameon=False, loc='lower left',
              title='solver (solid) / GravOp (dashed)')
    ax.set_title(f'(a) test run: $\\hat\\sigma_0$='
                 f'{10**d["log_sigma0"][ex]:.2f}, '
                 f'$\\hat w$={10**d["log_w"][ex]:.1f}', loc='left')
    ax = axes[1]
    for name, c, lab in (('gravop_s0', C['gravop'], 'GravOp'),
                         ('fno_nophys', C['nophys'], 'no-physics FNO'),
                         ('deeponet', C['deeponet'], 'DeepONet')):
        ax.semilogy(taus, curves[name], color=c, label=lab)
    ax.set(xlabel=r'$\tau = t/t_{\rm norm}$',
           ylabel=r'mean $|\Delta\rho|/\rho$ (test)')
    ax.legend(frameon=False)
    ax.set_title('(b) test error vs time', loc='left')
    save(fig, 'fig3_emulator_accuracy')


def fig_posterior():
    # the extended grid reaches the prior's upper edges, so neither posterior
    # is truncated by the plotted box; the original 91x91 box clipped both
    # (scripts/e9_zoom_grid.py)
    ext = os.path.join(ROOT, 'results', 'e9', 'grid_zoom_extended.npz')
    g = np.load(ext if os.path.exists(ext) else
                os.path.join(ROOT, 'results', 'e9', 'grid_zoom.npz'))
    ch = np.load(os.path.join(ROOT, 'results', 'e9', 'samples.npy'))
    lp = g['logpost'] - np.max(g['logpost'][np.isfinite(g['logpost'])])
    P = np.exp(lp)
    P /= P.sum()

    def levels(W):
        s = np.sort(W.ravel())[::-1]
        c = np.cumsum(s)
        return [s[np.searchsorted(c, q)] for q in (0.95, 0.68)]

    H, e1, e2 = np.histogram2d(
        ch[:, 0], ch[:, 1], bins=[60, 60],
        range=[[g['lsig'][0], g['lsig'][-1]], [g['lw'][0], g['lw'][-1]]])
    from scipy.ndimage import gaussian_filter
    Q = gaussian_filter(H / H.sum(), 1.2)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    A, B = np.meshgrid(g['lsig'], g['lw'], indexing='ij')
    ax.contour(A, B, P, levels=levels(P), colors=C['ref'],
               linewidths=(1.0, 1.8))
    c1 = 0.5 * (e1[:-1] + e1[1:])
    c2 = 0.5 * (e2[:-1] + e2[1:])
    A2, B2 = np.meshgrid(c1, c2, indexing='ij')
    ax.contour(A2, B2, Q, levels=levels(Q), colors=C['gravop'],
               linewidths=(1.0, 1.8), linestyles='--')
    ax.plot(np.log10(2.0), np.log10(1.5), '*', color='#333', ms=11,
            label='truth')
    ax.plot([], [], color=C['ref'], label='direct solver (grid)')
    ax.plot([], [], color=C['gravop'], ls='--', label='GravOp + MCMC')
    ax.set(xlabel=r'$\log_{10}\hat\sigma_0$', ylabel=r'$\log_{10}\hat w$',
           xlim=(g['lsig'][0], g['lsig'][-1]), ylim=(g['lw'][0], g['lw'][-1]))
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('68/95% regions', loc='left')
    save(fig, 'fig4_posterior')


def fig_speedup():
    bench = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    sol = np.mean([r['wall_s'] for r in bench['solver']])
    emu_gpu = min(r['seconds_per_item'] for r in bench['emulator']
                  if r['device'] == 'cuda:0' and r['kind'] == 'trajectory')
    emu_cpu = min(r['seconds_per_item'] for r in bench['emulator']
                  if r['device'] == 'cpu' and r['kind'] == 'trajectory')
    labels = ['explicit\nsolver', 'implicit solver\n(this work)',
              'GravOp (CPU)', 'GravOp (GPU,\nbatched)']
    vals = [500.0, sol, emu_cpu, emu_gpu]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    bars = ax.bar(labels, vals, color=['#b9b9b3', C['ref'], C['gravop'],
                                       C['gravop']], width=0.62)
    ax.set_yscale('log')
    ax.set_ylabel('wall time per trajectory [s]')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.4,
                f'{v:.2g}', ha='center', fontsize=8)
    ax.grid(axis='x', visible=False)
    ax.set_title(r'cost per full evolution ($\hat\sigma=0.5$ class)',
                 loc='left')
    save(fig, 'fig5_speedup')


def fig_ablation():
    """(a) data-efficiency of the physics constraints; (b) zero-shot
    resolution transfer."""
    de = json.load(open(os.path.join(ROOT, 'results', 'ablation',
                                     'data_efficiency.json')))
    ns = sorted(int(k) for k in de)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    for key, c, lab in (('phys', C['gravop'], 'with physics losses'),
                        ('nophys', C['nophys'], 'data loss only')):
        mu = np.array([de[str(n)][f'{key}_rel_mean'] for n in ns])
        sd = np.array([de[str(n)][f'{key}_rel_std'] for n in ns])
        ax.errorbar(ns, mu, yerr=sd, color=c, marker='o', ms=4, capsize=2,
                    label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='training trajectories', ylabel='median $|\\Delta\\rho|/\\rho$ [%]')
    ax.legend(frameon=False)
    ax.set_title('(a) data efficiency', loc='left')

    ax = axes[1]
    rs = json.load(open(os.path.join(ROOT, 'results', 'e10',
                                     'resolution.json')))
    xs = [int(k) for k in rs['resolutions']]
    ys = [100 * rs['per_resolution'][str(k)]['mean_rel_rho'] for k in xs]
    ax.plot(xs, ys, color=C['gravop'], marker='o', ms=5)
    trained = rs['trained_at']
    it = xs.index(trained)
    ax.plot([trained], [ys[it]], 'o', ms=10, mfc='none',
            mec=C['nophys'], mew=1.8)
    ax.annotate('training grid', (trained, ys[it]),
                textcoords='offset points', xytext=(6, 10), fontsize=8,
                color=C['nophys'])
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set(xlabel='radial grid points at evaluation',
           ylabel='mean $|\\Delta\\rho|/\\rho$ [%]', ylim=(0, 2.6))
    ax.set_title('(b) zero-shot resolution transfer', loc='left')
    save(fig, 'fig6_ablations')


def fig_universality():
    """(a) collapse-clock ratio vs Knudsen regime; (b) its independence of the
    velocity-dependence scale in the LMFP regime."""
    u = json.load(open(os.path.join(ROOT, 'results', 'e13',
                                    'universality.json')))
    runs = u['runs']
    kn = np.array([r['kn_at_min'] for r in runs])
    ratio = np.array([r['ratio'] for r in runs])
    yuk = np.array([r['model'] == 'yukawa_tu' for r in runs])
    w = np.array([r['w_hat'] for r in runs])
    lm = kn > 10.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(wspace=0.33)
    ax = axes[0]
    ax.axvspan(10, kn.max() * 2, color='#edf3f8')   # pre-blended: EPS has no alpha
    ax.scatter(kn[~yuk], ratio[~yuk], s=16, color=C['nophys'],
               label='constant $\\hat\\sigma$', zorder=3)
    ax.scatter(kn[yuk], ratio[yuk], s=16, color=C['gravop'],
               label='Yukawa-Born', zorder=3)
    band = u['lmfp_only']
    ax.axhline(band['mean'], color='#555', lw=1.0, ls='--', zorder=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel=r'Knudsen number at $\rho_c^{\rm min}$',
           ylabel=r'$t_{\rm coll}/t_{c,0}$')
    ax.text(11, band['mean'] * 1.28, 'LMFP regime',
            fontsize=7.5, color='#2a78d6')
    ax.legend(frameon=False, loc='upper right', fontsize=7.5)
    ax.set_title('(a) universality holds where predicted', loc='left')

    ax = axes[1]
    ax.scatter(w[lm & yuk], ratio[lm & yuk], s=20, color=C['gravop'],
               label='Yukawa-Born (LMFP)')
    ax.scatter(w[lm & ~yuk], ratio[lm & ~yuk], s=20, color=C['nophys'],
               marker='s', label='constant (LMFP)')
    m, sd = band['mean'], band['std']
    ax.axhspan(m - sd, m + sd, color='#e3ecf7')   # pre-blended: EPS has no alpha
    ax.axhline(m, color='#555', lw=1.0, ls='--')
    ax.set_xscale('log')
    ax.set(xlabel=r'velocity scale $\hat w$',
           ylabel=r'$t_{\rm coll}/t_{c,0}$',
           ylim=(m - 8 * sd, m + 8 * sd))
    ax.legend(frameon=False, loc='upper right', fontsize=7.5)
    ax.set_title(f'(b) scatter {100 * band["frac_scatter"]:.1f}% '
                 f'across $\\hat w$', loc='left')
    save(fig, 'fig7_universality')


def fig_workprecision():
    """Work--precision for the two time-stepping schemes.

    The matched-accuracy speed-up is read off a log--log interpolation between
    two measured points in each block, so the construction is worth seeing
    rather than only tabulating: the plot shows both curves, the brackets the
    interpolation runs between, and the vertical at the target where the two
    wall times are read.
    """
    wp = json.load(open(os.path.join(ROOT, 'results', 'e8',
                                     'workprecision.json')))

    def pts(curve):
        rows = [r for r in wp['curves'][curve] if r.get('rel_err')]
        rows.sort(key=lambda r: r['rel_err'])
        return (np.array([r['rel_err'] for r in rows]),
                np.array([r['wall_median'] for r in rows]))

    def interp(curve, target):
        e, w = pts(curve)
        i = int(np.searchsorted(e, target))
        i = max(1, min(i, len(e) - 1))
        x, y = np.log10(e[i - 1:i + 1]), np.log10(w[i - 1:i + 1])
        return 10 ** (y[0] + (y[1] - y[0])
                      * (np.log10(target) - x[0]) / (x[1] - x[0]))

    target = 3e-4
    fig, ax = plt.subplots(figsize=(4.7, 3.2))

    # the wall cap the two loosest explicit runs ran into; they carry no error
    # value, so the cap is drawn rather than the points placed
    cap = wp['curves']['classic'][0]['exceeded_cap_s']
    ax.axhspan(cap, cap * 6, color='#e9e9e6', zorder=0)
    ax.text(2.6e-3, cap * 2.3, f'{cap:.0f} s cap',
            fontsize=7, color='#555550', va='center', ha='left')

    for curve, hue, label in (('implicit', C['gravop'],
                               'semi-implicit (this work)'),
                              ('classic', C['ref'], 'explicit')):
        e, w = pts(curve)
        ax.plot(e, w, '-o', color=hue, lw=2.0, ms=6, label=label, zorder=3)

    ax.axvline(target, color='#8a8a85', lw=1.0, ls=':', zorder=1)
    lo, hi = interp('implicit', target), interp('classic', target)
    ax.plot([target, target], [lo, hi], color='#333330', lw=1.4, zorder=4,
            marker='_', ms=9)
    # the x axis is inverted, so a smaller target sits further right
    # Two significant figures, matching the manuscript: the explicit bracket
    # points are single runs, so further digits are not measured.
    ax.text(target * 0.80, np.sqrt(lo * hi), f'{round(hi / lo, -1):.0f}$\\times$',
            fontsize=10, va='center', ha='left')
    for y in (lo, hi):
        ax.plot([target], [y], 'o', ms=6, mfc='white', mec='#333330',
                mew=1.5, zorder=5)
    # beside the vertical rather than across it (the x axis is inverted, so
    # a smaller value sits further right)
    ax.text(target * 0.92, 700, 'matched accuracy',
            fontsize=7.5, color='#555550', ha='left', va='bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel(r'relative error in $t_c$   (accuracy improves $\rightarrow$)')
    ax.set_ylabel('wall time [s]')
    ax.set_ylim(0.05, cap * 6)
    ax.legend(loc='center left', frameon=False, fontsize=8)
    save(fig, 'fig20_work_precision')


if __name__ == '__main__':
    # The first four read per-run solver arrays, the N-body profiles and the
    # emulator weights, none of which the archival record carries; the rest
    # read stored summaries only. A figure whose inputs are absent is skipped
    # and named, so the others still regenerate from the archive alone.
    skipped = []
    for make in (fig_solver_validation, fig_nbody, fig_emulator, fig_posterior,
                 fig_speedup, fig_ablation, fig_universality, fig_workprecision):
        try:
            make()
        except FileNotFoundError as exc:
            skipped.append(f'{make.__name__}: needs {exc.filename or exc}')
    for line in skipped:
        print('skipped', line)
    print('figures done' + (f', {len(skipped)} skipped for missing inputs'
                            if skipped else ''))
