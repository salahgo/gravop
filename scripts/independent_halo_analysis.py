#!/usr/bin/env python3
"""Independent haloes B and C: fluid model, emulator and NSphere compared.

Calibration is transferred, not refitted: the fluid model runs at the
conductivity calibrated on halo A (Fischer, Yu & Dolag 2025), C_eff = 0.712 for
the production harmonic interpolation, and C = 0.878 for the gentler alpha = 1/2
form of van den Bosch & Tokayer (2026), each with no further freedom. Both
haloes start from NSphere's exact initial density (truncated NFW with the inner
softening of its initial conditions); the cross-section is NSphere's constant,
isotropic sigma/m, the same convention as halo A's 80 cm^2/g (Fischer et al.
quote the viscosity cross-section normalised to the total for isotropic
scattering).

Common observables, defined identically for particles and fluid shells:
  * Lagrangian radii r_f enclosing mass fractions f of the halo;
  * the mean density inside r_f (central density at f = 0.003);
  * the one-dimensional velocity dispersion inside r_f (mass weighted);
  * density profiles in logarithmic bins at matched times.

Stages:
  solver  [--alpha 1|0.5]   fluid references -> solver_<halo>_a<alpha>.npz
  nsphere <run>             particle measures -> nsphere_<run>.npz
  emulator                  production ensemble zero-shot -> emulator_<halo>.npz
  compare                   -> independent_halo.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'results', 'independent_halo_20261107')
NS_OUT = os.path.join(OUT, 'nsphere')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

G_NS = 4.3e-6                      # NSphere's G, kpc (km/s)^2 / Msun
KMS_TO_KPCMYR = 1.02271e-3         # NSphere's velocity unit conversion
MSUN_KPC2 = 2.08909e-10
HALO = {
    'B': dict(mass=1.15e9, rs=1.18, falloff=19.0, kappa=50.0, tfinal_tdyn=1100),
    'C': dict(mass=1.0e12, rs=25.0, falloff=8.0, kappa=58.82, tfinal_tdyn=400),
}
TFINAL_RUN = {'C3': 250, 'C4': 250, 'C5': 250, 'B5': 850, 'B6': 850}
RUNS = {'B1': 'B', 'B2': 'B', 'B5': 'B', 'B6': 'B', 'C1': 'C', 'C2': 'C', 'C3': 'C', 'C4': 'C', 'C5': 'C'}
C_CAL = {1.0: 0.712, 0.5: 0.878}  # calibrated on halo A (results/e2)
S_RESCALE = 0.945                  # emulator (C = 0.753) time = 0.945 x time at C_eff
FRACS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
RBINS = np.logspace(-1.3, 1.0, 24)  # r / r_s, 0.05 to 10


def halo_units(h):
    p = HALO[h]
    shape = lambda x: 1.0 / ((x + 0.01) * (1 + x) ** 2 * (1 + (x / p['falloff']) ** 10))
    I = integrate.quad(lambda x: x * x * shape(x), 0, 85, limit=500,
                       points=[0.01, 1, p['falloff']])[0]
    rho_s = p['mass'] / (4 * np.pi * p['rs'] ** 3 * I)          # Msun / kpc^3
    t0_myr = 1.0 / np.sqrt(4 * np.pi * G_NS * rho_s) / KMS_TO_KPCMYR
    tdyn_myr = 1.0 / np.sqrt(KMS_TO_KPCMYR ** 2 * G_NS * p['mass'] / p['rs'] ** 3)
    v0_kpcmyr = np.sqrt(4 * np.pi * G_NS * rho_s) * p['rs'] * KMS_TO_KPCMYR
    return dict(shape=shape, I=I, rho_s=rho_s, t0_myr=t0_myr, tdyn_myr=tdyn_myr,
                v0=v0_kpcmyr, sigma_hat=p['kappa'] * rho_s * p['rs'] * MSUN_KPC2,
                t_end=p['tfinal_tdyn'] * tdyn_myr / t0_myr)


def fluid_measures(r, m, rho, p, I):
    """Lagrangian radii, mean density and dispersion inside them, and the
    density profile on RBINS, from one fluid snapshot (code units)."""
    out_r, out_rho, out_s = [], [], []
    for f in FRACS:
        mf = f * I
        rf = np.interp(mf, m, r)
        k = np.searchsorted(m, mf)
        w = np.diff(np.concatenate([[0.0], m]))[:k + 1]
        s2 = (p[:k + 1] / rho[:k + 1])
        out_r.append(rf)
        out_rho.append(3 * mf / rf ** 3)
        out_s.append(np.sqrt(np.sum(w * s2) / np.sum(w)))
    rm = np.concatenate([[0.5 * r[0]], 0.5 * (r[1:] + r[:-1])])
    prof = np.interp(np.log10(np.sqrt(RBINS[1:] * RBINS[:-1])), np.log10(rm), np.log10(rho))
    return np.array(out_r), np.array(out_rho), np.array(out_s), prof


def stage_solver(alpha):
    os.environ['SOLVER_ALPHA'] = str(alpha)
    if alpha == 1.0:
        from gravop import solver
    else:
        sys.path.insert(0, os.path.join(HERE, 'e2_conductivity_alpha'))
        import solver_alpha as solver
    variants = [('', None)] + ([('_nfw', 'nfw')] if alpha == 1.0 else [])
    for h in HALO:
      for suf, ic in variants:
        u = halo_units(h)
        if ic == 'nfw':
            r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
        else:
            r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
        res = solver.run_halo(sigma_m=u['sigma_hat'], C=C_CAL[alpha], r=r, m=m, rho=rho, p=p,
                              eps_t=3e-4, timestep_mode='implicit', rho_c_factor=1e4,
                              kn_min=0.02, t_end=u['t_end'], max_snaps=3000)
        meas = [fluid_measures(res['r'][i], res['m'], res['rho'][i], res['p'][i], u['I'])
                for i in range(len(res['t']))]
        np.savez(os.path.join(OUT, f'solver_{h}_a{alpha}{suf}.npz'), t=res['t'],
                 lagr_r=np.array([x[0] for x in meas]), mean_rho=np.array([x[1] for x in meas]),
                 disp=np.array([x[2] for x in meas]), prof=np.array([x[3] for x in meas]),
                 t_final=res['t_final'], status=res['status'], sigma_hat=u['sigma_hat'],
                 C=C_CAL[alpha], alpha=alpha,
                 energy_drift=np.max(np.abs(res['E'] - res['E'][0])) / abs(res['E'][0]))
        print(h, alpha, suf, 't_final', res['t_final'], 'status', res['status'])


def stage_nsphere(run):
    h = RUNS[run]
    u = halo_units(h)
    meta = json.load(open(os.path.join(NS_OUT, run, 'run.json')))
    N, steps, dtw = meta['n'], meta['steps'], meta['dtwrite']
    path = [f for f in os.listdir(os.path.join(NS_OUT, run, 'data'))
            if f.startswith('all_particle_data_')][0]
    raw = np.memmap(os.path.join(NS_OUT, run, 'data', path), mode='r',
                    dtype=np.dtype([('rank', '<i4'), ('R', '<f4'), ('V', '<f4'), ('L', '<f4')]))
    ns = raw.size // N
    rs, I = HALO[h]['rs'], u['I']
    t, lagr, mrho, disp, prof = [], [], [], [], []
    vol = 4 / 3 * np.pi * (RBINS[1:] ** 3 - RBINS[:-1] ** 3)
    for k in range(ns):
        s = np.asarray(raw[k * N:(k + 1) * N])
        R = s['R'].astype(np.float64) / rs
        order = np.argsort(R)
        R = R[order]
        v2 = (s['V'].astype(np.float64) ** 2
              + (s['L'].astype(np.float64) / (s['R'].astype(np.float64))) ** 2)[order] / 3
        v2 /= u['v0'] ** 2
        lr, mr, sd = [], [], []
        for f in FRACS:
            j = max(int(round(f * N)), 1)
            rf = R[j - 1]
            lr.append(rf)
            mr.append(3 * f * I / rf ** 3)
            sd.append(np.sqrt(v2[:j].mean()))
        cnt = np.histogram(R, RBINS)[0]
        # mass in code units: total I, each particle I / N; rho in rho_s = 3 m / r^3 per volume
        with np.errstate(divide='ignore'):
            prof.append(np.log10(cnt * (I / N) / (vol / (4 * np.pi))))
        tf = TFINAL_RUN.get(run, HALO[h]['tfinal_tdyn'])
        t.append(k * dtw / (steps - 1) * tf * u['tdyn_myr'] / u['t0_myr'])
        lagr.append(lr)
        mrho.append(mr)
        disp.append(sd)
    np.savez(os.path.join(OUT, f'nsphere_{run}.npz'), t=np.array(t), lagr_r=np.array(lagr),
             mean_rho=np.array(mrho), disp=np.array(disp), prof=np.array(prof), N=N)
    print(run, ns, 'snapshots to t =', t[-1], 't0')


def stage_emulator():
    import torch
    from gravop import solver, dataset as ds
    from evaluate_all import load_model, NormObj
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models, n = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs', f'gravop_s{s}'))
        models.append(m.to(dev).eval())
        n = NormObj(nrm)
    lg = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
    for h, suf in [(h, suf) for h in HALO for suf in ('', '_nfw')]:
        u = halo_units(h)
        if suf == '_nfw':
            r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
        else:
            r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
        lr0, lv0 = ds.regrid_snapshot(r, rho, p)
        prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd, (lv0 - n.v_mu) / n.v_sd]),
                            dtype=torch.float32, device=dev)[None]
        par = torch.tensor([[(np.log10(u['sigma_hat']) - n.sig_mu) / n.sig_sd,
                             (0.0 - n.w_mu) / n.w_sd, 0.0]], dtype=torch.float32, device=dev)
        taus = np.linspace(0, 1, 97)
        with torch.no_grad():
            heads = [mm(prof, torch.cat([par, torch.zeros(1, 1, device=dev)], 1))[1][0, 0].item()
                     for mm in models]
            F = []
            for tau in taus:
                cond = torch.cat([par, torch.tensor([[tau]], dtype=torch.float32, device=dev)], 1)
                F.append([mm(prof, cond)[0][0].cpu().numpy() for mm in models])
        F = np.array(F)                                   # (tau, member, 2, N_R)
        lrho = F[:, :, 0] * n.rho_sd + n.rho_mu
        lv = F[:, :, 1] * n.v_sd + n.v_mu
        t_norm = 10 ** (np.mean(heads) * n.tc_sd + n.tc_mu)   # in C = 0.753 units
        # Lagrangian measures from the emulated grid profile (mass by quadrature in ln r,
        # the inner sphere below the grid taken at the innermost density)
        rg = 10 ** lg
        meas = []
        for i in range(len(taus)):
            rr = 10 ** lrho[i].mean(0)
            mm_ = np.concatenate([[rr[0] * rg[0] ** 3 / 3],
                                  rr[0] * rg[0] ** 3 / 3 + integrate.cumulative_trapezoid(
                                      rr * rg ** 3, np.log(rg))])
            pp = rr * (10 ** lv[i].mean(0)) ** 2
            meas.append(fluid_measures(rg, mm_, rr, pp, u['I']))
        np.savez(os.path.join(OUT, f'emulator_{h}{suf}.npz'), tau=taus, lrho=lrho, lv=lv,
                 t_norm_emulator_C0753=t_norm, t_norm_members=np.array(heads),
                 lagr_r=np.array([x[0] for x in meas]), mean_rho=np.array([x[1] for x in meas]),
                 disp=np.array([x[2] for x in meas]), prof=np.array([x[3] for x in meas]),
                 spread_dex=lrho.std(1).mean(-1))
        print(h, suf, 'emulator t_norm (C=0.753)', t_norm, '-> at C_eff', t_norm / S_RESCALE,
              'spread', np.median(lrho.std(1).mean(-1)))


def collapse_markers(t, rho):
    """Pre-collapse minimum of the central density, its time, and the times
    the density first exceeds 3, 10 and 30 times that minimum on the way to
    its peak (the late particle rebound is excluded by stopping at the peak)."""
    ipk = int(np.argmax(rho))
    i0 = int(np.argmin(rho[:ipk + 1]))
    out = {'rho_min': float(rho[i0]), 't_min': float(t[i0]), 't_peak': float(t[ipk]),
           'rho_peak': float(rho[ipk])}
    for fac in (3, 10, 30):
        j = np.where((np.arange(len(t)) > i0) & (np.arange(len(t)) <= ipk) & (rho > fac * rho[i0]))[0]
        out[f't_x{fac}'] = float(t[j[0]]) if len(j) else None
    return out


def interp_at(t_src, y_src, t):
    return np.array([np.interp(t, t_src, y_src[:, k]) for k in range(y_src.shape[1])])


def stage_compare():
    out = {'note': __doc__.strip(), 'fractions': FRACS, 'haloes': {}}
    for h, runs in (('B', ['B1', 'B2', 'B5', 'B6']), ('C', ['C1', 'C2', 'C3', 'C4', 'C5'])):
        u = halo_units(h)
        H = {'sigma_hat': u['sigma_hat'], 't0_Gyr': u['t0_myr'] / 1e3, 'nsphere': {}, 'models': {}}
        ns = {r: np.load(os.path.join(OUT, f'nsphere_{r}.npz')) for r in runs
              if os.path.exists(os.path.join(OUT, f'nsphere_{r}.npz'))}
        for r, d in ns.items():
            H['nsphere'][r] = dict(collapse_markers(d['t'], d['mean_rho'][:, 1]),
                                   t_end=float(d['t'][-1]), N=int(d['N']))
        seeds = [H['nsphere'][r]['t_x10'] for r in ns if r in ('B1', 'B2', 'C1', 'C2')
                 and H['nsphere'][r]['t_x10'] is not None]
        ref_t10 = float(np.mean(seeds)) if seeds else None
        H['nsphere_seed_mean_t_x10'] = ref_t10
        H['nsphere_seed_halfspread_t_x10'] = float(np.ptp(seeds) / 2) if len(seeds) > 1 else None
        refs = [ns[r] for r in ns if r in ('B1', 'B2', 'C1', 'C2')]
        models = {'fluid_harmonic_C0.712': f'solver_{h}_a1.0.npz',
                  'fluid_alpha0.5_C0.878': f'solver_{h}_a0.5.npz',
                  'fluid_harmonic_untruncated_NFW': f'solver_{h}_a1.0_nfw.npz'}
        for name, f in models.items():
            d = np.load(os.path.join(OUT, f))
            m = collapse_markers(d['t'], d['mean_rho'][:, 1])
            m['t_final'] = float(d['t_final'])
            m['t_x10_rel_to_nsphere'] = (m['t_x10'] / ref_t10 - 1) if (ref_t10 and m['t_x10']) else None
            # pre-collapse structure: compare at the NSphere snapshots before 0.8 t_x10
            if refs and ref_t10 and 'untruncated' not in name:
                dp, dr, ds_ = [], [], []
                for nd in refs:
                    for k, tt in enumerate(nd['t']):
                        if 0 < tt < 0.8 * ref_t10 and tt <= d['t'][-1]:
                            mp = interp_at(d['t'], d['prof'], tt)
                            # bins with at least 100 particles (Poisson error <= 0.04 dex)
                            vol = 4 / 3 * np.pi * (RBINS[1:] ** 3 - RBINS[:-1] ** 3)
                            cnt = 10 ** nd['prof'][k] * (vol / (4 * np.pi)) / (u['I'] / int(nd['N']))
                            # halo body: inside 0.6 of the truncation radius, where the fluid
                            # model's fixed outer shell does not act (edge reported separately)
                            rc_ = np.sqrt(RBINS[1:] * RBINS[:-1])
                            ok = (np.isfinite(nd['prof'][k]) & (cnt >= 100)
                                  & (rc_ < 0.6 * HALO[h]['falloff']))
                            dp.append(np.sqrt(np.mean((mp[ok] - nd['prof'][k][ok]) ** 2)))
                            dr.append(np.abs(np.log10(interp_at(d['t'], d['lagr_r'], tt)[1:]
                                                      / nd['lagr_r'][k][1:])))
                            ds_.append(np.abs(interp_at(d['t'], d['disp'], tt)[1:] / nd['disp'][k][1:] - 1))
                m['profile_rms_dex_median'] = float(np.median(dp))
                m['lagrangian_radius_dex_median'] = float(np.median(dr))
                m['dispersion_rel_median'] = float(np.median(ds_))
            H['models'][name] = m
        for suf, lab in (('_nfw', 'emulator_untruncated_NFW_input'), ('', 'emulator_exact_profile_input')):
            e = np.load(os.path.join(OUT, f'emulator_{h}{suf}.npz'))
            t_cal = float(e['t_norm_emulator_C0753']) / S_RESCALE
            m = collapse_markers(e['tau'] * t_cal, e['mean_rho'][:, 1])
            m['t_norm_at_C_eff'] = t_cal
            m['median_ensemble_spread_dex'] = float(np.median(e['spread_dex']))
            H['models'][lab] = m
        nsp = H['nsphere']
        if all(r in nsp and nsp[r]['t_x10'] for r in ('C1', 'C2', 'C3', 'C4')):
            mean12 = 0.5 * (nsp['C1']['t_x10'] + nsp['C2']['t_x10'])
            # 2e5-particle reference: mean of C4 and C5 (seeds 0 and 1) when both exist
            n2 = [nsp[r]['t_x10'] for r in ('C4', 'C5') if r in nsp and nsp[r]['t_x10']]
            mean2 = float(np.mean(n2))
            H['convergence'] = {
                'seed_spread_rel': abs(nsp['C2']['t_x10'] / nsp['C1']['t_x10'] - 1),
                'half_timestep_rel_to_C1': nsp['C3']['t_x10'] / nsp['C1']['t_x10'] - 1,
                'double_N_seed0_rel_to_seed_mean': nsp['C4']['t_x10'] / mean12 - 1,
                'double_N_rel_to_seed_mean': mean2 / mean12 - 1,
                'double_N_n_seeds': len(n2),
                'double_N_halfspread_rel': (0.5 * abs(n2[1] - n2[0]) / mean2) if len(n2) == 2 else None,
                'fluid_harmonic_rel_to_double_N':
                    H['models']['fluid_harmonic_C0.712']['t_x10'] / mean2 - 1,
                'fluid_alpha0.5_rel_to_double_N':
                    H['models']['fluid_alpha0.5_C0.878']['t_x10'] / mean2 - 1}
        out['haloes'][h] = H
    json.dump(out, open(os.path.join(OUT, 'independent_halo.json'), 'w'), indent=1)
    print(json.dumps(out['haloes'], indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['solver', 'nsphere', 'emulator', 'compare'])
    ap.add_argument('run', nargs='?')
    ap.add_argument('--alpha', type=float, default=1.0)
    a = ap.parse_args()
    if a.stage == 'solver':
        stage_solver(a.alpha)
    elif a.stage == 'nsphere':
        stage_nsphere(a.run)
    elif a.stage == 'emulator':
        stage_emulator()
    else:
        stage_compare()


if __name__ == '__main__':
    main()
