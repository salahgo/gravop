"""E1 crosscheck — run reference GravothermalSIDM (Boddy et al., GPLv3) at a
config matched to our solver and compare rho(r, t) and collapse time.

Matched config: NFW, n_shells=150, r in [0.01, 100], dimensionless sigma = 0.5
(constant), eps_t = 1e-3, evolve to rho_c = 300 x initial.

Outputs results/e1/crosscheck_ref.npz (their snapshots) and a comparison JSON.
Run our side first: e1_verify.py smoke produces smoke npz (or rerun here).
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'data', 'ref_kboddy', 'SourcePy'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import evolve as kb_evolve  # noqa: E402  (reference code)
import record as kb_record  # noqa: E402
from gravop import solver  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e1')
os.makedirs(OUT, exist_ok=True)
WORK = '/tmp/gravop_crosscheck'
SIGMA_HAT = 0.5
N_SHELLS = 150
RHO_END = 300.0
EPS_T = 1e-3


def run_reference():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    rec = kb_record.HaloRecord(os.path.join(WORK, 'xcheck'))
    # determine dimensionful sigma that gives SIGMA_HAT
    probe = kb_evolve.Halo(kb_record.HaloRecord(os.path.join(WORK, 'probe')),
                           sigma_m_with_units=1.0, n_shells=N_SHELLS)
    sigma_units = SIGMA_HAT / probe.sigma_m
    halo = kb_evolve.Halo(rec, sigma_m_with_units=sigma_units,
                          n_shells=N_SHELLS)
    assert abs(halo.sigma_m - SIGMA_HAT) < 1e-10, halo.sigma_m
    t0 = time.time()
    halo.evolve_halo(rho_factor_end=RHO_END, save_frequency_density=0.05,
                     t_epsilon=EPS_T)
    wall = time.time() - t0
    # collect snapshots
    import glob
    import pickle
    snaps = []
    for f in glob.glob(os.path.join(WORK, 'xcheck', 'time*.pickle')):
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        if isinstance(d, dict) and 'rho' in d:
            snaps.append(d)
    snaps.sort(key=lambda d: d['t'])
    ts = np.array([d['t'] for d in snaps])
    rr = np.array([d['r'] for d in snaps])
    rho = np.array([d['rho'] for d in snaps])
    p = np.array([d['p'] for d in snaps])
    np.savez_compressed(os.path.join(OUT, 'crosscheck_ref.npz'),
                        t=ts, r=rr, rho=rho, p=p, wall=wall)
    print(f'reference: {len(snaps)} snaps, wall {wall:.0f}s')
    return ts, rr, rho, p, wall


def run_ours():
    t0 = time.time()
    res = solver.run_halo(sigma_m=SIGMA_HAT, n_shells=N_SHELLS, eps_t=EPS_T,
                          rho_c_factor=RHO_END, snap_dlnrho=0.02)
    wall = time.time() - t0
    print(f'ours: {len(res["t"])} snaps, wall {wall:.0f}s, '
          f'status {res["status"]}')
    return res, wall


def tc_of(ts, rho_c, factor):
    target = factor * rho_c[0]
    idx = np.argmax(rho_c >= target)
    if rho_c[idx] < target:
        return None
    t0, t1 = ts[idx - 1], ts[idx]
    l0, l1 = np.log(rho_c[idx - 1]), np.log(rho_c[idx])
    return float(t0 + (t1 - t0) * (np.log(target) - l0) / (l1 - l0))


def main():
    res, wall_ours = run_ours()
    rts, rr, rrho, rp, wall_ref = run_reference()

    tc_ours = tc_of(res['t'], res['rho'][:, 3], 100.)
    tc_ref = tc_of(rts, rrho[:, 3], 100.)
    # profile comparison at matched central-density factors
    comps = {}
    for fac in (2., 10., 100.):
        i_o = np.argmax(res['rho'][:, 3] >= fac * res['rho'][0, 3])
        i_r = np.argmax(rrho[:, 3] >= fac * rrho[0, 3])
        lg = np.linspace(-2, 1.5, 100)
        rmid_o = np.log10(0.5 * (np.r_[res['r'][i_o][0],
                                       res['r'][i_o][:-1]] + res['r'][i_o]))
        rmid_r = np.log10(0.5 * (np.r_[rr[i_r][0], rr[i_r][:-1]] + rr[i_r]))
        lro = np.interp(lg, rmid_o, np.log10(res['rho'][i_o]))
        lrr = np.interp(lg, rmid_r, np.log10(rrho[i_r]))
        comps[f'max_abs_dex_fac{fac:g}'] = float(np.max(np.abs(lro - lrr)))
        comps[f'rms_dex_fac{fac:g}'] = float(np.sqrt(np.mean((lro - lrr) ** 2)))
    out = {'sigma_hat': SIGMA_HAT, 'n_shells': N_SHELLS, 'eps_t': EPS_T,
           'tc100_ours': tc_ours, 'tc100_ref': tc_ref,
           'tc100_rel_diff': abs(tc_ours - tc_ref) / tc_ref,
           'wall_ours_s': wall_ours, 'wall_ref_s': wall_ref,
           'speed_ratio_ref_over_ours': wall_ref / wall_ours,
           **comps}
    print(json.dumps(out, indent=1))
    with open(os.path.join(OUT, 'crosscheck.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
