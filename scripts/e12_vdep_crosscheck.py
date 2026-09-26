"""E12 — cross-code validation of the VELOCITY-DEPENDENT branch.

The Section-3.4 cross-check validated only the constant-cross-section branch
against the public GravothermalSIDM code. The emulator spans velocity-dependent
(Yukawa-Born) cross-sections, so that branch needs its own validation. Both
codes implement the same order-1, index-5 viscosity-averaged Yukawa-Born fit of
Outmezguine et al. (2023), so an exact comparison is possible.

Two parts:

  A. The velocity-dependence function F(v/w) itself is compared point by point
     between the two implementations over five decades in v/w, for both the
     t+u-channel and t-channel-only variants. This is the sharpest possible
     test of the physics input: it is a pure function comparison.

  B. Full trajectories are compared over a fixed dimensionless time window,
     covering core formation and the onset of collapse. A fixed time window
     rather than a central-density threshold is used because the reference
     implementation's explicit conduction step becomes prohibitively expensive
     deep in the collapse phase (that cost is precisely what this work removes,
     Section 3.2), so a threshold-based comparison would not terminate.

Usage: e12_vdep_crosscheck.py [--n_shells 150] [--t_end 100]
Output: results/e12/vdep_crosscheck.json
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'data', 'ref_kboddy', 'SourcePy'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import evolve as kb_evolve   # noqa: E402  reference code (GPLv3)
import record as kb_record   # noqa: E402
from gravop import solver, dataset as ds  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e12')
os.makedirs(OUT, exist_ok=True)
WORK = '/tmp/gravop_e12'

REF_MODEL_TU = 'YukawaBornViscosityApprox_K5_order1'
REF_MODEL_T = 'YukawaBornViscosityApproxTchannel_K5_order1'

# (sigma_0-hat, w-hat). Chosen so the *effective* cross-section sigma_0 F(v/w)
# is order unity: the Yukawa suppression at small w-hat is severe (F ~ 2e-3 at
# w-hat = 0.3), which would make the reference run take days without testing
# anything extra.
# (sigma_0-hat, w-hat, t_end). The per-case time window is set to the range in
# which the REFERENCE implementation remains numerically stable: at longer
# times its undamped hydrostatic solve overflows as the shells crowd (see
# Section 3.2 and the N=300 divide-by-zero of Section 3.4). Our solver runs all
# of these to collapse without difficulty.
CASES = [(5.0, 2.0, 30.0), (5.0, 5.0, 30.0),
         (2.0, 3.0, 60.0), (1.0, 10.0, 60.0)]


def compare_F():
    """Part A: point-by-point comparison of F(v/w) between the two codes."""
    s = np.logspace(-3, 2, 4000)
    out = {}
    for tag, t_only, ref_name in (('t+u channel', False, REF_MODEL_TU),
                                  ('t channel', True, REF_MODEL_T)):
        ref = kb_evolve.average_YukawaBornViscosityApprox(5, 1, t_only)
        mid, mp_ = solver.model_params(
            'yukawa_t' if t_only else 'yukawa_tu', yb_index=5)
        a = np.array([float(ref(x)) for x in s])
        b = np.array([solver._F_elastic(x, mid, mp_) for x in s])
        out[tag] = {'max_abs_diff': float(np.max(np.abs(a - b))),
                    'max_rel_diff': float(np.max(np.abs(a - b)
                                                 / np.maximum(np.abs(a), 1e-300))),
                    'bitwise_identical': bool(np.array_equal(a, b)),
                    'F_range': [float(a.min()), float(a.max())],
                    'n_points': int(len(s))}
        print(f'  F ({tag}): max rel diff {out[tag]["max_rel_diff"]:.2e}, '
              f'identical={out[tag]["bitwise_identical"]}', flush=True)
    return out


def run_reference(sigma_hat, w_hat, n_shells, t_end):
    tag = f's{sigma_hat}_w{w_hat}'.replace('.', 'p')
    d = os.path.join(WORK, tag)
    if os.path.exists(d):
        shutil.rmtree(d)
    probe = kb_evolve.Halo(
        kb_record.HaloRecord(os.path.join(WORK, 'probe_' + tag)),
        sigma_m_with_units=1.0, w_units=1.0, n_shells=n_shells,
        model_elastic_scattering_lmfp=REF_MODEL_TU,
        model_elastic_scattering_smfp=REF_MODEL_TU)
    halo = kb_evolve.Halo(
        kb_record.HaloRecord(d),
        sigma_m_with_units=sigma_hat / probe.sigma_m,
        w_units=w_hat / probe.w, n_shells=n_shells,
        model_elastic_scattering_lmfp=REF_MODEL_TU,
        model_elastic_scattering_smfp=REF_MODEL_TU)
    assert abs(halo.sigma_m - sigma_hat) / sigma_hat < 1e-9
    assert abs(halo.w - w_hat) / w_hat < 1e-9
    t0 = time.time()
    halo.evolve_halo(t_end=t_end, save_frequency_density=0.02, t_epsilon=1e-3)
    wall = time.time() - t0
    snaps = []
    for f in glob.glob(os.path.join(d, 'time*.pickle')):
        with open(f, 'rb') as fh:
            s_ = pickle.load(fh)
        if isinstance(s_, dict) and 'rho' in s_:
            snaps.append(s_)
    snaps.sort(key=lambda x: x['t'])
    return (np.array([x['t'] for x in snaps]),
            np.array([x['r'] for x in snaps]),
            np.array([x['rho'] for x in snaps]), wall)


def profile_at(ts, rr, rrho, t, lg):
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    w = np.clip((t - ts[i]) / max(ts[i + 1] - ts[i], 1e-300), 0, 1)
    out = []
    for j in (i, i + 1):
        rm = np.log10(ds.shell_midpoints(rr[j]))
        out.append(np.interp(lg, rm, np.log10(rrho[j])))
    return (1 - w) * out[0] + w * out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_shells', type=int, default=150)
    ap.add_argument('--t_end', type=float, default=100.0)
    args = ap.parse_args()
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    print('Part A: velocity-dependence function', flush=True)
    fcmp = compare_F()

    print('\nPart B: trajectories (per-case time windows)', flush=True)
    lg = np.linspace(-2, 1.5, 100)
    rows = []
    for sigma_hat, w_hat, t_end in CASES:
        t0 = time.time()
        ours = solver.run_halo(sigma_m=sigma_hat, w=w_hat,
                               model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
                               n_shells=args.n_shells, eps_t=3e-4,
                               t_end=t_end, rho_c_factor=1e30,
                               kn_min=0.0, timestep_mode='implicit')
        wall_ours = time.time() - t0
        rts, rr, rrho, wall_ref = run_reference(sigma_hat, w_hat,
                                                args.n_shells, t_end)
        # compare central density history and profiles at matched times
        probe_times = [t for t in (0.25, 0.5, 0.75, 1.0)
                       if t * t_end <= min(ours['t'][-1], rts[-1])]
        dprof, drhoc = [], []
        for frac in probe_times:
            t = frac * t_end
            po = profile_at(ours['t'], ours['r'], ours['rho'], t, lg)
            pr = profile_at(rts, rr, rrho, t, lg)
            dprof.append(float(np.sqrt(np.mean((po - pr) ** 2))))
            io = int(np.clip(np.searchsorted(ours['t'], t) - 1, 0,
                             len(ours['t']) - 1))
            ir = int(np.clip(np.searchsorted(rts, t) - 1, 0, len(rts) - 1))
            drhoc.append(float(abs(ours['rho'][io, 3] / rrho[ir, 3] - 1)))
        F1 = solver._F_elastic(
            1.0 / w_hat, *solver.model_params('yukawa_tu', yb_index=5))
        row = {'sigma_hat': sigma_hat, 'w_hat': w_hat, 't_end': t_end,
               'F_at_v1': float(F1), 'sigma_eff_at_v1': float(sigma_hat * F1),
               'probe_time_fractions': probe_times,
               'rms_dex_per_time': dprof,
               'worst_rms_dex': max(dprof) if dprof else None,
               'rho_c_rel_diff_per_time': drhoc,
               'worst_rho_c_rel_diff': max(drhoc) if drhoc else None,
               'wall_ours_s': wall_ours, 'wall_ref_s': wall_ref,
               'speedup': wall_ref / wall_ours,
               'n_steps_ours': ours['n_steps']}
        rows.append(row)
        print(f'  sigma_0={sigma_hat}, w={w_hat} '
              f'(sigma_eff~{sigma_hat * F1:.2f}): '
              f'worst profile {row["worst_rms_dex"]:.4f} dex, '
              f'worst rho_c {100 * row["worst_rho_c_rel_diff"]:.2f}%, '
              f'{wall_ours:.2f}s vs {wall_ref:.0f}s '
              f'({row["speedup"]:.0f}x)', flush=True)

    summary = {'F_function_comparison': fcmp,
               'n_shells': args.n_shells,
               'note_t_end': 'per-case, set by reference-code stability',
               'model': REF_MODEL_TU,
               'worst_profile_rms_dex': max(r['worst_rms_dex'] for r in rows),
               'worst_rho_c_rel_diff': max(r['worst_rho_c_rel_diff']
                                           for r in rows),
               'median_speedup': float(np.median([r['speedup']
                                                  for r in rows])),
               'cases': rows}
    json.dump(summary, open(os.path.join(OUT, 'vdep_crosscheck.json'), 'w'),
              indent=1)
    print(f'\nWORST profile disagreement: '
          f'{summary["worst_profile_rms_dex"]:.4f} dex')
    print(f'WORST central-density disagreement: '
          f'{100 * summary["worst_rho_c_rel_diff"]:.2f}%')
    print(f'median speed-up: {summary["median_speedup"]:.0f}x')


if __name__ == '__main__':
    main()
