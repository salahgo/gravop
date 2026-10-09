"""E11 — end-to-end validation: emulator predictions vs N-body simulation.

Sections 3.5 and 6 validate two links separately: solver-vs-N-body and
emulator-vs-solver. This script closes the chain by comparing the *emulator*
directly against the Fischer, Yu & Dolag (2025) N = 5e7 particle run, with no
solver call at prediction time.

The halo is NFW with a constant cross-section sigma-hat = 0.9455, inside the
constant-cross-section branch of the training distribution. The emulator
reproduces the C = 0.753 fluid model, so the same single time-rescaling
calibration used in Section 3.5 (s = 0.945, i.e. C_eff = C * s = 0.712) is applied to
the N-body times; no other freedom is used.

TWO CONDITIONINGS (P2). `--conditioning tau` is the original arm:
the emulator predicts its own termination time and the N-body time is converted
to tau, so the comparison inherits the deployment penalty E17 measured.
`--conditioning phase` instead reads the state coordinate off the N-body
snapshot being compared against -- the central density contrast and the
post-minimum flag -- and needs no termination time at all.

The phase arm is partly self-explaining and the honest treatment is to say so
and to measure it: phi is taken from the innermost radii of the very profile
the prediction is scored against. The same leakage control the cross-domain
study uses is therefore applied here, re-scoring outside `LEAK_EXCLUDE_RS`, and
both numbers are reported. Reading phi from N-body also samples the least
converged part of an N-body profile, which is a limitation of the measurement
rather than of the method.

Output: results/e11/emulator_vs_nbody{,_phase}.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import solver, dataset as ds, phase as gphase  # noqa: E402
from evaluate_all import load_model, NormObj  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e11')
os.makedirs(OUT, exist_ok=True)

R_S, RHO_S, T0G = 1.28, 4.42e7, 0.02045991174237841
SIGMA_HAT = 0.94553884672
S_RESCALE = 0.945          # from E2 (C_eff = 0.753 * s = 0.712); no further tuning
FIT_RANGE = (0.05, 10.0)   # r/r_s window used in Section 3.5
LEAK_EXCLUDE_RS = 0.2      # phi is read inside this radius; leak check drops it


def nbody_central_contrast(d, keys, lrho0_grid, r_phi):
    """phi and the post-minimum flag for every N-body snapshot.

    phi = log10(rho(r_phi, t) / rho(r_phi, 0)). Both densities are taken at
    the SAME radius `r_phi` -- the one the ensemble was trained to read its
    phase at -- because the quantity is a contrast, and evaluating numerator
    and denominator at different radii would silently fold the profile slope
    into it.

    `r_phi` must be resolved by the N-body profiles. The emulator grid starts
    at 0.0032 r_s while these snapshots resolve 0.0084 r_s, so an ensemble
    trained at grid index 0 cannot be conditioned here without extrapolating;
    the caller checks and refuses rather than silently extrapolating.

    The flag is set after the epoch of minimum density at r_phi, read from the
    ordered snapshot series -- observable, since it needs only the sign of the
    trend so far.
    """
    lg = gphase.grid_log_radii()
    lrho0_at = float(np.interp(np.log10(r_phi), lg, lrho0_grid))
    times, lc = [], []
    for k in keys:
        r_nb = d[k + '_r'] / R_S
        rho_nb = d[k + '_rho'] / (RHO_S * 1e-10)
        sel = (r_nb > 0) & (rho_nb > 0)
        order = np.argsort(r_nb[sel])
        times.append(float(d[k + '_t']))
        lc.append(float(np.interp(np.log10(r_phi),
                                  np.log10(r_nb[sel][order]),
                                  np.log10(rho_nb[sel][order]))))
    times, lc = np.asarray(times), np.asarray(lc)
    imin = int(np.argsort(times)[int(np.argmin(lc[np.argsort(times)]))])
    return {k: (float(lc[j] - lrho0_at),
                1.0 if times[j] > times[imin] else 0.0)
            for j, k in enumerate(keys)}


def nbody_min_radius(d, keys):
    """Innermost radius resolved by every snapshot, in r_s."""
    return min(float((d[k + '_r'] / R_S)[((d[k + '_r'] / R_S) > 0)
                                         & (d[k + '_rho'] > 0)].min())
               for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditioning', default='tau', choices=['tau', 'phase'])
    ap.add_argument('--runs', default=None,
                    help='ensemble directory prefix; defaults to gravop_s '
                         '(tau) or gravop_phase_nbody_s (phase), the latter '
                         'being the ensemble trained to read its phase at a '
                         'radius these N-body profiles resolve')
    args = ap.parse_args()
    prefix = args.runs or ('gravop_phase_nbody_s'
                           if args.conditioning == 'phase' else 'gravop_s')
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models, norm = [], None
    for s in range(5):
        m, nrm, a = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'{prefix}{s}'))
        models.append(m)
        norm = nrm
    n = NormObj(norm)

    # NFW initial condition on the emulator's grid
    r, m_, rho, p = solver.nfw_profile(n_shells=300)
    lr0, lv0 = ds.regrid_snapshot(r, rho, p)
    prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd,
                                  (lv0 - n.v_mu) / n.v_sd]),
                        dtype=torch.float32, device=dev)[None]
    par = torch.tensor([[(np.log10(SIGMA_HAT) - n.sig_mu) / n.sig_sd,
                         (0.0 - n.w_mu) / n.w_sd, 0.0]],
                       dtype=torch.float32, device=dev)

    # Scalar head -> termination time. ONLY the tau arm has one worth reading:
    # e28_phase_ensemble.py optimises the field loss alone, so the phase
    # members' scalar head is never trained and its output is meaningless.
    # Reporting it anyway would put a fabricated number in the record, so the
    # phase arm reports None and the tau column is left to speak for itself.
    t_norm = None
    if args.conditioning == 'tau':
        with torch.no_grad():
            cond0 = torch.cat([par, torch.zeros(1, 1, device=dev)], 1)
            ltc = np.mean([m(prof, cond0)[1][0, 0].item() for m in models])
        t_norm = float(10 ** (ltc * n.tc_sd + n.tc_mu))
        print(f'emulator predicted t_norm = {t_norm:.1f} t0', flush=True)
    else:
        print('phase-conditioned: no t_norm is computed or needed', flush=True)

    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    keys = sorted({k.rsplit('_', 1)[0] for k in d.files})
    lg = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
    phases, r_phi = None, None
    if args.conditioning == 'phase':
        phi_index = a.get('phi_index', 0)
        r_min = nbody_min_radius(d, keys)
        try:
            r_phi = gphase.check_phase_radius(
                phi_index, r_min, what='these N-body profiles',
                suffix='_nbody')
        except ValueError as exc:
            raise SystemExit(str(exc))
        phases = nbody_central_contrast(d, keys, lr0, r_phi)
        print(f'phase arm: reading phi at r = {r_phi:.5f} r_s '
              f'(grid index {phi_index}); N-body resolves to {r_min:.5f} r_s',
              flush=True)

    rows, sq, cnt = [], 0.0, 0
    sq_leak, cnt_leak = 0.0, 0
    for k in keys:
        t_nb = float(d[k + '_t']) / T0G * S_RESCALE
        if t_nb == 0.0:
            continue
        tau = (t_nb / t_norm) if t_norm else float('nan')
        if args.conditioning == 'tau':
            if tau > 1.0:
                rows.append({'snapshot': k, 'tau': float(tau),
                             'note': 'beyond emulator time horizon'})
                continue
            extra = torch.tensor([[tau]], dtype=torch.float32, device=dev)
        else:
            # no termination time anywhere in this branch: the conditioning is
            # read off the snapshot itself
            phi, flag = phases[k]
            extra = torch.tensor([[(phi - n.p_mu) / n.p_sd, flag]],
                                 dtype=torch.float32, device=dev)
        cond = torch.cat([par, extra], 1)
        with torch.no_grad():
            preds = [m(prof, cond)[0][0, 0].cpu().numpy() for m in models]
        mu = np.mean(preds, axis=0) * n.rho_sd + n.rho_mu
        sd = np.std(preds, axis=0) * n.rho_sd

        r_nb = d[k + '_r'] / R_S
        rho_nb = d[k + '_rho'] / (RHO_S * 1e-10)
        sel = ((r_nb > FIT_RANGE[0]) & (r_nb < FIT_RANGE[1]) & (rho_nb > 0))
        pred = np.interp(np.log10(r_nb[sel]), lg, mu)
        spread = np.interp(np.log10(r_nb[sel]), lg, sd)
        dif = pred - np.log10(rho_nb[sel])
        sq += np.sum(dif ** 2)
        cnt += len(dif)
        # leakage control: phi was read from the innermost radii, so re-score
        # outside them. Applied to both arms so the two stay comparable.
        outer = r_nb[sel] > LEAK_EXCLUDE_RS
        sq_leak += np.sum(dif[outer] ** 2)
        cnt_leak += int(outer.sum())
        row = {'snapshot': k, 't_gadget': float(d[k + '_t']),
               'tau': (float(tau) if t_norm else None),
               'rms_dex': float(np.sqrt(np.mean(dif ** 2))),
               'max_abs_dex': float(np.max(np.abs(dif))),
               'mean_ensemble_sd_dex': float(np.mean(spread))}
        if outer.any():
            row['rms_dex_outer'] = float(np.sqrt(np.mean(dif[outer] ** 2)))
        if args.conditioning == 'phase':
            row['phi'], row['flag'] = phases[k]
        rows.append(row)
        print(row, flush=True)

    suffix = '' if args.conditioning == 'tau' else '_phase'
    out = {'sigma_hat': SIGMA_HAT, 's_rescale': S_RESCALE,
           'conditioning': args.conditioning, 'ensemble_prefix': prefix,
           'emulator_t_norm': t_norm,
           't_norm_used': args.conditioning == 'tau',
           'global_rms_dex': float(np.sqrt(sq / cnt)),
           'global_rms_dex_outer': (float(np.sqrt(sq_leak / cnt_leak))
                                    if cnt_leak else None),
           'leak_exclude_rs': LEAK_EXCLUDE_RS,
           'fit_range_rs': list(FIT_RANGE),
           'reference': 'Fischer, Yu & Dolag 2025, arXiv:2506.06269, N=5e7',
           'note': 'emulator only; no solver call at prediction time',
           'per_snapshot': rows}
    json.dump(out, open(os.path.join(
        OUT, f'emulator_vs_nbody{suffix}.json'), 'w'), indent=1)
    print(f"\nGLOBAL ({args.conditioning}): emulator vs N-body = "
          f"{out['global_rms_dex']:.4f} dex RMS")
    if out['global_rms_dex_outer'] is not None:
        print(f"  outside {LEAK_EXCLUDE_RS} r_s (phi is read inside it): "
              f"{out['global_rms_dex_outer']:.4f} dex RMS")
    print('(solver vs N-body was 0.0197 dex, E2-production)')


if __name__ == '__main__':
    main()
