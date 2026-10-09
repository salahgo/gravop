"""P17 — the N-body error budget, by paired residuals instead of quadrature.

WHY THIS EXISTS
---------------
The manuscript reports the emulator at 0.0232 dex against the N = 5e7 N-body
run and the solver it emulates at 0.0197 dex, then writes

    sqrt(0.0232^2 - 0.0197^2) = 0.0123 dex

and calls that the emulation step's contribution. That subtraction is only
valid if the two residual fields are orthogonal, which was never tested. With
    e_emul = e_fluid + e_es
(exact, since all three are log densities on the same points),

    ||e_emul||^2 = ||e_fluid||^2 + ||e_es||^2 + 2 <e_fluid, e_es>

so the quadrature answer is wrong by whatever the cross term is. The referee
report was right that the triangle inequality alone leaves the emulator-solver
RMS difference anywhere in [0.0035, 0.0429] dex.

There is no need to bound it: all three fields are computable on the same
(snapshot, radius) points, so **the emulator's own error is a direct
measurement**, not an inference. This script makes it one.

It also asks whether calibration and validation can be separated at all. The
single time rescaling s = 0.945 (C_eff = 0.753 * s = 0.712) was fitted on snapshots 010-045
-- every epoch the comparison then scores. `snapshot_051` is the only other one,
and it lies past the end of the solver run, which terminates at collapse. So it
cannot be scored without extrapolating, and **no held-out validation of the
calibration is possible with this data.** That is a limitation to state, not a
test to report: scoring it anyway returns 0.4998 dex, which reads as a
catastrophic failure and is purely an extrapolation artefact.

Output: results/e2/error_budget.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import solver, dataset as ds            # noqa: E402
from evaluate_all import load_model, NormObj        # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
R_S, RHO_S, T0G = 1.28, 4.42e7, 0.02045991174237841
SIGMA_HAT = 0.94553884672
S_RESCALE = 0.945
FIT_RANGE = (0.05, 10.0)
# the epochs the s = 0.945 rescaling was fitted on (results/e2/compare_production.json)
CALIBRATION_SNAPSHOTS = ('snapshot_010', 'snapshot_020', 'snapshot_030',
                         'snapshot_040', 'snapshot_045')


def midpoints(r):
    """Shell mid-radii, matching e2_nbody.py's convention exactly."""
    return 0.5 * (np.r_[r[0], r[:-1]] + r)


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models, norm = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{s}'))
        models.append(m)
        norm = nrm
    n = NormObj(norm)

    r, _, rho, p = solver.nfw_profile(n_shells=300)
    lr0, lv0 = ds.regrid_snapshot(r, rho, p)
    prof = torch.tensor(np.stack([(lr0 - n.rho_mu) / n.rho_sd,
                                  (lv0 - n.v_mu) / n.v_sd]),
                        dtype=torch.float32, device=dev)[None]
    par = torch.tensor([[(np.log10(SIGMA_HAT) - n.sig_mu) / n.sig_sd,
                         (0.0 - n.w_mu) / n.w_sd, 0.0]],
                       dtype=torch.float32, device=dev)
    with torch.no_grad():
        c0 = torch.cat([par, torch.zeros(1, 1, device=dev)], 1)
        ltc = np.mean([m(prof, c0)[1][0, 0].item() for m in models])
    t_norm = float(10 ** (ltc * n.tc_sd + n.tc_mu))

    d = np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz'))
    f = np.load(os.path.join(ROOT, 'results', 'e2',
                             'fluid_run_production.npz'))
    lg = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
    keys = sorted({k.rsplit('_', 1)[0] for k in d.files})

    rows, pool = [], {'fluid': [], 'emul': [], 'es': [],
                      'fluid_all': [], 'held_all': []}
    for k in keys:
        t_raw = float(d[k + '_t'])
        if t_raw == 0.0:
            continue
        t_nb = t_raw / T0G * S_RESCALE
        tau = t_nb / t_norm
        emulator_scorable = tau <= 1.0            # the emulator horizon
        # ...but the solver run itself terminates AT collapse, so an epoch past
        # its last stored time cannot be scored either: interpolation would
        # silently compare the collapsed final profile against a later N-body
        # snapshot. Scoring it anyway gave 0.4998 dex, which reads as a
        # catastrophic calibration failure and is nothing of the kind.
        solver_scorable = t_nb <= float(f['t'][-1])

        r_nb = d[k + '_r'] / R_S
        rho_nb = d[k + '_rho'] / (RHO_S * 1e-10)
        sel = (r_nb > FIT_RANGE[0]) & (r_nb < FIT_RANGE[1]) & (rho_nb > 0)
        x = np.log10(r_nb[sel])
        y_nb = np.log10(rho_nb[sel])

        # Solver, onto exactly those radii. Interpolated in TIME between the two
        # bracketing stored profiles and in log-radius within each, which is the
        # convention e2_nbody.py established for the 0.0197 dex number; taking
        # the lower bracket alone shifts the pooled RMS by ~4%.
        ts = f['t']
        i = int(np.clip(np.searchsorted(ts, t_nb) - 1, 0, len(ts) - 2))
        w = float(np.clip((t_nb - ts[i]) / max(ts[i + 1] - ts[i], 1e-300), 0, 1))
        br = []
        for j in (i, i + 1):
            rf, rhof = midpoints(f['r'][j]), f['rho'][j]
            ok = rhof > 0
            br.append(np.interp(x, np.log10(rf[ok]), np.log10(rhof[ok])))
        y_fl = (1 - w) * br[0] + w * br[1]

        held = k not in CALIBRATION_SNAPSHOTS
        row = {'snapshot': k, 't_gadget': t_raw, 'tau': tau,
               'n_points': int(sel.sum()), 'held_out_of_calibration': held,
               'emulator_scorable': emulator_scorable,
               'solver_scorable': solver_scorable}
        if not solver_scorable:
            row['note'] = ('past the end of the solver run (it terminates at '
                           'collapse), so neither the solver nor the emulator '
                           'can be scored here without extrapolating')
            rows.append(row)
            print(row, flush=True)
            continue
        e_fluid = y_fl - y_nb          # solver vs N-body   (model error)
        row['rms_fluid_vs_nbody'] = float(np.sqrt(np.mean(e_fluid ** 2)))
        pool['fluid_all'].append(e_fluid)
        pool['held_all'].append(np.full(len(e_fluid), held))

        if emulator_scorable:
            cond = torch.cat([par, torch.tensor([[tau]], dtype=torch.float32,
                                                device=dev)], 1)
            with torch.no_grad():
                mu = np.mean([m(prof, cond)[0][0, 0].cpu().numpy()
                              for m in models], axis=0) * n.rho_sd + n.rho_mu
            y_em = np.interp(x, lg, mu)
            e_emul = y_em - y_nb       # emulator vs N-body (total)
            e_es = y_em - y_fl         # emulator vs solver (THE emulator error)
            assert np.abs(e_emul - (e_fluid + e_es)).max() < 1e-9
            row.update({
                'rms_emul_vs_nbody': float(np.sqrt(np.mean(e_emul ** 2))),
                'rms_emul_vs_fluid': float(np.sqrt(np.mean(e_es ** 2))),
                'cross_term_correlation': float(
                    np.corrcoef(e_fluid, e_es)[0, 1]),
            })
            for nm, v in (('fluid', e_fluid), ('emul', e_emul), ('es', e_es)):
                pool[nm].append(v)
        else:
            row['note'] = ('beyond the emulator time horizon; scores the '
                           'solver and the calibration only')
        rows.append(row)
        print(row, flush=True)

    ef = np.concatenate(pool['fluid'])
    ee = np.concatenate(pool['emul'])
    es = np.concatenate(pool['es'])
    # the solver is scorable everywhere, including past the emulator's horizon
    ef_all = np.concatenate(pool['fluid_all'])
    hm = np.concatenate(pool['held_all']).astype(bool)

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2)))

    cross = float(2 * np.mean(ef * es))
    quad_claim = float(np.sqrt(max(rms(ee) ** 2 - rms(ef) ** 2, 0.0)))
    out = {
        'reference': 'Fischer, Yu & Dolag 2025, arXiv:2506.06269, N=5e7',
        's_rescale': S_RESCALE, 'C_eff': 0.753 * S_RESCALE,
        'calibration_snapshots': list(CALIBRATION_SNAPSHOTS),
        'emulator_t_norm': t_norm,
        'n_points': int(len(ef)),
        'rms_fluid_vs_nbody': rms(ef),
        'rms_emul_vs_nbody': rms(ee),
        'rms_emul_vs_fluid_DIRECT': rms(es),
        'quadrature_subtraction_claims': quad_claim,
        'cross_term_2<e_fluid,e_es>': cross,
        'cross_term_correlation': float(np.corrcoef(ef, es)[0, 1]),
        'identity_check': {
            'lhs_rms_emul_sq': rms(ee) ** 2,
            'rhs_sum': rms(ef) ** 2 + rms(es) ** 2 + cross,
        },
        'calibration_transfer': {
            'note': ('NO held-out validation of the calibration is possible '
                     'with this data. The s = 0.945 rescaling was fitted on '
                     'snapshots 010-045, which is every epoch the comparison '
                     'scores. The only other epoch, snapshot_051, lies past '
                     'the end of the solver run (which terminates at collapse) '
                     'and so cannot be scored without extrapolating. '
                     'Calibration and validation therefore overlap completely, '
                     'and the manuscript must say so rather than imply an '
                     'independent test.'),
            'n_points_fitted': int((~hm).sum()),
            'n_points_held_out': int(hm.sum()),
            'rms_fluid_vs_nbody_fitted': rms(ef_all[~hm]),
            'rms_fluid_vs_nbody_held_out': (rms(ef_all[hm]) if hm.any()
                                            else None),
        },
        'per_snapshot': rows,
    }
    json.dump(out, open(os.path.join(ROOT, 'results', 'e2',
                                     'error_budget.json'), 'w'), indent=1)

    print(f'\n--- pooled over {len(ef)} (snapshot, radius) points ---')
    print(f'  solver   vs N-body           {rms(ef):.4f} dex')
    print(f'  emulator vs N-body           {rms(ee):.4f} dex')
    print(f'  emulator vs solver  DIRECT   {rms(es):.4f} dex')
    print(f'  quadrature subtraction says  {quad_claim:.4f} dex  '
          f'-> off by {abs(quad_claim - rms(es)) / rms(es) * 100:.0f}%')
    print(f'  cross term 2<e_f,e_es>       {cross:+.6f}   '
          f'(corr {out["cross_term_correlation"]:+.3f})')
    ct = out['calibration_transfer']
    if ct['rms_fluid_vs_nbody_held_out'] is not None:
        print(f'\n  calibration transfer (solver vs N-body):')
        print(f'    fitted epochs   {ct["rms_fluid_vs_nbody_fitted"]:.4f} dex '
              f'({ct["n_points_fitted"]} pts)')
        print(f'    HELD OUT epoch  {ct["rms_fluid_vs_nbody_held_out"]:.4f} dex '
              f'({ct["n_points_held_out"]} pts, snapshot_051)')


if __name__ == '__main__':
    main()
