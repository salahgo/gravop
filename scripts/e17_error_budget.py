"""E17 — decompose the emulator error into its two channels.

The emulator factorises the problem into (i) a scalar head predicting the
termination time t_norm and (ii) a field operator giving rho(r, tau) at
normalised time tau = t / t_norm. In use, both are needed: to predict the halo
at a physical time t one first predicts t_norm, forms tau = t / t_norm_pred,
and then evaluates the field there.

That means a t_norm error does not merely mispredict a collapse time - it makes
the field operator answer the *wrong question*, returning the profile at the
wrong phase of the evolution. No experiment so far has separated these. This
one does, by comparing

    (A) field error at the TRUE tau            (oracle t_norm)
    (B) field error at tau from PREDICTED t_norm (deployed behaviour)

The gap between them is the error the scalar head induces in the fields. If it
dominates, effort should go into the (small, cheap) scalar head rather than the
operator; if not, the operator is the limit.

Output: results/e17/error_budget.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds                                  # noqa: E402
from train import load_data, split, make_tensors                  # noqa: E402
from evaluate_all import load_model, NormObj                      # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e17')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    models, norm = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{s}'))
        models.append(m)
        norm = nrm
    n = NormObj(norm)

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    lrho, lv, prof, par, tc, nc = make_tensors(d, n, DEV)
    taus_np = ds.tau_grid()
    taus = torch.tensor(taus_np, dtype=torch.float32, device=DEV)

    # predicted t_norm per test run (ensemble mean of the scalar head)
    with torch.no_grad():
        cond0 = torch.cat([par[te], torch.zeros(len(te), 1, device=DEV)], 1)
        ltc_pred = np.mean([m(prof[te], cond0)[1][:, 0].cpu().numpy()
                            for m in models], axis=0)
    ltc_pred = ltc_pred * n.tc_sd + n.tc_mu
    ltc_true = d['log_tc'][te]
    tnorm_ratio = 10 ** (ltc_pred - ltc_true)      # predicted / true
    print(f'scalar head: median |t_norm error| = '
          f'{100 * np.median(np.abs(tnorm_ratio - 1)):.2f}%')

    def field_err(tau_query, tau_target_idx):
        """Mean |drho|/rho per (run, tau) with the model asked at tau_query
        and compared against the truth at index tau_target_idx."""
        errs = np.zeros((len(te), len(tau_target_idx)))
        with torch.no_grad():
            for j, ti in enumerate(tau_target_idx):
                q = torch.tensor(tau_query[:, j], dtype=torch.float32,
                                 device=DEV)[:, None]
                cond = torch.cat([par[te], q], 1)
                pr = np.mean([m(prof[te], cond)[0][:, 0].cpu().numpy()
                              for m in models], axis=0)
                tgt = lrho[te, ti].cpu().numpy()
                errs[:, j] = np.abs(10 ** ((pr - tgt) * n.rho_sd) - 1).mean(1)
        return errs

    idx = np.arange(len(taus_np))
    # (A) oracle: ask at the true tau
    tau_oracle = np.tile(taus_np, (len(te), 1))
    # (B) deployed: physical time t = tau_true * t_norm_true; with a predicted
    #     t_norm the queried phase is tau_true / (t_norm_pred/t_norm_true)
    tau_deployed = np.clip(tau_oracle / tnorm_ratio[:, None], 0.0, 1.0)

    eA = field_err(tau_oracle, idx)
    eB = field_err(tau_deployed, idx)

    late = taus_np > 0.92
    out = {
        'n_test': int(len(te)),
        'tnorm_median_abs_err_pct': float(100 * np.median(np.abs(tnorm_ratio - 1))),
        'A_oracle_tnorm': {
            'mean_pct': float(100 * eA.mean()),
            'median_pct': float(100 * np.median(eA)),
            'early_mean_pct': float(100 * eA[:, ~late].mean()),
            'late_mean_pct': float(100 * eA[:, late].mean())},
        'B_predicted_tnorm': {
            'mean_pct': float(100 * eB.mean()),
            'median_pct': float(100 * np.median(eB)),
            'early_mean_pct': float(100 * eB[:, ~late].mean()),
            'late_mean_pct': float(100 * eB[:, late].mean())},
    }
    infl = eB.mean() / eA.mean()
    out['inflation_from_scalar_head'] = float(infl)
    out['induced_error_pct'] = float(100 * (eB.mean() - eA.mean()))

    print(f"\n(A) oracle t_norm     : mean {out['A_oracle_tnorm']['mean_pct']:.3f}%"
          f"  (early {out['A_oracle_tnorm']['early_mean_pct']:.3f}%,"
          f" late {out['A_oracle_tnorm']['late_mean_pct']:.3f}%)")
    print(f"(B) predicted t_norm  : mean {out['B_predicted_tnorm']['mean_pct']:.3f}%"
          f"  (early {out['B_predicted_tnorm']['early_mean_pct']:.3f}%,"
          f" late {out['B_predicted_tnorm']['late_mean_pct']:.3f}%)")
    print(f"\n=> the scalar head inflates field error by {infl:.2f}x "
          f"({out['induced_error_pct']:+.3f} percentage points)")

    # how sensitive is the field to a tau error? d(rho)/d(tau) locally
    with torch.no_grad():
        eps = 0.01
        base = field_err(tau_oracle, idx)
        shifted = field_err(np.clip(tau_oracle - eps, 0, 1), idx)
    out['sensitivity_1pct_tau_shift_pct'] = float(
        100 * (shifted.mean() - base.mean()))
    print(f"   a uniform 1% tau shift alone costs "
          f"{out['sensitivity_1pct_tau_shift_pct']:+.3f} percentage points")

    json.dump(out, open(os.path.join(OUT, 'error_budget.json'), 'w'), indent=1)
    print('\nsaved results/e17/error_budget.json')


if __name__ == '__main__':
    main()
