"""Domain baseline: the Yang et al. (2024) parametric SIDM halo model.

The referee question this answers is "what does a neural operator buy over a
cheap calibrated formula?". Yang, Nadler, Yu & Zhong (JCAP 02:032, 2024;
arXiv:2305.16176) give a closed-form density profile for an isolated SIDM halo
at any point in its gravothermal evolution, calibrated against a controlled
N-body run. It costs no training and microseconds to evaluate, so if it were
competitive on our test split the surrogate would be hard to justify.

The model, their eqs. (2.1)-(2.3), for an initial NFW halo with scale density
rho_s0 and scale radius r_s0, at normalised time tau = t / t_c:

    rho(r) = rho_s / [ (r^4 + r_c^4)^(1/4) / r_s * (1 + r/r_s)^2 ]

    rho_s/rho_s0 = 2.033 + 0.7381 tau + 7.264 tau^5 - 12.73 tau^7
                   + 9.915 tau^9 + (1 - 2.033) ln(tau + 0.001)/ln(0.001)
    r_s/r_s0     = 0.7178 - 0.1026 tau + 0.2474 tau^2 - 0.4079 tau^3
                   + (1 - 0.7178) ln(tau + 0.001)/ln(0.001)
    r_c/r_s0     = 2.555 sqrt(tau) - 3.632 tau + 2.131 tau^2 - 1.415 tau^3
                   + 0.4683 tau^4

with their eq. (2.2) for the collapse timescale, which in our dimensionless
units (G = 1, times in t_0 = (4 pi G rho_s)^-1/2, rho_s0 = r_s0 = 1) reduces to
t_c = (150 / C) / sigma_hat_eff with C = 0.75.

SCOPE. The model is calibrated for *isolated NFW haloes with an effective
constant cross-section*. On our test split that is a strict subset: the cored
and gNFW initial-condition families and the velocity-dependent runs are outside
what it claims. We evaluate only where it is defined and report the count,
because scoring a model outside its stated domain is not a comparison.

FAIRNESS. Two variants are reported. `analytic` uses their t_c with their
C = 0.75. `calibrated` fits a single global time-rescaling on the *training*
split and applies it unchanged to test -- the same one-parameter courtesy we
extend to the fluid model in the N-body comparison, and it absorbs the
difference between their collapse-time convention and ours.

Metric is identical to train.py's: mean over radii of |rho_pred/rho_true - 1|,
then per-trajectory mean over the tau grid, then the median over trajectories.

Usage:  python scripts/baselines_parametric.py --out results/benchmark
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds          # noqa: E402
from train import load_data, split        # noqa: E402

IC_NFW, MODEL_CONST = 0, 0
C_CAL = 0.75          # their calibration constant, their eq. (2.2)
TC_PREFACTOR = 150.0  # their eq. (2.2)


def profile_params(tau):
    """Yang et al. eq. (2.3). tau may be an array; returns (rho_s, r_s, r_c)
    in units of the initial NFW rho_s0, r_s0."""
    tau = np.clip(tau, 0.0, 1.0)
    lg = np.log(tau + 0.001) / np.log(0.001)
    rho_s = (2.033 + 0.7381 * tau + 7.264 * tau ** 5 - 12.73 * tau ** 7
             + 9.915 * tau ** 9 + (1 - 2.033) * lg)
    r_s = (0.7178 - 0.1026 * tau + 0.2474 * tau ** 2 - 0.4079 * tau ** 3
           + (1 - 0.7178) * lg)
    r_c = (2.555 * np.sqrt(tau) - 3.632 * tau + 2.131 * tau ** 2
           - 1.415 * tau ** 3 + 0.4683 * tau ** 4)
    return rho_s, r_s, r_c


def density(r, rho_s, r_s, r_c):
    """Yang et al. eq. (2.1) with beta = 4. Shapes broadcast as (..., n_r)."""
    inner = (r ** 4 + r_c ** 4) ** 0.25 / r_s
    return rho_s / (inner * (1.0 + r / r_s) ** 2)


def predict(sigma_hat, t_norm, taus, r, scale=1.0):
    """Predicted density on the (tau x r) grid for one trajectory.

    `scale` multiplies the model's tau, i.e. rescales its collapse clock.
    """
    t_c = TC_PREFACTOR / C_CAL / sigma_hat          # their eq. (2.2), our units
    tau_model = scale * (taus * t_norm) / t_c
    rho_s, r_s, r_c = profile_params(tau_model)
    return density(r[None, :], rho_s[:, None], r_s[:, None], r_c[:, None])


def rel_err(pred, true):
    """train.py's metric: mean over radii, then mean over tau, per trajectory."""
    return np.abs(pred / true - 1.0).mean(axis=2).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/d1/train.h5')
    ap.add_argument('--out', default='results/benchmark')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py
    d = load_data(args.data)
    with h5py.File(args.data) as f:
        ic_type = f['ic_type'][:]
    n = len(d['log_tc'])
    tr, va, te = split(n)

    r = ds.r_grid()
    taus = ds.tau_grid()
    sigma_hat = 10.0 ** np.asarray(d['log_sigma0'], dtype=np.float64)
    t_norm = 10.0 ** np.asarray(d['log_tc'], dtype=np.float64)
    rho_true = 10.0 ** np.asarray(d['lrho'], dtype=np.float64)

    # Strictly in domain: NFW initial condition AND a constant cross-section,
    # which is exactly what their calibration run was. Their section 6 extends
    # the model to velocity-dependent scattering through an *effective* constant
    # cross-section, so we also report the NFW velocity-dependent runs under
    # sigma_eff = sigma_0 F(v_0 / w) with the same viscosity-averaged
    # Yukawa-Born F the solver uses. v_0 is read from the initial profile, so
    # this stays a deployable quantity rather than an oracle one.
    from gravop import solver as gsolver
    # the dataset's model_id is a binary flag; the non-constant runs are all
    # the order-1, index-5 viscosity-averaged Yukawa--Born t+u variant
    YB_MID, YB_MP = gsolver.model_params('yukawa_tu', yb_index=5)
    is_nfw = ic_type == IC_NFW
    is_const = np.asarray(d['model_id']) == MODEL_CONST
    strict = is_nfw & is_const

    v0 = 10.0 ** np.asarray(d['lv0'], dtype=np.float64)
    w_hat = 10.0 ** np.asarray(d['log_w'], dtype=np.float64)
    sigma_eff = sigma_hat.copy()
    for i in np.nonzero(is_nfw & ~is_const)[0]:
        # characteristic velocity of the initial halo: v at the scale radius
        j = int(np.argmin(np.abs(r - 1.0)))
        F = gsolver._F_elastic(v0[i, j] / w_hat[i], YB_MID, YB_MP)
        sigma_eff[i] = sigma_hat[i] * F

    def run(idx, scale):
        idx = np.asarray(idx)
        pred = np.stack([predict(sigma_eff[i], t_norm[i], taus, r, scale)
                         for i in idx])
        return rel_err(pred, rho_true[idx])

    res = {}
    for pop, mask in (('nfw_constant', strict), ('nfw_all', is_nfw)):
        tr_idx = np.array([i for i in tr if mask[i]])
        te_idx = np.array([i for i in te if mask[i]])
        # one global time-rescaling, fitted on TRAIN only
        grid = np.linspace(0.2, 3.0, 141)
        med_tr = np.array([np.median(run(tr_idx, s)) for s in grid])
        scale = float(grid[int(np.argmin(med_tr))])
        res[pop] = {'n_test': int(len(te_idx)), 'n_train_for_calib':
                    int(len(tr_idx))}
        for name, s in (('analytic', 1.0), ('calibrated', scale)):
            e = run(te_idx, s)
            res[pop][name] = {'median_rel_rho_pct': float(100 * np.median(e)),
                              'mean_rel_rho_pct': float(100 * e.mean()),
                              'time_scale': s}

    # ---- the same metric, same trajectories, for the shipped ensemble, so the
    # ---- comparison is like-for-like rather than against the full-split number
    gravop = {}
    try:
        import torch
        from gravop.model import GravOpNet
        from train import Norm, make_tensors, batch_inputs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        norm = Norm(d, tr)
        tensors = make_tensors(d, norm, device)
        lrho_t, lv_t, prof, par, _, _ = tensors
        taus_t = torch.tensor(taus, dtype=torch.float32, device=device)
        seeds = [0, 1, 2, 3, 4]
        for pop, mask in (('nfw_constant', strict), ('nfw_all', is_nfw)):
            idx = np.array([i for i in te if mask[i]])
            per_seed = []
            for s in seeds:
                ck = f'results/runs/gravop_s{s}/model.pt'
                if not os.path.exists(ck):
                    continue
                blob = torch.load(ck, map_location=device, weights_only=False)
                a = blob.get('args', {})
                m = GravOpNet(n_r=ds.N_R, width=a.get('width', 64),
                              modes=a.get('modes', 32),
                              n_layers=a.get('layers', 4))
                m.load_state_dict(blob['model'])
                m = m.to(device).eval()
                errs = []
                with torch.no_grad():
                    for i in idx:
                        ri = np.full(len(taus), i)
                        ti = np.arange(len(taus))
                        pi, ci = batch_inputs(prof, par, taus_t, ri, ti)
                        fields, _ = m(pi, ci)
                        pr = fields[:, 0] * norm.rho_sd + norm.rho_mu
                        tgt = lrho_t[ri, ti] * norm.rho_sd + norm.rho_mu
                        rel = (10 ** pr / 10 ** tgt - 1).abs().mean(dim=1)
                        errs.append(float(rel.mean()))
                per_seed.append(float(np.median(errs)))
            if per_seed:
                gravop[pop] = {
                    'median_rel_rho_pct': float(100 * np.mean(per_seed)),
                    'per_seed_median_pct': [100 * x for x in per_seed],
                    'n_test': int(len(idx)),
                    'definition': ('mean over production seeds of each seed\'s '
                                   'median per-trajectory error, oracle phase, '
                                   'same trajectories as the parametric model'),
                }
    except Exception as exc:                       # checkpoints may be absent
        gravop = {'unavailable': str(exc)}

    out = {
        'model': 'Yang, Nadler, Yu & Zhong (2024), JCAP 02:032, eqs. (2.1)-(2.3)',
        'gravop_same_trajectories': gravop,
        'doi': '10.1088/1475-7516/2024/02/032',
        'scope': ('calibrated for isolated NFW haloes with an effective '
                  'constant cross-section; evaluated only on that subset'),
        'n_test_total': int(len(te)),
        'metric': ("train.py's: mean over radii of |rho_pred/rho_true - 1|, "
                   'then per-trajectory mean over tau, then median over '
                   'trajectories'),
        'results': res,
    }
    p = os.path.join(args.out, 'parametric.json')
    json.dump(out, open(p, 'w'), indent=1)

    print(f'Yang et al. (2024) parametric model, {len(te)}-run test split')
    for pop in ('nfw_constant', 'nfw_all'):
        v = res[pop]
        print(f"  {pop} (n = {v['n_test']})")
        for k in ('analytic', 'calibrated'):
            print(f"    {k:11} median {v[k]['median_rel_rho_pct']:7.2f}%   "
                  f"mean {v[k]['mean_rel_rho_pct']:7.2f}%   "
                  f"time scale {v[k]['time_scale']:.3f}")
        g = gravop.get(pop)
        if g:
            print(f"    {'GravOp':11} median {g['median_rel_rho_pct']:7.2f}%"
                  f"   <- same trajectories, oracle phase")
    print(f'  -> {p}')


if __name__ == '__main__':
    main()
