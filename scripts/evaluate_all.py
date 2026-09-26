"""Consolidated evaluation: E3 accuracy, E5 OOD, E6 conservation, E7 calibration.

Evaluates every model under results/runs/ on the in-distribution test split and
the OOD sets; computes ensemble (gravop_s*) statistics, physics-consistency
residuals, and calibration. Output: results/evaluation/summary.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds  # noqa: E402
from gravop.model import GravOpNet, DeepONet, phys_residuals  # noqa: E402
from train import load_data, split, Norm, make_tensors, evaluate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'evaluation')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model(run_dir):
    ck = torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu',
                    weights_only=False)
    a = ck['args']
    if a['model'] in ('gravop', 'fno_nophys'):
        # cond_dim defaults to 4 (the three parameters plus tau). The
        # phase-conditioned ensemble from e28_phase_ensemble.py stores 5,
        # because it reads (phi, flag) off the profile instead of tau; older
        # checkpoints predate the key and must keep loading unchanged.
        m = GravOpNet(n_r=ds.N_R, width=a['width'], modes=a['modes'],
                      n_layers=a['layers'], cond_dim=a.get('cond_dim', 4))
    elif a['model'] == 'deeponet':
        m = DeepONet(n_r=ds.N_R)
    else:
        return None, None, None
    m.load_state_dict(ck['model'])
    return m.to(DEVICE).eval(), ck['norm'], a


class NormObj:
    def __init__(self, d):
        self.__dict__.update(d)


def ens_predict(models, norm, tensors, idx, taus):
    """Stacked per-model predictions of lrho for (runs x taus).
    Returns preds (M, K, T, N), targets (K, T, N)."""
    lrho, lv, prof, par, tc, nc = tensors
    n_tau = len(taus)
    K = len(idx)
    preds = []
    tgt = lrho[idx][:, :, :].cpu().numpy()
    for m in models:
        rows = np.empty((K, n_tau, ds.N_R), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, K, 256):
                ri = idx[s:s + 256]
                B = len(ri)
                pi = prof[ri].repeat_interleave(n_tau, dim=0)
                ci = torch.cat([par[ri].repeat_interleave(n_tau, dim=0),
                                taus.repeat(B)[:, None]], dim=1)
                f, _ = m(pi, ci)
                rows[s:s + B] = f[:, 0].reshape(B, n_tau, ds.N_R).cpu().numpy()
        preds.append(rows)
    return np.array(preds), tgt


def conservation_metrics(model, norm, tensors, idx, taus):
    """Mass/energy conservation + hydrostatic residual of predictions,
    with ground-truth fields as reference level."""
    lrho, lv, prof, par, tc, nc = tensors
    log_r = torch.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R, device=DEVICE)
    n = NormObj(norm)
    rows = {'pred_hyd': [], 'true_hyd': [], 'pred_dm': [], 'true_dm': [],
            'pred_de': [], 'true_de': []}
    n_tau = len(taus)
    with torch.no_grad():
        for s in range(0, len(idx), 128):
            ri = idx[s:s + 128]
            B = len(ri)
            pi = prof[ri].repeat_interleave(n_tau, dim=0)
            ci = torch.cat([par[ri].repeat_interleave(n_tau, dim=0),
                            taus.repeat(B)[:, None]], dim=1)
            f, _ = model(pi, ci)
            plrho = f[:, 0] * n.rho_sd + n.rho_mu
            plv = f[:, 1] * n.v_sd + n.v_mu
            tlrho = (lrho[ri].reshape(-1, ds.N_R))
            tlv = (lv[ri].reshape(-1, ds.N_R))
            tlrho = tlrho * 0 + tlrho  # already normalized? no: stored norm.
            # tensors store normalized fields; denormalize
            tlrho = lrho[ri].reshape(-1, ds.N_R) * n.rho_sd + n.rho_mu
            tlv = lv[ri].reshape(-1, ds.N_R) * n.v_sd + n.v_mu
            for tag, lr_, lv_ in (('pred', plrho, plv), ('true', tlrho, tlv)):
                hyd, mtot, etot = phys_residuals(lr_, lv_, log_r)
                m0 = mtot.reshape(B, n_tau)[:, 0:1]
                e0 = etot.reshape(B, n_tau)[:, 0:1]
                dm = ((mtot.reshape(B, n_tau) - m0) / m0).abs()
                de = ((etot.reshape(B, n_tau) - e0) / m0.abs()).abs()
                rows[f'{tag}_hyd'].append(
                    hyd.abs().mean(dim=1).cpu().numpy())
                rows[f'{tag}_dm'].append(dm.mean(dim=1).cpu().numpy())
                rows[f'{tag}_de'].append(de.mean(dim=1).cpu().numpy())
    return {k: float(np.mean(np.concatenate(v))) for k, v in rows.items()}


def main():
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=DEVICE)
    d_train = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    n = len(d_train['log_tc'])
    tr, va, te = split(n)

    summary = {}
    run_dirs = sorted(glob.glob(os.path.join(ROOT, 'results', 'runs', '*')))
    models_ens = []
    norm_d = None
    for rd in run_dirs:
        name = os.path.basename(rd)
        if not os.path.exists(os.path.join(rd, 'model.pt')):
            continue
        model, norm, a = load_model(rd)
        if model is None:
            continue
        norm_obj = NormObj(norm)
        tensors = make_tensors(d_train, norm_obj, DEVICE)
        res = {'test': evaluate(model, tensors, taus, te, norm_obj, DEVICE)}
        for ood_name in ('ood_param', 'ood_ic'):
            path = os.path.join(ROOT, 'data', 'd1', ood_name + '.h5')
            if os.path.exists(path):
                d_ood = load_data(path)
                t_ood = make_tensors(d_ood, norm_obj, DEVICE)
                res[ood_name] = evaluate(model, t_ood, taus,
                                         np.arange(len(d_ood['log_tc'])),
                                         norm_obj, DEVICE)
        res['conservation_test'] = conservation_metrics(
            model, norm, tensors, te, taus)
        summary[name] = res
        print(name, json.dumps(res['test']), flush=True)
        if name.startswith('gravop_s'):
            models_ens.append(model)
            norm_d = norm

    # ensemble: accuracy of mean prediction + calibration + OOD flagging
    if len(models_ens) >= 2:
        norm_obj = NormObj(norm_d)
        tensors = make_tensors(d_train, norm_obj, DEVICE)
        ens = {}
        preds, tgt = ens_predict(models_ens, norm_d, tensors, te, taus)
        mu, sd = preds.mean(0), preds.std(0)
        err = mu - tgt
        ens['test_rmse_lrho_mean_pred'] = float(np.sqrt((err ** 2).mean()))
        z = err / np.maximum(sd, 1e-8)
        ens['calib_z_abs_lt1'] = float(np.mean(np.abs(z) < 1))
        ens['calib_z_abs_lt2'] = float(np.mean(np.abs(z) < 2))
        ens['test_mean_sd'] = float(sd.mean())
        for ood_name in ('ood_param', 'ood_ic'):
            path = os.path.join(ROOT, 'data', 'd1', ood_name + '.h5')
            if os.path.exists(path):
                d_ood = load_data(path)
                t_ood = make_tensors(d_ood, norm_obj, DEVICE)
                po, to = ens_predict(models_ens, norm_d, t_ood,
                                     np.arange(len(d_ood['log_tc'])), taus)
                eo = po.mean(0) - to
                ens[f'{ood_name}_rmse_mean_pred'] = float(
                    np.sqrt((eo ** 2).mean()))
                ens[f'{ood_name}_mean_sd'] = float(po.std(0).mean())
        summary['ENSEMBLE'] = ens
        print('ENSEMBLE', json.dumps(ens), flush=True)

    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('saved', os.path.join(OUT, 'summary.json'))


if __name__ == '__main__':
    main()
