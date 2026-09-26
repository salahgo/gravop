"""Unified benchmark: model families x generalisation regimes.

Motivation. The project's headline accuracy was measured on a random split, in
which every test trajectory sits within ~4% of the parameter box from a training
trajectory (`results/e19_interpolation_audit.json`). Error is flat with distance
there, so the model is not memorising - but the split has almost no dynamic
range, and cannot distinguish interpolation from generalisation. This benchmark
adds splits that can, and evaluates every model family under identical
conditions on all of them.

SPLITS (all split whole trajectories, never rows, so no leakage)
  random      dense interpolation - the original protocol
  param_hole  a contiguous box in (log sigma0, log w) is removed from training
              and used as test: interpolation into an unsampled region
  param_edge  the high-sigma0 quartile is removed: extrapolation beyond the hull
  ic_family   the gNFW initial-condition family is removed: family generalisation

MODEL FAMILIES (each included to test a specific hypothesis)
  ridge_pod   Ridge on POD coefficients. If this is competitive the problem is
              essentially linear and no deep model is warranted. Cost control.
  mlp         Residual MLP on (params, profile) -> POD coefficients. Tests
              whether operator structure matters at all versus plain regression.
  pod_mlp     The classical projection emulator a referee will ask about.
  gp_pod      Gaussian process on POD coefficients. Uncertainty-aware by
              construction; tests whether calibrated UQ needs deep ensembles.
  deeponet    Standard operator baseline.
  fno         Our backbone without physics losses.
  fno_phys    GravOp: backbone plus physics-residual losses.

Usage: benchmark.py [--split random] [--models all] [--seeds 3]
Output: results/benchmark/<split>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds                                   # noqa: E402
from gravop.model import GravOpNet, DeepONet, phys_residuals       # noqa: E402
from train import Norm, make_tensors                               # noqa: E402
import h5py                                                        # noqa: E402


def load_data(path):
    """Like train.load_data but also returns ic_type/rc, needed for the
    initial-condition-family split."""
    keys = ('lrho', 'lv', 'lrho0', 'lv0', 'log_sigma0', 'log_w', 'model_id',
            'log_tc', 'no_collapse', 'status', 'ic_type', 'rc')
    with h5py.File(path, 'r') as h5:
        d = {k: h5[k][:] for k in keys}
    ok = d['status'] >= 0
    return {k: v[ok] for k, v in d.items()}

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'benchmark')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------- splits
def make_split(d, kind, seed=1234):
    """Return (train, val, test) trajectory indices for a generalisation regime."""
    n = len(d['log_tc'])
    rng = np.random.default_rng(seed)
    ls, lw, ic = d['log_sigma0'], d['log_w'], d['ic_type']
    if kind == 'random':
        idx = rng.permutation(n)
        a, b = int(0.8 * n), int(0.9 * n)
        return idx[:a], idx[a:b], idx[b:]
    if kind == 'param_hole':
        # central box in (log sigma0, log w) held out
        lo_s, hi_s = np.quantile(ls, [0.35, 0.65])
        lo_w, hi_w = np.quantile(lw, [0.35, 0.65])
        test = np.where((ls >= lo_s) & (ls <= hi_s) & (lw >= lo_w) & (lw <= hi_w))[0]
    elif kind == 'param_edge':
        # strongest-coupling quartile held out: extrapolation beyond the hull
        thr = np.quantile(ls, 0.75)
        test = np.where(ls > thr)[0]
    elif kind == 'ic_family':
        test = np.where(ic == 3)[0]          # gNFW family
    else:
        raise ValueError(kind)
    rest = np.setdiff1d(np.arange(n), test)
    rest = rng.permutation(rest)
    n_va = max(1, int(0.1 * len(rest)))
    return rest[n_va:], rest[:n_va], test


# ---------------------------------------------------------------- features
def pod_basis(traj, tr, n_comp=128):
    mu = traj[tr].mean(0)
    U, S, Vt = np.linalg.svd(traj[tr] - mu, full_matrices=False)
    V = Vt[:n_comp]
    evr = float((S[:n_comp] ** 2).sum() / (S ** 2).sum())
    return mu, V, evr


def tabular_features(d, idx, n):
    par = np.stack([(d['log_sigma0'][idx] - n.sig_mu) / n.sig_sd,
                    (d['log_w'][idx] - n.w_mu) / n.w_sd,
                    d['model_id'][idx]], 1)
    prof = np.concatenate([(d['lrho0'][idx] - n.rho_mu) / n.rho_sd,
                           (d['lv0'][idx] - n.v_mu) / n.v_sd], 1)
    return np.concatenate([par, prof], 1).astype(np.float32)


# ---------------------------------------------------------------- metrics
def metrics_from_fields(pred_lrho, true_lrho, taus, sd_lrho=None):
    """pred/true: (n_runs, n_tau, n_r) in log10. Returns the common metric set."""
    rel = np.abs(10 ** (pred_lrho - true_lrho) - 1)
    per_run = rel.mean(axis=(1, 2))
    late = taus > 0.92
    rg = ds.r_grid()
    # central density = innermost grid point; core radius = r where rho drops 2x
    def core_radius(l):
        rho = 10 ** l
        out = np.empty(rho.shape[:2])
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                half = rho[i, j, 0] / 2.0
                k = np.argmax(rho[i, j] < half)
                out[i, j] = rg[k] if rho[i, j, k] < half else rg[-1]
        return out
    rc_p, rc_t = core_radius(pred_lrho), core_radius(true_lrho)
    m = {
        'median_rel_rho': float(np.median(per_run)),
        'mean_rel_rho': float(per_run.mean()),
        'p90_rel_rho': float(np.quantile(per_run, 0.9)),
        'late_rel_rho': float(np.abs(10 ** (pred_lrho[:, late] -
                                            true_lrho[:, late]) - 1).mean()),
        'rho_c_rel_err': float(np.abs(10 ** (pred_lrho[..., 0] -
                                             true_lrho[..., 0]) - 1).mean()),
        'core_radius_rel_err': float(np.mean(np.abs(rc_p / rc_t - 1))),
    }
    if sd_lrho is not None:
        z = (pred_lrho - true_lrho) / np.maximum(sd_lrho, 1e-9)
        m['calib_z_lt1'] = float(np.mean(np.abs(z) < 1))
        m['calib_z_lt2'] = float(np.mean(np.abs(z) < 2))
    return m


def physics_violation(pred_lrho, pred_lv):
    log_r = torch.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R, device=DEV)
    a = torch.tensor(pred_lrho.reshape(-1, ds.N_R), dtype=torch.float32, device=DEV)
    b = torch.tensor(pred_lv.reshape(-1, ds.N_R), dtype=torch.float32, device=DEV)
    with torch.no_grad():
        hyd, mtot, etot = phys_residuals(a, b, log_r)
    n_tau = pred_lrho.shape[1]
    m0 = mtot.reshape(-1, n_tau)[:, :1]
    dm = ((mtot.reshape(-1, n_tau) - m0) / m0).abs()
    return {'hydrostatic_residual': float(hyd.abs().mean()),
            'mass_violation': float(dm.mean())}


# ---------------------------------------------------------------- models
class Runner:
    """Common interface: fit(train,val) then predict(test) -> (lrho, lv, sd)."""
    needs_torch = True

    def __init__(self, d, n, taus):
        self.d, self.n, self.taus = d, n, taus
        self.n_tau, self.n_r = len(taus), ds.N_R

    def _traj(self, idx):
        return np.concatenate([self.d['lrho'][idx].reshape(len(idx), -1),
                               self.d['lv'][idx].reshape(len(idx), -1)], 1)

    def _unpack(self, flat):
        k = self.n_tau * self.n_r
        return (flat[:, :k].reshape(-1, self.n_tau, self.n_r),
                flat[:, k:].reshape(-1, self.n_tau, self.n_r))


class PODLinear(Runner):
    """Ridge on POD coefficients: is the map essentially linear?"""
    name = 'ridge_pod'

    def fit(self, tr, va, seed=0):
        traj = self._traj(np.arange(len(self.d['log_tc'])))
        self.mu, self.V, self.evr = pod_basis(traj, tr)
        X = tabular_features(self.d, tr, self.n)
        Y = (traj[tr] - self.mu) @ self.V.T
        lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
        self.W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        self.traj = traj

    def predict(self, te):
        Y = tabular_features(self.d, te, self.n) @ self.W
        return (*self._unpack(Y @ self.V + self.mu), None)


class PODMLP(Runner):
    """MLP on POD coefficients (classical emulator) or full residual MLP."""
    name = 'pod_mlp'
    width, epochs = 512, 400

    def fit(self, tr, va, seed=0):
        torch.manual_seed(seed)
        traj = self._traj(np.arange(len(self.d['log_tc'])))
        self.mu, self.V, self.evr = pod_basis(traj, tr)
        X = tabular_features(self.d, tr, self.n)
        Y = (traj[tr] - self.mu) @ self.V.T
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-9
        net = nn.Sequential(nn.Linear(X.shape[1], self.width), nn.GELU(),
                            nn.Linear(self.width, self.width), nn.GELU(),
                            nn.Linear(self.width, Y.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        Xt = torch.tensor((X - self.xm) / self.xs, device=DEV)
        Yt = torch.tensor(((Y - self.ym) / self.ys).astype(np.float32), device=DEV)
        for _ in range(self.epochs):
            net.train(); opt.zero_grad()
            nn.functional.mse_loss(net(Xt), Yt).backward(); opt.step(); sch.step()
        self.net = net.eval()

    def predict(self, te):
        X = (tabular_features(self.d, te, self.n) - self.xm) / self.xs
        with torch.no_grad():
            Y = self.net(torch.tensor(X, device=DEV)).cpu().numpy()
        return (*self._unpack((Y * self.ys + self.ym) @ self.V + self.mu), None)


class GPPOD(Runner):
    """Exact GP (RBF) per POD coefficient: calibrated UQ without ensembling."""
    name = 'gp_pod'
    n_comp = 24

    def fit(self, tr, va, seed=0):
        traj = self._traj(np.arange(len(self.d['log_tc'])))
        self.mu, self.V, self.evr = pod_basis(traj, tr, self.n_comp)
        X = tabular_features(self.d, tr, self.n)[:, :3]      # params only
        Y = (traj[tr] - self.mu) @ self.V.T
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.xm) / self.xs
        d2 = ((Xs[:, None] - Xs[None]) ** 2).sum(-1)
        self.ell = np.sqrt(np.median(d2[d2 > 0]))
        # The constant-cross-section runs all sit at w_hat = 1 exactly, so the
        # kernel matrix has near-duplicate rows and needs a real nugget rather
        # than a token jitter; escalate until it factorises.
        base = np.exp(-0.5 * d2 / self.ell ** 2)
        for jit in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2):
            try:
                self.L = np.linalg.cholesky(base + jit * np.eye(len(Xs)))
                self.jitter = jit
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError('GP kernel not factorisable')
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
        self.Xs, self.ym = Xs, Y.mean(0)

    def predict(self, te):
        X = (tabular_features(self.d, te, self.n)[:, :3] - self.xm) / self.xs
        d2 = ((X[:, None] - self.Xs[None]) ** 2).sum(-1)
        Ks = np.exp(-0.5 * d2 / self.ell ** 2)
        Y = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
        flat = Y @ self.V + self.mu
        sd = np.sqrt(var)[:, None] * np.abs(self.V).sum(0)[None, :]
        lrho, lv = self._unpack(flat)
        sd_lrho, _ = self._unpack(sd)
        return lrho, lv, sd_lrho


class NeuralOp(Runner):
    """FNO backbone, optionally with physics-residual losses."""
    def __init__(self, d, n, taus, physics, kind='fno'):
        super().__init__(d, n, taus)
        self.physics, self.kind = physics, kind
        self.name = {'fno': 'fno_phys' if physics else 'fno',
                     'deeponet': 'deeponet'}[kind]

    def fit(self, tr, va, seed=0, epochs=200):
        torch.manual_seed(seed); np.random.seed(seed)
        net = (GravOpNet(n_r=ds.N_R) if self.kind == 'fno'
               else DeepONet(n_r=ds.N_R)).to(DEV)
        lrho, lv, prof, par, tc, nc = make_tensors(self.d, self.n, DEV)
        taus = torch.tensor(self.taus, dtype=torch.float32, device=DEV)
        log_r = torch.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R, device=DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        pr_ = np.repeat(tr, self.n_tau); pt_ = np.tile(np.arange(self.n_tau), len(tr))
        for ep in range(epochs):
            net.train(); perm = np.random.permutation(len(pr_))
            for s in range(0, len(perm), 1024):
                sel = perm[s:s + 1024]; ri, ti = pr_[sel], pt_[sel]
                cond = torch.cat([par[ri], taus[ti][:, None]], 1)
                f, sc = net(prof[ri], cond)
                tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], 1)
                loss = nn.functional.mse_loss(f, tgt) \
                    + 0.1 * nn.functional.mse_loss(sc[:, 0], tc[ri]) \
                    + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                        sc[:, 1], nc[ri].float())
                if self.physics:
                    ramp = min(1.0, (ep + 1) / max(1, int(0.2 * epochs)))
                    pl = f[:, 0] * self.n.rho_sd + self.n.rho_mu
                    pv = f[:, 1] * self.n.v_sd + self.n.v_mu
                    hyd, mt, et = phys_residuals(pl, pv, log_r)
                    il = prof[ri][:, 0] * self.n.rho_sd + self.n.rho_mu
                    iv = prof[ri][:, 1] * self.n.v_sd + self.n.v_mu
                    _, m0, e0 = phys_residuals(il, iv, log_r)
                    loss = loss + ramp * (
                        0.05 * hyd.clamp(-10, 10).pow(2).mean()
                        + 0.1 * ((mt - m0) / m0).abs().clamp(max=10).pow(2).mean()
                        + 0.02 * ((et - e0) / m0.abs()).abs().clamp(max=10).pow(2).mean())
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        self.net = net.eval()
        self.tensors = (prof, par)

    def predict(self, te):
        prof, par = self.tensors
        taus = torch.tensor(self.taus, dtype=torch.float32, device=DEV)
        out = np.empty((len(te), self.n_tau, self.n_r, 2), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(te), 64):
                ri = te[s:s + 64]; B = len(ri)
                pi = prof[ri].repeat_interleave(self.n_tau, 0)
                ci = torch.cat([par[ri].repeat_interleave(self.n_tau, 0),
                                taus.repeat(B)[:, None]], 1)
                f, _ = self.net(pi, ci)
                out[s:s + B] = f.reshape(B, self.n_tau, 2, self.n_r
                                         ).permute(0, 1, 3, 2).cpu().numpy()
        lrho = out[..., 0] * self.n.rho_sd + self.n.rho_mu
        lv = out[..., 1] * self.n.v_sd + self.n.v_mu
        return lrho, lv, None


REGISTRY = {
    'ridge_pod': lambda d, n, t: PODLinear(d, n, t),
    'pod_mlp':   lambda d, n, t: PODMLP(d, n, t),
    'gp_pod':    lambda d, n, t: GPPOD(d, n, t),
    'deeponet':  lambda d, n, t: NeuralOp(d, n, t, False, 'deeponet'),
    'fno':       lambda d, n, t: NeuralOp(d, n, t, False, 'fno'),
    'fno_phys':  lambda d, n, t: NeuralOp(d, n, t, True, 'fno'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='random')
    ap.add_argument('--models', default='all')
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    taus = ds.tau_grid()
    tr, va, te = make_split(d, args.split)
    print(f'split={args.split}: train {len(tr)}, val {len(va)}, test {len(te)}',
          flush=True)
    n = Norm(d, tr)
    true_lrho = d['lrho'][te]

    names = list(REGISTRY) if args.models == 'all' else args.models.split(',')
    out = {'split': args.split, 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'models': {}}
    for name in names:
        seeds = 1 if name in ('ridge_pod', 'gp_pod') else args.seeds
        runs = []
        for s in range(seeds):
            t0 = time.time()
            mdl = REGISTRY[name](d, n, taus)
            mdl.fit(tr, va, seed=s)
            t_fit = time.time() - t0
            t1 = time.time()
            pl, pv, sd = mdl.predict(te)
            t_pred = (time.time() - t1) / max(len(te), 1)
            m = metrics_from_fields(pl, true_lrho, taus, sd)
            m.update(physics_violation(pl, pv))
            m['fit_seconds'] = t_fit
            m['predict_seconds_per_traj'] = t_pred
            if hasattr(mdl, 'net'):
                m['n_params'] = sum(p.numel() for p in mdl.net.parameters())
            runs.append(m)
        agg = {k: float(np.mean([r[k] for r in runs]))
               for k in runs[0] if isinstance(runs[0][k], (int, float))}
        agg['n_seeds'] = seeds
        agg['median_rel_rho_std'] = float(np.std([r['median_rel_rho']
                                                  for r in runs]))
        out['models'][name] = agg
        print(f"  {name:10s} median {100*agg['median_rel_rho']:6.2f}%  "
              f"late {100*agg['late_rel_rho']:6.2f}%  "
              f"rho_c {100*agg['rho_c_rel_err']:6.2f}%  "
              f"hyd {agg['hydrostatic_residual']:.4f}  "
              f"fit {agg['fit_seconds']:.0f}s", flush=True)

    json.dump(out, open(os.path.join(OUT, f'{args.split}.json'), 'w'), indent=1)
    print(f'saved results/benchmark/{args.split}.json')


if __name__ == '__main__':
    main()
