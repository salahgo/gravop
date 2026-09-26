"""Track F: can a solver-only signal gate a surrogate-solver fallback?

Hybrid Neural World Models (arXiv:2605.28317) gates a reference-solver fallback
on a per-trajectory error map recovered from the network's forward passes. That
signal, like deep ensembles and learned error heads, needs a trained network.

The coordinate displacement does not: it is computed from the reference solution
and is available before any network exists. This measures whether it ranks
individual trajectories by deployed error as well as a network-derived signal.

Evaluation only; existing checkpoints.
Pre-registration: docs/GATING_PREREGISTRATION.md, committed first.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.stats import spearmanr

from sc_experiment import Norm, tensors, load_checkpoint
import statecond.data as scd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(RES, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
QS = [0.0, 0.1, 0.2, 0.3]


def run(name, seeds, dev):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    u, u0, par, phi, flag, lts = tensors(d, n, dev)
    taus = scd.tau_grid()
    n_tau = len(taus)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    ri = np.asarray(te)
    un = u.cpu().numpy()

    preds, ratios = [], None
    for s in range(seeds):
        tau_ck = os.path.join(CKPT, f'{name}_fno_tau_seed{s}.pt')
        head_ck = os.path.join(CKPT, f'{name}_fno_time_conv_seed{s}.pt')
        if not (os.path.exists(tau_ck) and os.path.exists(head_ck)):
            continue
        net, _ = load_checkpoint(tau_ck, dev); net.eval()
        head, _ = load_checkpoint(head_ck, dev); head.eval()
        with torch.no_grad():
            lts_hat = head(u0, par).squeeze(-1)
        # x n.t_sd: both terms are normalised log-time, so the physical
        # ratio needs the training split's scale (see sc_decomposition.py).
        rr = torch.pow(10.0, (lts - lts_hat) * n.t_sd).cpu().numpy()[ri]
        if ratios is None:
            ratios = rr
        tq = np.clip(taus[None, :] * rr[:, None], 0.0, 1.0)
        out = []
        for j in range(n_tau):
            q = torch.tensor(tq[:, j], dtype=torch.float32, device=dev)
            cond = torch.cat([par[ri], q[:, None]], 1)
            with torch.no_grad():
                out.append(net(u0[ri], cond).cpu().numpy())
        preds.append(np.stack(out, 1))
        del net, head; torch.cuda.empty_cache()
    if not preds:
        return None
    P = np.stack(preds)                                   # (seed, run, tau, c, x)

    G = un[ri]
    dep = np.sqrt(((P.mean(0) - G) ** 2).mean(axis=(1, 2, 3)))       # truth
    spread = P.std(0).mean(axis=(1, 2, 3))                            # network
    # solver-only: displacement of the reference along the estimated clock error
    F = []
    for i in range(len(ri)):
        shifted = np.stack([[np.interp(tq[i], taus, un[ri[i], :, c, x])
                             for x in range(un.shape[3])]
                            for c in range(un.shape[2])]).transpose(2, 0, 1)
        F.append(float(np.sqrt(((shifted - G[i]) ** 2).mean())))
    F = np.asarray(F)

    def gate(sig):
        out = {}
        order = np.argsort(-sig)                 # defer the worst first
        for q in QS:
            k = int(round(q * len(sig)))
            keep = order[k:]
            out[str(q)] = float(dep[keep].mean())
        return out

    return dict(n_test=int(len(ri)),
                spearman_solver=float(spearmanr(F, dep).statistic),
                spearman_ensemble=float(spearmanr(spread, dep).statistic),
                gate_solver=gate(F), gate_ensemble=gate(spread),
                gate_oracle=gate(dep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = {}
    print(f"{'system':8s} {'rho_solver':>11s} {'rho_ens':>9s}   gate@20%: "
          f"{'base':>8s} {'solver':>8s} {'ens':>8s} {'oracle':>8s}")
    for name in a.systems:
        r = run(name, a.seeds, a.device)
        if r is None:
            continue
        out[name] = r
        b = r['gate_oracle']['0.0']
        print(f"{name:8s} {r['spearman_solver']:11.3f} {r['spearman_ensemble']:9.3f}   "
              f"          {b:8.5f} {r['gate_solver']['0.2']:8.5f} "
              f"{r['gate_ensemble']['0.2']:8.5f} {r['gate_oracle']['0.2']:8.5f}",
              flush=True)
    json.dump(out, open(os.path.join(RES, 'gating.json'), 'w'), indent=1)
    print('wrote gating.json')


if __name__ == '__main__':
    main()
