"""Track A: the exact deployed-error decomposition, including its cross term.

The referee's correction. With a learned operator written as Ghat = G + e,

    Ghat(v~) - G(v) = [G(v~) - G(v)] + e(v~) = Delta + e(v~),

so the deployed squared error is

    E_dep^2 = ||Delta||^2 + ||e(v~)||^2 + 2<Delta, e(v~)>.

The quadrature this project has used, sqrt(E_A^2 + F^2), discards the cross term
and silently replaces ||e(v~)||, the operator error at the CORRUPTED coordinate,
with E_A, its error at the correct one. Both are assumptions. This measures all
four quantities on the frozen test split.

Evaluation only: the trained arms and the converged clock heads are restored
from checkpoints.

Pre-registration: docs/CROSS_TERM_PREREGISTRATION.md, committed first.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from sc_experiment import Norm, tensors, load_checkpoint
import statecond.data as scd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(RES, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def fields_at(net, cond_fn, u0, te, n_tau, dev):
    """Operator output for every (test run, tau) slot."""
    out = []
    ri = np.asarray(te)
    for j in range(n_tau):
        ti = np.full(len(te), j)
        with torch.no_grad():
            out.append(net(u0[ri], cond_fn(ri, ti)).cpu().numpy())
    return np.stack(out, axis=1)          # (runs, tau, channel, x)


def run(name, backbone, seeds, dev):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, lts = T
    taus = scd.tau_grid()
    n_tau = len(taus)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    G = u[np.asarray(te)].cpu().numpy()            # reference, correct coord

    rows = []
    for s in range(seeds):
        tau_ck = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        head_ck = os.path.join(CKPT, f'{name}_fno_time_conv_seed{s}.pt')
        if not (os.path.exists(tau_ck) and os.path.exists(head_ck)):
            continue
        net, _ = load_checkpoint(tau_ck, dev)
        net.eval()
        head, _ = load_checkpoint(head_ck, dev)
        head.eval()

        # the deployed coordinate: tau built from the predicted terminal time
        with torch.no_grad():
            lts_hat = head(u0, par).squeeze(-1)
        # n.t_sd is NOT optional. `lts` and the head's output are both in
        # normalised log-time; the physical log-ratio is the difference times
        # the training split's standard deviation. Omitting it inflates the
        # clock error by 1/t_sd -- 6.7x in `burgers`, 2.9x in `film` -- and
        # measures the decomposition at a clock nobody trained.
        ratio = torch.pow(10.0, (lts - lts_hat) * n.t_sd).cpu().numpy()
        ri = np.asarray(te)

        def cond_true(r, t):
            return torch.cat([par[r], taus_t[t][:, None]], 1)

        # G(v~): the REFERENCE field resampled to the corrupted coordinate
        tq = np.clip(taus[None, :] * ratio[ri][:, None], 0.0, 1.0)
        Gv = np.stack([np.stack([np.interp(tq[i], taus, u[ri[i], :, c, x].cpu().numpy())
                                 for x in range(u.shape[3])], -1)
                       for i in range(len(ri)) for c in range(u.shape[2])])
        Gv = Gv.reshape(len(ri), u.shape[2], n_tau, u.shape[3]).transpose(0, 2, 1, 3)

        def cond_dep(r, t):
            q = torch.tensor(tq[[list(ri).index(x) for x in r], t],
                             dtype=torch.float32, device=dev)
            return torch.cat([par[r], q[:, None]], 1)

        Ghat_v = fields_at(net, cond_true, u0, te, n_tau, dev)    # at correct
        Ghat_dep = fields_at(net, cond_dep, u0, te, n_tau, dev)   # at corrupted

        Delta = Gv - G                     # displacement of the exact map
        e_dep = Ghat_dep - Gv              # operator error at corrupted coord
        e_true = Ghat_v - G                # operator error at correct coord
        dep = Ghat_dep - G                 # total deployed error

        m = lambda a: float(np.sqrt((a ** 2).mean()))
        F, Ee, EA, Ed = m(Delta), m(e_dep), m(e_true), m(dep)
        cross = 2.0 * float((Delta * e_dep).mean())
        rows.append(dict(seed=s, F=F, e_dep=Ee, E_A=EA, E_dep=Ed,
                         cross=cross,
                         chi=cross / max(Ed ** 2, 1e-30),
                         rho=Ee / max(EA, 1e-30),
                         quadrature=float(np.sqrt(EA ** 2 + F ** 2)),
                         quad_over_measured=float(np.sqrt(EA ** 2 + F ** 2)) / max(Ed, 1e-30)))
        del net, head
        torch.cuda.empty_cache()
    return rows


def aggregate(rows):
    """Combine seeds under ONE convention, so the identity survives the average.

    E_dep^2 = F^2 + |e|^2 + cross is quadratic, so it is preserved by a mean over
    seeds of the SQUARED norms and of cross -- i.e. by an RMS of each norm -- and
    is destroyed by a mean of the norms themselves. Averaging each column
    independently, as an earlier version did, left rows closing to 0.974 rather
    than 1. Ratios are likewise derived from the aggregate, never averaged.
    """
    rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows])))
    F, Ee, EA, Ed = rms('F'), rms('e_dep'), rms('E_A'), rms('E_dep')
    cross = float(np.mean([r['cross'] for r in rows]))
    quad = float(np.sqrt(EA ** 2 + F ** 2))
    agg = dict(F=F, e_dep=Ee, E_A=EA, E_dep=Ed, cross=cross,
               chi=cross / max(Ed ** 2, 1e-30),
               rho=Ee / max(EA, 1e-30),
               quadrature=quad,
               quad_over_measured=quad / max(Ed, 1e-30))
    closure = (F ** 2 + Ee ** 2 + cross) / max(Ed ** 2, 1e-30)
    assert abs(closure - 1.0) < 1e-6, f'identity does not close: {closure!r}'
    agg['closure'] = closure
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    ap.add_argument('--backbones', nargs='+',
                    default=['fno', 'cno', 'transformer', 'deeponet'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(2)          # F9 owns the CPU
    out = {}
    for name in a.systems:
        for bb in a.backbones:
            rows = run(name, bb, a.seeds, a.device)
            if not rows:
                continue
            agg = aggregate(rows)
            out[f'{name}_{bb}'] = dict(agg, per_seed=rows)
            print(f"{name:8s} {bb:12s} F {agg['F']:.4f}  e(v~) {agg['e_dep']:.4f}  "
                  f"E_dep {agg['E_dep']:.4f}  chi {agg['chi']:+.3f}  rho {agg['rho']:.2f}",
                  flush=True)
    json.dump(out, open(os.path.join(RES, 'decomposition.json'), 'w'), indent=1)
    print('wrote decomposition.json')


if __name__ == '__main__':
    main()
