"""P15 -- the direct physical-time baseline the paper never built.

WHY THIS EXISTS
---------------
Every arm in this study conditions on a time that has been factorised through
the singular time: arm A on tau = t/t_sing with t_sing supplied, arm B on tau
from a predicted t_sing, arm C on the state coordinate that replaces t_sing
altogether. The referee's objection is structural: with no arm conditioned on
PHYSICAL time, the paper silently implies that the tau factorisation is
unavoidable, and the deployment penalty it reports is a penalty for a choice
the reader was never shown an alternative to.

Arm D fixes that. It is conditioned on standardised log10 t with no t_sing
anywhere -- not in training, not at deployment. That makes it the honest
"do nothing clever" baseline: at deployment the user knows what time they are
asking about, so arm D needs no prediction and pays no clock error at all. Its
cost is different: it must learn the map from (parameters, time) to state
across trajectories whose singular times span up to 3.9 decades, so what tau
buys it is a common coordinate, and what tau costs it is having to know
t_sing.

PRE-REGISTERED (Track D):
    direct-time loses to state-conditioning in the four strongly singular
    systems and is competitive in `burgers`. If direct-time wins anywhere
    else, the factorisation this paper critiques was never necessary there,
    and we say so.

WHAT IS RETRAINED AND WHAT IS NOT
---------------------------------
Only arm D is trained. Arms A and C are restored from the P6d checkpoints --
the exact published weights -- and arm B is recomputed from the published time
head under the endpoint policy of `sc_endpoint.py`. So the comparison is
against the numbers in the paper rather than against a fresh run that would
differ by training noise, and the cost is ~15 trainings rather than ~60.

Output: results/statecond/direct.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                    # noqa: E402
from sc_provenance import stamp                                 # noqa: E402
from sc_experiment import (LATE, Norm, field_err, load_checkpoint,  # noqa: E402
                           phys_time, save_checkpoint, tensors,
                           train_operator)

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def check_plain_arm(meta, path):
    """Refuse a checkpoint that is not the plain arm it claims to be.

    The H1 arm overwrote the plain-FNO checkpoints once (see
    `scripts/sc_ckpt_audit.py`); the fix is a filename tag, and this is the
    belt to that brace. Checkpoints written before 2026-09-05 carry no `h1`
    key at all, and those cannot be trusted from their name alone -- they are
    rejected rather than assumed innocent.
    """
    if 'h1' not in meta:
        raise SystemExit(
            f'{os.path.basename(path)} predates the h1 tag (2026-09-05) and '
            'cannot be identified from its filename. Regenerate it with '
            'sc_experiment.py --ckpt-dir, or audit it with sc_ckpt_audit.py.')
    if meta['h1'] != 0.0:
        raise SystemExit(f'{os.path.basename(path)} is an H1 arm (h1='
                         f"{meta['h1']}), not the plain arm this script "
                         'compares against.')


def run_system(name, seeds, dev, epochs, backbone, save, reuse=False):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, lts = T
    taus = scd.tau_grid()
    late = taus > LATE

    ltab, lt_mu, lt_sd = phys_time(d, tr, taus)
    ltab_t = torch.tensor(ltab, device=dev)

    acc = {k: [] for k in ('A_tau_oracle', 'B_tau_deployed', 'C_state',
                           'D_time_direct')}
    late_acc = {k: [] for k in acc}
    # P23: keep the per-trajectory vector so sc_significance can resample test
    # trajectories. Shape per key: (seeds, n_test).
    per_traj = {k: [] for k in acc}

    for s in seeds:
        # arm D: the only thing that is trained here
        dpath = os.path.join(CKPT, f'{name}_{backbone}_phys_seed{s}.pt')
        if reuse and os.path.exists(dpath):
            net_d, _ = load_checkpoint(dpath, dev)

            def cond_d(ri, ti):
                return torch.cat([par[ri], ltab_t[ri, ti][:, None]], 1)
        else:
            net_d, cond_d, _ = train_operator('phys', T, tr, te, s, dev,
                                              epochs, n_fields, n_par,
                                              backbone, extra=ltab_t)
        eD = field_err(net_d, cond_d, u, u0, te)
        if save and not (reuse and os.path.exists(dpath)):
            os.makedirs(CKPT, exist_ok=True)
            save_checkpoint(os.path.join(
                CKPT, f'{name}_{backbone}_phys_seed{s}.pt'),
                net_d, 'phys', backbone, n, s, epochs,
                (n_fields, n_par, n_x))

        # arms A, B, C from the published weights
        ptau = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        ppha = os.path.join(CKPT, f'{name}_{backbone}_phase_seed{s}.pt')
        phed = os.path.join(CKPT, f'{name}_{backbone}_time_seed{s}.pt')
        net_a, ma = load_checkpoint(ptau, dev)
        net_c, mc = load_checkpoint(ppha, dev)
        head, mh = load_checkpoint(phed, dev)
        for meta, path in ((ma, ptau), (mc, ppha), (mh, phed)):
            check_plain_arm(meta, path)

        taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)

        def cond_a(ri, ti):
            return torch.cat([par[ri], taus_t[ti][:, None]], 1)

        def cond_c(ri, ti):
            return torch.cat([par[ri], phi[ri, ti][:, None],
                              flag[ri, ti][:, None]], 1)

        with torch.no_grad():
            lt_pred = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt_pred - d['log_tsing'][te])
        tau_dep = np.clip(np.tile(taus, (len(te), 1)) / ratio[:, None], 0, 1)

        eA = field_err(net_a, cond_a, u, u0, te)
        eB = field_err(net_a, cond_a, u, u0, te, tau_query=tau_dep)
        eC = field_err(net_c, cond_c, u, u0, te)

        for k, e in (('A_tau_oracle', eA), ('B_tau_deployed', eB),
                     ('C_state', eC), ('D_time_direct', eD)):
            acc[k].append(float(e.mean()))
            late_acc[k].append(float(e[:, late].mean()))
            per_traj[k].append(e.mean(axis=1))

        del net_d, net_a, net_c, head
        torch.cuda.empty_cache()

    def ms(v):
        return {'mean': float(np.mean(v)), 'std': float(np.std(v))}

    out = {
        'system': name,
        'backbone': backbone,
        'seeds': list(seeds),
        'epochs': epochs,
        'n_test': len(te),
        'tsing_spread_decades': float(np.ptp(d['log_tsing'])),
        'log_t_mu': lt_mu,
        'log_t_sd': lt_sd,
        'nrmse': {k: ms(v) for k, v in acc.items()},
        'nrmse_late': {k: ms(v) for k, v in late_acc.items()},
        'per_trajectory': {k: np.asarray(v).tolist()
                           for k, v in per_traj.items()},
        'test_index': np.asarray(te).tolist(),
        '_provenance': stamp('sc_direct.py'),
    }
    D = out['nrmse']['D_time_direct']['mean']
    out['ratios'] = {
        'D_over_A_oracle': D / out['nrmse']['A_tau_oracle']['mean'],
        'D_over_B_deployed': D / out['nrmse']['B_tau_deployed']['mean'],
        'D_over_C_state': D / out['nrmse']['C_state']['mean'],
    }
    out['state_beats_direct'] = out['ratios']['D_over_C_state'] > 1.0
    out['direct_beats_deployed'] = out['ratios']['D_over_B_deployed'] < 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--backbone', default='fno')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--reuse-d', action='store_true',
                    help='load arm D from its checkpoint instead of training '
                         'it, so the A/B/C reference columns can be '
                         'recomputed cheaply')
    args = ap.parse_args()

    names = SYSTEMS if args.all else [args.system]
    path = os.path.join(OUT, 'direct.json')
    out = json.load(open(path)) if os.path.exists(path) else {}

    for name in names:
        r = run_system(name, range(args.seeds), args.device, args.epochs,
                       args.backbone, not args.no_save, args.reuse_d)
        out[name] = r
        n = r['nrmse']
        print(f"{name:8s} A {n['A_tau_oracle']['mean']:.4f}  "
              f"B {n['B_tau_deployed']['mean']:.4f}  "
              f"C {n['C_state']['mean']:.4f}  "
              f"D {n['D_time_direct']['mean']:.4f}   "
              f"D/C {r['ratios']['D_over_C_state']:.2f}x  "
              f"D/B {r['ratios']['D_over_B_deployed']:.2f}x", flush=True)
        json.dump(out, open(path, 'w'), indent=1)

    done = [k for k in SYSTEMS if k in out]
    if len(done) == len(SYSTEMS):
        cw = [k for k in done if not out[k]['state_beats_direct']]
        bw = [k for k in done if out[k]['direct_beats_deployed']]
        out['_summary'] = {
            'prereg': ('direct-time loses to state-conditioning in the four '
                       'strongly singular systems and is competitive in '
                       'burgers'),
            'state_conditioning_wins_in': [k for k in done
                                           if out[k]['state_beats_direct']],
            'direct_time_wins_over_state_in': cw,
            'direct_time_beats_deployed_time_in': bw,
            'D_over_C_range': [min(out[k]['ratios']['D_over_C_state']
                                   for k in done),
                               max(out[k]['ratios']['D_over_C_state']
                                   for k in done)],
        }
        json.dump(out, open(path, 'w'), indent=1)
        print('\nstate-conditioning beats direct time in:',
              out['_summary']['state_conditioning_wins_in'])
        print('direct time beats DEPLOYED time-conditioning in:', bw)
    print('wrote', path)


if __name__ == '__main__':
    main()
