"""P14 -- clock recovery from the state, and whether it lifts the E29 boundary.

THE QUESTION
------------
The time head predicts `t_sing` from INITIAL DATA at 1.8-7.1% median relative
error, and that error is what arm B pays at deployment. A clock recovered from
the STATE is conditioned on a measurement taken at the query time: it knows
where the solution already is, not merely where it started. If its error is
materially lower, then state-conditioning plus clock recovery beats
time-conditioning at time-conditioning's own task, and the scope boundary of
E29 -- the paper's sharpest negative result, that state-conditioning discards
exactly the information an inverse problem needs -- is an artefact of our
factorisation rather than something fundamental.

TWO HEADS, both trained on the same split, same budget, same architecture as
the baseline `TimeHead` so that only the INPUTS differ:

  clock_t      (u0, par, phi, b) -> log10 t         "where on the clock am I?"
  clock_tsing  (u0, par, phi, b) -> log10 t_sing    LIKE-FOR-LIKE with the time
                                                    head, which sees (u0, par)
                                                    alone. Same target, strictly
                                                    more information, so the
                                                    comparison is clean.

The second is the one that speaks to E29: if an observation of the state
recovers `t_sing`, the parameter information that lives in `t_sing` was never
destroyed by conditioning on state -- it was simply not being read back out.

PRE-REGISTERED (Track C): clock recovery from measured phi has
lower relative error than t_sing prediction from initial data in >= 4 of 5
systems. If it does not, the boundary is fundamental rather than an artefact,
which is a sharper and still publishable statement.

Monotonicity is checked, not enforced: a recovered clock that is not monotone in
phi within a branch would be unusable as a coordinate, so the fraction of
non-monotone steps is reported.

Output: results/statecond/clock.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                    # noqa: E402
from sc_experiment import Norm, tensors                          # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


class ClockHead(nn.Module):
    """The baseline TimeHead plus the two state scalars, and nothing else.

    Identical width and depth to `statecond.model.TimeHead`, so a difference in
    error is a difference in what the head is TOLD, not in what it can express.
    """

    def __init__(self, n_fields=1, n_par=4, n_x=128, width=256, n_state=2,
                 use_par=True):
        super().__init__()
        self.use_par = use_par
        n_in = n_fields * n_x + (n_par if use_par else 0) + n_state
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, prof, par, state):
        parts = [prof.reshape(prof.shape[0], -1)]
        if self.use_par:
            parts.append(par)
        parts.append(state)
        return self.net(torch.cat(parts, 1))[:, 0]


def train_clock(T, tr, target, dev, epochs, seed, n_fields, n_par, n_x,
                use_state=True, use_par=True, use_current_field=False):
    """One head. `target` is (n_runs, n_tau) in log10 of whatever is predicted."""
    u, u0, par, phi, flag, _ = T
    torch.manual_seed(seed + 991)
    np.random.seed(seed + 991)
    n_tau = u.shape[1]
    head = ClockHead(n_fields, n_par, n_x, n_state=2 if use_state else 0,
                     use_par=use_par).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tgt = torch.tensor(target, dtype=torch.float32, device=dev)
    mu, sd = float(tgt[tr].mean()), float(tgt[tr].std()) + 1e-9
    tgt_n = (tgt - mu) / sd

    ri_all = np.repeat(tr, n_tau)
    ti_all = np.tile(np.arange(n_tau), len(tr))
    for _ in range(epochs):
        head.train()
        perm = np.random.permutation(len(ri_all))
        for s in range(0, len(perm), 512):
            sel = perm[s:s + 512]
            ri, ti = ri_all[sel], ti_all[sel]
            st = (torch.stack([phi[ri, ti], flag[ri, ti]], 1) if use_state
                  else torch.zeros(len(ri), 0, device=dev))
            prof = u[ri, ti] if use_current_field else u0[ri]
            loss = nn.functional.mse_loss(head(prof, par[ri], st),
                                          tgt_n[ri, ti])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    head.eval()
    return head, mu, sd


@torch.no_grad()
def predict(head, T, te, mu, sd, use_state=True, phi_alt=None,
            use_current_field=False):
    u, u0, par, phi, flag, _ = T
    if phi_alt is not None:
        phi = phi_alt
    n_tau = u.shape[1]
    out = np.zeros((len(te), n_tau))
    ri = np.asarray(te)
    for j in range(n_tau):
        st = (torch.stack([phi[ri, j], flag[ri, j]], 1) if use_state
              else torch.zeros(len(ri), 0, device=u.device))
        prof = u[ri, j] if use_current_field else u0[ri]
        out[:, j] = (head(prof, par[ri], st).cpu().numpy() * sd + mu)
    return out


def measured_phi(name, d, n, dev, aperture=0.05, sigma_a0=0.05, seed=20260905):
    """The P13 measurement chain: phase read off the stored field, through an
    aperture, with A(0) known only to `sigma_a0` dex. Standardised as
    `tensors` does. A clock that works only on an exact phase is not a clock
    anyone can deploy, so the recovered error is reported both ways."""
    from sc_contract import aperture_log_a
    la = aperture_log_a(name, d['u'], aperture)
    rng = np.random.default_rng(seed)
    off = rng.normal(0.0, sigma_a0, size=(d['u'].shape[0], 1))
    ph = (la - la[:, :1]) - off
    return torch.tensor((ph - n.phi_mu) / n.phi_sd, dtype=torch.float32,
                        device=dev)


def rel_err_pct(log_pred, log_true):
    """Median |pred/true - 1| in percent, the same statistic the time head is
    reported with (`tsing_median_err_pct`)."""
    return float(100 * np.median(np.abs(10.0 ** (log_pred - log_true) - 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--system', default=None)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    path = os.path.join(OUT, 'clock.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    names = SYSTEMS if args.all else [args.system]

    for name in names:
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        n_runs, n_tau, n_fields, n_x = d['u'].shape
        n_par = d['par'].shape[1]
        tr, va, te = scd.split(n_runs)
        n = Norm(d, tr)
        T = tensors(d, n, dev)
        taus = scd.tau_grid()

        lts = np.asarray(d['log_tsing'], dtype=np.float64)
        # physical time of every (run, tau): t = tau * t_sing. tau = 0 is
        # floored as in sc_experiment.phys_time -- one grid point of 48.
        floor = 0.5 * taus[taus > 0].min()
        lt = np.log10(np.maximum(taus, floor))[None, :] + lts[:, None]
        lts_grid = np.repeat(lts[:, None], n_tau, axis=1)

        phi_meas = measured_phi(name, d, n, dev)
        acc = {'clock_t_pct': [], 'clock_tsing_pct': [],
               'baseline_tsing_pct': [], 'nonmonotone_frac': [],
               'clock_t_measured_pct': [], 'clock_tsing_measured_pct': [],
               'inverse_observed_pct': [], 'inverse_initial_pct': []}
        for s in range(args.seeds):
            h1, m1, s1 = train_clock(T, tr, lt, dev, args.epochs, s,
                                     n_fields, n_par, n_x)
            p1 = predict(h1, T, te, m1, s1)
            acc['clock_t_pct'].append(rel_err_pct(p1, lt[te]))

            h2, m2, s2 = train_clock(T, tr, lts_grid, dev, args.epochs, s,
                                     n_fields, n_par, n_x)
            p2 = predict(h2, T, te, m2, s2)
            acc['clock_tsing_pct'].append(rel_err_pct(p2, lts_grid[te]))

            # the SAME head with the state channels removed: the honest
            # baseline, trained here rather than quoted from another run
            h0, m0, s0 = train_clock(T, tr, lts_grid, dev, args.epochs, s,
                                     n_fields, n_par, n_x, use_state=False)
            p0 = predict(h0, T, te, m0, s0, use_state=False)
            acc['baseline_tsing_pct'].append(rel_err_pct(p0, lts_grid[te]))

            acc['clock_t_measured_pct'].append(rel_err_pct(
                predict(h1, T, te, m1, s1, phi_alt=phi_meas), lt[te]))
            acc['clock_tsing_measured_pct'].append(rel_err_pct(
                predict(h2, T, te, m2, s2, phi_alt=phi_meas), lts_grid[te]))

            # THE E29 QUESTION. In an inverse problem the parameters are what
            # you are inferring, so a clock that is handed them is circular.
            # These two heads see no parameters at all: one sees the field as
            # observed at the query time plus its state coordinate, the other
            # only the initial field. If the observation localises t_sing much
            # better than the initial field does, the information E29 says
            # state-conditioning discards is recoverable from an observation.
            hi, mi, si = train_clock(T, tr, lts_grid, dev, args.epochs, s,
                                     n_fields, n_par, n_x, use_par=False,
                                     use_current_field=True)
            acc['inverse_observed_pct'].append(rel_err_pct(
                predict(hi, T, te, mi, si, use_current_field=True),
                lts_grid[te]))
            hj, mj, sj = train_clock(T, tr, lts_grid, dev, args.epochs, s,
                                     n_fields, n_par, n_x, use_state=False,
                                     use_par=False)
            acc['inverse_initial_pct'].append(rel_err_pct(
                predict(hj, T, te, mj, sj, use_state=False), lts_grid[te]))
            del hi, hj

            # is the recovered clock usable as a coordinate? it must increase
            # with tau along a trajectory
            dif = np.diff(p1, axis=1)
            acc['nonmonotone_frac'].append(float((dif <= 0).mean()))

            del h1, h2, h0
            torch.cuda.empty_cache()

        def ms(v):
            return {'mean': float(np.mean(v)), 'std': float(np.std(v))}

        rec = {k: ms(v) for k, v in acc.items()}
        rec['system'] = name
        rec['seeds'] = args.seeds
        rec['published_time_head_pct'] = float(
            json.load(open(os.path.join(OUT, f'{name}.json')))
            ['tsing_median_err_pct']['mean'])
        base = rec['baseline_tsing_pct']['mean']
        rec['improvement_vs_same_head_no_state'] = (
            base / rec['clock_tsing_pct']['mean'])
        rec['improvement_vs_published_head'] = (
            rec['published_time_head_pct'] / rec['clock_tsing_pct']['mean'])
        rec['state_helps'] = rec['clock_tsing_pct']['mean'] < base
        rec['state_helps_measured'] = (
            rec['clock_tsing_measured_pct']['mean'] < base)
        rec['measured_phi_cost'] = (rec['clock_tsing_measured_pct']['mean']
                                    / rec['clock_tsing_pct']['mean'])
        io_, ii = (rec['inverse_observed_pct']['mean'],
                   rec['inverse_initial_pct']['mean'])
        rec['inverse_gain'] = ii / io_
        rec['observation_localises_clock'] = io_ < ii
        out[name] = rec
        print(f"{name:8s} t_sing from state {rec['clock_tsing_pct']['mean']:6.2f}%"
              f"  vs same head without state {base:6.2f}%"
              f"  (published head {rec['published_time_head_pct']:.2f}%)"
              f"  -> {rec['improvement_vs_same_head_no_state']:5.2f}x  |"
              f"  t recovered to {rec['clock_t_pct']['mean']:6.2f}%"
              f"  non-monotone {100 * rec['nonmonotone_frac']['mean']:.2f}%"
              f" | measured phi {rec['clock_tsing_measured_pct']['mean']:6.2f}%"
              f" | inverse: observed {rec['inverse_observed_pct']['mean']:6.2f}%"
              f" vs initial {rec['inverse_initial_pct']['mean']:6.2f}%"
              f" ({rec['inverse_gain']:.2f}x)",
              flush=True)
        json.dump(out, open(path, 'w'), indent=1)

    done = [k for k in SYSTEMS if k in out]
    if len(done) == len(SYSTEMS):
        wins = [k for k in done if out[k]['state_helps']]
        inv = [k for k in done if out[k]['observation_localises_clock']]
        out['_summary'] = {
            'prereg': ('clock recovery from measured phi has lower relative '
                       'error than t_sing prediction from initial data in '
                       '>= 4 of 5 systems'),
            'state_helps_in': wins,
            'observation_localises_clock_in': inv,
            'inverse_gain_range': [min(out[k]['inverse_gain'] for k in done),
                                   max(out[k]['inverse_gain'] for k in done)],
            'prereg_met': len(wins) >= 4,
            'improvement_range': [
                min(out[k]['improvement_vs_same_head_no_state'] for k in done),
                max(out[k]['improvement_vs_same_head_no_state'] for k in done)],
            'verdict': ('the E29 boundary is an artefact of the factorisation: '
                        'an observation of the state recovers t_sing'
                        if len(wins) >= 4 else
                        'the E29 boundary is FUNDAMENTAL: observing the state '
                        'does not recover t_sing'),
        }
        json.dump(out, open(path, 'w'), indent=1)
        print('\nstate helps in:', wins, '| prereg met:',
              out['_summary']['prereg_met'])
        print(out['_summary']['verdict'])
    print('wrote', path)


if __name__ == '__main__':
    main()
