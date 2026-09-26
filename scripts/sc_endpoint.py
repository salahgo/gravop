"""P16 - the endpoint policy for a predicted tau, measured rather than asserted.

WHY THIS EXISTS
---------------
Every deployed number in this paper is produced by querying an operator trained
on tau in [0, 1] at tau_hat = t / t_sing_hat, and `sc_experiment.py:296` clips
that query to [0, 1]. The clip is load-bearing and the manuscript never states
it. When the time head *under*-predicts t_sing (ratio = t_sing_hat/t_sing < 1)
every query with tau > ratio lands above 1 and is answered with the operator's
tau = 1 output -- the model is never asked to extrapolate, but it is also never
allowed to say "past the end", so the last stretch of the trajectory is served
by a single saturated prediction.

Two things a referee is entitled to know, and neither can be read off the
existing artefacts:

1. **How much of the deployment is served by the clip**, per system.
2. **Whether the clip flatters arm B.** The alternative policy is to let the
   query run past 1 into extrapolation. If unclipped extrapolation were better,
   the reported penalty would be an artefact of a conservative convention.

This script answers both from the saved fno checkpoints (P6d), so it costs
seconds and no retraining. It re-derives `ratio` exactly as `sc_experiment.py`
does, and reuses that module's `field_err` so the error definition is identical.

Output: results/statecond/endpoint.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                   # noqa: E402
from sc_experiment import (Norm, tensors, field_err,            # noqa: E402
                           load_checkpoint)

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
SEEDS = [0, 1, 2]


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


def run_system(name, dev):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    n_runs = d['u'].shape[0]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, lts = T
    taus = scd.tau_grid()
    n_fields, n_par = d['u'].shape[2], d['par'].shape[1]

    rows = []
    for s in SEEDS:
        hp = os.path.join(CKPT, f'{name}_fno_time_seed{s}.pt')
        op = os.path.join(CKPT, f'{name}_fno_tau_seed{s}.pt')
        if not (os.path.exists(hp) and os.path.exists(op)):
            continue
        head, mh = load_checkpoint(hp, dev)
        net, mo = load_checkpoint(op, dev)
        check_plain_arm(mh, hp)
        check_plain_arm(mo, op)
        with torch.no_grad():
            lt_pred = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt_pred - d['log_tsing'][te])          # pred / true

        raw = np.tile(taus, (len(te), 1)) / ratio[:, None]
        clipped = raw > 1.0

        def cond_t(ri, ti):
            return torch.cat([par[ri],
                              torch.tensor(taus[ti], dtype=torch.float32,
                                           device=dev)[:, None]], 1)

        e_clip = field_err(net, cond_t, u, u0, te,
                           tau_query=np.clip(raw, 0, 1))
        e_free = field_err(net, cond_t, u, u0, te, tau_query=raw)

        m = clipped
        rows.append({
            'seed': s,
            'frac_queries_clipped': float(m.mean()),
            'frac_trajectories_underpredicted': float((ratio < 1).mean()),
            'max_query_tau': float(raw.max()),
            'p99_query_tau': float(np.quantile(raw, 0.99)),
            'err_clipped_region': float(e_clip[m].mean()) if m.any() else None,
            'err_unclipped_region': (float(e_clip[~m].mean())
                                     if (~m).any() else None),
            'deployed_err_clip_policy': float(e_clip.mean()),
            'deployed_err_extrapolate': float(e_free.mean()),
            'frac_deployed_error_from_clipped': (
                float(e_clip[m].sum() / e_clip.sum()) if m.any() else 0.0),
        })
        del head, net
        torch.cuda.empty_cache()
    return rows


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = {}
    for name in SYSTEMS:
        rows = run_system(name, dev)
        if not rows:
            print(f'{name}: no checkpoints, skipped')
            continue
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in rows[0] if k != 'seed' and rows[0][k] is not None}
        agg['n_seeds'] = len(rows)
        agg['per_seed'] = rows
        agg['clip_penalty_ratio'] = (agg['deployed_err_clip_policy']
                                     / agg['deployed_err_extrapolate'])
        # Reproduction check. The checkpoints are from the P6d re-run, not the
        # run that wrote <system>.json, so the two agree only up to training
        # nondeterminism. Record the comparison instead of assuming it.
        pub = json.load(open(os.path.join(OUT, f'{name}.json')))
        pm = pub['nrmse']['B_tau_deployed']['mean']
        ps = pub['nrmse']['B_tau_deployed']['std']
        agg['published_deployed_err'] = pm
        dep = agg['deployed_err_clip_policy']
        agg['checkpoint_vs_published_ratio'] = dep / pm
        agg['checkpoint_vs_published_z'] = (dep - pm) / ps

        out[name] = agg
        print(f"{name}: {100*agg['frac_queries_clipped']:5.1f}% of queries "
              f"clipped, {100*agg['frac_deployed_error_from_clipped']:5.1f}% "
              f"of deployed error there; clip/extrapolate = "
              f"{agg['clip_penalty_ratio']:.3f}")

    zs = [abs(v['checkpoint_vs_published_z']) for v in out.values()]
    fr = [v['frac_queries_clipped'] for v in out.values()]
    cp = [v['clip_penalty_ratio'] for v in out.values()]
    out['_summary'] = {
        'policy': ('tau_hat = t / t_sing_hat is clipped to [0, 1]; the operator '
                   'is never queried outside its training interval'),
        'frac_queries_clipped_range': [min(fr), max(fr)],
        'clip_penalty_ratio_range': [min(cp), max(cp)],
        'verdict': ('the clip is favourable to arm B in every system where the '
                    'ratio is below 1, so the reported deployment penalty is a '
                    'lower bound on what free extrapolation would give'
                    if max(cp) < 1.0 else
                    'the clip is not uniformly favourable; see per-system '
                    'clip_penalty_ratio'),
        'max_abs_z_vs_published': max(zs),
        'reproduction': ('the checkpointed arms reproduce the published '
                         'deployed error to within %.1f seed-sigma in all five '
                         'systems' % max(zs)),
    }
    p = os.path.join(OUT, 'endpoint.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
