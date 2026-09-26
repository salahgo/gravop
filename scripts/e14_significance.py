"""E14 — paired statistical tests for the model comparison.

Table 2 of the paper compares GravOp against baselines by median error, which
does not establish that the differences are statistically meaningful. This was
raised in external review. Here we compute, for every model, the per-trajectory
mean relative density error on the identical test split, and compare models
with a paired Wilcoxon signed-rank test (paired by trajectory, non-parametric
because the error distribution is strongly right-skewed), together with a
rank-biserial effect size and a bootstrap confidence interval on the median.

The GravOp entry is the 5-member ensemble mean, which is what the package
actually ships; single-seed members are also reported for reference.

Output: results/e14/significance.json
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds  # noqa: E402
from train import load_data, split, make_tensors  # noqa: E402
from evaluate_all import load_model, NormObj  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e14')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def per_run_errors(models, norm, d, idx):
    """Mean relative density error per trajectory for an (ensemble of) model."""
    n = NormObj(norm)
    lrho, lv, prof, par, tc, nc = make_tensors(d, n, DEV)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=DEV)
    out = np.empty(len(idx))
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            ri = idx[s:s + 64]
            B = len(ri)
            pi = prof[ri].repeat_interleave(ds.N_TAU, dim=0)
            ci = torch.cat([par[ri].repeat_interleave(ds.N_TAU, dim=0),
                            taus.repeat(B)[:, None]], dim=1)
            preds = []
            for m in models:
                f, _ = m(pi, ci)
                preds.append(f[:, 0].reshape(B, ds.N_TAU, ds.N_R))
            mu = torch.stack(preds).mean(0)
            rel = (10 ** ((mu - lrho[ri]) * n.rho_sd) - 1).abs()
            out[s:s + B] = rel.mean(dim=(1, 2)).cpu().numpy()
    return out


def compare(a, b, name_a, name_b):
    """Paired comparison of two error vectors (lower is better)."""
    diff = a - b
    stat, p = stats.wilcoxon(a, b)
    # rank-biserial effect size for the signed-rank test
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    rb = (n_neg - n_pos) / max(n_pos + n_neg, 1)   # >0 means a is better
    rng = np.random.default_rng(0)
    boot = [np.median(diff[rng.integers(0, len(diff), len(diff))])
            for _ in range(5000)]
    return {'model_a': name_a, 'model_b': name_b,
            'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
            'median_paired_diff': float(np.median(diff)),
            'diff_ci95': [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))],
            'wilcoxon_p': float(p),
            'rank_biserial': float(rb),
            'a_better_on': n_neg, 'b_better_on': n_pos, 'n_pairs': len(a)}


def main():
    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))

    errors, norm = {}, None
    ens = []
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{s}'))
        ens.append(m)
        norm = nrm
    errors['GravOp (ensemble)'] = per_run_errors(ens, norm, d, te)
    errors['GravOp (seed 0)'] = per_run_errors([ens[0]], norm, d, te)
    for name, path in (('FNO no-physics', 'fno_nophys'),
                       ('DeepONet', 'deeponet')):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs', path))
        errors[name] = per_run_errors([m], nrm, d, te)

    print(f'{"model":22s} {"median":>9} {"mean":>9}')
    for k, v in errors.items():
        print(f'{k:22s} {100 * np.median(v):8.3f}% {100 * v.mean():8.3f}%')

    # Seed-matched pairs. Comparing the 5-member ensemble against a single
    # no-physics model would confound the physics losses with ensembling, so
    # the physics ablation is tested single-seed against single-seed.
    pairs = [('GravOp (ensemble)', 'DeepONet'),
             ('GravOp (ensemble)', 'GravOp (seed 0)'),
             ('GravOp (seed 0)', 'FNO no-physics')]
    tests = []
    for ref, other in pairs:
        r = compare(errors[ref], errors[other], ref, other)
        tests.append(r)
        sig = 'SIGNIFICANT' if r['wilcoxon_p'] < 0.05 else 'NOT significant'
        print(f"\n{ref} vs {other}:")
        print(f"   median paired difference "
              f"{100 * r['median_paired_diff']:+.4f}% "
              f"(95% CI {100 * r['diff_ci95'][0]:+.4f}, "
              f"{100 * r['diff_ci95'][1]:+.4f})")
        print(f"   Wilcoxon p = {r['wilcoxon_p']:.3g}  -> {sig}; "
              f"GravOp better on {r['a_better_on']}/{r['n_pairs']} "
              f"trajectories (rank-biserial {r['rank_biserial']:+.2f})")

    json.dump({'note': 'per-trajectory mean relative density error on the '
                       'common test split; paired Wilcoxon signed-rank',
               'n_test_trajectories': len(te),
               'summary': {k: {'median': float(np.median(v)),
                               'mean': float(v.mean()),
                               'std': float(v.std())}
                           for k, v in errors.items()},
               'tests': tests},
              open(os.path.join(OUT, 'significance.json'), 'w'), indent=1)
    print('\nsaved results/e14/significance.json')


if __name__ == '__main__':
    main()
