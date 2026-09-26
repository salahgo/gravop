"""Identify which experiment each saved checkpoint actually came from.

WHY THIS EXISTS
---------------
`sc_experiment.py` wrote results to `<system><backbone><_h1>.json` but
checkpoints to `<system>_<backbone>_<kind>_seed<s>.pt` -- the h1 flag was in
the results name and NOT in the checkpoint name. So the P10 spectral-bias run
(`--h1 > 0`, same backbone) silently overwrote the P6d checkpoints, and every
later script that loaded `<system>_fno_tau_seed*.pt` believing it to be the
plain FNO arm was loading the H1 arm instead.

That went unnoticed because the DEPLOYED error is nearly backbone-invariant --
which is precisely what P12's floor predicts -- so the arm-B numbers agreed to
within seed noise. Only the ORACLE error distinguishes the arms, and nothing
was comparing it.

This script settles it by behaviour rather than by filename: it evaluates each
checkpointed tau arm's oracle error on the test split and matches it against
every published arm's `A_tau_oracle`. The nearest published arm, in units of
that arm's own seed spread, is what the file contains.

Output: results/statecond/ckpt_audit.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                   # noqa: E402
from sc_experiment import Norm, field_err, load_checkpoint, tensors  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def published(name):
    """Every published arm for one system: {label: (mean, std)}."""
    out = {}
    for p in glob.glob(os.path.join(OUT, f'{name}.json')) + \
            glob.glob(os.path.join(OUT, f'{name}_*.json')):
        base = os.path.basename(p)[:-5]
        if base.split('_')[0] != name:
            continue
        label = base[len(name) + 1:] or 'fno'
        try:
            d = json.load(open(p))
            a = d['nrmse']['A_tau_oracle']
        except (KeyError, TypeError):
            continue
        out[label] = (a['mean'], a['std'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    report = {}

    for name in SYSTEMS:
        pub = published(name)
        if not pub:
            continue
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        tr, va, te = scd.split(d['u'].shape[0])
        n = Norm(d, tr)
        u, u0, par, phi, flag, lts = T = tensors(d, n, dev)
        taus = torch.tensor(scd.tau_grid(), dtype=torch.float32, device=dev)

        def cond_tau(ri, ti):
            return torch.cat([par[ri], taus[ti][:, None]], 1)

        for path in sorted(glob.glob(
                os.path.join(CKPT, f'{name}_*_tau_seed*.pt'))):
            m = re.match(rf'{name}_(.+)_tau_seed(\d+)\.pt',
                         os.path.basename(path))
            # `fno_h1` is the checkpoint name for what the results call `h1`
            backbone, seed = m.group(1), int(m.group(2))
            claim_label = 'h1' if backbone == 'fno_h1' else backbone
            net, _ = load_checkpoint(path, dev)
            e = float(field_err(net, cond_tau, u, u0, te).mean())
            del net
            torch.cuda.empty_cache()

            # nearest published arm, in units of that arm's own seed spread
            z = {k: abs(e - mu) / (sd + 1e-12) for k, (mu, sd) in pub.items()}
            best = min(z, key=z.get)
            claim = claim_label
            report[os.path.basename(path)] = {
                'system': name, 'filename_says': claim, 'seed': seed,
                'oracle_error': e,
                'best_match': best, 'best_match_z': z[best],
                'z_against_filename': z.get(claim),
                'mislabelled': best != claim and z[best] < 2.0
                and (z.get(claim) is None or z.get(claim) > 3.0),
            }
            r = report[os.path.basename(path)]
            print(f"{os.path.basename(path):34s} A={e:.4f}  says {claim:11s} "
                  f"looks like {best:11s} (z={z[best]:.2f}, "
                  f"z_vs_filename={r['z_against_filename']:.2f})"
                  f"{'   <-- MISLABELLED' if r['mislabelled'] else ''}",
                  flush=True)

    # PER-ARM verdict. The per-seed z above is too weak to have caught the
    # original defect: a single seed of the H1 arm sits within ~1 sigma of the
    # plain arm in three of five systems. The 3-seed MEAN does not -- it is the
    # statistic the published table reports, and it matched the H1 arm to four
    # decimals in every system. Group first, judge second.
    arms = {}
    for k, v in report.items():
        arms.setdefault((v['system'], v['filename_says']), []).append(
            v['oracle_error'])
    per_arm, bad = {}, []
    for (sysname, label), errs in sorted(arms.items()):
        pub = published(sysname)
        mean = float(np.mean(errs))
        z = {kk: abs(mean - mu) / (sd / np.sqrt(len(errs)) + 1e-12)
             for kk, (mu, sd) in pub.items()}
        best = min(z, key=z.get)
        rec = {'n_seeds': len(errs), 'oracle_mean': mean,
               'best_match': best, 'best_match_z': z[best],
               'z_against_filename': z.get(label),
               'mislabelled': best != label and z[best] < 3.0}
        per_arm[f'{sysname}/{label}'] = rec
        if rec['mislabelled']:
            bad.append(f'{sysname}/{label}')
        print(f"  arm {sysname:8s}/{label:7s} mean A={mean:.4f} -> {best:11s}"
              f" (z={z[best]:.2f}, z_vs_filename="
              f"{rec['z_against_filename']:.2f})"
              f"{'   <-- MISLABELLED' if rec['mislabelled'] else ''}")

    report['_per_arm'] = per_arm
    report['_summary'] = {
        'n_checked': len([k for k in report if not k.startswith('_')]),
        'mislabelled_arms': bad,
        'cause': ('checkpoint names omitted the h1 flag that the results '
                  'names carried, so an --h1 run overwrote the plain arm'),
        'note': ('judged on the 3-seed mean; per-seed z cannot separate arms '
                 'whose oracle errors overlap within one seed spread'),
    }
    json.dump(report, open(os.path.join(OUT, 'ckpt_audit.json'), 'w'), indent=1)
    print('\nmislabelled arms:', bad or 'none')


if __name__ == '__main__':
    main()
