"""P21b follow-up: the sidm label-error correction with each test trajectory's own termination.

`sc_clock_truth.py --sidm` applies the density-terminated label error (P21, twofold
reference) to every test trajectory, but 35 of the 61 test trajectories end on the
Knudsen floor. `sc_sidm_reference_knudsen.py` measured those labels. Here each test
trajectory is displaced by hypot(head error, label error of ITS termination type), with
the floor construction of `sc_clock_truth.floor_at` (sign-symmetrised, test split,
training normalisation), and the frozen quadrature rule gives the predicted penalty.
Two variants: both label errors from twofold references (like-for-like with P21), and the
Knudsen labels from their three-grid Richardson limit (density stays twofold, so this is
still a lower bound for those). Nothing is fitted.
Output: results/statecond/clock_truth_sidm_mixed.json
"""
import json, os, sys
import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(__file__))
import sc_clock_truth as ct  # noqa: E402
from sc_provenance import stamp  # noqa: E402


def per_trajectory_floor(delta_by_traj):
    d = ct.scd.load(os.path.join(ct.DATA, 'sidm.h5')); tr, _, te = ct.scd.split(d['u'].shape[0])
    n = ct.Norm(d, tr); un = (d['u'] - n.f_mu) / n.f_sd; taus = ct.scd.tau_grid(); acc = []
    for sgn in (+1., -1.):
        e = []
        for j, i in enumerate(te):
            tq = np.clip(taus / (1. + sgn * delta_by_traj[j]), 0., 1.)
            up = ct.resample_tau(un[i:i + 1], taus, tq, 'linear')
            e.append(float(np.sqrt(((up[0] - un[i]) ** 2).mean(axis=(1, 2))).mean()))
        acc.append(np.mean(e))
    return float(np.mean(acc))


def main():
    full = h5py.File(os.path.join(ct.DATA, 'sidm_full.h5'), 'r'); st = full['status'][:]; keep = np.flatnonzero(st != 1)
    bench = h5py.File(os.path.join(ct.DATA, 'sidm.h5'), 'r')
    assert np.allclose(bench['log_tsing'][:], full['log_tsing'][keep]), 'benchmark rows are not the non-horizon rows of sidm_full'
    _, _, te = ct.scd.split(len(keep)); kind = st[keep][te]
    sr = json.load(open(os.path.join(ct.OUT, 'sidm_reference.json'))); kr = json.load(open(os.path.join(ct.OUT, 'sidm_reference_knudsen.json')))
    th = json.load(open(os.path.join(ct.OUT, 'timehead.json')))['sidm']; head = th['converged_head_err_pct'] / 100.; E_A = th['armA']
    old = json.load(open(os.path.join(ct.OUT, 'clock_truth_sidm.json')))
    # reproduce the stored single-population construction first
    F_head = per_trajectory_floor(np.full(len(te), head))
    assert abs(F_head / old['floor_at_head_error'] - 1) < 1e-6, (F_head, old['floor_at_head_error'])
    out = dict(protocol=__doc__.split('Output:')[0].strip(), test_trajectories=int(len(te)), density_terminated=int((kind == 0).sum()),
               knudsen_terminated=int((kind == 2).sum()), head_vs_label=head, armA_oracle_error=E_A, floor_at_head_error=F_head,
               penalty_predicted_at_head_error=float(np.hypot(E_A, F_head) / E_A), penalty_measured_converged=th['penalty_converged'],
               single_population_record=dict(label_error=old['label_vs_truth_grid_2x'], penalty_understated_by=old['penalty_understated_by']))
    for label, kn in [('twofold_both', kr['rel_dt_sing_median_vs600']), ('knudsen_richardson', kr['rel_dt_richardson_median'])]:
        lab = np.where(kind == 2, kn, sr['rel_dt_sing_median']); eff = np.hypot(head, lab); F = per_trajectory_floor(eff)
        pen = float(np.hypot(E_A, F) / E_A)
        out[label] = dict(label_error_density=sr['rel_dt_sing_median'], label_error_knudsen=kn, floor_at_effective_error=F,
                          penalty_predicted_at_effective_error=pen, penalty_understated_by=pen / out['penalty_predicted_at_head_error'])
    out['_provenance'] = stamp('sc_clock_truth_sidm_mixed.py')
    path = os.path.join(ct.OUT, 'clock_truth_sidm_mixed.json'); json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('protocol', '_provenance')}, indent=1))


if __name__ == '__main__': main()
