"""Track D: coordinate conditioning, and the first-order law for displacement.

The referee's correction. Where the inversion is differentiable,

    dG_c/dc = d_t G / cdot(c),

so an injective coordinate can still be badly conditioned wherever its time
derivative vanishes. The testable consequence is that the displacement obeys a
first-order law

    F_c(delta) ~ kappa_c * delta,      kappa_c = RMS( ||d_t G|| / |cdot(c)| ),

with kappa computed pointwise on the reference solution and F measured by
resampling it. Reference solutions only; no network.

Pre-registration: docs/CONDITIONING_PREREGISTRATION.md, committed first.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from sc_coord_displacement import psi_table, OUT, DATA, SYSTEMS, Norm
import statecond.data as scd

DELTAS = [0.01, 0.02, 0.05, 0.10, 0.20]


def kappa(un, ctab, taus, te):
    """RMS of ||d_t G|| / |cdot c| over the trajectory, in coordinate units.

    Both derivatives are taken with the same central differences on the same
    grid, so the ratio is the local sensitivity of the state to the coordinate.
    """
    vals, amb = [], []
    for i in te:
        c, u = ctab[i], un[i]
        dc = np.gradient(c, taus)
        du = np.gradient(u, taus, axis=0)
        gnorm = np.sqrt((du ** 2).mean(axis=(1, 2)))       # ||d_t G|| per tau
        good = np.abs(dc) > 1e-12
        vals.append(gnorm[good] / np.abs(dc[good]))
        amb.append(1.0 - good.mean())
    v = np.concatenate(vals)
    return float(np.sqrt((v ** 2).mean())), float(np.median(v)), float(np.mean(amb))


def displacement(un, ctab, taus, te, delta):
    """F_c(delta): resample the reference along a coordinate perturbed by delta.

    The perturbation is applied in the coordinate's own range so that delta is
    dimensionless in every coordinate, and symmetrised over sign.
    """
    acc = []
    for sgn in (+1.0, -1.0):
        sq = []
        for i in te:
            c, u = ctab[i], un[i]
            span = float(c.max() - c.min())
            if span <= 0:
                continue
            target = c + sgn * delta * span
            idx = np.interp(target, c, taus) if c[-1] > c[0] else \
                  np.interp(target[::-1], c[::-1], taus[::-1])[::-1]
            idx = np.clip(idx, taus[0], taus[-1])
            up = np.stack([[np.interp(idx, taus, u[:, ch, x])
                            for x in range(u.shape[2])] for ch in range(u.shape[1])])
            up = up.transpose(2, 0, 1)
            sq.append(float(((up - u) ** 2).mean()))
        acc.append(np.sqrt(np.mean(sq)))
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    a = ap.parse_args()
    out = {}
    print(f"{'system':8s} {'coord':6s} {'kappaRMS':>10s} {'kappaMed':>8s}  " +
          "  ".join(f'd={d}' for d in DELTAS))
    for name in a.systems:
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        tr, _, te = scd.split(d['u'].shape[0])
        n = Norm(d, tr)
        un = (np.asarray(d['u'], dtype=np.float64) - n.f_mu) / n.f_sd
        taus = scd.tau_grid()
        coords = {'tau': np.tile(taus, (un.shape[0], 1)),
                  'phi': np.asarray(d['phi'], dtype=np.float64),
                  'psi': psi_table(d)}
        for cname, ctab in coords.items():
            k_rms, k_med, zero_dc = kappa(un, ctab, taus, te)
            F = {str(dd): displacement(un, ctab, taus, te, dd) for dd in DELTAS}
            ratio = {k: (v / (k_rms * float(k))) if k_rms > 0 else None
                     for k, v in F.items()}
            out[f'{name}_{cname}'] = dict(kappa_rms=k_rms, kappa_median=k_med,
                                          zero_derivative_frac=zero_dc, F=F, ratio=ratio)
            print(f"{name:8s} {cname:6s} {k_rms:10.4f} {k_med:8.4f}  " +
                  "  ".join(f'{ratio[str(dd)]:5.2f}' for dd in DELTAS), flush=True)
    json.dump(out, open(os.path.join(OUT, 'conditioning.json'), 'w'), indent=1)
    print('wrote conditioning.json')


if __name__ == '__main__':
    main()
