"""C1: the ambiguity barrier, in the SAME statistic as the measured error.

The 2026-09-12 scoping probe computed RMS over samples while `sc_auxcoord`
reports the MEAN of per-sample RMSE. By Jensen the first is the larger, and the
gap grows with the tail -- which is exactly what a few heavily-folded instants
produce. This recomputes the barrier as a mean of per-sample RMSE so the bound
and the measurement are the same object.
"""
import sys, os, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
import numpy as np
from sc_coord_displacement import psi_table, OUT, DATA, SYSTEMS, Norm
import statecond.data as scd

def barrier(un, ctab, taus, te, use_flag=None):
    """Mean over ALL (traj, tau) of the per-sample RMSE left by collisions."""
    per = []
    for i in te:
        c, u = ctab[i], un[i]
        for j in range(len(taus)):
            target = c[j]
            diff = c - target
            hits = {j}
            for k in np.nonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]:
                a = k if abs(c[k] - target) < abs(c[k + 1] - target) else k + 1
                if use_flag is not None and use_flag[i][a] != use_flag[i][j]:
                    continue
                hits.add(int(a))
            if len(hits) <= 1:
                per.append(0.0)                    # injective here
                continue
            h = sorted(hits)
            per.append(float(np.sqrt(((u[j] - u[h].mean(axis=0)) ** 2).mean())))
    return float(np.mean(per)), float(np.mean(np.asarray(per) > 0))

aux = json.load(open(os.path.join(OUT, 'auxcoord.json')))
print(f"{'sys':8s} {'E_C':>8s} {'E_E meas':>9s} {'B_psi':>8s} {'bound':>8s} {'b/meas':>7s} {'amb':>6s}")
rows = []
for name in SYSTEMS:
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    un = (np.asarray(d['u'], dtype=np.float64) - n.f_mu) / n.f_sd
    taus, flag = scd.tau_grid(), np.asarray(d['flag'])
    B_psi, amb = barrier(un, psi_table(d), taus, te)
    B_phi, _ = barrier(un, np.asarray(d['phi'], dtype=np.float64), taus, te, use_flag=flag)
    E_C = aux[name]['nrmse']['C_state']['mean']
    E_E = aux[name]['nrmse']['E_aux_state']['mean']
    bound = np.sqrt(E_C ** 2 + B_psi ** 2)
    rows.append(dict(system=name, B_psi=B_psi, B_phi=B_phi, amb=amb,
                     E_C=E_C, E_E=E_E, bound=bound, ratio=bound / E_E))
    print(f"{name:8s} {E_C:8.4f} {E_E:9.4f} {B_psi:8.4f} {bound:8.4f} "
          f"{bound/E_E:7.2f} {amb:6.3f}")
r = [x['ratio'] for x in rows]
print(f"\nbound holds (ratio>=1) in {sum(x>=1 for x in r)}/5 ; median ratio {np.median(r):.2f}")
print(f"B_phi range: {min(x['B_phi'] for x in rows):.2e} - {max(x['B_phi'] for x in rows):.2e}")
json.dump(rows, open('/tmp/c1_rows.json','w'), indent=1)
