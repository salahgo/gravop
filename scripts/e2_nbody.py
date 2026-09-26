"""E2 — N-body validation against Fischer+2025 (arXiv:2506.06269) public data.

Stage 'inspect': print HDF5 structure + header of a snapshot.
Stage 'profiles': compute radial density profiles about the shrinking-sphere
    center for all downloaded snapshots -> results/e2/nbody_profiles.npz
Stage 'compare': run gravop solver at the matched dimensionless cross-section
    and compare central-density evolution + profiles -> results/e2/*.json/png

Halo (Fischer+2025): NFW r_s = 1.28 kpc, rho_s = 4.42e7 Msun/kpc^3,
constant sigma/m = 80 cm^2/g (run label r80; exact matching convention for the
fluid sigma recorded by the compare stage).
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'd3_fischer2025')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'e2')
os.makedirs(OUT, exist_ok=True)

R_S_KPC = 1.28
RHO_S_MSUN_KPC3 = 4.42e7


def inspect(path):
    with h5py.File(path, 'r') as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'  {name}: {obj.shape} {obj.dtype}')
        print(path)
        for k, v in f.attrs.items():
            print(' attr', k, v)
        for grp in f:
            print(' group', grp)
            for k, v in f[grp].attrs.items():
                print('   attr', k, v)
            f[grp].visititems(show)


def shrinking_sphere_center(pos, mass, r_start=None, shrink=0.9, min_n=1000):
    c = np.average(pos, axis=0, weights=mass)
    r = r_start or np.percentile(np.linalg.norm(pos - c, axis=1), 90)
    idx = np.arange(len(pos))
    while True:
        d = np.linalg.norm(pos[idx] - c, axis=1)
        keep = d < r
        if keep.sum() < min_n or r < 1e-3:
            break
        idx = idx[keep]
        c = np.average(pos[idx], axis=0, weights=mass[idx])
        r *= shrink
    return c


def radial_profile(pos, mass, center, n_bins=60, r_min=1e-2, r_max=1e2):
    d = np.linalg.norm(pos - center, axis=1)
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    m_in, _ = np.histogram(d, bins=bins, weights=mass)
    vol = 4. / 3. * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    r_mid = np.sqrt(bins[1:] * bins[:-1])
    return r_mid, m_in / vol


def profiles():
    snaps = sorted(glob.glob(os.path.join(DATA, 'snapshot_*.hdf5')))
    out = {}
    for s in snaps:
        with h5py.File(s, 'r') as f:
            hdr = dict(f['Header'].attrs)
            t = float(hdr['Time'])
            pt = f['PartType1']
            pos = pt['Coordinates'][:].astype(np.float64)
            if 'Masses' in pt:
                mass = pt['Masses'][:].astype(np.float64)
            else:
                mass = np.full(len(pos), hdr['MassTable'][1])
        c = shrinking_sphere_center(pos, mass)
        r, rho = radial_profile(pos, mass, c)
        key = os.path.basename(s).replace('.hdf5', '')
        out[key + '_t'] = t
        out[key + '_r'] = r
        out[key + '_rho'] = rho
        print(key, 't =', t, 'center =', c)
    np.savez(os.path.join(OUT, 'nbody_profiles.npz'), **out)
    print('saved', os.path.join(OUT, 'nbody_profiles.npz'))


def compare():
    """Fluid solver vs N-body: central density evolution + profiles.

    Unit conversion (Fischer+2025 r80 run): lengths kpc, velocities km/s,
    masses 1e10 Msun, time kpc/(km/s) = 0.978 Gyr. Dimensionless units:
    r_s = 1.28 kpc, rho_s = 4.42e7 Msun/kpc^3 = 4.42e-3 (1e10 Msun)/kpc^3,
    t0 = 1/sqrt(4 pi G rho_s), sigma-hat = 80 cm^2/g * rho_s r_s = 0.9455.
    C is the fluid-model calibration constant: report both default 0.753 and
    a best-fit (time-rescaling) value, as in Mace+2025-style calibrations.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from gravop import solver

    G_KPC = 4.30091e-6  # kpc (km/s)^2 / Msun
    rho_s = 4.42e7  # Msun/kpc^3
    t0_gadget = 1.0 / np.sqrt(4 * np.pi * G_KPC * rho_s)  # in kpc/(km/s)
    sigma_hat = 80.0 * (rho_s * R_S_KPC) * 2.08909e-10  # Msun/kpc^2 -> g/cm^2

    d = np.load(os.path.join(OUT, 'nbody_profiles.npz'))
    keys = sorted({k.rsplit('_', 1)[0] for k in d.files})
    print('sigma_hat =', sigma_hat, ' t0 [gadget] =', t0_gadget)

    res = solver.run_halo(sigma_m=sigma_hat, n_shells=200, eps_t=1e-3,
                          rho_c_factor=3e3, timestep_mode='physical')
    ts = res['t']

    rows = []
    for k in keys:
        t_nb = float(d[k + '_t']) / t0_gadget  # dimensionless
        r_nb = d[k + '_r'] / R_S_KPC
        rho_nb = d[k + '_rho'] / (rho_s * 1e-10)  # data in 1e10Msun/kpc^3
        # fluid profile at the same dimensionless time
        if t_nb > ts[-1]:
            rows.append({'snap': k, 't_dimless': t_nb, 'note': 'beyond fluid run'})
            continue
        i = np.clip(np.searchsorted(ts, t_nb) - 1, 0, len(ts) - 2)
        w = np.clip((t_nb - ts[i]) / max(ts[i + 1] - ts[i], 1e-300), 0, 1)
        lg = np.log10(np.clip(r_nb, 1e-3, None))
        prof = []
        for j in (i, i + 1):
            rm = np.log10(0.5 * (np.r_[res['r'][j][0], res['r'][j][:-1]]
                                 + res['r'][j]))
            prof.append(np.interp(lg, rm, np.log10(res['rho'][j])))
        lrho_fl = (1 - w) * prof[0] + w * prof[1]
        # compare in the trustworthy range 0.05 < r/rs < 10
        sel = (r_nb > 0.05) & (r_nb < 10) & (rho_nb > 0)
        dif = lrho_fl[sel] - np.log10(rho_nb[sel])
        rows.append({'snap': k, 't_dimless': float(t_nb),
                     'rms_dex': float(np.sqrt(np.mean(dif ** 2))),
                     'max_abs_dex': float(np.max(np.abs(dif))),
                     'rho_c_nb': float(np.max(rho_nb[sel])),
                     'rho_c_fluid': float(res['rho'][i, 3])})
        print(rows[-1], flush=True)
    out = {'sigma_hat': sigma_hat, 't0_gadget': t0_gadget, 'C': 0.753,
           'fluid_t_final': float(res['t_final']),
           'fluid_status': int(res['status']), 'rows': rows}
    with open(os.path.join(OUT, 'compare.json'), 'w') as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(OUT, 'fluid_run.npz'), t=res['t'], r=res['r'],
             rho=res['rho'], p=res['p'])
    print('saved compare.json')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'inspect'
    if stage == 'inspect':
        files = sorted(glob.glob(os.path.join(DATA, '*.hdf5')))
        inspect(files[0])
    elif stage == 'profiles':
        profiles()
    elif stage == 'compare':
        compare()
