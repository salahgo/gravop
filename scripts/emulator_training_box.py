"""Record the emulator's training box in dimensionless and physical units.

Reads the sampling ranges of scripts/gen_dataset.py (draw_params, 'train') and
the composition of data/d1/train.h5, and converts the cross-section and
velocity-scale ranges to cm^2/g and km/s for three reference NFW haloes with
gravop.units. Writes results/emulator_scope/training_box.json.
"""
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from gravop.units import G_KPC, KPC_PER_KMS_TO_GYR, HaloUnits  # noqa: E402
import gen_dataset  # noqa: E402

BOX = {'yukawa_tu': {'log10_sigma0_hat': [-1.5, 0.7], 'log10_w_hat': [-1.0, 1.5], 'fraction': 0.7},
       'constant': {'log10_sigma_hat': [-2.0, 0.5], 'fraction': 0.3}}
HALOES = {'dwarf': (1e10, 15.0), 'milky_way': (1e12, 10.0), 'cluster': (1e15, 5.0)}


def check_box_matches_generator():
    """The ranges above must be the generator's: draw many and compare hulls."""
    rng = np.random.default_rng(0)
    s = {'yukawa_tu': [], 'constant': []}
    w = []
    for _ in range(20000):
        d, _, _ = gen_dataset.draw_params(rng, 'train')
        s[d['model_lmfp']].append(np.log10(d['sigma_m']))
        if d['model_lmfp'] == 'yukawa_tu':
            w.append(np.log10(d['w']))
    ok = (abs(min(s['yukawa_tu']) + 1.5) < 0.01 and abs(max(s['yukawa_tu']) - 0.7) < 0.01
          and abs(min(s['constant']) + 2.0) < 0.01 and abs(max(s['constant']) - 0.5) < 0.01
          and abs(min(w) + 1.0) < 0.01 and abs(max(w) - 1.5) < 0.01)
    return bool(ok)


def main():
    out = {'box': BOX, 'box_matches_generator': check_box_matches_generator(),
           'solver_settings': gen_dataset.SOLVER_KW}
    with h5py.File(os.path.join(ROOT, 'data', 'd1', 'train.h5'), 'r') as h:
        st, ic, mid = h['status'][:], h['ic_type'][:], h['model_id'][:]
    out['dataset'] = {'n': int(st.size),
                      'status': {'density_event': int((st == 0).sum()), 'no_event_by_t_end': int((st == 1).sum()),
                                 'knudsen_floor': int((st == 2).sum())},
                      'ic_family': {'nfw': int((ic == 0).sum()), 'cored_nfw': int((ic == 1).sum()),
                                    'gnfw': int((ic == 3).sum())},
                      'velocity_dependent': int((mid == 1).sum()), 'constant': int((mid == 0).sum())}
    phys = {}
    for name, (m, c) in HALOES.items():
        u = HaloUnits.from_nfw(m, c)
        t0 = 1 / np.sqrt(4 * np.pi * G_KPC * u.rho_s) * KPC_PER_KMS_TO_GYR
        phys[name] = {'m200_msun': m, 'c200': c, 'r_s_kpc': float(u.r_s),
                      'sigma_over_m_for_sigma_hat_1_cm2_g': float(u.sigma_over_m(1.0)),
                      'v0_kms': float(u.w_kms(1.0)),
                      'yukawa_sigma0_over_m_cm2_g': [float(u.sigma_over_m(10 ** x)) for x in BOX['yukawa_tu']['log10_sigma0_hat']],
                      'yukawa_w_kms': [float(u.w_kms(10 ** x)) for x in BOX['yukawa_tu']['log10_w_hat']],
                      'constant_sigma_over_m_cm2_g': [float(u.sigma_over_m(10 ** x)) for x in BOX['constant']['log10_sigma_hat']],
                      't0_gyr': float(t0), 't_end_gyr': float(gen_dataset.SOLVER_KW['t_end'] * t0)}
    out['physical'] = phys
    json.dump(out, open(os.path.join(ROOT, 'results', 'emulator_scope', 'training_box.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
