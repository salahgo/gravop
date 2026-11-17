#!/usr/bin/env python3
"""NSphere particle references for the research extensions (isolated from the
CPC run matrix in scripts/nsphere_reference_runs.py, whose code and image it reuses).

Same code, build, image, truncated-NFW initial conditions, isotropic constant
cross-section and time step per dynamical time (11364 steps / t_dyn, as B1-B6,
C1-C5) as the published-reference runs. Each run lives in
experiments/research_extensions/<group>/nsphere/<run>/ with run.json (exact
command, seed, commit, image, threads, wall time, return code).

  nsphere_ext.py launch <run> [--threads T]   (blocking; use setsid/nohup)
  nsphere_ext.py measure <run>                -> <run>/measures.npz (+ t_x10 in run.json)
  nsphere_ext.py list
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import nsphere_reference_runs as NR  # noqa: E402

STEPS_PER_TDYN = 12_500_000 / 1100.0     # B1/B2 and C1/C2 cadence
SNAPS = 500                              # snapshots written over the run

HALOES = {   # mass [Msun], rs [kpc], falloff c, kappa = sigma/m [cm^2/g]
    'B': dict(mass=1.15e9, rs=1.18, falloff=19.0, kappa=50.0),
    'C': dict(mass=1.0e12, rs=25.0, falloff=8.0, kappa=58.82),
}
RUNS = {
    # Multi-fidelity ladder: a third particle number (5e4) for haloes B and C,
    # new seeds, same horizons as the 2e5 runs (B5/B6: 850 t_dyn; C4/C5: 250).
    'B_N5e4_s11': dict(group='multifidelity', halo='B', n=50_000, seed=11, tfinal=850, steps=9659001, dtwrite=19318),
    'B_N5e4_s12': dict(group='multifidelity', halo='B', n=50_000, seed=12, tfinal=850, steps=9659001, dtwrite=19318),
    'C_N5e4_s11': dict(group='multifidelity', halo='C', n=50_000, seed=11, tfinal=250, steps=2850001, dtwrite=5700),
    'C_N5e4_s12': dict(group='multifidelity', halo='C', n=50_000, seed=12, tfinal=250, steps=2850001, dtwrite=5700),
}
RUNS_FILE = os.path.join(EXT, 'scripts', 'nsphere_runs_extra.json')
if os.path.exists(RUNS_FILE):          # runs added later (designed haloes) are appended here
    _x = json.load(open(RUNS_FILE))
    HALOES.update(_x.get('haloes', {}))
    RUNS.update(_x.get('runs', {}))


def run_dir(run):
    return os.path.join(EXT, RUNS[run]['group'], 'nsphere', run)


def args_for(run):
    r = RUNS[run]
    h = HALOES[r['halo']]
    # NSphere silently readjusts ntimesteps unless (steps - 1) is a multiple of
    # dtwrite, which changes the time step; pin both (explicit values match the
    # reference runs B5/B6 and C4/C5 exactly)
    dtw = r.get('dtwrite') or int(round(STEPS_PER_TDYN * r['tfinal'] / SNAPS))
    steps = r.get('steps') or SNAPS * dtw + 1
    return steps, dtw, ['--sidm', '--sidm-kappa', f"{h['kappa']}", '--profile', 'nfw',
                        '--halo-mass', f"{h['mass']:.6g}", '--scale-radius', f"{h['rs']}",
                        '--falloff-factor', f"{h['falloff']}", '--cutoff-factor', '85',
                        '--nparticles', str(r['n']), '--ntimesteps', str(steps),
                        '--dtwrite', str(dtw), '--nout', '250', '--tfinal', str(r['tfinal']),
                        '--sort', '3', '--master-seed', str(r['seed']), '--save', 'raw-data',
                        '--tag', run]


def launch(run, threads):
    d = run_dir(run)
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, 'nsphere')
    if not os.path.exists(exe):
        subprocess.run(['cp', os.path.join(NR.NS, 'nsphere'), exe], check=True)
    steps, dtw, a = args_for(run)
    cmd = ['docker', 'run', '--rm', '--name', f'nsx_{run}', '-u', f'{os.getuid()}:{os.getgid()}',
           '-e', f'OMP_NUM_THREADS={threads}', '-v', f'{d}:/w', '-w', '/w', NR.IMAGE, './nsphere', *a]
    meta = dict(run=run, **{k: v for k, v in RUNS[run].items() if k not in ('steps', 'dtwrite')},
                halo_params=HALOES[RUNS[run]['halo']], steps=steps,
                dtwrite=dtw, nsphere_commit=NR.commit(), image=NR.IMAGE, build=NR.BUILD,
                threads=threads, command=' '.join(cmd), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(meta, open(os.path.join(d, 'run.json'), 'w'), indent=1)
    t0 = time.time()
    with open(os.path.join(d, 'stdout.log'), 'w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    meta.update(returncode=rc, wall_s=time.time() - t0)
    json.dump(meta, open(os.path.join(d, 'run.json'), 'w'), indent=1)
    if os.path.exists(exe):
        os.remove(exe)
    return rc


def halo_units(hname):
    import independent_halo_analysis as IH
    h = HALOES[hname]
    IH.HALO[hname] = dict(mass=h['mass'], rs=h['rs'], falloff=h['falloff'], kappa=h['kappa'],
                          tfinal_tdyn=1.0)
    return IH, IH.halo_units(hname)


def measure(run):
    d = run_dir(run)
    meta = json.load(open(os.path.join(d, 'run.json')))
    IH, u = halo_units(RUNS[run]['halo'])
    N, steps, dtw, tf = meta['n'], meta['steps'], meta['dtwrite'], meta['tfinal']
    path = [f for f in os.listdir(os.path.join(d, 'data')) if f.startswith('all_particle_data_')][0]
    raw = np.memmap(os.path.join(d, 'data', path), mode='r',
                    dtype=np.dtype([('rank', '<i4'), ('R', '<f4'), ('V', '<f4'), ('L', '<f4')]))
    ns = raw.size // N
    rs, I = HALOES[RUNS[run]['halo']]['rs'], u['I']
    vol = 4 / 3 * np.pi * (IH.RBINS[1:] ** 3 - IH.RBINS[:-1] ** 3)
    t, lagr, mrho, disp, prof = [], [], [], [], []
    for k in range(ns):
        s = np.asarray(raw[k * N:(k + 1) * N])
        R = s['R'].astype(np.float64) / rs
        o = np.argsort(R)
        R = R[o]
        v2 = (s['V'].astype(np.float64) ** 2 + (s['L'].astype(np.float64) / s['R'].astype(np.float64)) ** 2)[o] / 3
        v2 /= u['v0'] ** 2
        lr, mr, sd = [], [], []
        for f in IH.FRACS:
            j = max(int(round(f * N)), 1)
            lr.append(R[j - 1]); mr.append(3 * f * I / R[j - 1] ** 3); sd.append(np.sqrt(v2[:j].mean()))
        cnt = np.histogram(R, IH.RBINS)[0]
        with np.errstate(divide='ignore'):
            prof.append(np.log10(cnt * (I / N) / (vol / (4 * np.pi))))
        t.append(k * dtw / (steps - 1) * tf * u['tdyn_myr'] / u['t0_myr'])
        lagr.append(lr); mrho.append(mr); disp.append(sd)
    t, mrho = np.array(t), np.array(mrho)
    np.savez(os.path.join(d, 'measures.npz'), t=t, lagr_r=np.array(lagr), mean_rho=mrho,
             disp=np.array(disp), prof=np.array(prof), N=N, sigma_hat=u['sigma_hat'],
             t0_myr=u['t0_myr'], tdyn_myr=u['tdyn_myr'])
    mk = IH.collapse_markers(t, mrho[:, 1])
    meta['markers_t0'] = mk
    meta['sigma_hat'] = u['sigma_hat']
    json.dump(meta, open(os.path.join(d, 'run.json'), 'w'), indent=1)
    print(run, ns, 'snapshots; t_x10 =', mk['t_x10'], 't0')
    return mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['launch', 'measure', 'list'])
    ap.add_argument('run', nargs='?')
    ap.add_argument('--threads', type=int, default=5)
    a = ap.parse_args()
    if a.mode == 'list':
        for k, v in RUNS.items():
            print(k, v, ' '.join(args_for(k)[2]))
        return 0
    return launch(a.run, a.threads) if a.mode == 'launch' else (measure(a.run) and 0)


if __name__ == '__main__':
    sys.exit(main())
