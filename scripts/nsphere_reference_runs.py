#!/usr/bin/env python3
"""Independent-halo references: SIDM runs with the public NSphere code.

NSphere (Kamionkowski & Sigurdson 2025, arXiv:2504.13996; SIDM physics in
Kamionkowski, Sigurdson & Slone 2025, arXiv:2506.04334) evolves a spherical
system of N particles under the shell-theorem force with direct Monte Carlo
SIDM scattering. It shares no equation with the gravothermal fluid model, so it
is an independent kinetic reference. Apache-2.0; github.com/kris-sigurdson/NSphere
at the commit recorded below, built with clang in the container image below.

Haloes (truncated NFW of NSphere, rho ~ 1/[(x+0.01)(1+x)^2(1+(x/c)^10)], cut at
85 r_s; constant, isotropic cross-section):

  B  the published SIDM run of Kamionkowski et al.: M = 1.15e9 Msun,
     r_s = 1.18 kpc, c = 19, sigma/m = 50 cm^2/g (sigma-hat = 0.337), 1100 t_dyn
  C  a Milky-Way-mass, low-concentration halo: M = 1e12 Msun, r_s = 25 kpc,
     c = 8, sigma/m chosen for sigma-hat = 1.20, 400 t_dyn

Runs: B1/B2 and C1/C2 differ only in the random seed (scattering noise).
Convergence is tested on halo C, which collapses within 250 t_dyn: C3 halves
the timestep, C4 doubles the particle number, both to 250 t_dyn at C1's
cadence. (The same tests on halo B, B3 and B4, would take about a day each at
the available core count and are defined but not run.) B5 and B6 double the
particle number of halo B with two seeds, to 850 t_dyn.

Usage: nsphere_reference_runs.py launch <run> [--threads T]
       nsphere_reference_runs.py commands          (print the run matrix)
Outputs: results/independent_halo_20261107/nsphere/<run>/data/...
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NS = os.path.join(ROOT, 'data', 'ref_nsphere')
OUT = os.path.join(ROOT, 'results', 'independent_halo_20261107', 'nsphere')
IMAGE = 'nsphere-env:24.04'   # ubuntu:24.04 + clang, libomp, libgsl, libfftw3
BUILD = ('clang -Ofast -march=native -DQUAD_CACHE=16777216 -fopenmp -funroll-loops '
         '-ffast-math -fno-finite-math-only -fno-math-errno -ffp-contract=fast '
         '-Isrc/c src/c/nsphere.c -o nsphere -fopenmp -lgsl -lgslcblas -lm -lfftw3 '
         '-lfftw3_threads')

HALO = {
    'B': dict(mass=1.15e9, rs=1.18, falloff=19.0, kappa=50.0, tfinal=1100),
    'C': dict(mass=1.0e12, rs=25.0, falloff=8.0, kappa=58.82, tfinal=400),
    'Cs': dict(mass=1.0e12, rs=25.0, falloff=8.0, kappa=58.82, tfinal=250),
    'Bs': dict(mass=1.15e9, rs=1.18, falloff=19.0, kappa=50.0, tfinal=850),
}
# same timestep in units of t_dyn as the published run (12.5e6 steps / 1100 t_dyn)
RUNS = {
    'B1': dict(halo='B', n=100_000, steps=12_500_001, dtwrite=25_000, seed=1),
    'B2': dict(halo='B', n=100_000, steps=12_500_001, dtwrite=25_000, seed=2),
    'B3': dict(halo='B', n=100_000, steps=25_000_001, dtwrite=50_000, seed=1),
    'B4': dict(halo='B', n=200_000, steps=12_500_001, dtwrite=25_000, seed=3),
    'C1': dict(halo='C', n=100_000, steps=4_550_001, dtwrite=9_100, seed=1),
    'C2': dict(halo='C', n=100_000, steps=4_550_001, dtwrite=9_100, seed=2),
    'C3': dict(halo='Cs', n=100_000, steps=5_700_001, dtwrite=11_400, seed=1),
    'C4': dict(halo='Cs', n=200_000, steps=2_850_001, dtwrite=5_700, seed=3),
    # halo B at twice the particles, two seeds, to 850 t_dyn (past every collapse
    # seen at 1e5 particles, allowing for the 6% delay that doubling gave in halo C);
    # same time step per t_dyn as B1/B2
    'B5': dict(halo='Bs', n=200_000, steps=9_659_001, dtwrite=19_318, seed=1),
    'B6': dict(halo='Bs', n=200_000, steps=9_659_001, dtwrite=19_318, seed=2),
}
NOUT = 250


def args_for(run):
    r = RUNS[run]
    h = HALO[r['halo']]
    return ['--sidm', '--sidm-kappa', f"{h['kappa']}", '--profile', 'nfw',
            '--halo-mass', f"{h['mass']:.6g}", '--scale-radius', f"{h['rs']}",
            '--falloff-factor', f"{h['falloff']}", '--cutoff-factor', '85',
            '--nparticles', str(r['n']), '--ntimesteps', str(r['steps']),
            '--dtwrite', str(r['dtwrite']), '--nout', str(NOUT),
            '--tfinal', str(h['tfinal']), '--sort', '3',
            '--master-seed', str(r['seed']), '--save', 'raw-data', '--tag', run]


def commit():
    return subprocess.run(['git', '-C', NS, 'rev-parse', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()


def launch(run, threads):
    d = os.path.join(OUT, run)
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, 'nsphere')
    if not os.path.exists(exe):
        subprocess.run(['cp', os.path.join(NS, 'nsphere'), exe], check=True)
    cmd = ['docker', 'run', '--rm', '--name', f'nsphere_{run}', '-u',
           f'{os.getuid()}:{os.getgid()}', '-e', f'OMP_NUM_THREADS={threads}',
           '-v', f'{d}:/w', '-w', '/w', IMAGE, './nsphere', *args_for(run)]
    meta = dict(run=run, **RUNS[run], halo_params=HALO[RUNS[run]['halo']],
                nsphere_commit=commit(), image=IMAGE, build=BUILD, threads=threads,
                command=' '.join(cmd[cmd.index(IMAGE) + 1:]))
    json.dump(meta, open(os.path.join(d, 'run.json'), 'w'), indent=1)
    with open(os.path.join(d, 'stdout.log'), 'w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    meta['returncode'] = rc
    json.dump(meta, open(os.path.join(d, 'run.json'), 'w'), indent=1)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['launch', 'commands'])
    ap.add_argument('run', nargs='?')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    if a.mode == 'commands':
        for k in RUNS:
            print(k, './nsphere ' + ' '.join(args_for(k)))
        return 0
    return launch(a.run, a.threads)


if __name__ == '__main__':
    sys.exit(main())
