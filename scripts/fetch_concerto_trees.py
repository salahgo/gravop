"""Fetch the SIDM Concerto merger trees for LMC Halo104 (CDM and GroupSIDM).

WHY THIS EXISTS
---------------
The public SIDM Concerto release (Zenodo 10.5281/zenodo.14933624) ships the
Rockstar `hlist` for `a = 1.00000` ONLY, and states of the merger trees:
"The RCT tree*.dat files are too large to upload here; please contact the
authors."  P28/E30 therefore had to record that a second calibration of `C_eff`
was NOT POSSIBLE from the release, because `C_eff` sets a *rate* and a rate
needs time evolution.

The trees were subsequently supplied by the authors on request. This script
records exactly how they were obtained, so the provenance of `data/d3_concerto/
Halo104_trees/` is auditable rather than folklore.

The transfer is resumable: `tree_0_0_0.dat` is >5 GB per run, and gdown leaves a
`.part` file that `resume=True` continues from. Re-running this script after an
interruption costs only the remaining bytes.

Sizes are NOT asserted against hard-coded constants -- we were not told them in
advance. Instead each tree file is checked against the largest byte offset in
its own `locations.dat`, which is an internal consistency condition the file
must satisfy to be complete.
"""
import argparse
import json
import os
import sys

FOLDER = '11ETHxzD02nAs85vhAKO-3kxMsbXFn5Qv'

# (drive id, relative path) as listed from the shared folder on 2026-09-14.
FILES = [
    ('1_Q-nZgUQXJg_IAz17zW7xMx-faOUbOh3', 'cdm_trees/forests.list'),
    ('1CuU6GdMo4XpkiIrtJVJmNhi-gx2YXVD5', 'cdm_trees/locations.dat'),
    ('1N5fNQ4EzYJq9JsCXFFBOr9o0EDtRydWa', 'cdm_trees/tree_0_0_0.dat'),
    ('17HTcDqUfmK5KVh_NHdwtatJ-0x6Xwpbb', 'GroupSIDM_trees/forests.list'),
    ('1P8Kxdk7YnxKHjtlkPCaTgQ1YKP7Wkco7', 'GroupSIDM_trees/locations.dat'),
    ('1j2EYtqRl-Mo2QKPTHMPjMdZlQlS6CQ7H', 'GroupSIDM_trees/tree_0_0_0.dat'),
]


def max_offset(locations):
    """Largest byte offset named in a consistent-trees locations.dat."""
    m = 0
    with open(locations) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                m = max(m, int(parts[2]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'data', 'd3_concerto', 'Halo104_trees'))
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    if not args.verify_only:
        import gdown
        for fid, rel in FILES:
            dest = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f'--- {rel}', flush=True)
            got = gdown.download(id=fid, output=dest, resume=True,
                                 use_cookies=False, quiet=False)
            if not got:
                print(f'FAILED: {rel}', file=sys.stderr)
                return 1

    report = {'source': f'https://drive.google.com/drive/folders/{FOLDER}',
              'files': {}, 'ok': True}
    for _, rel in FILES:
        dest = os.path.join(out, rel)
        if not os.path.exists(dest):
            report['files'][rel] = {'present': False}
            report['ok'] = False
            continue
        size = os.path.getsize(dest)
        entry = {'present': True, 'bytes': size}
        if rel.endswith('tree_0_0_0.dat'):
            loc = os.path.join(os.path.dirname(dest), 'locations.dat')
            if os.path.exists(loc):
                mo = max_offset(loc)
                entry['max_offset_in_locations'] = mo
                # every offset must lie inside the file, or it is truncated
                entry['covers_all_offsets'] = bool(size > mo)
                if not entry['covers_all_offsets']:
                    report['ok'] = False
        report['files'][rel] = entry

    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
