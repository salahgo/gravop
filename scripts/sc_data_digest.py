"""Content digests of the benchmark trajectory sets, so a regeneration can be
checked against the sets the study used.

The HDF5 sets under data/sc/ are not archived; they are regenerated from the
released generators and seeds (REPRODUCING.md). A file checksum cannot verify a
regeneration, because each file also records per-run wall-clock times (dataset
`wall`) and HDF5 metadata that differ between runs of an identical computation.
The digest here is the SHA-256 of every other dataset, visited in sorted order,
with its name, dtype, shape and raw little-endian bytes.

  sc_data_digest.py                 # print digests of data/sc/<system>.h5
  sc_data_digest.py --write         # record them in data/sc/manifest.json
  sc_data_digest.py --check DIR     # compare DIR/<system>.h5 with the manifest
"""
import argparse
import hashlib
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data', 'sc')
MANIFEST = os.path.join(DATA, 'manifest.json')
EXCLUDE = {'wall'}


def digest(path):
    h = hashlib.sha256()
    names = []
    with h5py.File(path, 'r') as f:
        f.visititems(lambda n, o: names.append(n)
                     if isinstance(o, h5py.Dataset) else None)
        for n in sorted(names):
            if n.split('/')[-1] in EXCLUDE:
                continue
            a = np.ascontiguousarray(f[n][()])
            if a.dtype.byteorder == '>':
                a = a.byteswap().view(a.dtype.newbyteorder('<'))
            h.update(f'{n}|{a.dtype.str}|{a.shape}|'.encode())
            h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--check', metavar='DIR', default=None)
    a = ap.parse_args()
    man = json.load(open(MANIFEST))
    systems = sorted(man)
    if a.check:
        bad = 0
        for s in systems:
            p = os.path.join(a.check, f'{s}.h5')
            if not os.path.exists(p):
                continue
            want = man[s].get('content_sha256')
            got = digest(p)
            ok = want == got
            bad += not ok
            print(f"{s:8s} {'OK' if ok else 'MISMATCH'}  {got[:16]}")
        return 1 if bad else 0
    for s in systems:
        p = os.path.join(DATA, f'{s}.h5')
        if not os.path.exists(p):
            continue
        d = digest(p)
        print(f'{s:8s} {d}')
        if a.write:
            man[s]['content_sha256'] = d
            man[s]['content_sha256_excludes'] = sorted(EXCLUDE)
    if a.write:
        json.dump(man, open(MANIFEST, 'w'), indent=1)
        print('recorded in', os.path.relpath(MANIFEST, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
