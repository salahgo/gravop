"""Reader and inventory for the SIDM Concerto consistent-trees files.

The tree files are >5 GB of text each, and F9 owns all 24 CPU threads, so this
module never streams a whole file. `locations.dat` gives the byte offset of every
tree root, which is exactly what it is for: we seek to the tree we want and read
only that block. Extracting one halo's history costs a seek and a few kB.

Column semantics are taken from the file's own header, not assumed:
  scale(0)      scale factor of the row  -> time
  Mvir(10)      Msun/h
  Rvir(11)      kpc/h COMOVING
  rs(12)        kpc/h COMOVING
  vmax(16)      km/s PHYSICAL
  RVmax (last)  radius at which vmax occurs
  pid(5)        -1 for a distinct (non-sub) halo
  Snap_idx(31)  snapshot index, 235 at a = 1

Note the mixed convention: radii are comoving kpc/h while velocities are
physical km/s. DATA.md records the same trap for `out_sidm.txt` vs
`log.txt`; do not apply h twice.
"""
import os

COL = {'scale': 0, 'id': 1, 'desc_scale': 2, 'desc_id': 3, 'num_prog': 4,
       'pid': 5, 'upid': 6, 'Mvir': 10, 'Rvir': 11, 'rs': 12, 'vrms': 13,
       'mmp': 14, 'scale_of_last_MM': 15, 'vmax': 16, 'x': 17, 'y': 18,
       'z': 19, 'Tree_root_ID': 29, 'Orig_halo_ID': 30, 'Snap_idx': 31}
RVMAX = -1          # RVmax is the final column
HALFMASS = -2


def read_header(path):
    """Return (column_line, cosmology dict) from the file header."""
    cosmo = {}
    colline = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            if colline is None:
                colline = line.strip()
            if 'Omega_M' in line:
                for part in line.lstrip('#').split(';'):
                    if '=' in part:
                        k, v = part.split('=')
                        cosmo[k.strip()] = float(v)
    return colline, cosmo


def load_locations(path):
    """root id -> byte offset, from locations.dat."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 3:
                out[int(p[0])] = int(p[2])
    return out


def tree_span(locations, root_id, file_size):
    """Exact [start, end) byte span of one tree.

    The block ends where the next tree begins, so we take the smallest offset
    strictly greater than ours. Guessing a cap instead silently truncates the
    big trees -- the Halo104 host alone is ~107 MB -- and a truncated block
    yields a main branch that just stops, which reads like a short merger
    history rather than the bug it is.
    """
    start = locations[root_id]
    later = [o for o in locations.values() if o > start]
    return start, (min(later) if later else file_size)


def read_tree_block(tree_path, offset, end=None):
    """Read the rows of the single tree occupying [offset, end)."""
    rows = []
    with open(tree_path, 'rb') as fh:
        fh.seek(offset)
        limit = None if end is None else end - offset
        data = fh.read(limit) if limit is not None else fh.read()
    for ln in data.split(b'\n'):
        if ln.startswith(b'#'):
            continue
        if ln.strip():
            rows.append(ln.decode().split())
    return rows


def main_branch(rows):
    """The main progenitor branch: follow mmp? == 1 back from the root.

    Row 0 of a tree block is the root (the halo at the final timestep). A
    progenitor is on the main branch when its `mmp?` flag is 1 AND it descends
    from the halo we are currently on.
    """
    if not rows:
        return []
    by_desc = {}
    for r in rows:
        by_desc.setdefault(int(r[COL['desc_id']]), []).append(r)
    branch = [rows[0]]
    cur = int(rows[0][COL['id']])
    while True:
        progs = [r for r in by_desc.get(cur, []) if r[COL['mmp']] == '1']
        if not progs:
            break
        nxt = progs[0]
        branch.append(nxt)
        cur = int(nxt[COL['id']])
    return branch
