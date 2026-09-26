"""statecond.data — trajectories to fixed-grid samples, uniformly across systems.

Every trajectory is resampled onto the same normalised-time grid used for the
gravothermal runs (`gravop.dataset.tau_grid`): 24 points uniform on [0, 0.92]
and 24 more on (0.92, 1], refined near the singularity where the dynamics are
fastest.

Two conditioning variables are built for each sample, and they are the object
of the whole study:

  tau(t)   = t / t_sing
             Normalised time. Known during training; at deployment it requires
             t_sing, which must be predicted from the initial data.

  phi(t)   = log10( A(t) / A(0) ),  together with  flag(t) = [A has passed its
             minimum]
             The state coordinate. A is the system's singular observable, so
             phi is read directly off the current field: it needs no knowledge
             of t_sing. The flag is required because A is not monotone -- in
             several of these systems the observable first relaxes and then
             runs away, so phi alone is two-valued.

Both are defined by the same two lines of code for every system, which is what
makes the comparison a statement about the conditioning variable rather than
about any one domain.
"""
import numpy as np

N_TAU = 48


def tau_grid():
    """24 uniform points on [0, 0.92), 24 on [0.92, 1]."""
    a = np.linspace(0.0, 0.92, 24, endpoint=False)
    b = np.linspace(0.92, 1.0, 24)
    return np.concatenate([a, b])


def resample(tr, taus=None):
    """Interpolate a Trajectory onto the tau grid.

    Returns (u, a) with u of shape (N_TAU, n_fields, N_X) and a of shape
    (N_TAU,). Interpolation is linear in t on the stored (log) fields; the
    observable is interpolated in log A, which is the quantity that varies over
    decades.
    """
    if taus is None:
        taus = tau_grid()
    ts = tr.t
    targets = np.clip(taus * tr.t_sing, ts[0], ts[-1])
    idx = np.clip(np.searchsorted(ts, targets, side='right') - 1,
                  0, len(ts) - 2)
    dt = np.maximum(ts[idx + 1] - ts[idx], 1e-300)
    w = np.clip((targets - ts[idx]) / dt, 0.0, 1.0).astype(np.float32)
    u = (1 - w)[:, None, None] * tr.u[idx] + w[:, None, None] * tr.u[idx + 1]
    la = np.log10(tr.a)
    a = 10.0 ** ((1 - w) * la[idx] + w * la[idx + 1])
    return u.astype(np.float32), a.astype(np.float64)


def phase_variables(a):
    """State coordinate from the singular observable.

    a: (n_runs, N_TAU) observable A on the tau grid.
    Returns (phi, flag), both (n_runs, N_TAU) float32:
        phi  = log10(A / A(0))
        flag = 1 after the epoch of minimum A, else 0
    """
    la = np.log10(a)
    phi = la - la[:, :1]
    imin = np.argmin(la, axis=1)
    j = np.arange(a.shape[1])[None, :]
    flag = (j > imin[:, None]).astype(np.float32)
    return phi.astype(np.float32), flag


def build(system, trajectories, keep_nonsingular=False):
    """Assemble a dataset dict from a list of Trajectory objects."""
    trs = [t for t in trajectories
           if t is not None and (keep_nonsingular or t.status == 0)]
    if not trs:
        raise ValueError('no usable trajectories')
    us, avs = zip(*(resample(t) for t in trs))
    u = np.stack(us)                                   # (R, T, C, N)
    a = np.stack(avs)                                  # (R, T)
    phi, flag = phase_variables(a)
    par = np.array([[t.params[k] for k in system.param_names] for t in trs],
                   dtype=np.float32)
    return {
        'u': u,
        'u0': u[:, 0].copy(),
        'a': a.astype(np.float32),
        'phi': phi,
        'flag': flag,
        'par': par,
        'log_tsing': np.array([np.log10(t.t_sing) for t in trs], np.float32),
        'status': np.array([t.status for t in trs], np.int8),
        'wall': np.array([t.wall for t in trs], np.float32),
        'x': system.store_grid().astype(np.float32),
        'param_names': list(system.param_names),
        'system': system.name,
    }


def save(path, d):
    import h5py
    with h5py.File(path, 'w') as f:
        for k, v in d.items():
            if k in ('param_names', 'system'):
                f.attrs[k] = v
            else:
                f.create_dataset(k, data=v, compression='gzip')


def load(path):
    import h5py
    with h5py.File(path, 'r') as f:
        d = {k: f[k][:] for k in f.keys()}
        d['param_names'] = list(f.attrs['param_names'])
        d['system'] = str(f.attrs['system'])
    return d


def split(n, seed=1234, frac=(0.8, 0.1)):
    """Trajectory-level split; identical convention to scripts/train.py."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(frac[0] * n)
    n_va = int(frac[1] * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
