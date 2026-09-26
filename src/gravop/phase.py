"""The state coordinate, and the radius it is legitimately read at.

State-conditioning replaces normalised time with an observable coordinate of the
solution's state. For a halo that coordinate is a density *contrast at a radius*
-- `phi = log10( rho(r_phi, t) / rho(r_phi, 0) )` -- and E29 established the
constraint the rest of the project had missed: **`r_phi` must be a radius the
observation actually resolves.**

The emulator's radial grid starts at 0.0032 r_s, but nothing that observes a
halo reaches that far in:

    cross-domain / E20   innermost grid point   0.0032 r_s   index 0
    E11 N-body profiles  resolve only to        0.0084 r_s   index 13
    E9 mock observation  sampled from           0.02   r_s   index 23

An ensemble trained at index 0 therefore cannot be conditioned by either
application without extrapolating below the data. That is a silent failure --
the arrays broadcast, the numbers come out, and they are meaningless -- so the
check lives here as a function that raises, rather than inline in a script where
it could be forgotten on the next call site.

`e28_phase_ensemble.py --phi-index` trains at a chosen radius; `e9_inference.py`
and `e11_emulator_vs_nbody.py` check before conditioning.
"""
import numpy as np

from . import dataset as ds


def grid_log_radii():
    """log10(r / r_s) at each grid point, the axis phi indices refer to."""
    return np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)


def phase_radius(phi_index):
    """The radius, in r_s, that `phi_index` names."""
    return float(10.0 ** grid_log_radii()[int(phi_index)])


def required_phi_index(r_min):
    """Smallest grid index whose radius is at or outside `r_min`.

    "At or outside" rather than "nearest": the point is to stay within the
    observed region, so rounding to a nearer but smaller radius would defeat
    the check.
    """
    lg = grid_log_radii()
    idx = int(np.searchsorted(lg, np.log10(r_min)))
    if idx >= len(lg):
        raise ValueError(
            f'no grid radius reaches {r_min:.5f} r_s; the grid ends at '
            f'{10 ** lg[-1]:.5f} r_s')
    return idx


def check_phase_radius(phi_index, r_min, what='the observation', suffix=''):
    """Raise unless `phi_index` is inside the observed region.

    Returns the phase radius when it is legitimate, so callers can use it.
    The message names the index to retrain with, because the fix is always
    "retrain the ensemble at a bigger radius" and it should not have to be
    worked out by hand.
    """
    r_phi = phase_radius(phi_index)
    if r_phi < r_min:
        want = required_phi_index(r_min)
        raise ValueError(
            f'this ensemble reads its phase at r = {r_phi:.5f} r_s, inside '
            f'{what}, which reaches only {r_min:.5f} r_s. Conditioning on it '
            f'would extrapolate rather than measure. Retrain with '
            f'scripts/e28_phase_ensemble.py --phi-index {want}'
            f'{f" --suffix {suffix}" if suffix else ""} '
            f'(index {want} is r = {phase_radius(want):.5f} r_s).')
    return r_phi


def contrast_at(log_radii, log_rho, log_rho_initial, r_phi):
    """phi at `r_phi`, both densities interpolated to the SAME radius.

    Evaluating numerator and denominator at different radii would fold the
    profile slope into what is meant to be a pure contrast; that bug was made
    and caught once already, so the interpolation is done here rather than at
    each call site.
    """
    x = np.log10(r_phi)
    return float(np.interp(x, log_radii, log_rho)
                 - np.interp(x, log_radii, log_rho_initial))
