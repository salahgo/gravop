"""Split conformal prediction over an existing ensemble.

Post-hoc. The published weights are never touched, and nothing is retrained:
calibration is a quantile of held-out residuals.

The guarantee is the standard split-conformal one. For scores
`s_1..s_n` drawn exchangeably with a test score `s_{n+1}`, taking

    q = the ceil((n+1)(1-alpha))/n empirical quantile of s_1..s_n

gives `P(s_{n+1} <= q) >= 1 - alpha`, with no assumption on the distribution
and none on the model that produced the residuals. It is MARGINAL coverage,
averaged over the calibration distribution, and it is void under distribution
shift. Both limits are properties of the method, not of this implementation.

Normalising the score by an uncertainty estimate does not change the guarantee.
It changes the SHAPE of the interval: a normalised score buys intervals that
widen where the model is unsure, which is what makes the interval informative
rather than merely valid.
"""
import numpy as np

EPS = 1e-8


def conformal_quantile(scores, alpha):
    """The finite-sample-corrected empirical quantile.

    The correction is the (n+1) in the numerator. Using the plain (1-alpha)
    quantile undercovers by O(1/n), which at n of a few hundred is the whole
    effect being measured.
    """
    s = np.sort(np.asarray(scores, float).ravel())
    n = s.size
    if n == 0:
        raise ValueError('no calibration scores')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        # alpha is too small for this calibration set to certify at all
        return np.inf
    return float(s[k - 1])


def finite_sample_band(n, alpha):
    """The band nominal coverage is expected to land in, at calibration size n.

    Split conformal is conservative by construction: coverage lies in
    [1-alpha, 1-alpha+1/(n+1)]. Empirical coverage on a finite TEST set adds
    its own binomial scatter, which the caller reports separately.
    """
    return (1.0 - alpha, 1.0 - alpha + 1.0 / (n + 1))


def calibrate(residual, sigma, alpha, normalise=True):
    """Fit the threshold on calibration residuals.

    residual : |truth - prediction|, any shape
    sigma    : an uncertainty estimate of the same shape (ensemble spread here)
    """
    r = np.asarray(residual, float).ravel()
    if normalise:
        s = r / (np.asarray(sigma, float).ravel() + EPS)
    else:
        s = r
    return {'q': conformal_quantile(s, alpha),
            'alpha': float(alpha),
            'normalised': bool(normalise),
            'n_calibration': int(s.size)}


def covers(residual, sigma, cal):
    """Elementwise: does the calibrated interval contain the truth?"""
    r = np.asarray(residual, float)
    if cal['normalised']:
        return r <= cal['q'] * (np.asarray(sigma, float) + EPS)
    return r <= cal['q']


def interval_width(sigma, cal):
    """Half-width of the prediction interval, in the units of the field."""
    if cal['normalised']:
        return cal['q'] * (np.asarray(sigma, float) + EPS)
    return np.full(np.shape(sigma), cal['q'])


def gaussian_coverage(residual, sigma, nominal):
    """The baseline this replaces: treat the ensemble spread as a Gaussian sd.

    This is what the project did before, and what the neural-operator conformal
    literature reports as badly undercovering.
    """
    from math import erf, sqrt
    # two-sided z for the nominal level, without pulling in scipy
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if erf(mid / sqrt(2.0)) < nominal:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    return float(np.mean(np.asarray(residual, float)
                         <= z * (np.asarray(sigma, float) + EPS))), z


def break_even_deferral(r, g):
    """Deferral fraction at which a gated loop stops paying (arXiv:2608.23075).

    r : surrogate cost / solver cost, per call
    g : the gate's own cost, as a share of the solver call
    Above this deferral fraction the gated loop is slower than the solver.

    This is the reference formula, and it assumes the gate is a side
    computation: every served call pays r on top of g. Where the gate IS the
    surrogate -- an ensemble whose mean is the served answer and whose spread
    is the interval -- r is already inside g, and `break_even_deferral_fused`
    applies instead. Using this one there double-counts the surrogate.
    """
    if r >= 1.0:
        return 0.0
    return float((1.0 - g - r) / (1.0 - r))


def break_even_deferral_fused(g):
    """Break-even deferral when the gate's own cost produces the served answer.

    The hybrid cost per call, in units of a solver call, is g + d: every call
    pays the ensemble, and a deferred one pays the solver as well. There is no
    separate surrogate term, because the ensemble mean IS the surrogate. Setting
    that to one gives d_break = 1 - g, and the speed-up at deferral d is
    1 / (g + d).
    """
    return float(max(0.0, 1.0 - g))


def gated_speedup(g, d):
    """Speed-up over the solver at deferral d, for the fused gate above."""
    return float(1.0 / (g + d))
