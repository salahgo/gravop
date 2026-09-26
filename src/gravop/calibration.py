"""Numerically checked Gaussian discrepancy and finite-posterior diagnostics.

No model fitting on test observations. Discrete SBC uses exact posterior CDFs
with randomized ties, including data-dependent test quantities.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import kstest, norm


def posterior_weights(log_weight):
    values = np.asarray(log_weight, dtype=float)
    if values.ndim != 1 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('log weights must be a vector without NaN or positive infinity')
    if not np.isfinite(values).any():
        raise ValueError('posterior has no finite support')
    return np.exp(values-logsumexp(values))


class GaussianDiscrepancy:
    """Add bias to the emulator mean and covariance to observational variance."""
    def __init__(self, bias, covariance, noise_sd):
        self.bias = np.asarray(bias, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        if self.bias.ndim != 1 or self.covariance.shape != (len(self.bias),)*2:
            raise ValueError('bias and covariance shapes disagree')
        if not np.isfinite(self.bias).all() or not np.isfinite(self.covariance).all():
            raise ValueError('discrepancy must be finite')
        if not np.isfinite(noise_sd) or noise_sd <= 0:
            raise ValueError('observation noise must be finite and positive')
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12):
            raise ValueError('covariance must be symmetric')
        if np.linalg.eigvalsh(self.covariance).min() < -1e-12:
            raise ValueError('discrepancy covariance must be positive semidefinite')
        self.noise_sd = float(noise_sd)
        self.cholesky = np.linalg.cholesky(self.covariance+noise_sd**2*np.eye(len(self.bias)))
        self.logdet = float(2*np.log(np.diag(self.cholesky)).sum())

    def logpdf(self, residual):
        """Residual is observation minus uncorrected emulator, (..., n_radii)."""
        residual = np.asarray(residual, dtype=float)
        if residual.shape[-1:] != self.bias.shape or not np.isfinite(residual).all():
            raise ValueError('residual must be finite with the calibrated radius count')
        x = (residual-self.bias).reshape(-1, len(self.bias))
        z = solve_triangular(self.cholesky, x.T, lower=True, check_finite=False).T
        val = -.5*(np.square(z).sum(1)+self.logdet+len(self.bias)*np.log(2*np.pi))
        return val.reshape(residual.shape[:-1])

    def to_dict(self):
        return dict(bias=self.bias.tolist(), covariance=self.covariance.tolist(), noise_sd=self.noise_sd)


def select_discrepancy(train_residual, selection_residual, noise_sd, rng):
    train, validation = np.asarray(train_residual), np.asarray(selection_residual)
    if train.ndim != 2 or validation.ndim != 2 or len(train)<2 or not len(validation):
        raise ValueError('independent nonempty fit and selection residuals required')
    bias = train.mean(0)
    cov = np.cov(train, rowvar=False, ddof=1)
    zeros = np.zeros_like(bias); zero_cov = np.zeros_like(cov)
    recipes = [('none', zeros, zero_cov), ('bias', bias, zero_cov)]
    for shrink in (.1, .5, 1.):
        recipes.append((f'bias_cov_{shrink:g}', bias,
                        (1-shrink)*cov+shrink*np.diag(np.diag(cov))))
    # The observation-noise draw is shared across all options in this comparison.
    observed = validation+rng.normal(0, noise_sd, validation.shape)
    models = {name: GaussianDiscrepancy(b, c, noise_sd) for name, b, c in recipes}
    scores = {name: float(model.logpdf(observed).mean()) for name, model in models.items()}
    chosen = max(scores, key=scores.get)
    return models[chosen], dict(selected=chosen, selection_log_scores=scores,
                               n_fit=len(train), n_selection=len(validation))


def randomized_pit(values, weights, truth_value, uniform):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape or np.isnan(values).any():
        raise ValueError('test quantity and weights must be aligned vectors without NaN')
    if not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('weights must be normalized probabilities')
    if not 0 <= uniform <= 1 or np.isnan(truth_value):
        raise ValueError('invalid randomizer or truth quantity')
    result=float(weights[values<truth_value].sum()+uniform*weights[values==truth_value].sum())
    # A normalized floating-point sum can exceed one by a few ulps when all
    # mass is below the truth (e.g. a method rejects the true parameter).
    # Preserve boundary PITs as diagnostic failures, without treating roundoff
    # as an invalid probability or hiding any material normalization error.
    if result < -1e-12 or result > 1+1e-12:
        raise ValueError('posterior CDF lies outside probability bounds')
    return float(np.clip(result,0.,1.))


def wilson(successes, n):
    if n <= 0: raise ValueError('sample count must be positive')
    z = norm.ppf(.975); p = successes/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half), float(center+half)]


def pit_summary(pit):
    pit = np.asarray(pit, dtype=float)
    if pit.ndim != 1 or not len(pit) or not np.isfinite(pit).all() or ((pit<0)|(pit>1)).any():
        raise ValueError('PIT values must be a nonempty probability vector')
    result = kstest(pit, 'uniform')
    coverage = {}
    for level in (.68, .90, .95):
        hits = int(((pit>=(1-level)/2)&(pit<=(1+level)/2)).sum())
        coverage[str(level)] = dict(estimate=hits/len(pit), ci95=wilson(hits,len(pit)))
    return dict(n=len(pit), ks_statistic=float(result.statistic), ks_pvalue=float(result.pvalue),
                histogram=np.histogram(pit,bins=np.linspace(0,1,11))[0].tolist(),coverage=coverage)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('p-values must be probabilities')
    order = np.argsort(p); out = np.empty_like(p)
    out[order] = np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out
