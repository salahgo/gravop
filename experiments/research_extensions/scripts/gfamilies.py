"""Closure families as correction tables f(ln y) for gclosure.run_halo_t.

y = K_l / K_s; ln G = -ln(1 + y) + f(ln y); f -> 0 at both ends.
  alpha(a)        generalised p-norm (van den Bosch & Tokayer form)
  rational(a, b)  G = (1 + a y) / (1 + b y + a y^2), [1/2] Pade with the exact
                  limits; positivity needs b > -2 sqrt(a), a > 0
  gate(eps, l0, w) regime-adaptive: f = eps * exp(-(ln y - l0)^2 / (2 w^2)),
                  a localised enhancement (eps > 0) or suppression of conduction
                  centred on the transition ln y = l0
"""
import numpy as np

GX = np.linspace(-30.0, 30.0, 6001)


def base(lx):
    return -np.log1p(np.exp(lx))


def alpha(a):
    y = np.exp(GX)
    lnG = -(1.0 / a) * np.log1p(y ** a)
    f = lnG - base(GX)
    f[0] = f[-1] = 0.0
    return GX, f


def rational(a, b):
    y = np.exp(GX)
    with np.errstate(over='ignore', invalid='ignore'):
        lnG = np.log1p(a * y) - np.log(1 + b * y + a * y * y)
        f = lnG - base(GX)
    f = np.where(np.isfinite(f), f, 0.0)
    f[0] = f[-1] = 0.0
    return GX, f


def gate(eps, l0, w):
    return GX, eps * np.exp(-0.5 * ((GX - l0) / w) ** 2)
