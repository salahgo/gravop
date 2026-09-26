"""gravop.units — physical <-> dimensionless conversion for SIDM haloes.

The solver and the emulator work entirely in the dimensionless variables of
Section 2 of the paper. This module is the bridge for users who start from
physical halo parameters.

The important consequence, and the reason this module is short: **the
gravothermal fluid equations in these units contain no residual halo-mass
scale.** Given a dimensionless cross-section

    sigma_hat = (sigma/m) * rho_s * r_s ,

a dimensionless velocity scale w_hat = w / v_0, and a dimensionless initial
profile, the solution is the same function for every physical halo that maps to
those numbers. Halo-mass generalisation is therefore *exact by construction*,
not something the emulator has to learn: a 10^9 M_sun dwarf and a
10^15 M_sun cluster with the same (sigma_hat, w_hat) follow identical
dimensionless trajectories, and differ only in the units used to read them out.
This is a structural advantage over surrogates trained directly in physical
units, which must cover the mass range in their training set.

Units used here: lengths kpc, masses M_sun, velocities km/s, densities
M_sun/kpc^3, cross-sections cm^2/g, times Gyr.
"""
import numpy as np

G_KPC = 4.30091e-6              # kpc (km/s)^2 / M_sun
CM2_PER_G_TO_KPC2_PER_MSUN = 2.089087e-10  # (cm^2/g) -> kpc^2/M_sun
KPC_PER_KMS_TO_GYR = 0.977792   # kpc/(km/s) -> Gyr


def nfw_scales(m200, c200, rho_crit=136.05):
    """Scale radius and scale density of an NFW halo.

    m200 in M_sun, c200 the concentration, rho_crit in M_sun/kpc^3
    (default: z = 0, h = 0.674).
    Returns (r_s [kpc], rho_s [M_sun/kpc^3]).
    """
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    r_s = r200 / c200
    mu = np.log(1.0 + c200) - c200 / (1.0 + c200)
    rho_s = m200 / (4.0 * np.pi * r_s ** 3 * mu)
    return r_s, rho_s


class HaloUnits:
    """Unit system for one physical halo.

    >>> u = HaloUnits.from_nfw(m200=1e9, c200=15)
    >>> sigma_hat = u.sigma_hat(5.0)          # sigma/m = 5 cm^2/g
    >>> t_gyr = u.time_to_gyr(410.0)          # dimensionless time -> Gyr
    """

    def __init__(self, r_s, rho_s):
        self.r_s = float(r_s)                       # kpc
        self.rho_s = float(rho_s)                   # M_sun/kpc^3
        self.m_0 = 4.0 * np.pi * self.rho_s * self.r_s ** 3          # M_sun
        self.v_0 = np.sqrt(G_KPC * self.m_0 / self.r_s)              # km/s
        # t_0 = 1/sqrt(4 pi G rho_s), in kpc/(km/s), then to Gyr
        self.t_0 = (1.0 / np.sqrt(4.0 * np.pi * G_KPC * self.rho_s)
                    * KPC_PER_KMS_TO_GYR)                            # Gyr

    @classmethod
    def from_nfw(cls, m200, c200, rho_crit=136.05):
        return cls(*nfw_scales(m200, c200, rho_crit))

    # --- forward conversions (physical -> dimensionless) ---
    def sigma_hat(self, sigma_over_m):
        """sigma/m in cm^2/g -> dimensionless sigma_hat."""
        return (sigma_over_m * CM2_PER_G_TO_KPC2_PER_MSUN
                * self.rho_s * self.r_s)

    def w_hat(self, w_kms):
        """Velocity scale in km/s -> dimensionless w_hat."""
        return w_kms / self.v_0

    # --- inverse conversions (dimensionless -> physical) ---
    def sigma_over_m(self, sigma_hat):
        return sigma_hat / (CM2_PER_G_TO_KPC2_PER_MSUN
                            * self.rho_s * self.r_s)

    def w_kms(self, w_hat):
        return w_hat * self.v_0

    def radius_to_kpc(self, r_hat):
        return np.asarray(r_hat) * self.r_s

    def density_to_msun_kpc3(self, rho_hat):
        return np.asarray(rho_hat) * self.rho_s

    def velocity_to_kms(self, v_hat):
        return np.asarray(v_hat) * self.v_0

    def time_to_gyr(self, t_hat):
        return np.asarray(t_hat) * self.t_0

    def gyr_to_time(self, t_gyr):
        return np.asarray(t_gyr) / self.t_0

    def __repr__(self):
        return (f'HaloUnits(r_s={self.r_s:.4g} kpc, '
                f'rho_s={self.rho_s:.4g} Msun/kpc^3, '
                f'v_0={self.v_0:.4g} km/s, t_0={self.t_0:.4g} Gyr)')


def matched_cross_section(units_a, units_b, sigma_over_m_a):
    """The sigma/m that gives halo B the same dimensionless cross-section that
    sigma_over_m_a gives halo A. Two haloes matched this way follow identical
    dimensionless evolutions."""
    return units_b.sigma_over_m(units_a.sigma_hat(sigma_over_m_a))
