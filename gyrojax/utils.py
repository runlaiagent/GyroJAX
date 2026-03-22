"""Physical constants and normalization utilities for GyroJAX."""

import jax
# Default float dtype — float32 for GPU performance
DTYPE = jax.numpy.float32

from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalConstants:
    """SI physical constants."""
    e: float = 1.602176634e-19    # elementary charge [C]
    mi: float = 1.6726219e-27     # proton mass [kg]
    me: float = 9.1093837e-31     # electron mass [kg]
    eps0: float = 8.8541878e-12   # vacuum permittivity [F/m]
    c: float = 2.99792458e8       # speed of light [m/s]


PHYS = PhysicalConstants()


@dataclass(frozen=True)
class NormalizationParams:
    """Normalization parameters for a gyrokinetic simulation.

    All quantities normalized to ion gyroradius ρ_i = v_ti / Ω_i
    and thermal velocity v_ti = sqrt(T_i / m_i).

    Attributes
    ----------
    R0 : float  Major radius [m]
    B0 : float  On-axis field [T]
    Ti0 : float Ion temperature on axis [eV]
    ni0 : float Ion density on axis [m^-3]
    """
    R0: float
    B0: float
    Ti0: float          # [eV]
    ni0: float          # [m^-3]
    mi: float = PHYS.mi
    e: float = PHYS.e

    @property
    def vti(self) -> float:
        """Ion thermal velocity sqrt(Ti/mi) [m/s]."""
        return float(jnp.sqrt(self.Ti0 * self.e / self.mi))

    @property
    def Omega_i(self) -> float:
        """Ion cyclotron frequency [rad/s]."""
        return float(self.e * self.B0 / self.mi)

    @property
    def rho_i(self) -> float:
        """Ion gyroradius vti/Omega_i [m]."""
        return self.vti / self.Omega_i

    @property
    def time_unit(self) -> float:
        """Time unit R0/vti [s]."""
        return self.R0 / self.vti


def gyroradius(mu: float, B: float, mi: float, e: float) -> float:
    """Compute thermal Larmor radius sqrt(2*mu*mi) / (e*B)."""
    return float(jnp.sqrt(2.0 * mu * mi) / (e * B))
