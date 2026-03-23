"""
Non-dimensional normalization layer for GyroJAX — GENE/GX hat-unit conventions.

Reference: Lref = a (minor radius), following GENE/GX standard.

Normalization table (physical → hat):
    x̂ = x / a_ref
    v̂ = v / vt_ref          where vt_ref = sqrt(T_ref / m_ref)
    t̂ = t * vt_ref / a_ref
    B̂ = B / B_ref
    φ̂ = e_ref * φ / T_ref
    n̂ = n / n_ref
    T̂ = T / T_ref

Derived:
    vt_ref  = sqrt(T_ref / m_ref)
    Omega_ref = e_ref * B_ref / m_ref
    rho_ref = vt_ref / Omega_ref
    rho_star = rho_ref / a_ref    (KEY dimensionless parameter)
"""

from __future__ import annotations
import math
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class NormParams:
    """Reference normalization parameters (GENE/GX hat-unit convention).

    All quantities are in the code's internal unit system (may be normalized
    units, not necessarily SI).  CBC standard uses e_ref=1000 (Omega_i unit)
    so that rho_star = 1/180.
    """
    a_ref: float   # minor radius (length unit)
    B_ref: float   # reference magnetic field
    T_ref: float   # reference ion temperature
    n_ref: float   # reference density
    m_ref: float   # reference ion mass
    e_ref: float   # charge-to-mass factor (code units: e_ref = q*B/m normalized)
    Lref: str = 'a'  # reference length; 'a' or 'R0'
    R0_ref: float = 1.0  # major radius (used when Lref='R0')

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def vt_ref(self) -> float:
        """Thermal velocity = sqrt(T_ref / m_ref)."""
        return math.sqrt(self.T_ref / self.m_ref)

    @property
    def Omega_ref(self) -> float:
        """Cyclotron frequency = e_ref * B_ref / m_ref."""
        return self.e_ref * self.B_ref / self.m_ref

    @property
    def rho_ref(self) -> float:
        """Thermal Larmor radius = vt_ref / Omega_ref."""
        return self.vt_ref / self.Omega_ref

    @property
    def rho_star(self) -> float:
        """rho_ref / Lref — the key dimensionless gyroradius parameter."""
        if self.Lref == 'R0':
            return self.rho_ref / self.R0_ref
        return self.rho_ref / self.a_ref

    # ------------------------------------------------------------------ #
    # Named constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cbc(cls) -> 'NormParams':
        """Cyclone Base Case reference parameters (CBC).

        In CBC code units: vti=1, mi=1, B0=1, a=0.18.
        Omega_i = 1/rho_star * vti*mi/(a*B0) = 180/0.18 = 1000
        => e_ref = Omega_i * m_ref / B_ref = 1000.
        rho_star = sqrt(T/m) / (e/m * B * a) = 1/(1000 * 1 * 0.18) = 1/180 ✓
        """
        return cls(
            a_ref=0.18,
            B_ref=1.0,
            T_ref=1.0,
            n_ref=1.0,
            m_ref=1.0,
            e_ref=1000.0,
            Lref='a',
        )

    @classmethod
    def from_vmec(
        cls,
        geom: Any,
        Ti: float = 1.0,
        n0: float = 1.0,
        mi: float = 1.0,
        e_ref: float = None,
        rho_star: float = 1.0 / 180.0,
    ) -> 'NormParams':
        """Construct NormParams from a VMEC geometry object.

        If e_ref is None, it is derived from the desired rho_star:
            rho_star = vti / (e_ref/mi * B0 * a)
            => e_ref = vti * mi / (rho_star * B0 * a)
        """
        a = float(geom.a)
        B0 = float(geom.B0)
        vti = math.sqrt(Ti / mi)
        if e_ref is None:
            e_ref = vti * mi / (rho_star * B0 * a)
        return cls(
            a_ref=a,
            B_ref=B0,
            T_ref=Ti,
            n_ref=n0,
            m_ref=mi,
            e_ref=e_ref,
            Lref='a',
        )


# ------------------------------------------------------------------ #
# Geometry normalization
# ------------------------------------------------------------------ #

def normalize_geometry(geom: Any, norm: NormParams) -> Any:
    """Return a copy of *geom* with all fields in hat units.

    Scaling rules:
        psi_grid → psi_grid / a_ref          (r/a, dimensionless)
        B_field  → B_field / B_ref
        gradB_*  → gradB_* * a_ref / B_ref   ([B/m] → dimensionless)
        kappa_*  → kappa_* * a_ref           ([1/m] → dimensionless)
        galphaalpha → galphaalpha * a_ref²   ([1/m²] → dimensionless)
    """
    import dataclasses as _dc
    a = norm.a_ref
    B = norm.B_ref
    fields = _dc.fields(geom)
    updates = {}
    for f in fields:
        name = f.name
        val = getattr(geom, name)
        if name == 'psi_grid':
            updates[name] = val / a
        elif name == 'B_field':
            updates[name] = val / B
        elif name in ('gradB_psi', 'gradB_th'):
            updates[name] = val * a / B
        elif name in ('kappa_psi', 'kappa_th'):
            updates[name] = val * a
        elif name == 'galphaalpha':
            updates[name] = val * a ** 2
        # gpsipsi, gpsialpha, q_profile, shat, theta_grid, alpha_grid: dimensionless already
    return _dc.replace(geom, **updates)


# ------------------------------------------------------------------ #
# De-normalization helpers
# ------------------------------------------------------------------ #

def denormalize_phi(phi_hat, norm: NormParams):
    """Convert hat potential φ̂ to physical φ.

    φ = φ̂ * T_ref / e_ref
    """
    return phi_hat * norm.T_ref / norm.e_ref


def denormalize_growth_rate(gamma_hat: float, norm: NormParams) -> float:
    """Convert hat growth rate γ̂ to physical γ in [s⁻¹] (or code 1/t units).

    γ̂ has units of [vt_ref / a_ref], so:
        γ_phys = γ̂ * vt_ref / a_ref
    """
    return gamma_hat * norm.vt_ref / norm.a_ref


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

def norm_summary(norm: NormParams) -> dict:
    """Return a dict summarising the key normalization quantities."""
    return {
        'a_ref': norm.a_ref,
        'B_ref': norm.B_ref,
        'T_ref': norm.T_ref,
        'n_ref': norm.n_ref,
        'm_ref': norm.m_ref,
        'e_ref': norm.e_ref,
        'Lref': norm.Lref,
        'vt_ref': norm.vt_ref,
        'Omega_ref': norm.Omega_ref,
        'rho_ref': norm.rho_ref,
        'rho_star': norm.rho_star,
        '1/rho_star': 1.0 / norm.rho_star,
    }
