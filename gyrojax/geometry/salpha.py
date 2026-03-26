# FILE: gyrojax/geometry/salpha.py
"""
s-α magnetic geometry for large aspect ratio tokamak (circular flux surfaces).

The s-α model assumes:
  - Large aspect ratio: ε = r/R0 << 1
  - Circular, concentric flux surfaces
  - Simple safety factor profile q(r) = q0 + q1*(r/a)²

Magnetic field: B(r,θ) = B0 / (1 + ε·cosθ)   [leading-order large-aspect-ratio]

Optionally includes Shafranov shift (Beer et al. 1995):
  Δ(r) ≈ -ε²·q²·R0/2   (large-aspect-ratio approximation)
  R(r,θ) = R0 + r·cosθ + Δ(r)   (shifted flux surface centers)

All geometry quantities are precomputed on a 3-D (r, θ, ζ) grid and stored in
a frozen dataclass registered as a JAX pytree.

References:
    Beer, Cowley & Hammett (1995) Phys. Plasmas 2, 2687
    Dimits et al. (2000) Phys. Plasmas 7, 969
"""

from __future__ import annotations
import dataclasses
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SAlphaGeometry:
    """
    Precomputed s-α geometry on a (r, θ, ζ) grid.

    All array shapes documented in field comments.
    """
    r_grid: jnp.ndarray        # (Nr,)
    theta_grid: jnp.ndarray    # (Ntheta,)
    zeta_grid: jnp.ndarray     # (Nzeta,)
    B_field: jnp.ndarray       # (Nr, Ntheta, Nzeta)  |B|
    gradB_r: jnp.ndarray       # (Nr, Ntheta, Nzeta)  ∂B/∂r
    gradB_theta: jnp.ndarray   # (Nr, Ntheta, Nzeta)  ∂B/∂θ
    kappa_r: jnp.ndarray       # (Nr, Ntheta, Nzeta)  normal curvature
    kappa_theta: jnp.ndarray   # (Nr, Ntheta, Nzeta)  geodesic curvature
    g_tt: jnp.ndarray          # (Nr,)  metric g_θθ = r²
    g_zz: jnp.ndarray          # (Nr, Ntheta)  metric g_ζζ = R²
    q_profile: jnp.ndarray     # (Nr,)  safety factor q(r)
    delta_shafranov: jnp.ndarray  # (Nr,)  Shafranov shift Δ(r), zero if disabled
    R0: float
    a: float
    B0: float
    shafranov: bool             # whether Shafranov shift was applied


# Register as JAX pytree
def _flatten(g):
    leaves = [g.r_grid, g.theta_grid, g.zeta_grid,
              g.B_field, g.gradB_r, g.gradB_theta,
              g.kappa_r, g.kappa_theta, g.g_tt, g.g_zz,
              g.q_profile, g.delta_shafranov]
    return leaves, (g.R0, g.a, g.B0, g.shafranov)

def _unflatten(aux, leaves):
    R0, a, B0, shafranov = aux
    return SAlphaGeometry(
        r_grid=leaves[0], theta_grid=leaves[1], zeta_grid=leaves[2],
        B_field=leaves[3], gradB_r=leaves[4], gradB_theta=leaves[5],
        kappa_r=leaves[6], kappa_theta=leaves[7],
        g_tt=leaves[8], g_zz=leaves[9], q_profile=leaves[10],
        delta_shafranov=leaves[11],
        R0=R0, a=a, B0=B0, shafranov=shafranov,
    )

jax.tree_util.register_pytree_node(SAlphaGeometry, _flatten, _unflatten)


def build_salpha_geometry(
    Nr: int,
    Ntheta: int,
    Nzeta: int,
    R0: float,
    a: float,
    B0: float,
    q0: float = 1.4,
    q1: float = 0.5,
    r_inner: float = 0.1,
    shafranov_shift: bool = True,
) -> SAlphaGeometry:
    """
    Build s-α geometry on a uniform (r, θ, ζ) grid.

    Safety factor: q(r) = q0 + q1*(r/a)²
    Field:         B(r,θ) = B0*R0 / R(r,θ)
    Without Shafranov: R(r,θ) = R0*(1 + ε·cosθ)
    With Shafranov:    R(r,θ) = R0 + r·cosθ + Δ(r),  Δ(r) = -ε²·q²·R0/2

    Parameters
    ----------
    Nr, Ntheta, Nzeta : grid sizes
    R0, a : major and minor radii
    B0    : on-axis field
    q0, q1: safety factor coefficients
    r_inner: inner radial boundary (fraction of a)
    shafranov_shift: if True, include Shafranov shift in R(r,θ)
    """
    r_np     = np.linspace(r_inner * a, a, Nr,     dtype=np.float32)
    theta_np = np.linspace(0.0, 2*np.pi, Ntheta,   endpoint=False, dtype=np.float32)
    zeta_np  = np.linspace(0.0, 2*np.pi, Nzeta,    endpoint=False, dtype=np.float32)

    # Safety factor
    q_np = (q0 + q1 * (r_np / a)**2).astype(np.float32)
    dqdr = (2 * q1 * r_np / a**2).astype(np.float32)

    eps_np = r_np / R0  # ε(r)

    # 3-D broadcast grids
    r3  = r_np[:, None, None]
    th3 = theta_np[None, :, None]
    # These will broadcast correctly — no need to expand with zeros

    if shafranov_shift:
        # Shafranov shift: Δ(r) = -ε²·q²·R0/2 (large-aspect-ratio, Beer 1995)
        Delta_np = (-eps_np**2 * q_np**2 * R0 / 2.0).astype(np.float32)  # (Nr,)
        # dΔ/dr = d/dr[-ε²·q²·R0/2] = -R0/2 * d/dr[(r/R0)²*q²]
        #       = -R0/2 * [2r/R0² * q² + (r/R0)² * 2q*dq/dr]
        #       = -[r*q²/R0 + r²*q*dq/dr/R0²]  (but simpler: finite difference or symbolic)
        # dΔ/dr = -R0/2 * 2*(r/R0)*(1/R0)*q² + (-R0/2)*(r/R0)²*2*q*dqdr
        dDelta_dr = (-(r_np / R0) * q_np**2 - (r_np / R0)**2 * q_np * dqdr).astype(np.float32)

        Delta3 = Delta_np[:, None, None]  # broadcast (Nr,1,1)
        dDdr3  = dDelta_dr[:, None, None]

        R3 = (R0 + r3 * np.cos(th3) + Delta3).astype(np.float32)
    else:
        Delta_np = np.zeros(Nr, dtype=np.float32)
        dDdr3    = 0.0
        eps3     = r3 / R0
        R3 = (R0 * (1.0 + eps3 * np.cos(th3))).astype(np.float32)

    B3 = (B0 * R0 / R3).astype(np.float32)

    # ∂B/∂r : B = B0*R0/R, so ∂B/∂r = -B0*R0/R² * ∂R/∂r
    # ∂R/∂r = cosθ + dΔ/dr  (with Shafranov)  or  cosθ/R0*(R0/R0) ... let's redo cleanly
    # ∂R/∂r = cosθ + dΔ/dr
    dRdr = (np.cos(th3) + dDdr3).astype(np.float32)
    gradB_r3 = (-B0 * R0 / R3**2 * dRdr).astype(np.float32)

    # ∂B/∂θ = -B0*R0/R² * ∂R/∂θ
    # ∂R/∂θ = -r*sinθ
    dRdth = (-r3 * np.sin(th3)).astype(np.float32)
    gradB_th3 = (-B0 * R0 / R3**2 * dRdth).astype(np.float32)

    # Curvature (from Beer et al. 1995, modified for Shafranov):
    # κ_r     = -∂R/∂r / R  =  -(cosθ + dΔ/dr)/R  ≈ -cosθ/R  for small shift
    # κ_theta =  sinθ / R   (geodesic, unchanged to leading order)
    kappa_r3  = (-(np.cos(th3) + dDdr3) / R3).astype(np.float32)
    kappa_th3 = ( np.sin(th3) / R3).astype(np.float32)

    # Metric
    g_tt_np = (r_np**2).astype(np.float32)
    g_zz_np = (R3[:, :, 0]**2).astype(np.float32)

    return SAlphaGeometry(
        r_grid            = jnp.array(r_np),
        theta_grid        = jnp.array(theta_np),
        zeta_grid         = jnp.array(zeta_np),
        B_field           = jnp.array(B3),
        gradB_r           = jnp.array(gradB_r3),
        gradB_theta       = jnp.array(gradB_th3),
        kappa_r           = jnp.array(kappa_r3),
        kappa_theta       = jnp.array(kappa_th3),
        g_tt              = jnp.array(g_tt_np),
        g_zz              = jnp.array(g_zz_np),
        q_profile         = jnp.array(q_np),
        delta_shafranov   = jnp.array(Delta_np),
        R0=float(R0), a=float(a), B0=float(B0),
        shafranov=bool(shafranov_shift),
    )


def interp_geometry_to_particles(
    geom: SAlphaGeometry,
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """
    Trilinearly interpolate geometry quantities to particle positions.

    Parameters
    ----------
    geom : SAlphaGeometry
    r, theta, zeta : shape (N,), particle positions

    Returns
    -------
    B, gradB_r, gradB_theta, kappa_r, kappa_theta  each shape (N,)
    """
    from jax.scipy.ndimage import map_coordinates

    Nr, Ntheta, Nzeta = geom.B_field.shape

    dr  = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    idx_r  = (r - geom.r_grid[0]) / dr
    idx_th = theta / (2.0 * jnp.pi) * Ntheta
    idx_ze = zeta  / (2.0 * jnp.pi) * Nzeta

    r_clamped  = jnp.clip(idx_r,  0.0, Nr     - 1.001)
    th_wrapped = idx_th % Ntheta
    ze_wrapped = idx_ze % Nzeta

    coords = jnp.stack([r_clamped, th_wrapped, ze_wrapped], axis=0)  # (3, N)

    def interp(field):
        return map_coordinates(field, coords, order=1, mode='nearest')

    return (
        interp(geom.B_field),
        interp(geom.gradB_r),
        interp(geom.gradB_theta),
        interp(geom.kappa_r),
        interp(geom.kappa_theta),
    )


def interp_geometry_to_particles_full(
    geom: SAlphaGeometry,
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """
    Like interp_geometry_to_particles but also returns q_at_r and g_aa.

    Uses manual bilinear interpolation for r and theta, consolidating the
    q interpolation that was previously scattered across the codebase.

    Parameters
    ----------
    geom  : SAlphaGeometry
    r, theta, zeta : shape (N,) particle positions

    Returns
    -------
    B, gradB_r, gradB_theta, kappa_r, kappa_theta, q_at_r, g_aa
        each shape (N,)
        g_aa = (q/r)² — contravariant metric element g^{αα} at leading order
    """
    Nr, Ntheta, Nzeta = geom.B_field.shape

    # --- radial index (clamped, bilinear) ---
    dr = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    idx_r_f  = (r - geom.r_grid[0]) / dr
    idx_r_f  = jnp.clip(idx_r_f, 0.0, Nr - 1.001)
    ir0 = jnp.floor(idx_r_f).astype(jnp.int32)
    ir1 = jnp.minimum(ir0 + 1, Nr - 1)
    wr1 = idx_r_f - ir0.astype(jnp.float32)
    wr0 = 1.0 - wr1

    # --- theta index (periodic) ---
    idx_th_f = (theta % (2.0 * jnp.pi)) / (2.0 * jnp.pi) * Ntheta
    it0 = jnp.floor(idx_th_f).astype(jnp.int32) % Ntheta
    it1 = (it0 + 1) % Ntheta
    wt1 = idx_th_f - jnp.floor(idx_th_f)
    wt0 = 1.0 - wt1

    # bilinear helper: field shape (Nr, Ntheta, Nzeta) or (Nr, Ntheta, ...)
    def bilinear_3d(field):
        # take zeta=0 slice for axisymmetric fields, then bilinear in (r, theta)
        f00 = field[ir0, it0, 0]
        f10 = field[ir1, it0, 0]
        f01 = field[ir0, it1, 0]
        f11 = field[ir1, it1, 0]
        return wr0*wt0*f00 + wr1*wt0*f10 + wr0*wt1*f01 + wr1*wt1*f11

    B            = bilinear_3d(geom.B_field)
    gradB_r_p    = bilinear_3d(geom.gradB_r)
    gradB_theta_p = bilinear_3d(geom.gradB_theta)
    kappa_r_p    = bilinear_3d(geom.kappa_r)
    kappa_theta_p = bilinear_3d(geom.kappa_theta)

    # q at r: 1-D linear interpolation
    q0_ = geom.q_profile[ir0]
    q1_ = geom.q_profile[ir1]
    q_at_r = wr0 * q0_ + wr1 * q1_

    # g_aa = (q/r)²  (leading-order, no shear term)
    g_aa = (q_at_r / r)**2

    return B, gradB_r_p, gradB_theta_p, kappa_r_p, kappa_theta_p, q_at_r, g_aa
