# FILE: gyrojax/geometry/salpha.py
"""
s-α magnetic geometry for large aspect ratio tokamak (circular flux surfaces).

The s-α model assumes:
  - Large aspect ratio: ε = r/R0 << 1
  - Circular, concentric flux surfaces
  - Simple safety factor profile q(r) = q0 + q1*(r/a)²

Magnetic field: B(r,θ) = B0 / (1 + ε·cosθ)   [leading-order large-aspect-ratio]

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
    R0: float
    a: float
    B0: float


# Register as JAX pytree
def _flatten(g):
    leaves = [g.r_grid, g.theta_grid, g.zeta_grid,
              g.B_field, g.gradB_r, g.gradB_theta,
              g.kappa_r, g.kappa_theta, g.g_tt, g.g_zz, g.q_profile]
    return leaves, (g.R0, g.a, g.B0)

def _unflatten(aux, leaves):
    R0, a, B0 = aux
    return SAlphaGeometry(
        r_grid=leaves[0], theta_grid=leaves[1], zeta_grid=leaves[2],
        B_field=leaves[3], gradB_r=leaves[4], gradB_theta=leaves[5],
        kappa_r=leaves[6], kappa_theta=leaves[7],
        g_tt=leaves[8], g_zz=leaves[9], q_profile=leaves[10],
        R0=R0, a=a, B0=B0,
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
) -> SAlphaGeometry:
    """
    Build s-α geometry on a uniform (r, θ, ζ) grid.

    Safety factor: q(r) = q0 + q1*(r/a)²
    Field:         B(r,θ) = B0 / (1 + ε·cosθ),  ε = r/R0

    Parameters
    ----------
    Nr, Ntheta, Nzeta : grid sizes
    R0, a : major and minor radii
    B0    : on-axis field
    q0, q1: safety factor coefficients
    r_inner: inner radial boundary (fraction of a)
    """
    r_np    = np.linspace(r_inner * a, a, Nr,      dtype=np.float32)
    theta_np = np.linspace(0.0, 2*np.pi, Ntheta,   endpoint=False, dtype=np.float32)
    zeta_np  = np.linspace(0.0, 2*np.pi, Nzeta,    endpoint=False, dtype=np.float32)

    # 3-D broadcast grids
    r3  = r_np[:, None, None]      + np.zeros((Nr, Ntheta, Nzeta), dtype=np.float32)
    th3 = theta_np[None, :, None]  + np.zeros((Nr, Ntheta, Nzeta), dtype=np.float32)

    eps3 = r3 / R0                                   # ε(r)
    R3   = R0 * (1.0 + eps3 * np.cos(th3))           # R(r,θ)
    B3   = (B0 / (1.0 + eps3 * np.cos(th3))).astype(np.float32)   # |B|

    # ∂B/∂r  = -B² cosθ / (B0·R0)
    gradB_r3  = (-B3**2 * np.cos(th3) / (B0 * R0)).astype(np.float32)
    # ∂B/∂θ  = B² ε sinθ / B0
    gradB_th3 = ( B3**2 * eps3 * np.sin(th3) / B0).astype(np.float32)

    # Curvature (Beer et al. 1995):
    #   κ_r     = -cosθ / R   (points toward axis → negative at θ=0)
    #   κ_theta =  sinθ / R
    kappa_r3  = (-np.cos(th3) / R3).astype(np.float32)
    kappa_th3 = ( np.sin(th3) / R3).astype(np.float32)

    # Safety factor q(r) = q0 + q1*(r/a)²
    q_np = (q0 + q1 * (r_np / a)**2).astype(np.float32)

    # Metric
    g_tt_np = (r_np**2).astype(np.float32)
    g_zz_np = (R3[:, :, 0]**2).astype(np.float32)

    return SAlphaGeometry(
        r_grid      = jnp.array(r_np),
        theta_grid  = jnp.array(theta_np),
        zeta_grid   = jnp.array(zeta_np),
        B_field     = jnp.array(B3),
        gradB_r     = jnp.array(gradB_r3),
        gradB_theta = jnp.array(gradB_th3),
        kappa_r     = jnp.array(kappa_r3),
        kappa_theta = jnp.array(kappa_th3),
        g_tt        = jnp.array(g_tt_np),
        g_zz        = jnp.array(g_zz_np),
        q_profile   = jnp.array(q_np),
        R0=float(R0), a=float(a), B0=float(B0),
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
