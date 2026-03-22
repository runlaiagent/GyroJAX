"""
Field-aligned coordinate geometry for s-α tokamak.

In standard (r, θ, ζ) coordinates, turbulent structures are highly elongated
along B (k∥ << k⊥). Representing them on a grid aligned with B dramatically
reduces the number of grid points needed in the parallel direction.

Field-aligned coordinates: (ψ, θ, α)
--------------------------------------
  ψ = r               — flux surface label (radial, ≈ poloidal flux)
  θ                   — poloidal angle (parallel coordinate along B)
  α = ζ - q(r)·θ      — field-line label (perpendicular to B in ζ-θ plane)

Properties:
  - B·∇α = 0 exactly  → α is constant along field lines
  - B·∇θ ≠ 0          → θ parameterizes arclength along B
  - Grid point (ψ, θ_i, α_j) sits on the field line labelled α_j

Twist-and-shift boundary condition:
  At θ = ±π, field lines connect back with a shift in α:
      f(ψ, θ + 2π, α) = f(ψ, θ, α + Δα)
      Δα = -2π·ŝ     where ŝ = (r/q)·dq/dr is the magnetic shear

Metric coefficients (large-aspect-ratio s-α):
  g^ψψ  = 1
  g^αα  = (q/r)²·(1 + ŝ²·θ²)   (field-line bending)
  g^ψα  = 0   (orthogonal in ψ-α)
  g^θθ  = 1/r²                   (poloidal metric)

References:
  Beer, Cowley & Hammett (1995) Phys. Plasmas 2, 2687  (s-α geometry)
  Lapillonne et al. (2009) Phys. Plasmas 16, 032308    (field-aligned coords)
  Jolliet et al. (2007) Comput. Phys. Commun. 177, 409 (ORB5 approach)
"""

from __future__ import annotations
import dataclasses
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

from gyrojax.geometry.salpha import SAlphaGeometry


@dataclasses.dataclass(frozen=True)
class FieldAlignedGeometry:
    """
    Field-aligned geometry on a (Npsi, Ntheta, Nalpha) grid.

    Coordinates:
      psi   = r  (radial, flux label)
      theta      (poloidal angle, parallel to B)
      alpha = zeta - q(r)*theta  (field-line label, perp to B)

    All metric and field arrays are shape (Npsi, Ntheta, Nalpha) unless noted.
    """
    # 1-D grids
    psi_grid:   jnp.ndarray   # (Npsi,)   r values
    theta_grid: jnp.ndarray   # (Ntheta,) poloidal angle ∈ [-π, π]
    alpha_grid: jnp.ndarray   # (Nalpha,) field-line label ∈ [0, 2π)

    # Magnetic field magnitude
    B_field:    jnp.ndarray   # (Npsi, Ntheta, Nalpha)

    # Gradient and curvature (needed for drifts and weight equation)
    gradB_psi:  jnp.ndarray   # ∂B/∂ψ   (Npsi, Ntheta, Nalpha)
    gradB_th:   jnp.ndarray   # ∂B/∂θ   (Npsi, Ntheta, Nalpha)
    kappa_psi:  jnp.ndarray   # κ_ψ     (Npsi, Ntheta, Nalpha)
    kappa_th:   jnp.ndarray   # κ_θ     (Npsi, Ntheta, Nalpha)

    # Contravariant metric (for GK Poisson FLR operator)
    gpsipsi:    jnp.ndarray   # g^{ψψ}  (Npsi,)          = 1
    galphaalpha: jnp.ndarray  # g^{αα}  (Npsi, Ntheta)   = (q/r)²(1+ŝ²θ²)
    gpsialpha:  jnp.ndarray   # g^{ψα}  ≈ 0 in s-α

    # Safety factor and shear
    q_profile:  jnp.ndarray   # (Npsi,)
    shat:       jnp.ndarray   # (Npsi,)  ŝ = (r/q)*dq/dr

    # Twist-and-shift BC: shift in α when θ → θ + 2π
    twist_shift: jnp.ndarray  # (Npsi,)  Δα = -2π·ŝ

    # Scalar params
    R0: float
    a:  float
    B0: float


def _flatten_fa(g):
    leaves = [g.psi_grid, g.theta_grid, g.alpha_grid,
              g.B_field, g.gradB_psi, g.gradB_th,
              g.kappa_psi, g.kappa_th,
              g.gpsipsi, g.galphaalpha, g.gpsialpha,
              g.q_profile, g.shat, g.twist_shift]
    return leaves, (g.R0, g.a, g.B0)


def _unflatten_fa(aux, leaves):
    R0, a, B0 = aux
    return FieldAlignedGeometry(
        psi_grid=leaves[0], theta_grid=leaves[1], alpha_grid=leaves[2],
        B_field=leaves[3], gradB_psi=leaves[4], gradB_th=leaves[5],
        kappa_psi=leaves[6], kappa_th=leaves[7],
        gpsipsi=leaves[8], galphaalpha=leaves[9], gpsialpha=leaves[10],
        q_profile=leaves[11], shat=leaves[12], twist_shift=leaves[13],
        R0=R0, a=a, B0=B0,
    )


jax.tree_util.register_pytree_node(FieldAlignedGeometry, _flatten_fa, _unflatten_fa)


def build_field_aligned_geometry(
    Npsi: int,
    Ntheta: int,
    Nalpha: int,
    R0: float,
    a: float,
    B0: float,
    q0: float = 1.4,
    q1: float = 0.5,
    r_inner: float = 0.1,
) -> FieldAlignedGeometry:
    """
    Build field-aligned geometry on a (ψ, θ, α) grid.

    Safety factor:  q(r) = q0 + q1*(r/a)²
    Magnetic shear: ŝ(r) = (r/q) * dq/dr = 2*q1*(r/a)² / q(r)
    Field:          B(r,θ) = B0 / (1 + ε·cosθ)

    Grid:
      ψ ∈ [r_inner*a, a]   (Npsi points, uniform in r)
      θ ∈ [-π, π]          (Ntheta points, uniform, field-parallel direction)
      α ∈ [0, 2π)          (Nalpha points, uniform, field-line label)

    Parameters
    ----------
    Npsi, Ntheta, Nalpha : grid dimensions
    R0, a : major/minor radii [m]
    B0    : on-axis field [T]
    q0, q1: safety factor: q = q0 + q1*(r/a)²
    r_inner: inner boundary as fraction of a
    """
    r_np    = np.linspace(r_inner * a, a, Npsi,    dtype=np.float32)
    th_np   = np.linspace(-np.pi, np.pi, Ntheta,  endpoint=False, dtype=np.float32)
    al_np   = np.linspace(0.0, 2*np.pi, Nalpha,   endpoint=False, dtype=np.float32)

    # Safety factor and shear on r grid
    q_np  = (q0 + q1 * (r_np / a)**2).astype(np.float32)
    dqdr  = (2 * q1 * r_np / a**2).astype(np.float32)
    s_np  = (r_np / q_np * dqdr).astype(np.float32)         # ŝ = (r/q)*dq/dr
    twist = (-2 * np.pi * s_np).astype(np.float32)          # Δα = -2π·ŝ

    # 3-D grids (Npsi, Ntheta, Nalpha)
    r3   = r_np[:, None, None]  + np.zeros((Npsi, Ntheta, Nalpha), dtype=np.float32)
    th3  = th_np[None, :, None] + np.zeros((Npsi, Ntheta, Nalpha), dtype=np.float32)
    # α doesn't enter B in s-α (B is axisymmetric)

    eps3 = r3 / R0
    B3   = (B0 / (1.0 + eps3 * np.cos(th3))).astype(np.float32)

    # ∂B/∂ψ = ∂B/∂r  =  -B² cosθ / (B0·R0)
    gradB_psi3 = (-B3**2 * np.cos(th3) / (B0 * R0)).astype(np.float32)

    # ∂B/∂θ  =  B² ε sinθ / B0
    gradB_th3  = ( B3**2 * eps3 * np.sin(th3) / B0).astype(np.float32)

    # Curvature: κ_ψ = -cosθ/R,  κ_θ = sinθ/R
    R3 = R0 * (1.0 + eps3 * np.cos(th3))
    kappa_psi3 = (-np.cos(th3) / R3).astype(np.float32)
    kappa_th3  = ( np.sin(th3) / R3).astype(np.float32)

    # Contravariant metric
    # g^{ψψ} = 1  (ψ=r, orthonormal radially)
    gpsipsi_1d = np.ones(Npsi, dtype=np.float32)

    # g^{αα}(ψ,θ) = (q/r)²·(1 + ŝ²·θ²)  — field-line bending (Lapillonne 2009)
    q2d = q_np[:, None]   # (Npsi, 1)
    r2d = r_np[:, None]
    s2d = s_np[:, None]
    th2d = th_np[None, :]  # (1, Ntheta)
    galphaalpha_2d = ((q2d / r2d)**2 * (1.0 + s2d**2 * th2d**2)).astype(np.float32)

    gpsialpha_1d = np.zeros(Npsi, dtype=np.float32)  # ≈ 0 in s-α

    return FieldAlignedGeometry(
        psi_grid    = jnp.array(r_np),
        theta_grid  = jnp.array(th_np),
        alpha_grid  = jnp.array(al_np),
        B_field     = jnp.array(B3),
        gradB_psi   = jnp.array(gradB_psi3),
        gradB_th    = jnp.array(gradB_th3),
        kappa_psi   = jnp.array(kappa_psi3),
        kappa_th    = jnp.array(kappa_th3),
        gpsipsi     = jnp.array(gpsipsi_1d),
        galphaalpha = jnp.array(galphaalpha_2d),
        gpsialpha   = jnp.array(gpsialpha_1d),
        q_profile   = jnp.array(q_np),
        shat        = jnp.array(s_np),
        twist_shift = jnp.array(twist),
        R0=float(R0), a=float(a), B0=float(B0),
    )


def fa_to_salpha_coords(
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
    geom: FieldAlignedGeometry,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert field-aligned (ψ, θ, α) to standard toroidal (r, θ, ζ).

    ζ = α + q(r)·θ
    """
    # Interpolate q at psi positions
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (len(geom.psi_grid) - 1)
    ir = jnp.clip((psi - geom.psi_grid[0]) / dr, 0.0, len(geom.psi_grid) - 1.001)
    q_at_psi = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]
    zeta = alpha + q_at_psi * theta
    return psi, theta, zeta % (2 * jnp.pi)


def salpha_to_fa_coords(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    geom: FieldAlignedGeometry,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert standard (r, θ, ζ) to field-aligned (ψ, θ, α).

    α = ζ - q(r)·θ
    """
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (len(geom.psi_grid) - 1)
    ir = jnp.clip((r - geom.psi_grid[0]) / dr, 0.0, len(geom.psi_grid) - 1.001)
    q_at_r = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]
    alpha = zeta - q_at_r * theta
    return r, theta, alpha


def interp_fa_to_particles(
    geom: FieldAlignedGeometry,
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """
    Interpolate geometry to particle positions in field-aligned coords.

    Returns B, gradB_psi, gradB_th, kappa_psi, kappa_th — each (N,).
    Note: α doesn't enter B in s-α (axisymmetric), so we only need (ψ, θ).
    """
    from jax.scipy.ndimage import map_coordinates

    Npsi, Ntheta, Nalpha = geom.B_field.shape

    dr  = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    idx_psi = jnp.clip((psi - geom.psi_grid[0]) / dr, 0.0, Npsi - 1.001)

    # θ ∈ [-π, π), map to [0, Ntheta)
    idx_th = ((theta + jnp.pi) % (2 * jnp.pi)) / (2 * jnp.pi) * Ntheta

    # α doesn't affect B in s-α — use α=0 slice
    idx_al = jnp.zeros_like(psi)

    coords = jnp.stack([idx_psi, idx_th, idx_al], axis=0)  # (3, N)

    def interp(field):
        return map_coordinates(field, coords, order=1, mode='nearest')

    return (
        interp(geom.B_field),
        interp(geom.gradB_psi),
        interp(geom.gradB_th),
        interp(geom.kappa_psi),
        interp(geom.kappa_th),
    )
