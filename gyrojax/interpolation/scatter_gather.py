# FILE: gyrojax/interpolation/scatter_gather.py
"""
Gyroaveraged scatter and gather operations for gyrokinetic PIC.

Gyroaveraging replaces the full particle position with an average over the
Larmor ring. For a 4-point gyroaverage:
    <φ>(R) = (1/4) Σ_{k=0}^{3} φ(R + ρ_L * [cos(kπ/2), sin(kπ/2)] · e_perp)

where ρ_L = sqrt(2μm) / (eB) is the local Larmor radius.

In s-α geometry, the perpendicular plane is approximately the (r,θ) plane,
so the 4 ring points shift in (r, θ) directions.

GPU-safe scatter uses jax.lax.scatter with SUM mode (atomic add equivalent).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.particles.guiding_center import GCState
from gyrojax.geometry.salpha import SAlphaGeometry, interp_geometry_to_particles


def _larmor_ring_points(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    rho_L: jnp.ndarray,
    n_points: int = 4,
) -> tuple:
    """
    Compute n_points positions on the Larmor ring around guiding center (r, θ, ζ).

    The ring lies in the (r, θ) plane (perpendicular to B in s-α).

    Returns
    -------
    ring_r, ring_theta, ring_zeta : each shape (n_points, N_particles)
    """
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_points, endpoint=False)  # (n_points,)
    # Shape: (n_points, N)
    dr    = rho_L[None, :] * jnp.cos(angles[:, None])
    dtheta_arc = rho_L[None, :] * jnp.sin(angles[:, None]) / jnp.maximum(r[None, :], 1e-6)

    ring_r     = r[None, :]     + dr
    ring_theta = theta[None, :] + dtheta_arc
    ring_zeta  = jnp.broadcast_to(zeta[None, :], (n_points, len(zeta)))

    return ring_r, ring_theta % (2*jnp.pi), ring_zeta % (2*jnp.pi)


def _trilinear_index_weights(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    geom: SAlphaGeometry,
    Nr: int, Ntheta: int, Nzeta: int,
) -> tuple:
    """
    Compute trilinear interpolation indices and weights for positions (r, θ, ζ).

    Returns
    -------
    i0, j0, k0 : integer base indices, each shape (N,)
    wr, wt, wz : fractional weights in [0,1], each shape (N,)
    """
    # Radial
    dr   = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    ir   = (r - geom.r_grid[0]) / dr
    i0   = jnp.clip(ir.astype(jnp.int32), 0, Nr - 2)
    wr   = jnp.clip(ir - i0.astype(jnp.float32), 0.0, 1.0)

    # Poloidal (periodic)
    dth  = 2.0 * jnp.pi / Ntheta
    jth  = theta / dth
    j0   = (jth.astype(jnp.int32)) % Ntheta
    wt   = jnp.clip(jth - jnp.floor(jth), 0.0, 1.0)

    # Toroidal (periodic)
    dze  = 2.0 * jnp.pi / Nzeta
    kze  = zeta / dze
    k0   = (kze.astype(jnp.int32)) % Nzeta
    wz   = jnp.clip(kze - jnp.floor(kze), 0.0, 1.0)

    return i0, j0, k0, wr, wt, wz


def scatter_to_grid(
    state: GCState,
    geom: SAlphaGeometry,
    grid_shape: tuple,
    mi: float,
    e: float,
    n_gyro: int = 4,
) -> jnp.ndarray:
    """
    Scatter gyroaveraged delta-f density δn onto the (r, θ, ζ) grid.

    For each particle, deposit weight to n_gyro ring points with equal
    1/n_gyro weighting (gyroaveraging). Uses jax.lax.scatter for GPU safety.

    Parameters
    ----------
    state      : GCState
    geom       : SAlphaGeometry
    grid_shape : (Nr, Ntheta, Nzeta)
    mi, e      : ion mass and charge
    n_gyro     : number of gyroaverage points (default 4)

    Returns
    -------
    delta_n : (Nr, Ntheta, Nzeta) gyroaveraged perturbed density
    """
    Nr, Ntheta, Nzeta = grid_shape
    N = state.r.shape[0]

    # Compute Larmor radius at guiding center B
    B_gc, _, _, _, _ = interp_geometry_to_particles(geom, state.r, state.theta, state.zeta)
    rho_L = jnp.sqrt(2.0 * state.mu * mi) / (e * B_gc)

    # Ring points: (n_gyro, N)
    ring_r, ring_th, ring_ze = _larmor_ring_points(
        state.r, state.theta, state.zeta, rho_L, n_gyro
    )

    delta_n = jnp.zeros(grid_shape)

    # Process each ring point
    for k in range(n_gyro):
        rk = ring_r[k]
        thk = ring_th[k]
        zek = ring_ze[k]

        i0, j0, k0, wr, wt, wz = _trilinear_index_weights(
            rk, thk, zek, geom, Nr, Ntheta, Nzeta
        )
        i1 = (i0 + 1) % Nr
        j1 = (j0 + 1) % Ntheta
        k1 = (k0 + 1) % Nzeta

        val = state.weight / n_gyro   # gyroaveraged weight

        # Trilinear 8-corner deposit
        for (ii, jj, kk, wx, wy_w, wz_w) in [
            (i0, j0, k0, (1-wr)*(1-wt)*(1-wz)),
            (i1, j0, k0,    wr *(1-wt)*(1-wz)),
            (i0, j1, k0, (1-wr)*   wt *(1-wz)),
            (i0, j0, k1, (1-wr)*(1-wt)*   wz ),
            (i1, j1, k0,    wr *   wt *(1-wz)),
            (i1, j0, k1,    wr *(1-wt)*   wz ),
            (i0, j1, k1, (1-wr)*   wt *   wz ),
            (i1, j1, k1,    wr *   wt *   wz ),
        ]:
            flat_idx = ii * (Ntheta * Nzeta) + jj * Nzeta + kk
            flat_n   = delta_n.reshape(-1)
            flat_n   = flat_n.at[flat_idx].add(val * wx)
            delta_n  = flat_n.reshape(grid_shape)

    return delta_n


def gather_from_grid(
    phi: jnp.ndarray,
    state: GCState,
    geom: SAlphaGeometry,
    mi: float,
    e: float,
    n_gyro: int = 4,
) -> tuple:
    """
    Gather gyroaveraged electric field at particle guiding center positions.

    Computes E = -∇φ by finite differences on the grid, then interpolates
    to guiding center positions with gyroaveraging.

    Parameters
    ----------
    phi   : (Nr, Ntheta, Nzeta) electrostatic potential
    state : GCState
    geom  : SAlphaGeometry
    mi, e : ion mass and charge
    n_gyro: gyroaverage points

    Returns
    -------
    E_r, E_theta, E_zeta : electric field components at particles, shape (N,)
    """
    Nr, Ntheta, Nzeta = phi.shape

    # Compute E = -∇φ via finite differences on the grid
    dr  = (geom.r_grid[-1]  - geom.r_grid[0])  / (Nr - 1)
    dth = 2.0 * jnp.pi / Ntheta
    dze = 2.0 * jnp.pi / Nzeta

    E_r_grid     = -(jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2*dr)
    E_theta_grid = -(jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2*dth)
    E_zeta_grid  = -(jnp.roll(phi, -1, axis=2) - jnp.roll(phi, 1, axis=2)) / (2*dze)

    # Larmor radius
    B_gc, _, _, _, _ = interp_geometry_to_particles(geom, state.r, state.theta, state.zeta)
    rho_L = jnp.sqrt(2.0 * state.mu * mi) / (e * B_gc)

    ring_r, ring_th, ring_ze = _larmor_ring_points(
        state.r, state.theta, state.zeta, rho_L, n_gyro
    )

    E_r_p     = jnp.zeros(state.r.shape[0])
    E_theta_p = jnp.zeros(state.r.shape[0])
    E_zeta_p  = jnp.zeros(state.r.shape[0])

    def interp_field(field, rk, thk, zek):
        i0, j0, k0, wr, wt, wz = _trilinear_index_weights(
            rk, thk, zek, geom, Nr, Ntheta, Nzeta
        )
        i1 = (i0 + 1) % Nr
        j1 = (j0 + 1) % Ntheta
        k1 = (k0 + 1) % Nzeta

        return (
            field[i0, j0, k0] * (1-wr)*(1-wt)*(1-wz) +
            field[i1, j0, k0] *    wr *(1-wt)*(1-wz) +
            field[i0, j1, k0] * (1-wr)*   wt *(1-wz) +
            field[i0, j0, k1] * (1-wr)*(1-wt)*   wz  +
            field[i1, j1, k0] *    wr *   wt *(1-wz) +
            field[i1, j0, k1] *    wr *(1-wt)*   wz  +
            field[i0, j1, k1] * (1-wr)*   wt *   wz  +
            field[i1, j1, k1] *    wr *   wt *   wz
        )

    for k in range(n_gyro):
        rk = ring_r[k]; thk = ring_th[k]; zek = ring_ze[k]
        E_r_p     = E_r_p     + interp_field(E_r_grid,     rk, thk, zek) / n_gyro
        E_theta_p = E_theta_p + interp_field(E_theta_grid, rk, thk, zek) / n_gyro
        E_zeta_p  = E_zeta_p  + interp_field(E_zeta_grid,  rk, thk, zek) / n_gyro

    return E_r_p, E_theta_p, E_zeta_p
