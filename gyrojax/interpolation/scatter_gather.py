"""
Gyroaveraged scatter (particle → grid) and gather (grid → particle).

Uses fully-vectorized JAX operations — no Python for-loops.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.geometry.salpha import SAlphaGeometry, interp_geometry_to_particles


# ---------------------------------------------------------------------------
# Trilinear interpolation helpers  (fully vectorized over N particles)
# ---------------------------------------------------------------------------

def _get_trilinear_weights(
    r: jnp.ndarray,        # (N,)
    theta: jnp.ndarray,    # (N,)
    zeta: jnp.ndarray,     # (N,)
    geom: SAlphaGeometry,
    grid_shape: tuple,
):
    """
    Return indices and weights for trilinear interp / scatter, vectorized.

    Returns:
        i0, i1  : (N,)  lower/upper r  indices
        j0, j1  : (N,)  lower/upper θ  indices
        k0, k1  : (N,)  lower/upper ζ  indices
        wr, wt, wz : (N,) fractional positions in each direction
    """
    Nr, Ntheta, Nzeta = grid_shape

    # --- r (clamped, non-periodic) ---
    r0, r1 = geom.r_grid[0], geom.r_grid[-1]
    dr = (r1 - r0) / (Nr - 1)
    ir  = jnp.clip((r - r0) / dr, 0.0, Nr - 1.001)
    i0  = jnp.floor(ir).astype(jnp.int32)
    i1  = jnp.clip(i0 + 1, 0, Nr - 1)
    wr  = ir - i0.astype(jnp.float32)

    # --- θ (periodic) ---
    it  = (theta % (2 * jnp.pi)) / (2 * jnp.pi) * Ntheta
    j0  = jnp.floor(it).astype(jnp.int32) % Ntheta
    j1  = (j0 + 1) % Ntheta
    wt  = it - jnp.floor(it)

    # --- ζ (periodic) ---
    iz  = (zeta % (2 * jnp.pi)) / (2 * jnp.pi) * Nzeta
    k0  = jnp.floor(iz).astype(jnp.int32) % Nzeta
    k1  = (k0 + 1) % Nzeta
    wz  = iz - jnp.floor(iz)

    return i0, i1, j0, j1, k0, k1, wr, wt, wz


def _scatter_one_corner(
    grid: jnp.ndarray,    # (Nr*Ntheta*Nzeta,) flat
    ii: jnp.ndarray,      # (N,) int32
    jj: jnp.ndarray,      # (N,) int32
    kk: jnp.ndarray,      # (N,) int32
    w:  jnp.ndarray,      # (N,) float32  combined weight
    grid_shape: tuple,
) -> jnp.ndarray:
    Nr, Ntheta, Nzeta = grid_shape
    flat = ii * (Ntheta * Nzeta) + jj * Nzeta + kk
    return grid.at[flat].add(w)


def scatter_to_grid(
    state,
    geom: SAlphaGeometry,
    grid_shape: tuple,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    Scatter particle weights to grid (no gyroaveraging for now — point scatter).

    Returns delta_n: (Nr, Ntheta, Nzeta) charge density perturbation.
    """
    Nr, Ntheta, Nzeta = grid_shape
    i0, i1, j0, j1, k0, k1, wr, wt, wz = _get_trilinear_weights(
        state.r, state.theta, state.zeta, geom, grid_shape
    )
    val = state.weight.astype(jnp.float32)
    size = Nr * Ntheta * Nzeta
    grid = jnp.zeros(size, dtype=jnp.float32)

    # 8 corners (vectorized at-scatter, no Python loops in hot path)
    grid = _scatter_one_corner(grid, i0, j0, k0, val*(1-wr)*(1-wt)*(1-wz), grid_shape)
    grid = _scatter_one_corner(grid, i1, j0, k0, val*   wr *(1-wt)*(1-wz), grid_shape)
    grid = _scatter_one_corner(grid, i0, j1, k0, val*(1-wr)*   wt *(1-wz), grid_shape)
    grid = _scatter_one_corner(grid, i0, j0, k1, val*(1-wr)*(1-wt)*   wz , grid_shape)
    grid = _scatter_one_corner(grid, i1, j1, k0, val*   wr *   wt *(1-wz), grid_shape)
    grid = _scatter_one_corner(grid, i1, j0, k1, val*   wr *(1-wt)*   wz , grid_shape)
    grid = _scatter_one_corner(grid, i0, j1, k1, val*(1-wr)*   wt *   wz , grid_shape)
    grid = _scatter_one_corner(grid, i1, j1, k1, val*   wr *   wt *   wz , grid_shape)

    # Normalize by cell volume (convert sum of weights → density perturbation)
    n_particles = state.weight.shape[0]
    cell_vol = (geom.r_grid[-1] - geom.r_grid[0]) * 2*jnp.pi * 2*jnp.pi / size
    delta_n = grid.reshape(grid_shape) / (n_particles * cell_vol + 1e-30)
    return delta_n


def gather_from_grid(
    phi: jnp.ndarray,
    state,
    geom: SAlphaGeometry,
    mi: float,
    e: float,
) -> tuple:
    """
    Gather E-field at particle positions (trilinear interpolation from phi).

    Returns E_r, E_theta, E_zeta each shape (N,).
    """
    from gyrojax.fields.poisson import compute_efield
    E_r_g, E_th_g, E_ze_g = compute_efield(phi, geom)

    grid_shape = phi.shape
    i0, i1, j0, j1, k0, k1, wr, wt, wz = _get_trilinear_weights(
        state.r, state.theta, state.zeta, geom, grid_shape
    )

    def trilinear_gather(field):
        # 8 corners
        f = (  field[i0, j0, k0] * (1-wr)*(1-wt)*(1-wz)
             + field[i1, j0, k0] *    wr *(1-wt)*(1-wz)
             + field[i0, j1, k0] * (1-wr)*   wt *(1-wz)
             + field[i0, j0, k1] * (1-wr)*(1-wt)*   wz
             + field[i1, j1, k0] *    wr *   wt *(1-wz)
             + field[i1, j0, k1] *    wr *(1-wt)*   wz
             + field[i0, j1, k1] * (1-wr)*   wt *   wz
             + field[i1, j1, k1] *    wr *   wt *   wz )
        return f

    return (
        trilinear_gather(E_r_g),
        trilinear_gather(E_th_g),
        trilinear_gather(E_ze_g),
    )
