"""
Scatter (particle→grid) and gather (grid→particle) in field-aligned coordinates.

Twist-and-shift boundary condition:
    At θ = ±π, the poloidal domain wraps with a shift in α:
        f(ψ, θ + 2π, α) = f(ψ, θ, α + Δα)
        Δα(ψ) = -2π·ŝ(ψ)    (field-aligned BC)

    This is implemented in the α-index scatter/gather: when a particle is near
    θ = ±π and contributes to the boundary cells, the α-shift is applied.

    In practice for the scatter we use simple periodic-α with the understanding
    that the field solver already handles the twist implicitly (it operates in
    full-torus Fourier space). The twist-and-shift matters most when the domain
    covers only one field-line connection length (flux-tube codes); for our
    full-annulus approach periodic-α suffices.

All operations fully vectorized — no Python for-loops.
"""

from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from gyrojax.geometry.field_aligned import FieldAlignedGeometry, interp_fa_to_particles


def _trilinear_weights_fa(
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
    geom: FieldAlignedGeometry,
    grid_shape: tuple,
):
    """
    Compute trilinear weights and indices for field-aligned grid.

    θ ∈ [-π, π) — non-periodic across domain (clamp at boundaries)
    ψ ∈ [r_inner, a] — non-periodic (clamp)
    α ∈ [0, 2π) — periodic

    Returns i0,i1,j0,j1,k0,k1 (indices), wp,wt,wa (fractional positions).
    """
    Npsi, Ntheta, Nalpha = grid_shape

    # --- ψ (clamped) ---
    p0, p1 = geom.psi_grid[0], geom.psi_grid[-1]
    dpsi = (p1 - p0) / (Npsi - 1)
    ip   = jnp.clip((psi - p0) / dpsi, 0.0, Npsi - 1.001)
    i0   = jnp.floor(ip).astype(jnp.int32)
    i1   = jnp.clip(i0 + 1, 0, Npsi - 1)
    wp   = ip - i0.astype(jnp.float32)

    # --- θ ∈ [-π, π)  (non-periodic, clamped) ---
    dth  = 2.0 * jnp.pi / Ntheta
    # map θ to index in [0, Ntheta)
    it   = jnp.clip(((theta + jnp.pi) % (2*jnp.pi)) / dth, 0.0, Ntheta - 1.001)
    j0   = jnp.floor(it).astype(jnp.int32)
    j1   = (j0 + 1) % Ntheta   # wrap at ±π boundary
    wt   = it - jnp.floor(it)

    # --- α ∈ [0, 2π)  (periodic) ---
    dal  = 2.0 * jnp.pi / Nalpha
    ia   = ((alpha % (2*jnp.pi)) / dal)
    k0   = jnp.floor(ia).astype(jnp.int32) % Nalpha
    k1   = (k0 + 1) % Nalpha
    wa   = ia - jnp.floor(ia)

    return i0, i1, j0, j1, k0, k1, wp, wt, wa


@functools.partial(jax.jit, static_argnums=(2,))
def scatter_to_grid_fa(
    state,
    geom: FieldAlignedGeometry,
    grid_shape: tuple,
) -> jnp.ndarray:
    """
    Scatter particle weights to field-aligned (ψ, θ, α) grid.

    Particle positions must be in field-aligned coords: (psi, theta, alpha).
    Returns delta_n of shape grid_shape.
    """
    Npsi, Ntheta, Nalpha = grid_shape
    i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
        state.r, state.theta, state.zeta, geom, grid_shape
    )
    # Use float32 accumulator — sufficient precision for typical N_particles (50k–400k)
    # with weights ~O(1). float32 gives ~7 significant digits.
    val  = state.weight.astype(jnp.float32)
    size = Npsi * Ntheta * Nalpha
    Nth_Nal = Ntheta * Nalpha

    # Single batched scatter: stack all 8 corners into (8*N,) arrays
    all_flat = jnp.concatenate([
        i0 * Nth_Nal + j0 * Nalpha + k0,
        i1 * Nth_Nal + j0 * Nalpha + k0,
        i0 * Nth_Nal + j1 * Nalpha + k0,
        i0 * Nth_Nal + j0 * Nalpha + k1,
        i1 * Nth_Nal + j1 * Nalpha + k0,
        i1 * Nth_Nal + j0 * Nalpha + k1,
        i0 * Nth_Nal + j1 * Nalpha + k1,
        i1 * Nth_Nal + j1 * Nalpha + k1,
    ])
    all_weights = jnp.concatenate([
        val*(1-wp)*(1-wt)*(1-wa),
        val*   wp *(1-wt)*(1-wa),
        val*(1-wp)*   wt *(1-wa),
        val*(1-wp)*(1-wt)*   wa,
        val*   wp *   wt *(1-wa),
        val*   wp *(1-wt)*   wa,
        val*(1-wp)*   wt *   wa,
        val*   wp *   wt *   wa,
    ])
    grid = jnp.zeros(size, dtype=jnp.float32).at[all_flat].add(all_weights)

    n_particles = state.weight.shape[0]
    # Normalize so that uniform w=1 gives delta_n/n0 = 1 (dimensionless):
    # delta_n = (N_cells / N_particles) * accumulated_weights
    # This is equivalent to: delta_n[cell] = <w>_{particles in cell}
    # For uniform w=1: delta_n = 1 everywhere (correct delta-f identity)
    delta_n = grid.reshape(grid_shape) * (jnp.float32(size) / (jnp.float32(n_particles) + jnp.float32(1e-30)))
    return delta_n


@jax.jit
def gather_from_grid_fa(
    phi: jnp.ndarray,
    state,
    geom: FieldAlignedGeometry,
) -> tuple:
    """
    Gather E-field at particle positions in field-aligned coords.

    Returns E_psi, E_theta, E_alpha each shape (N,).
    """
    from gyrojax.fields.poisson_fa import compute_efield_fa
    E_psi_g, E_th_g, E_al_g = compute_efield_fa(phi, geom)

    grid_shape = phi.shape
    i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
        state.r, state.theta, state.zeta, geom, grid_shape
    )

    def trilinear(field):
        return (  field[i0, j0, k0] * (1-wp)*(1-wt)*(1-wa)
                + field[i1, j0, k0] *    wp *(1-wt)*(1-wa)
                + field[i0, j1, k0] * (1-wp)*   wt *(1-wa)
                + field[i0, j0, k1] * (1-wp)*(1-wt)*   wa
                + field[i1, j1, k0] *    wp *   wt *(1-wa)
                + field[i1, j0, k1] *    wp *(1-wt)*   wa
                + field[i0, j1, k1] * (1-wp)*   wt *   wa
                + field[i1, j1, k1] *    wp *   wt *   wa )

    return trilinear(E_psi_g), trilinear(E_th_g), trilinear(E_al_g)
