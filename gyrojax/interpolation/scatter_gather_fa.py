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

    # Ensure weights are float32 (jnp.pi is float64 by default, which would
    # cause FutureWarning when scattering into a float32 grid).
    wp = wp.astype(jnp.float32)
    wt = wt.astype(jnp.float32)
    wa = wa.astype(jnp.float32)

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


@functools.partial(jax.jit, static_argnums=(2,))
def scatter_weights_raw_fa(
    state,
    geom: FieldAlignedGeometry,
    grid_shape: tuple,
) -> jnp.ndarray:
    """
    Scatter particle weights to grid WITHOUT normalization.
    Used for weight spreading (GTC technique).
    Returns raw accumulated weight grid of shape grid_shape.
    """
    Npsi, Ntheta, Nalpha = grid_shape
    i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
        state.r, state.theta, state.zeta, geom, grid_shape
    )
    val  = state.weight.astype(jnp.float32)
    size = Npsi * Ntheta * Nalpha
    Nth_Nal = Ntheta * Nalpha

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
    return grid.reshape(grid_shape)


@jax.jit
def gather_scalar_from_grid_fa(
    grid: jnp.ndarray,
    state,
    geom: FieldAlignedGeometry,
) -> jnp.ndarray:
    """
    Gather scalar field from grid to particle positions (trilinear interpolation).
    Used for weight spreading.
    Returns scalar values at each particle position, shape (N,).
    """
    grid_shape = grid.shape
    i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
        state.r, state.theta, state.zeta, geom, grid_shape
    )
    g = grid
    val = (   (1-wp)*(1-wt)*(1-wa) * g[i0,j0,k0]
            +    wp *(1-wt)*(1-wa) * g[i1,j0,k0]
            + (1-wp)*   wt *(1-wa) * g[i0,j1,k0]
            + (1-wp)*(1-wt)*   wa  * g[i0,j0,k1]
            +    wp *   wt *(1-wa) * g[i1,j1,k0]
            +    wp *(1-wt)*   wa  * g[i1,j0,k1]
            + (1-wp)*   wt *   wa  * g[i0,j1,k1]
            +    wp *   wt *   wa  * g[i1,j1,k1])
    return val.astype(jnp.float32)


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


@functools.partial(jax.jit, static_argnums=(1,))
def compute_particle_indices(
    state,
    grid_shape_tuple: tuple,
    geom,
):
    """
    Compute trilinear indices + weights for particle positions.
    Returns (i0,i1,j0,j1,k0,k1,wp,wt,wa) — cache between scatter and gather.
    """
    return _trilinear_weights_fa(state.r, state.theta, state.zeta, geom, grid_shape_tuple)


@functools.partial(jax.jit, static_argnums=(2,))
def scatter_with_indices(
    weight,
    indices: tuple,
    grid_shape: tuple,
):
    """Scatter particle weights to grid using pre-computed trilinear indices."""
    Npsi, Ntheta, Nalpha = grid_shape
    i0, i1, j0, j1, k0, k1, wp, wt, wa = indices
    val  = weight.astype(jnp.float32)
    size = Npsi * Ntheta * Nalpha
    Nth_Nal = Ntheta * Nalpha

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
    n_particles = weight.shape[0]
    delta_n = grid.reshape(grid_shape) * (jnp.float32(size) / (jnp.float32(n_particles) + jnp.float32(1e-30)))
    return delta_n


@jax.jit
def gather_with_indices(
    E_psi_g,
    E_th_g,
    E_al_g,
    indices: tuple,
) -> tuple:
    """Gather pre-computed E-field grids to particle positions using cached trilinear indices."""
    i0, i1, j0, j1, k0, k1, wp, wt, wa = indices

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


# ---------------------------------------------------------------------------
# Blocked scatter — L2 cache efficiency
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2, 3))
def scatter_blocked(
    state,
    geom: FieldAlignedGeometry,
    grid_shape: tuple,
    block_size: int = 4096,
) -> jnp.ndarray:
    """Scatter particles to grid in blocks for L2 cache efficiency.

    Processes particles in blocks of `block_size` and accumulates results.
    Pads the particle array to a multiple of block_size with zero-weight
    particles (which contribute nothing to the grid).

    On RTX 3070 Ti the L2 cache is 4 MB; a 16×32×32 float32 grid is 64 KB,
    fitting ~64 times in L2.  With block_size=4096 each scatter block touches
    only 4096 random grid cells instead of the full N, keeping the grid hot in
    L2 and eliminating cache misses.

    Args:
        state: GCState with N particles
        geom: FieldAlignedGeometry
        grid_shape: (Npsi, Ntheta, Nalpha)
        block_size: particles per block (default 4096)

    Returns:
        delta_n of shape grid_shape (normalised as in scatter_to_grid_fa)
    """
    N = state.r.shape[0]

    # Pad to a multiple of block_size with zero-weight particles.
    pad = (-N) % block_size
    if pad > 0:
        def _pad(x):
            return jnp.concatenate([x, jnp.zeros(pad, dtype=x.dtype)])
        state_p = jax.tree_util.tree_map(_pad, state)
    else:
        state_p = state

    N_padded = N + pad
    n_blocks  = N_padded // block_size

    # Reshape each array: [N_padded] → [n_blocks, block_size]
    state_blocked = jax.tree_util.tree_map(
        lambda x: x.reshape(n_blocks, block_size), state_p
    )

    # Scatter a single block of shape [block_size] → grid (unnormalised)
    def _scatter_raw_block(block_state) -> jnp.ndarray:
        """Return unnormalised scatter grid for one block."""
        Npsi, Ntheta, Nalpha = grid_shape
        i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
            block_state.r, block_state.theta, block_state.zeta, geom, grid_shape
        )
        val  = block_state.weight.astype(jnp.float32)
        size = Npsi * Ntheta * Nalpha
        Nth_Nal = Ntheta * Nalpha
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
        return jnp.zeros(size, dtype=jnp.float32).at[all_flat].add(all_weights).reshape(grid_shape)

    # vmap over the leading n_blocks axis
    grids = jax.vmap(_scatter_raw_block)(state_blocked)   # [n_blocks, Npsi, Ntheta, Nalpha]

    raw_sum = grids.sum(axis=0)   # [Npsi, Ntheta, Nalpha]

    # Apply the same normalisation as scatter_to_grid_fa:
    # delta_n = (N_cells / N_particles) * raw_sum
    size = grid_shape[0] * grid_shape[1] * grid_shape[2]
    delta_n = raw_sum * (jnp.float32(size) / (jnp.float32(N) + jnp.float32(1e-30)))
    return delta_n
