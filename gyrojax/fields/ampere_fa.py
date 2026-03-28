"""
Ampere's law solver for δA∥ in field-aligned coordinates.

∇²⊥ δA∥ = -β · δj∥

where δj∥ = Σ_p q_p · vpar_p · W_p · shape(x - X_p)

Uses same FFT spectral method as poisson_fa.
β = n₀T / (B²/2μ₀) controls EM coupling (β=0 → electrostatic limit).
"""

from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from gyrojax.geometry.field_aligned import FieldAlignedGeometry


@functools.partial(jax.jit, static_argnums=(2,))
def scatter_jpar_to_grid(state, geom: FieldAlignedGeometry, grid_shape: tuple) -> jnp.ndarray:
    """
    Scatter parallel current δj∥ = Σ vpar·W·shape to grid.

    j_markers = vpar_p * weight_p (q already absorbed in weight normalization)
    Returns j_par_grid of shape grid_shape.
    """
    # We reuse the scatter logic from scatter_gather_fa but with j_markers
    from gyrojax.interpolation.scatter_gather_fa import _trilinear_weights_fa
    Npsi, Ntheta, Nalpha = grid_shape

    i0, i1, j0, j1, k0, k1, wp, wt, wa = _trilinear_weights_fa(
        state.r, state.theta, state.zeta, geom, grid_shape
    )
    # j_p = vpar_p * weight_p
    val = (state.vpar * state.weight).astype(jnp.float32)
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
    n_particles = state.weight.shape[0]
    j_grid = grid.reshape(grid_shape) * (jnp.float32(size) / (jnp.float32(n_particles) + jnp.float32(1e-30)))
    return j_grid


@functools.partial(jax.jit, static_argnums=(2,))
def solve_ampere_fa(
    jpar_grid: jnp.ndarray,
    geom: FieldAlignedGeometry,
    beta: float,
) -> jnp.ndarray:
    """
    Solve ∇²⊥ A∥ = -β · j∥ in field-aligned coordinates.

    In Fourier space:
        -(kpsi² + kα²·g^{αα}) · Â∥ = -β · ĵ∥
        Â∥ = β · ĵ∥ / (kpsi² + kα²·g^{αα} + ε)

    Parameters
    ----------
    jpar_grid : (Npsi, Ntheta, Nalpha)  parallel current density
    geom      : FieldAlignedGeometry
    beta      : plasma beta (0.0 = electrostatic limit → returns zeros)

    Returns
    -------
    A_par : (Npsi, Ntheta, Nalpha)  parallel vector potential
    """
    if beta == 0.0:
        return jnp.zeros_like(jpar_grid, dtype=jnp.float32)

    Npsi, Ntheta, Nalpha = jpar_grid.shape

    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dal = 2.0 * jnp.pi / Nalpha

    kpsi = (jnp.fft.fftfreq(Npsi, d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal) * 2 * jnp.pi).astype(jnp.float32)
    kth  = (jnp.fft.fftfreq(Ntheta, d=2.0 * jnp.pi / Ntheta) * 2 * jnp.pi).astype(jnp.float32)

    KPSI, KTH, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi, kth, kal, indexing='ij')]

    # g^{αα} at mid-radius, theta=0 (flux-tube approximation, same as poisson_fa)
    Npsi_half = Npsi // 2
    Ntheta_half = Ntheta // 2
    g_aa_ref = geom.galphaalpha[Npsi_half, Ntheta_half]  # scalar

    # ∇²⊥ = -(kpsi² + kα²·g^{αα})
    kperp_sq = KPSI**2 * 1.0 + KAL**2 * g_aa_ref   # (Npsi, Ntheta, Nalpha)

    # Avoid division by zero at k=0
    kperp_zero = kperp_sq < 1e-10
    kperp_sq_safe = jnp.where(kperp_zero, 1.0, kperp_sq)

    # ĵ∥ via FFT
    jpar_hat = jnp.fft.fftn(jpar_grid.astype(jnp.complex64))

    # Solve: Â∥ = β · ĵ∥ / kperp²
    A_hat = beta * jpar_hat / kperp_sq_safe

    # Zero out k=0 modes (gauge choice: ∫A∥ dV = 0)
    A_hat = jnp.where(kperp_zero, 0.0 + 0j, A_hat)

    A_par = jnp.fft.ifftn(A_hat).real
    return A_par.astype(jnp.float32)
