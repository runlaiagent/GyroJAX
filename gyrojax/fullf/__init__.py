"""
Full-f gyrokinetic PIC for GyroJAX — Phase 3.

In the δf method, each marker has weight w = δf/f0.
In full-f, each marker carries weight W = f/f0_ref — it represents
the full distribution function, not just the perturbation.

Key differences from δf:
  1. No weight equation: weights evolve only via the Poisson bracket
     (they are characteristics of the full Vlasov equation)
  2. The density perturbation for Poisson is the FULL δn, not just
     the perturbation part:
       δn_i(x) = ∫ f dv - n0  ≈  Σ_p W_p · δ(x - X_p) - n0
  3. Nonlinear terms are naturally included (no linearization)
  4. Phase-space weight variance grows: need periodic resampling
     (importance sampling / particle splitting) to control noise

Full-f weight evolution:
  The marker weights are CONSTANTS along phase-space trajectories
  (characteristics of Vlasov). They do NOT change unless resampled.
  
  Instead: the particle positions evolve under the full fields (E×B + drifts),
  and the changing particle density gives the self-consistent δn.

  dW_p/dt = 0    (constant marker weight in full-f)
  
  But we still use the GC equations of motion:
    dX/dt = vE + vd + v∥ b̂
    dv∥/dt = -(μ/m)·∂B/∂s - (q/m)·E∥

  The δn for Poisson:
    δn(x) = [Σ_p W_p · δ(x - X_p)] / n0(x)  - 1

Marker initialization (full-f Maxwellian):
  Each marker p gets weight W_p = f0(X_p, v_p) · Δ(phase space volume)
  
  In practice: initialize markers uniformly in phase space, then
  weight W_p = f0(X_p, v_p) / (marker density), so that the
  weighted sum recovers n0.

References:
  Grandgirard et al. (2006) J. Comput. Phys. 217, 395  (GYSELA full-f)
  Idomura et al. (2008) Nucl. Fusion 48, 035002  (GT5D full-f)
  Lin & Lee (1995) Phys. Rev. E 52, 5646  (particle weight schemes)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.particles.guiding_center import GCState


def init_fullf_particles(
    N: int,
    geom,
    vti: float,
    Ti: float,
    mi: float,
    n0_avg: float,
    key: jax.random.PRNGKey,
) -> GCState:
    """
    Initialize full-f markers.

    Markers are sampled from the background Maxwellian f0.
    Each marker weight W_p = 1 (all markers represent equal phase-space volume).
    The density weighting is handled by f0 being the sampling distribution.

    Since we sample from f0, the weighted sum naturally gives n0:
      n(x) = Σ_p δ(x - X_p)  →  mean = n0  (by construction)

    For δn computation:
      δn_i(x) = [Σ_p δ(x - X_p)] / n_cells - n0

    Parameters
    ----------
    N      : number of markers
    geom   : geometry (FieldAlignedGeometry or SAlphaGeometry)
    vti    : thermal velocity
    Ti     : ion temperature
    mi     : ion mass
    n0_avg : average background density (for normalization)
    key    : JAX random key
    """
    from gyrojax.particles.guiding_center import init_maxwellian_particles
    # Sample positions from Maxwellian — same as δf init
    state = init_maxwellian_particles(N, geom, vti, Ti, mi, key)
    # Full-f: all markers start with equal unit weight
    # (since they're sampled from f0, equal weights = f0)
    W = jnp.ones(N, dtype=jnp.float32) * n0_avg
    return state._replace(weight=W)


def scatter_fullf_to_grid(
    state: GCState,
    geom,
    grid_shape: tuple,
    n0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute δn for full-f: δn = n_particles - n0.

    n_particles(x) = Σ_p W_p · trilinear(x - X_p) / cell_volume
    δn(x) = n_particles(x) - n0(x)

    Parameters
    ----------
    state      : GCState with marker positions and weights
    geom       : geometry
    grid_shape : (Npsi, Ntheta, Nalpha)
    n0_grid    : background density on grid, shape grid_shape

    Returns
    -------
    delta_n : shape grid_shape
    """
    from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa
    # scatter_to_grid_fa already computes density (weighted sum / vol)
    n_particles = scatter_to_grid_fa(state, geom, grid_shape)
    # Renormalize: scatter_to_grid_fa normalizes by N_particles * cell_vol
    # We need actual density → multiply back by n0_avg (total weight / volume)
    n_actual = n_particles * float(jnp.mean(state.weight))
    delta_n = n_actual - n0_grid
    return delta_n


def compute_n0_grid(
    geom,
    grid_shape: tuple,
    n0_avg: float,
    R0_over_Ln: float,
    R0: float,
) -> jnp.ndarray:
    """
    Compute background density n0 on the grid.

    n0(ψ) = n0_avg · exp(-(ψ - ψ_mid) / Ln)
    """
    Npsi, Ntheta, Nalpha = grid_shape
    psi = geom.psi_grid               # (Npsi,)
    psi_mid = (psi[0] + psi[-1]) * 0.5
    Ln = R0 / R0_over_Ln
    n0_1d = n0_avg * jnp.exp(-(psi - psi_mid) / Ln)  # (Npsi,)
    return n0_1d[:, None, None] * jnp.ones((Npsi, Ntheta, Nalpha))
