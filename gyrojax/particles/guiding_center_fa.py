"""
Guiding-center pusher in field-aligned (ψ, θ, α) coordinates.

Equations of motion in field-aligned coords:
  dψ/dt  = vE_ψ + vd_ψ
  dθ/dt  = v∥/(q·R0) + (vE_θ + vd_θ)/r          (θ is parallel direction)
  dα/dt  = (vE_α + vd_α)                           (α is field-line label)
  dv∥/dt = -(μ/m)·∂B/∂s - (q/m)·E∥

where:
  E×B drifts: vE_ψ = -E_θ/B,  vE_α = E_ψ/B  (in s-α metric)
  ∇B drifts:  vd_ψ = -(μ/mΩ)·∂B/∂θ / r
  Curv drifts: vd_ψ = -(v∥²/Ω)·κ_θ / r

Note: this module keeps the same GCState NamedTuple structure as the original
pusher, but treats (r, theta, zeta) as (ψ, θ, α) in field-aligned coords.
The GCState fields are reused with semantic rename:
  state.r     = ψ  (= r in s-α)
  state.theta = θ  (same)
  state.zeta  = α  (field-line label, NOT toroidal angle)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from gyrojax.particles.guiding_center import GCState
from gyrojax.geometry.field_aligned import FieldAlignedGeometry, interp_fa_to_particles


def _gc_rhs_fa(
    psi: jnp.ndarray,      # scalar
    theta: jnp.ndarray,    # scalar
    alpha: jnp.ndarray,    # scalar
    vpar: jnp.ndarray,     # scalar
    mu: jnp.ndarray,       # scalar (constant)
    E_psi: jnp.ndarray,    # scalar — E at this particle
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    geom: FieldAlignedGeometry,
    q_over_m: float,
    mi: float,
):
    """RHS of guiding-center equations in field-aligned coords (single particle)."""
    # Interpolate geometry at (ψ, θ) — α doesn't enter B in s-α
    state_1p = GCState(
        r=jnp.array([psi]), theta=jnp.array([theta]),
        zeta=jnp.array([alpha]), vpar=jnp.array([vpar]),
        mu=jnp.array([mu]), weight=jnp.array([0.0])
    )
    B_arr, gBpsi_arr, gBth_arr, kpsi_arr, kth_arr = interp_fa_to_particles(
        geom, state_1p.r, state_1p.theta, state_1p.zeta
    )
    B = B_arr[0]; gBpsi = gBpsi_arr[0]; gBth = gBth_arr[0]
    kpsi = kpsi_arr[0]; kth = kth_arr[0]

    Omega = q_over_m * B
    safe_r = jnp.maximum(psi, 1e-4)
    safe_O = jnp.maximum(jnp.abs(Omega), 1e-10) * jnp.sign(Omega + 1e-20)

    # Interpolate q at ψ
    Nr = geom.psi_grid.shape[0]
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
    ir = jnp.clip((psi - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
    q_at_psi = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]

    # --- E×B drifts ---
    # In s-α (ψ=r, α=ζ-qθ) the covariant basis gives:
    #   vE_ψ = -E_θ / B   (radial ExB)
    #   vE_α =  E_ψ / B   (field-line drift)
    vE_psi   = -E_theta / B
    vE_alpha =  E_psi   / B

    # --- ∇B drifts: vd = (μ/mΩ)(b̂×∇B) ---
    prefac_grad = mu / (mi * safe_O)
    vd_psi  = -prefac_grad * gBth / safe_r     # radial ∇B drift

    # --- Curvature drifts: vd = (v∥²/Ω)(b̂×κ) ---
    prefac_curv = vpar**2 / safe_O
    vd_psi_curv = -prefac_curv * kth / safe_r

    # --- Equations of motion ---
    dpsi_dt   = vE_psi + vd_psi + vd_psi_curv
    dtheta_dt = vpar / (q_at_psi * geom.R0)    # parallel streaming
    dalpha_dt = vE_alpha                        # ExB in α direction

    # Parallel force: mirror + electric
    E_par      = 0.0    # electrostatic: E∥ ≈ 0 for perpendicular modes
    dvpar_dt   = -(mu / mi) * gBpsi - q_over_m * E_par

    return dpsi_dt, dtheta_dt, dalpha_dt, dvpar_dt


def _rk4_step_fa(
    state: GCState,
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    geom: FieldAlignedGeometry,
    q_over_m: float,
    mi: float,
    dt: float,
) -> GCState:
    """RK4 step for a single particle in field-aligned coords."""
    psi, theta, alpha, vpar, mu, w = (
        state.r, state.theta, state.zeta, state.vpar, state.mu, state.weight
    )

    def rhs(p_, th_, al_, vp_):
        return _gc_rhs_fa(p_, th_, al_, vp_, mu, E_psi, E_theta, E_alpha,
                          geom, q_over_m, mi)

    k1 = rhs(psi,                theta,                alpha,                vpar)
    k2 = rhs(psi+0.5*dt*k1[0],  theta+0.5*dt*k1[1],   alpha+0.5*dt*k1[2],   vpar+0.5*dt*k1[3])
    k3 = rhs(psi+0.5*dt*k2[0],  theta+0.5*dt*k2[1],   alpha+0.5*dt*k2[2],   vpar+0.5*dt*k2[3])
    k4 = rhs(psi+dt*k3[0],       theta+dt*k3[1],        alpha+dt*k3[2],        vpar+dt*k3[3])

    new_psi   = psi   + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    new_theta = theta + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    new_alpha = alpha + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    new_vpar  = vpar  + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    # Wrap θ to [-π, π) and α to [0, 2π)
    new_theta = ((new_theta + jnp.pi) % (2*jnp.pi)) - jnp.pi
    new_alpha = new_alpha % (2*jnp.pi)

    return GCState(r=new_psi, theta=new_theta, zeta=new_alpha,
                   vpar=new_vpar, mu=mu, weight=w)


@partial(jax.jit, static_argnames=('q_over_m', 'mi', 'dt'))
def push_particles_fa(
    state: GCState,
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    geom: FieldAlignedGeometry,
    q_over_m: float,
    mi: float,
    dt: float,
) -> GCState:
    """
    Push all particles one timestep in field-aligned coords using RK4.

    Vectorized via jax.vmap over particles.

    Parameters
    ----------
    state    : GCState with (ψ, θ, α, v∥, μ, w) for N particles
    E_psi, E_theta, E_alpha : E-field at particle positions, shape (N,)
    geom     : FieldAlignedGeometry
    q_over_m : e/m
    mi       : ion mass
    dt       : timestep
    """
    return jax.vmap(
        lambda r, th, ze, vp, mu, w, ep, eth, eal: _rk4_step_fa(
            GCState(r, th, ze, vp, mu, w), ep, eth, eal, geom, q_over_m, mi, dt
        )
    )(state.r, state.theta, state.zeta, state.vpar, state.mu, state.weight,
      E_psi, E_theta, E_alpha)
