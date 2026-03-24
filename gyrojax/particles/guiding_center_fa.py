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


def _gc_rhs_fa_batched(
    psi: jnp.ndarray,       # (N,)
    theta: jnp.ndarray,     # (N,)
    alpha: jnp.ndarray,     # (N,)
    vpar: jnp.ndarray,      # (N,)
    mu: jnp.ndarray,        # (N,)
    E_psi: jnp.ndarray,     # (N,)
    E_theta: jnp.ndarray,   # (N,)
    E_alpha: jnp.ndarray,   # (N,)
    B: jnp.ndarray,         # (N,) — pre-interpolated geometry
    gBpsi: jnp.ndarray,     # (N,)
    gBth: jnp.ndarray,      # (N,)
    kpsi: jnp.ndarray,      # (N,)
    kth: jnp.ndarray,       # (N,)
    q_at_psi: jnp.ndarray,  # (N,)
    q_over_m: float,
    mi: float,
    R0: float,
):
    """Batched (N,) RHS of guiding-center equations in field-aligned coords."""
    Omega = q_over_m * B
    safe_r = jnp.maximum(psi, 1e-4)
    safe_O = jnp.sign(Omega + 1e-20) * jnp.maximum(jnp.abs(Omega), 1e-10)

    # --- E×B drifts ---
    vE_psi   = -E_theta / B
    vE_alpha =  E_psi   / B

    # --- Radial ∇B drift: uses RADIAL gradient (gBpsi = ∂B/∂r = -B²cos(θ)/(B₀R₀)) ---
    # vd_r = -(μ/mΩ) * gradB_psi  →  positive (outward) at θ=0 (bad curvature)
    prefac_grad = mu / (mi * safe_O)
    vd_psi  = -prefac_grad * gBpsi

    # --- Radial curvature drift: uses RADIAL curvature (kpsi = -cos(θ)/R) ---
    # vd_r = -(v∥²/Ω) * kappa_psi  →  positive (outward) at θ=0
    prefac_curv = vpar**2 / safe_O
    vd_psi_curv = -prefac_curv * kpsi

    # --- Equations of motion ---
    dpsi_dt   = vE_psi + vd_psi + vd_psi_curv
    dtheta_dt = vpar / (q_at_psi * R0)    # parallel streaming
    dalpha_dt = vE_alpha                   # ExB in α direction

    # Parallel force: mirror force = -(μ/m) * ∂B/∂s
    # ∂B/∂s = ∂B/∂θ / (q·R₀)  (arc length along field line)
    # gBth = ∂B/∂θ / (q·R₀)  already normalized (see geometry builder)
    dvpar_dt   = -(mu / mi) * gBth

    return dpsi_dt, dtheta_dt, dalpha_dt, dvpar_dt


@partial(jax.jit, static_argnames=('q_over_m', 'mi', 'dt', 'R0'))
def push_particles_fa(
    state: GCState,
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    B: jnp.ndarray,
    gBpsi: jnp.ndarray,
    gBth: jnp.ndarray,
    kpsi_geom: jnp.ndarray,
    kth_geom: jnp.ndarray,
    q_at_psi: jnp.ndarray,   # pre-interpolated
    q_over_m: float,
    mi: float,
    dt: float,
    R0: float,
) -> GCState:
    """
    Push all particles one timestep in field-aligned coords using RK4.

    Geometry must be pre-interpolated by the caller (avoids redundant interp).

    Parameters
    ----------
    state    : GCState with (ψ, θ, α, v∥, μ, w) for N particles
    E_psi, E_theta, E_alpha : E-field at particle positions, shape (N,)
    B, gBpsi, gBth, kpsi_geom, kth_geom : pre-interpolated geometry, shape (N,)
    q_at_psi : safety factor at particle positions, shape (N,)
    q_over_m : e/m
    mi       : ion mass
    dt       : timestep
    R0       : major radius
    """
    kpsi, kth = kpsi_geom, kth_geom

    def rhs_at(psi_, theta_, alpha_, vpar_):
        return _gc_rhs_fa_batched(
            psi_, theta_, alpha_, vpar_,
            state.mu, E_psi, E_theta, E_alpha,
            B, gBpsi, gBth, kpsi, kth, q_at_psi,
            q_over_m, mi, R0,
        )

    # RK4 — all operations are batched over N particles with no vmap needed
    k1 = rhs_at(state.r, state.theta, state.zeta, state.vpar)
    k2 = rhs_at(state.r + 0.5*dt*k1[0], state.theta + 0.5*dt*k1[1],
                 state.zeta + 0.5*dt*k1[2], state.vpar + 0.5*dt*k1[3])
    k3 = rhs_at(state.r + 0.5*dt*k2[0], state.theta + 0.5*dt*k2[1],
                 state.zeta + 0.5*dt*k2[2], state.vpar + 0.5*dt*k2[3])
    k4 = rhs_at(state.r + dt*k3[0], state.theta + dt*k3[1],
                 state.zeta + dt*k3[2], state.vpar + dt*k3[3])

    new_psi   = state.r     + (dt/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    new_theta = state.theta + (dt/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    new_alpha = state.zeta  + (dt/6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    new_vpar  = state.vpar  + (dt/6) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    # Wrap θ to [-π, π) and α to [0, 2π)
    new_theta = ((new_theta + jnp.pi) % (2*jnp.pi)) - jnp.pi
    new_alpha = new_alpha % (2*jnp.pi)

    return GCState(r=new_psi, theta=new_theta, zeta=new_alpha,
                   vpar=new_vpar, mu=state.mu, weight=state.weight)
