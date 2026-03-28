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
    g_aa: jnp.ndarray,      # (N,) — g^{αα} metric at particle positions
    q_over_m: float,
    mi: float,
    R0: float,
    E_par_em: jnp.ndarray = None,  # (N,) inductive E∥ = -∂A∥/∂t (EM correction)
):
    """Batched (N,) RHS of guiding-center equations in field-aligned coords."""
    Omega = q_over_m * B
    safe_r = jnp.maximum(psi, 1e-4)
    safe_O = jnp.sign(Omega + 1e-20) * jnp.maximum(jnp.abs(Omega), 1e-10)

    # ExB drift in field-aligned Clebsch coords (ψ, θ, α), where α = ζ - q(ψ)·θ.
    # The radial ExB in the original (ψ, θ, ζ) coords is v_ExB^ψ = -E_θ^{phys}/B.
    # In FA coords, ∂φ/∂θ|_{ψ,ζ} = -q(ψ)·∂φ/∂α, so E_θ^{phys} = -q(ψ)·E_α^{FA}.
    # Therefore: v_ExB^ψ = q(ψ) * E_α / B
    #            v_ExB^α ≈ -E_ψ / B  (plus twist correction, small)
    safe_B = jnp.maximum(B, 1e-10)
    vE_psi   =  q_at_psi * E_alpha / safe_B
    vE_alpha = -E_psi   / safe_B

    # --- ∇B drifts ---
    prefac_grad = mu / (mi * safe_O)
    vd_psi  = -prefac_grad * gBth / safe_r

    # --- Curvature drifts ---
    prefac_curv = vpar**2 / safe_O
    vd_psi_curv = -prefac_curv * kth / safe_r

    # --- Equations of motion ---
    dpsi_dt   = vE_psi + vd_psi + vd_psi_curv
    dtheta_dt = vpar / (q_at_psi * R0)    # parallel streaming
    dalpha_dt = vE_alpha                   # ExB in α direction

    # Parallel force: mirror + inductive E∥ = -∂A∥/∂t (EM correction)
    dvpar_dt = -(mu / mi) * gBpsi
    if E_par_em is not None:
        dvpar_dt = dvpar_dt + q_over_m * E_par_em

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
    g_aa_p: jnp.ndarray,     # g^{αα} at particle positions
    q_over_m: float,
    mi: float,
    dt: float,
    R0: float,
    E_par_em: jnp.ndarray = None,  # (N,) inductive E∥ = -∂A∥/∂t (EM only)
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
    g_aa_p   : g^{αα} metric at particle positions, shape (N,)
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
            B, gBpsi, gBth, kpsi, kth, q_at_psi, g_aa_p,
            q_over_m, mi, R0, E_par_em,
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


@partial(jax.jit, static_argnames=('q_over_m', 'mi', 'dt', 'R0'))
def push_particles_and_weights_fa(
    state: GCState,
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    B: jnp.ndarray,
    gBpsi: jnp.ndarray,
    gBth: jnp.ndarray,
    kpsi_geom: jnp.ndarray,
    kth_geom: jnp.ndarray,
    q_at_psi: jnp.ndarray,
    g_aa_p: jnp.ndarray,
    n0_p: jnp.ndarray,
    T_p: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    q_over_m: float,
    mi: float,
    dt: float,
    R0: float,
    E_par_em: jnp.ndarray = None,
) -> GCState:
    """
    Fused RK4 integrator for GC equations + weight equation.

    Integrates all 5 equations (ψ, θ, α, v∥, w) together in one RK4 pass,
    computing drifts ONCE per RK4 stage instead of twice (as in separate
    push_particles_fa + update_weights calls).
    """
    mu = state.mu  # constant adiabatic invariant

    def full_rhs(psi, theta, alpha, vpar, weight):
        Omega = q_over_m * B
        safe_B = jnp.maximum(B, 1e-10)
        safe_O = jnp.sign(Omega + 1e-20) * jnp.maximum(jnp.abs(Omega), 1e-10)
        safe_r = jnp.maximum(psi, 1e-4)

        # ExB drifts
        vE_psi   =  q_at_psi * E_alpha / safe_B
        vE_alpha = -E_psi / safe_B

        # ∇B + curvature drifts (radial component)
        vd_psi = -(mu / (mi * safe_O)) * gBth / safe_r \
                 - (vpar**2 / safe_O) * kth_geom / safe_r

        # GC equations of motion
        dpsi_dt   = vE_psi + vd_psi
        dtheta_dt = vpar / (q_at_psi * R0)
        dalpha_dt = vE_alpha
        dvpar_dt  = -(mu / mi) * gBpsi
        if E_par_em is not None:
            dvpar_dt = dvpar_dt + q_over_m * E_par_em

        # Weight equation — reuses same drift (no redundant computation)
        v_total_r = vE_psi + vd_psi

        H = 0.5 * mi * vpar**2 + mu * B
        d_lnf0_dr = d_ln_n0_dr - mu * gBpsi / T_p - H * d_ln_T_dr / T_p

        dw_dt = -(1.0 - weight) * (v_total_r * d_lnf0_dr)

        return dpsi_dt, dtheta_dt, dalpha_dt, dvpar_dt, dw_dt

    r0, th0, al0, vp0, w0 = state.r, state.theta, state.zeta, state.vpar, state.weight

    k1 = full_rhs(r0, th0, al0, vp0, w0)
    k2 = full_rhs(r0 + 0.5*dt*k1[0], th0 + 0.5*dt*k1[1],
                  al0 + 0.5*dt*k1[2], vp0 + 0.5*dt*k1[3], w0 + 0.5*dt*k1[4])
    k3 = full_rhs(r0 + 0.5*dt*k2[0], th0 + 0.5*dt*k2[1],
                  al0 + 0.5*dt*k2[2], vp0 + 0.5*dt*k2[3], w0 + 0.5*dt*k2[4])
    k4 = full_rhs(r0 + dt*k3[0], th0 + dt*k3[1],
                  al0 + dt*k3[2], vp0 + dt*k3[3], w0 + dt*k3[4])

    def rk4_update(x0, k1i, k2i, k3i, k4i):
        return x0 + (dt / 6) * (k1i + 2*k2i + 2*k3i + k4i)

    new_psi   = rk4_update(r0,  k1[0], k2[0], k3[0], k4[0])
    new_theta = rk4_update(th0, k1[1], k2[1], k3[1], k4[1])
    new_alpha = rk4_update(al0, k1[2], k2[2], k3[2], k4[2])
    new_vpar  = rk4_update(vp0, k1[3], k2[3], k3[3], k4[3])
    new_w     = rk4_update(w0,  k1[4], k2[4], k3[4], k4[4])

    # Wrap angles
    new_theta = ((new_theta + jnp.pi) % (2*jnp.pi)) - jnp.pi
    new_alpha = new_alpha % (2*jnp.pi)

    # Soft weight limiter (consistent with update_weights)
    new_w = 10.0 * jnp.tanh(new_w / 10.0)

    return GCState(r=new_psi, theta=new_theta, zeta=new_alpha,
                   vpar=new_vpar, mu=state.mu, weight=new_w.astype(jnp.float32))
