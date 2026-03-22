# FILE: gyrojax/particles/guiding_center.py
"""
Guiding center particle state and RK4 pusher.

The guiding center approximation eliminates fast gyromotion by averaging
over the cyclotron orbit. Particles move in 5D phase space (r, θ, ζ, v∥, μ)
where μ = mv⊥²/2B is the adiabatic invariant (conserved along orbits).

Equations of motion (electrostatic, s-α geometry):
    dr/dt     = v_Er + v_∇Br + v_curvr
    dθ/dt     = v_par/(q*R0) + (v_Eθ + v_∇Bθ + v_curvθ) / r
    dζ/dt     = v_par/R0   (leading order)
    dv∥/dt    = -(μ/m)*b̂·∇B - (q/m)*E∥
    dμ/dt     = 0           (adiabatic invariant)
    dw/dt     = see deltaf/weights.py

References:
    Littlejohn (1983) J. Plasma Phys. 29, 111
    Brizard & Hahm (2007) Rev. Mod. Phys. 79, 421
"""

from __future__ import annotations
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from gyrojax.geometry.salpha import SAlphaGeometry, interp_geometry_to_particles


class GCState(NamedTuple):
    """
    Guiding center particle state. All arrays shape (N_particles,).

    Fields
    ------
    r     : radial position [m]
    theta : poloidal angle [rad]
    zeta  : toroidal angle [rad]
    vpar  : parallel velocity [m/s]
    mu    : magnetic moment mv⊥²/2B [J/T] — adiabatic invariant
    weight: delta-f weight w = δf/f
    """
    r: jnp.ndarray
    theta: jnp.ndarray
    zeta: jnp.ndarray
    vpar: jnp.ndarray
    mu: jnp.ndarray
    weight: jnp.ndarray


# Register as JAX pytree so it can pass through jit/vmap/scan
jax.tree_util.register_pytree_node(
    GCState,
    lambda s: (list(s), None),
    lambda _, xs: GCState(*xs),
)


def _gc_rhs(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_zeta: jnp.ndarray,
    geom: SAlphaGeometry,
    q_over_m: float,
    mi: float,
) -> Tuple[jnp.ndarray, ...]:
    """
    Compute RHS of guiding center equations of motion for a single particle.

    Returns (dr_dt, dtheta_dt, dzeta_dt, dvpar_dt).
    """
    # Interpolate geometry to particle position
    B, gradB_r, gradB_th, kappa_r, kappa_th = interp_geometry_to_particles(
        geom, r[None], theta[None], zeta[None]
    )
    B = B[0]; gradB_r = gradB_r[0]; gradB_th = gradB_th[0]
    kappa_r = kappa_r[0]; kappa_th = kappa_th[0]

    q = geom.r_grid[0] * 0.0 + q_over_m * mi   # charge from q/m ratio
    Omega = q_over_m * B   # cyclotron freq q*B/m (using q_over_m = q/m)

    # ---- E×B drift (in toroidal geometry) ----
    # vE = (E × B) / B²
    # In (r, θ, ζ) with B mostly in ζ direction:
    # vE_r   = -E_θ / B   (approximate for large aspect ratio)
    # vE_θ   =  E_r / B
    vE_r   = -E_theta / B
    vE_th  =  E_r / B

    # ---- ∇B drift: v_∇B = (μ/m·Ω) (b̂ × ∇B) ----
    # b̂ ≈ ζ̂ in s-α, so b̂ × ∇B has r and θ components:
    # v_∇B_r  = -(μ/mΩ) * (1/r) * ∂B/∂θ
    # v_∇B_θ  =  (μ/mΩ) * ∂B/∂r
    prefac_grad = mu / (mi * Omega)
    v_gradB_r  = -prefac_grad * gradB_th / jnp.maximum(r, 1e-6)
    v_gradB_th =  prefac_grad * gradB_r

    # ---- Curvature drift: v_curv = (v∥²/Ω)(b̂ × κ) ----
    # Same structure as ∇B drift with κ replacing ∇B/B
    prefac_curv = vpar**2 / Omega
    v_curv_r  = -prefac_curv * kappa_th
    v_curv_th =  prefac_curv * kappa_r

    # ---- q(r) for parallel streaming ----
    # Interpolate safety factor
    Nr = geom.r_grid.shape[0]
    dr = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    ir = jnp.clip((r - geom.r_grid[0]) / dr, 0.0, Nr - 1.001)
    ir0 = ir.astype(jnp.int32)
    ir1 = jnp.minimum(ir0 + 1, Nr - 1)
    frac = ir - ir0.astype(jnp.float32)
    q_at_r = geom.q_profile[ir0] * (1 - frac) + geom.q_profile[ir1] * frac

    R_approx = geom.R0 * (1.0 + (r / geom.R0) * jnp.cos(theta))

    # ---- Equations of motion ----
    dr_dt     = vE_r + v_gradB_r + v_curv_r
    dtheta_dt = vpar / (q_at_r * geom.R0) + (vE_th + v_gradB_th + v_curv_th) / jnp.maximum(r, 1e-6)
    dzeta_dt  = vpar / R_approx

    # Parallel force: -μ b̂·∇B/m  - (q/m) E∥
    E_par     = E_zeta   # ζ is approximately parallel in s-α
    dvpar_dt  = -(mu / mi) * gradB_r - q_over_m * E_par  # simplified: only radial ∇B term

    return dr_dt, dtheta_dt, dzeta_dt, dvpar_dt


def _rk4_step(
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_zeta: jnp.ndarray,
    geom: SAlphaGeometry,
    q_over_m: float,
    mi: float,
    dt: float,
) -> GCState:
    """RK4 step for a single guiding center particle."""
    r, theta, zeta, vpar, mu, w = state

    def rhs(r_, th_, ze_, vp_):
        return _gc_rhs(r_, th_, ze_, vp_, mu, E_r, E_theta, E_zeta, geom, q_over_m, mi)

    k1r, k1th, k1ze, k1vp = rhs(r, theta, zeta, vpar)
    k2r, k2th, k2ze, k2vp = rhs(r + 0.5*dt*k1r, theta + 0.5*dt*k1th,
                                  zeta + 0.5*dt*k1ze, vpar + 0.5*dt*k1vp)
    k3r, k3th, k3ze, k3vp = rhs(r + 0.5*dt*k2r, theta + 0.5*dt*k2th,
                                  zeta + 0.5*dt*k2ze, vpar + 0.5*dt*k2vp)
    k4r, k4th, k4ze, k4vp = rhs(r + dt*k3r, theta + dt*k3th,
                                  zeta + dt*k3ze, vpar + dt*k3vp)

    new_r    = r     + (dt/6.0)*(k1r  + 2*k2r  + 2*k3r  + k4r)
    new_th   = theta + (dt/6.0)*(k1th + 2*k2th + 2*k3th + k4th)
    new_ze   = zeta  + (dt/6.0)*(k1ze + 2*k2ze + 2*k3ze + k4ze)
    new_vp   = vpar  + (dt/6.0)*(k1vp + 2*k2vp + 2*k3vp + k4vp)

    # Wrap angles to [0, 2π)
    new_th = new_th % (2.0 * jnp.pi)
    new_ze = new_ze % (2.0 * jnp.pi)

    # Radial boundary: reflect if out of bounds
    r_min, r_max = geom.r_grid[0], geom.r_grid[-1]
    new_r = jnp.clip(new_r, r_min, r_max)

    return GCState(r=new_r, theta=new_th, zeta=new_ze, vpar=new_vp, mu=mu, weight=w)


@jax.jit
def push_particles(
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_zeta: jnp.ndarray,
    geom: SAlphaGeometry,
    q_over_m: float,
    mi: float,
    dt: float,
) -> GCState:
    """
    Push all guiding center particles by one RK4 timestep.

    Uses jax.vmap over particles for GPU parallelism.

    Parameters
    ----------
    state    : GCState, all fields shape (N_particles,)
    E_r/theta/zeta : electric field components at particle positions, (N_particles,)
    geom     : SAlphaGeometry
    q_over_m : charge-to-mass ratio q/m [C/kg]
    mi       : ion mass [kg]
    dt       : timestep [s]

    Returns
    -------
    GCState with updated (r, theta, zeta, vpar), mu and weight unchanged
    """
    step_one = jax.vmap(
        lambda r, th, ze, vp, mu, w, er, eth, eze: _rk4_step(
            GCState(r, th, ze, vp, mu, w), er, eth, eze, geom, q_over_m, mi, dt
        )
    )
    return step_one(state.r, state.theta, state.zeta, state.vpar,
                    state.mu, state.weight, E_r, E_theta, E_zeta)


def init_maxwellian_particles(
    N: int,
    geom: SAlphaGeometry,
    vti: float,
    Ti: float,
    mi: float,
    key: jax.random.PRNGKey,
) -> GCState:
    """
    Initialize N particles with a Maxwellian distribution.

    Particles are placed uniformly in (r, θ, ζ) and given
    velocities from a Maxwellian with thermal speed vti.
    μ = mv⊥²/2B is sampled from an exponential distribution.
    All weights initialized to zero (unperturbed state).

    Parameters
    ----------
    N    : number of particles
    geom : geometry
    vti  : ion thermal velocity [m/s]
    Ti   : ion temperature [J]
    mi   : ion mass [kg]
    key  : JAX PRNG key
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    r_min, r_max = float(geom.r_grid[0]), float(geom.r_grid[-1])
    r     = jax.random.uniform(k1, (N,), minval=r_min, maxval=r_max)
    theta = jax.random.uniform(k2, (N,), minval=0.0, maxval=2.0*jnp.pi)
    zeta  = jax.random.uniform(k3, (N,), minval=0.0, maxval=2.0*jnp.pi)
    vpar  = jax.random.normal(k4, (N,)) * vti

    # μ ~ Exp(Ti/B0): sample v_perp from Rayleigh, compute μ = m*v_perp²/(2B)
    # Clamp to 4*vti to avoid extreme-tail particles with huge drifts
    u = jax.random.uniform(k5, (N,))
    v_perp = vti * jnp.sqrt(-2.0 * jnp.log(jnp.clip(u, 1e-10, 1.0)))
    v_perp = jnp.clip(v_perp, 0.0, 4.0 * vti)   # cap at 4 vti
    vpar   = jnp.clip(vpar,  -4.0 * vti, 4.0 * vti)  # same for vpar
    B_on_axis = geom.B0
    mu = 0.5 * mi * v_perp**2 / B_on_axis

    weight = jnp.zeros(N)

    return GCState(r=r, theta=theta, zeta=zeta, vpar=vpar, mu=mu, weight=weight)
