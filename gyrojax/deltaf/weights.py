# FILE: gyrojax/deltaf/weights.py
"""
Delta-f weight evolution for gyrokinetic PIC.

The delta-f method decomposes the distribution function as:
    f = f0 + δf = f0(1 + w)
where f0 is a known Maxwellian background and w = δf/f0 is the marker weight.

Weight equation (from the Vlasov equation applied to δf):
    dw/dt = -(1-w) * C[f0]/f0
where C[f0] = (∂f0/∂t + v·∇f0 + F·∂f0/∂v) is the phase-space advection of f0.

For electrostatic ITG with adiabatic electrons:
    dw/dt = -(1-w) * [vE·∇ln(f0) + (q/m)*E∥ * ∂ln(f0)/∂v∥]

With a Maxwellian f0(r, v∥, μ):
    ln(f0) = ln(n0(r)) - (m*v∥²/2 + μ*B(r,θ)) / T(r) + const
    ∂ln(f0)/∂r   = d_ln_n0/dr - μ*(∂B/∂r)/T - (m*v∥²/2 + μ*B)*(d_ln_T/dr)/T
    ∂ln(f0)/∂v∥  = -m*v∥/T

References:
    Lin et al. (1998) Science 281, 1835
    Jolliet et al. (2007) Comput. Phys. Commun. 177, 409
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.particles.guiding_center import GCState
from gyrojax.geometry.salpha import SAlphaGeometry, interp_geometry_to_particles


def maxwellian_f0(
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    B: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    mi: float,
) -> jnp.ndarray:
    """
    Evaluate the background Maxwellian f0 at particle positions.

    f0(r, v∥, μ) = n0(r) / (sqrt(π)*v_t)³ * exp(-H/T)
    H = m*v∥²/2 + μ*B   (guiding center Hamiltonian)
    """
    vt = jnp.sqrt(2.0 * T / mi)
    H  = 0.5 * mi * vpar**2 + mu * B
    return n0 / (jnp.pi**1.5 * vt**3) * jnp.exp(-H / T)


def log_f0_gradients(
    r: jnp.ndarray,
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    mi: float,
) -> tuple:
    """
    Compute gradients of ln(f0) needed for weight evolution.

    Returns
    -------
    d_lnf0_dr  : ∂ln(f0)/∂r  at each particle
    d_lnf0_dvp : ∂ln(f0)/∂v∥ at each particle
    """
    H = 0.5 * mi * vpar**2 + mu * B

    # ∂ln(f0)/∂r = d_ln_n0/dr - μ*(∂B/∂r)/T - H*(d_ln_T/dr)/T
    d_lnf0_dr  = d_ln_n0_dr - mu * gradB_r / T - H * d_ln_T_dr / T

    # ∂ln(f0)/∂v∥ = -m*v∥/T
    d_lnf0_dvp = -mi * vpar / T

    return d_lnf0_dr, d_lnf0_dvp


def _weight_rhs(
    w: jnp.ndarray,
    vE_r: jnp.ndarray,
    dvpar_dt: jnp.ndarray,
    d_lnf0_dr: jnp.ndarray,
    d_lnf0_dvp: jnp.ndarray,
) -> jnp.ndarray:
    """
    RHS of the weight evolution equation.
        dw/dt = -(1-w) * [vE_r * d_lnf0_dr + dvpar_dt * d_lnf0_dvp]
    """
    return -(1.0 - w) * (vE_r * d_lnf0_dr + dvpar_dt * d_lnf0_dvp)


@jax.jit
def update_weights(
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    q_over_m: float,
    mi: float,
    dt: float,
) -> GCState:
    """
    Advance delta-f weights by one timestep using RK4.

    Parameters
    ----------
    state       : GCState with current particle positions/velocities
    E_r/theta   : electric field at particle positions, shape (N,)
    B           : |B| at particle positions, shape (N,)
    gradB_r     : ∂B/∂r at particle positions, shape (N,)
    n0, T       : background density and temperature at particle r, shape (N,)
    d_ln_n0_dr  : d(ln n0)/dr at particle r, shape (N,)
    d_ln_T_dr   : d(ln T)/dr at particle r, shape (N,)
    q_over_m    : charge-to-mass ratio [C/kg]
    mi          : ion mass [kg]
    dt          : timestep [s]
    """
    # E×B radial drift
    vE_r = -E_theta / B

    # Parallel force (for weight drive)
    # dvpar/dt from electric field only (for weight equation, not position push)
    E_par = jnp.zeros_like(E_r)   # electrostatic: E_par ≈ 0 for perp modes
    dvpar_dt = -(state.mu / mi) * gradB_r + q_over_m * E_par

    d_lnf0_dr, d_lnf0_dvp = log_f0_gradients(
        state.r, state.vpar, state.mu, B, gradB_r,
        n0, T, d_ln_n0_dr, d_ln_T_dr, mi
    )

    # RK4 on weights
    k1 = _weight_rhs(state.weight, vE_r, dvpar_dt, d_lnf0_dr, d_lnf0_dvp)
    k2 = _weight_rhs(state.weight + 0.5*dt*k1, vE_r, dvpar_dt, d_lnf0_dr, d_lnf0_dvp)
    k3 = _weight_rhs(state.weight + 0.5*dt*k2, vE_r, dvpar_dt, d_lnf0_dr, d_lnf0_dvp)
    k4 = _weight_rhs(state.weight + dt*k3,      vE_r, dvpar_dt, d_lnf0_dr, d_lnf0_dvp)

    new_weight = state.weight + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return GCState(
        r=state.r, theta=state.theta, zeta=state.zeta,
        vpar=state.vpar, mu=state.mu, weight=new_weight
    )
