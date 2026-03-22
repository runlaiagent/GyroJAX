# FILE: gyrojax/simulation.py
"""
Main simulation loop for GyroJAX.

The time loop order is:
  1. scatter δf → δn on grid (gyroaveraged)
  2. solve GK Poisson: δn → φ
  3. compute E = -∇φ
  4. gather E to particle positions (gyroaveraged)
  5. push guiding centers (RK4)
  6. update δf weights
  7. diagnostics (phi_rms, energy)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp

from gyrojax.geometry.salpha import SAlphaGeometry, build_salpha_geometry
from gyrojax.particles.guiding_center import GCState, push_particles, init_maxwellian_particles
from gyrojax.deltaf.weights import update_weights
from gyrojax.fields.poisson import solve_poisson_gk, compute_efield
from gyrojax.interpolation.scatter_gather import scatter_to_grid, gather_from_grid


@dataclass
class SimConfig:
    """Simulation configuration parameters."""
    # Grid
    Nr: int = 32
    Ntheta: int = 64
    Nzeta: int = 32
    # Particles
    N_particles: int = 100_000
    # Time
    n_steps: int = 500
    dt: float = 0.05           # normalized to R0/vti
    # Physics (CBC defaults, normalized)
    R0: float = 1.0            # major radius [normalized]
    a: float = 0.18            # minor radius = eps * R0
    B0: float = 1.0
    q0: float = 1.4
    q1: float = 0.5
    Ti: float = 1.0            # [normalized to reference energy]
    Te: float = 1.0
    mi: float = 1.0
    e: float  = 1.0
    # Profiles (CBC: R0/LT = 6.9, R0/Ln = 2.2)
    R0_over_LT: float = 6.9
    R0_over_Ln: float = 2.2
    vti: float = 1.0           # thermal velocity
    n0_avg: float = 1.0        # average density


class Diagnostics(NamedTuple):
    phi_rms: jnp.ndarray   # rms potential
    phi_max: jnp.ndarray   # max |phi|
    weight_rms: jnp.ndarray


def _get_profiles(r: jnp.ndarray, cfg: SimConfig):
    """Gaussian density and temperature profiles."""
    r_mid = 0.5 * (cfg.R0 * cfg.a + cfg.R0 * cfg.a * 0.1)
    # CBC-style profiles: n0 = n_ref * exp(-Δn * (r-r0)/a), T = T_ref * exp(-ΔT*(r-r0)/a)
    Ln = cfg.R0 / cfg.R0_over_Ln
    LT = cfg.R0 / cfg.R0_over_LT
    r_mid = cfg.a * 0.5
    n0 = cfg.n0_avg * jnp.exp(-(r - r_mid) / Ln)
    T  = cfg.Ti     * jnp.exp(-(r - r_mid) / LT)
    return n0, T


def run_simulation(cfg: SimConfig, key: jax.random.PRNGKey = None):
    """
    Run a gyrokinetic simulation.

    Parameters
    ----------
    cfg : SimConfig
    key : JAX random key (default: fixed seed)

    Returns
    -------
    diagnostics : list of Diagnostics namedtuples, length n_steps
    state       : final GCState
    phi         : final phi field
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    print(f"[GyroJAX] Initializing simulation: {cfg.N_particles} particles, "
          f"grid ({cfg.Nr},{cfg.Ntheta},{cfg.Nzeta}), {cfg.n_steps} steps")

    # Build geometry
    geom = build_salpha_geometry(
        cfg.Nr, cfg.Ntheta, cfg.Nzeta,
        R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1
    )

    # Initialize particles
    state = init_maxwellian_particles(
        cfg.N_particles, geom, cfg.vti, cfg.Ti, cfg.mi, key
    )

    # Seed small ITG perturbation: w = ε·sin(m·θ) at m=2
    # This seeds the CBC poloidal mode structure
    pert_amp = 1e-4
    pert = pert_amp * jnp.sin(2.0 * state.theta + state.zeta)
    state = state._replace(weight=pert)
    phi = jnp.zeros((cfg.Nr, cfg.Ntheta, cfg.Nzeta))

    q_over_m = cfg.e / cfg.mi
    grid_shape = (cfg.Nr, cfg.Ntheta, cfg.Nzeta)

    diags = []

    for step in range(cfg.n_steps):
        # 1. Scatter δn
        delta_n = scatter_to_grid(state, geom, grid_shape, cfg.mi, cfg.e)

        # 2. Solve Poisson
        phi = solve_poisson_gk(delta_n, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e)

        # 3. Gather E to particles
        E_r_g, E_th_g, E_ze_g = compute_efield(phi, geom)
        E_r_p, E_th_p, E_ze_p = gather_from_grid(phi, state, geom, cfg.mi, cfg.e)

        # 4. Push particles
        state = push_particles(state, E_r_p, E_th_p, E_ze_p, geom, q_over_m, cfg.mi, cfg.dt)
        # clamp radial position (simple BC)
        state = state._replace(r=jnp.clip(state.r, geom.r_grid[0]*1.001, geom.r_grid[-1]*0.999))

        # 5. Update weights (need full geometry + profiles at particle positions)
        from gyrojax.geometry.salpha import interp_geometry_to_particles
        B_p, gradB_r_p, gradB_th_p, kappa_r_p, kappa_th_p = interp_geometry_to_particles(
            geom, state.r, state.theta, state.zeta
        )
        # Profile values at particles
        n0_p, T_p = _get_profiles(state.r, cfg)
        Ln = cfg.R0 / cfg.R0_over_Ln
        LT = cfg.R0 / cfg.R0_over_LT
        d_ln_n0_dr = jnp.full_like(state.r, -1.0 / Ln)
        d_ln_T_dr  = jnp.full_like(state.r, -1.0 / LT)
        # Safety factor at particle positions (for drift normalization)
        dr = (geom.r_grid[-1] - geom.r_grid[0]) / (len(geom.r_grid) - 1)
        ir = jnp.clip((state.r - geom.r_grid[0]) / dr, 0, len(geom.r_grid) - 1.001)
        q_at_r_p = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]

        state = update_weights(
            state, E_r_p, E_th_p, B_p, gradB_r_p, gradB_th_p,
            kappa_r_p, kappa_th_p, q_at_r_p,
            n0_p, T_p, d_ln_n0_dr, d_ln_T_dr,
            q_over_m, cfg.mi, cfg.R0, cfg.dt
        )

        # 6. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi**2))
        phi_max = jnp.max(jnp.abs(phi))
        w_rms   = jnp.sqrt(jnp.mean(state.weight**2))
        diags.append(Diagnostics(phi_rms=phi_rms, phi_max=phi_max, weight_rms=w_rms))

        if step % 50 == 0:
            print(f"  step {step:4d}/{cfg.n_steps}  |phi|_max={float(phi_max):.3e}  "
                  f"|w|_rms={float(w_rms):.3e}")

    return diags, state, phi
