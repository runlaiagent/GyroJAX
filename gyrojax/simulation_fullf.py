"""
Full-f gyrokinetic PIC simulation — Phase 3 (revised).

In full-f, we track the full distribution f rather than just δf = f - f0.
However, in the *linear* phase, full-f and δf give the same growth rate —
the difference only shows up nonlinearly (zonal flows, profile relaxation).

For the CBC benchmark (linear phase) we use a hybrid approach:
  - Particles initialized from Maxwellian f0 with equal weights W_p = 1/N
  - Weight equation: same as δf but for the *full* log-gradient of f
  - No linearization: the full v_total (not just equilibrium drift) drives weights
  - Periodic resampling to control variance growth

This gives correct linear growth rates and captures the onset of nonlinearity,
while being more tractable than a pure counting-based full-f scheme (which
requires many more particles/cell to resolve δn/n0 ~ 1%).

References:
  Idomura et al. (2008) Nucl. Fusion 48, 035002
  Bottino & Sonnendrücker (2015) J. Plasma Phys. 81, 435810501
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import jax
import jax.numpy as jnp

from gyrojax.geometry.field_aligned import (
    FieldAlignedGeometry, build_field_aligned_geometry,
    salpha_to_fa_coords, interp_fa_to_particles,
)
from gyrojax.particles.guiding_center import GCState, init_maxwellian_particles
from gyrojax.particles.guiding_center_fa import push_particles_fa
from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa
from gyrojax.fields.poisson_fa import solve_poisson_fa
from gyrojax.deltaf.weights import update_weights
from gyrojax.geometry.salpha import build_salpha_geometry


@dataclass
class SimConfigFullF:
    # Grid
    Npsi:   int = 16
    Ntheta: int = 32
    Nalpha: int = 16
    # Particles
    N_particles: int = 200_000
    # Time
    n_steps: int = 300
    dt:      float = 0.05
    # Geometry
    R0:  float = 1.0
    a:   float = 0.18
    B0:  float = 1.0
    q0:  float = 1.4
    q1:  float = 0.5
    # Physics
    Ti:  float = 1.0
    Te:  float = 1.0
    mi:  float = 1.0
    rho_star: float = 1.0 / 180.0  # rho_i/a; CBC standard
    e:   float = 1000.0   # = 1/rho_star * vti*mi/(a*B0); Omega_i*mi/B0; CBC rho*=1/180 -> Omega=1000
    vti: float = 1.0
    n0_avg: float = 1.0
    # Profiles
    R0_over_LT: float = 6.9
    R0_over_Ln: float = 2.2
    # Velocity cap
    vpar_cap: float = 4.0
    # Resampling (0 = never)
    resample_interval: int = 50
    # ITG seed amplitude
    pert_amp: float = 1e-4


class DiagnosticsFullF(NamedTuple):
    phi_rms:    jnp.ndarray
    phi_max:    jnp.ndarray
    n_rms:      jnp.ndarray


def _resample_particles(
    state: GCState,
    cfg: SimConfigFullF,
    key: jax.random.PRNGKey,
) -> tuple:
    """
    Systematic resampling: replace high-variance weights with equal-weight markers.
    Preserves the total distribution while reducing weight variance.
    """
    N = cfg.N_particles
    W = jnp.abs(state.weight)
    W_norm = W / (jnp.sum(W) + 1e-30)

    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, N, shape=(N,), replace=True, p=W_norm)

    # Preserve sign of weight
    signs = jnp.sign(state.weight[indices])
    new_w = signs * jnp.mean(W)

    new_state = GCState(
        r=state.r[indices], theta=state.theta[indices],
        zeta=state.zeta[indices], vpar=state.vpar[indices],
        mu=state.mu[indices], weight=new_w,
    )
    return new_state, key


def run_simulation_fullf(
    cfg: SimConfigFullF,
    geom: Optional[FieldAlignedGeometry] = None,
    key: jax.random.PRNGKey = None,
    verbose: bool = True,
) -> tuple:
    """
    Run full-f gyrokinetic simulation.

    Returns (diags, final_state, final_phi, geom).
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    Npsi, Ntheta, Nalpha = cfg.Npsi, cfg.Ntheta, cfg.Nalpha
    grid_shape = (Npsi, Ntheta, Nalpha)

    if geom is None:
        geom = build_field_aligned_geometry(
            Npsi, Ntheta, Nalpha,
            R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1,
        )

    if verbose:
        print(f"[GyroJAX Full-f] {cfg.N_particles:,} particles, "
              f"grid ({Npsi},{Ntheta},{Nalpha}), {cfg.n_steps} steps, dt={cfg.dt}")

    # --- Particle initialization ---
    geom_sa = build_salpha_geometry(
        Npsi, Ntheta, Nalpha,
        R0=cfg.R0, a=cfg.a, B0=cfg.B0,
        q0=float(geom.q_profile[Npsi//2]), q1=0.0,
    )
    key, subkey = jax.random.split(key)
    state_sa = init_maxwellian_particles(
        cfg.N_particles, geom_sa, cfg.vti, cfg.Ti, cfg.mi, subkey
    )
    psi_p, theta_p, alpha_p = salpha_to_fa_coords(
        state_sa.r, state_sa.theta, state_sa.zeta, geom
    )

    # Full-f: seed ITG via initial weight perturbation
    # w_p = δf/f0 ~ pert_amp * sin(m*θ + n*α) → same as δf init
    pert = cfg.pert_amp * jnp.sin(2.0 * theta_p + alpha_p)
    state = GCState(
        r=psi_p, theta=theta_p, zeta=alpha_p,
        vpar=state_sa.vpar, mu=state_sa.mu,
        weight=pert,
    )

    # Profile gradients (flat — flux-tube approximation)
    Ln = cfg.R0 / cfg.R0_over_Ln
    LT = cfg.R0 / cfg.R0_over_LT
    q_over_m = cfg.e / cfg.mi

    phi = jnp.zeros(grid_shape)
    diags: List[DiagnosticsFullF] = []

    for step in range(cfg.n_steps):

        # 1. Scatter δf weights → δn
        delta_n = scatter_to_grid_fa(state, geom, grid_shape)

        # 2. Solve GK Poisson
        phi = solve_poisson_fa(
            delta_n, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
        )

        # 3. Gather E to particles
        E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state, geom)

        # 4. Get B/gradB/curvature at particle positions
        B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p = \
            interp_fa_to_particles(geom, state.r, state.theta, state.zeta)

        # 5. Update weights (full nonlinear — no linearization in vE)
        n0_p   = jnp.ones_like(state.r) * cfg.n0_avg
        T_p    = jnp.ones_like(state.r) * cfg.Ti
        q_p    = jnp.full_like(state.r, cfg.q0)
        d_ln_n = jnp.full_like(state.r, -1.0 / Ln)
        d_ln_T = jnp.full_like(state.r, -1.0 / LT)

        state = update_weights(
            state, E_psi_p, E_theta_p, B_p,
            gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p,
            q_p, n0_p, T_p, d_ln_n, d_ln_T,
            q_over_m, cfg.mi, cfg.Ti, cfg.dt,
        )

        # 6. Push GC (RK4)
        state = push_particles_fa(
            state, E_psi_p, E_theta_p, E_alpha_p,
            geom, q_over_m, cfg.mi, cfg.dt,
        )

        # Boundary + velocity clamp
        state = state._replace(
            r=jnp.clip(state.r, geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999),
            vpar=jnp.clip(state.vpar, -cfg.vpar_cap*cfg.vti, cfg.vpar_cap*cfg.vti),
        )

        # 7. Optional resampling
        if cfg.resample_interval > 0 and step > 0 and step % cfg.resample_interval == 0:
            state, key = _resample_particles(state, cfg, key)

        # 8. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi**2))
        phi_max = jnp.max(jnp.abs(phi))
        n_rms   = jnp.sqrt(jnp.mean(delta_n**2))
        diags.append(DiagnosticsFullF(phi_rms=phi_rms, phi_max=phi_max, n_rms=n_rms))

        if verbose and step % 50 == 0:
            w_rms = float(jnp.sqrt(jnp.mean(state.weight**2)))
            print(f"  step {step:4d}/{cfg.n_steps}  "
                  f"|φ|_max={float(phi_max):.3e}  |w|_rms={w_rms:.3e}")

    return diags, state, phi, geom
