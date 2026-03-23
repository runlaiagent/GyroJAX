"""
Full-f gyrokinetic simulation loop — Phase 3.

Key differences from δf (simulation_fa.py):
  - Marker weights W_p are CONSTANT (no weight equation)
  - δn = n_markers(x) - n0(x)  [full density, not just perturbation]
  - Resampling step every N_resample steps (optional, for noise control)
  - Nonlinear turbulence naturally included

The GC pusher and Poisson solver are reused unchanged from Phase 2a.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, List, Optional

import jax
import jax.numpy as jnp

from gyrojax.geometry.field_aligned import (
    FieldAlignedGeometry,
    build_field_aligned_geometry,
    interp_fa_to_particles,
    salpha_to_fa_coords,
)
from gyrojax.particles.guiding_center import GCState, init_maxwellian_particles
from gyrojax.particles.guiding_center_fa import push_particles_fa
from gyrojax.fields.poisson_fa import solve_poisson_fa, compute_efield_fa
from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa
from gyrojax.fullf import init_fullf_particles, compute_n0_grid


@dataclass
class SimConfigFullF:
    """Simulation configuration for full-f run."""
    # Grid
    Npsi:   int = 32
    Ntheta: int = 64
    Nalpha: int = 32
    # Particles
    N_particles: int = 500_000
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
    e:   float = 1.0
    vti: float = 1.0
    n0_avg: float = 1.0
    # Profiles
    R0_over_LT: float = 6.9
    R0_over_Ln: float = 2.2
    # Velocity cap
    vpar_cap: float = 4.0
    # Resampling: resample every N steps (0 = never)
    resample_interval: int = 0


class DiagnosticsFullF(NamedTuple):
    phi_rms:    jnp.ndarray
    phi_max:    jnp.ndarray
    n_rms:      jnp.ndarray   # rms of δn/n0


def _interp_q(r: jnp.ndarray, geom: FieldAlignedGeometry) -> jnp.ndarray:
    Nr  = geom.psi_grid.shape[0]
    dr  = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
    ir  = jnp.clip((r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
    return geom.q_profile[jnp.floor(ir).astype(jnp.int32)]


def _resample_particles(
    state: GCState,
    geom: FieldAlignedGeometry,
    cfg: SimConfigFullF,
    key: jax.random.PRNGKey,
) -> tuple:
    """
    Importance resampling: replace markers with high/low weights by equal-weight
    markers sampled proportional to the current weight distribution.

    This controls the variance growth inherent in full-f PIC and prevents
    a few markers from dominating the statistics.

    Simple implementation: multinomial resampling (standard in particle filters).
    """
    N = cfg.N_particles
    W = state.weight
    W_total = jnp.sum(W)
    probs = W / (W_total + 1e-30)

    # Multinomial resampling: sample N indices according to probs
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, N, shape=(N,), replace=True, p=probs)

    new_state = GCState(
        r      = state.r[indices],
        theta  = state.theta[indices],
        zeta   = state.zeta[indices],
        vpar   = state.vpar[indices],
        mu     = state.mu[indices],
        weight = jnp.full(N, W_total / N, dtype=jnp.float32),
    )
    return new_state, key


def run_simulation_fullf(
    cfg: SimConfigFullF,
    geom: Optional[FieldAlignedGeometry] = None,
    key: jax.random.PRNGKey = None,
    verbose: bool = True,
) -> tuple:
    """
    Run a full-f gyrokinetic PIC simulation.

    Parameters
    ----------
    cfg  : SimConfigFullF
    geom : optional pre-built geometry (e.g. VMEC). If None, builds s-α.
    key  : JAX random key
    verbose : print progress

    Returns
    -------
    diags : list of DiagnosticsFullF
    state : final GCState
    phi   : final potential
    geom  : geometry used
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # --- Geometry ---
    if geom is None:
        geom = build_field_aligned_geometry(
            Npsi=cfg.Npsi, Ntheta=cfg.Ntheta, Nalpha=cfg.Nalpha,
            R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1,
        )

    Npsi   = geom.B_field.shape[0]
    Ntheta = geom.B_field.shape[1]
    Nalpha = geom.B_field.shape[2]
    grid_shape = (Npsi, Ntheta, Nalpha)

    if verbose:
        print(f"[GyroJAX full-f] {cfg.N_particles:,} particles, "
              f"grid ({Npsi},{Ntheta},{Nalpha}), {cfg.n_steps} steps")

    # --- Background density on grid ---
    n0_grid = compute_n0_grid(geom, grid_shape, cfg.n0_avg, cfg.R0_over_Ln, cfg.R0)

    # --- Particle initialization ---
    # Full-f: sample from Maxwellian, set equal weights
    from gyrojax.geometry.salpha import build_salpha_geometry
    geom_sa = build_salpha_geometry(
        Npsi, Ntheta, Nalpha,
        R0=geom.R0, a=geom.a, B0=geom.B0,
        q0=float(geom.q_profile[Npsi//2]), q1=0.0,
    )
    state_sa = init_maxwellian_particles(
        cfg.N_particles, geom_sa, cfg.vti, cfg.Ti, cfg.mi, key
    )
    psi_p, theta_p, alpha_p = salpha_to_fa_coords(
        state_sa.r, state_sa.theta, state_sa.zeta, geom
    )

    # Full-f weights: constant = n0_avg / N (each marker represents n0_avg/N density)
    W0 = jnp.full(cfg.N_particles, cfg.n0_avg / cfg.N_particles, dtype=jnp.float32)
    state = GCState(r=psi_p, theta=theta_p, zeta=alpha_p,
                    vpar=state_sa.vpar, mu=state_sa.mu, weight=W0)

    # Small ITG seed: perturb a fraction of markers by displacing them radially
    # (In full-f we perturb positions slightly rather than weights)
    key, subkey = jax.random.split(key)
    dr_pert = 1e-3 * geom.a * jnp.sin(2.0 * state.theta + state.zeta)
    state = state._replace(
        r=jnp.clip(state.r + dr_pert,
                   geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999)
    )

    phi = jnp.zeros(grid_shape)
    q_over_m = cfg.e / cfg.mi

    diags: List[DiagnosticsFullF] = []

    for step in range(cfg.n_steps):

        # 1. Scatter markers → density; compute δn = n - n0
        n_markers = scatter_to_grid_fa(state, geom, grid_shape)
        # scatter_to_grid_fa returns normalized density (markers/vol/N)
        # Scale to physical units: multiply by n0_avg
        n_phys = n_markers * cfg.n0_avg * cfg.N_particles
        delta_n = n_phys - n0_grid

        # 2. Solve GK Poisson with δn
        phi = solve_poisson_fa(
            delta_n, geom,
            cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
        )

        # 3. Gather E to particles
        E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state, geom)

        # 4. Push GC (RK4)
        state = push_particles_fa(
            state, E_psi_p, E_theta_p, E_alpha_p,
            geom, q_over_m, cfg.mi, cfg.dt
        )

        # Boundary + velocity clamp
        state = state._replace(
            r=jnp.clip(state.r, geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999),
            vpar=jnp.clip(state.vpar, -cfg.vpar_cap*cfg.vti, cfg.vpar_cap*cfg.vti),
        )

        # 5. Full-f: weights are constant — NO weight equation step

        # 6. Optional resampling
        if cfg.resample_interval > 0 and step > 0 and step % cfg.resample_interval == 0:
            state, key = _resample_particles(state, geom, cfg, key)
            if verbose:
                print(f"    [resample at step {step}]")

        # 7. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi**2))
        phi_max = jnp.max(jnp.abs(phi))
        n_rms   = jnp.sqrt(jnp.mean((delta_n / (n0_grid + 1e-30))**2))
        diags.append(DiagnosticsFullF(phi_rms=phi_rms, phi_max=phi_max, n_rms=n_rms))

        if verbose and step % 50 == 0:
            print(f"  step {step:4d}/{cfg.n_steps}  |φ|_max={float(phi_max):.3e}  "
                  f"δn/n0_rms={float(n_rms):.3e}")

    return diags, state, phi, geom
