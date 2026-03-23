"""
GyroJAX Phase 2a simulation loop — field-aligned coordinates.

Uses:
  - FieldAlignedGeometry  (ψ, θ, α) grid aligned with B
  - Full Γ₀(b) GK Poisson solver (exact FLR, not Padé)
  - Field-aligned guiding-center pusher (RK4)
  - Twist-and-shift scatter/gather

Time loop order:
  1. Scatter δf weights → δn on (ψ, θ, α) grid
  2. Solve GK Poisson [exact Γ₀(b)]: δn → φ
  3. Compute E = -∇φ on grid
  4. Gather E to particle positions (trilinear)
  5. Push guiding centers (RK4) in FA coords
  6. Update δf weights (weight equation with ∇B + curvature drives)
  7. Diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, List

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
from gyrojax.deltaf.weights import update_weights
from gyrojax.fields.poisson_fa import solve_poisson_fa, compute_efield_fa
from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa
from gyrojax.geometry.profiles import build_cbc_profiles, interp_profiles, krook_damping


@dataclass
class SimConfigFA:
    """Simulation configuration for field-aligned (Phase 2a) run."""
    # Grid
    Npsi:   int = 32
    Ntheta: int = 64
    Nalpha: int = 32
    # Particles
    N_particles: int = 200_000
    # Time
    n_steps: int = 500
    dt:      float = 0.05        # normalized to R0/vti
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
    # rho_star = rho_i / a = 1/180 for CBC (standard gyrokinetic ordering)
    rho_star: float = 1.0 / 180.0
    # e is the charge-to-mass ratio factor = 1/rho_star * vti*mi/(a*B0)
    # CBC: Omega_i = vti/(rho_star*a) = 1/(1/180*0.18) = 1000 => e=1000
    e:   float = 1000.0
    vti: float = 1.0
    n0_avg: float = 1.0
    # CBC profiles
    R0_over_LT: float = 6.9
    R0_over_Ln: float = 2.2
    # Velocity cap (multiples of vti) to prevent runaway particles
    vpar_cap: float = 4.0
    # Global geometry flag
    use_global: bool = False   # True = global profiles, False = flux-tube


class DiagnosticsFA(NamedTuple):
    phi_rms:    jnp.ndarray
    phi_max:    jnp.ndarray
    weight_rms: jnp.ndarray


def _get_profiles(r: jnp.ndarray, cfg: SimConfigFA):
    """Gaussian density and temperature profiles (CBC-style)."""
    Ln   = cfg.R0 / cfg.R0_over_Ln
    LT   = cfg.R0 / cfg.R0_over_LT
    r_mid = cfg.a * 0.5
    n0 = cfg.n0_avg * jnp.exp(-(r - r_mid) / Ln)
    T  = cfg.Ti     * jnp.exp(-(r - r_mid) / LT)
    return n0, T


def _interp_q(r: jnp.ndarray, geom: FieldAlignedGeometry) -> jnp.ndarray:
    """Linear interpolation of safety factor at particle positions."""
    Nr  = geom.psi_grid.shape[0]
    dr  = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
    ir  = jnp.clip((r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
    i0  = jnp.floor(ir).astype(jnp.int32)
    return geom.q_profile[i0]


def _run_with_geom(
    cfg: SimConfigFA,
    geom: FieldAlignedGeometry,
    key: jax.random.PRNGKey,
    verbose: bool = True,
) -> tuple:
    """
    Core simulation loop given a pre-built FieldAlignedGeometry.

    Used by both run_simulation_fa (s-α) and external callers that supply
    a VMEC geometry (Phase 2b).

    Returns (diags, state, phi, geom).
    """
    Npsi   = geom.B_field.shape[0]
    Ntheta = geom.B_field.shape[1]
    Nalpha = geom.B_field.shape[2]

    if verbose:
        print(f"[GyroJAX FA] {cfg.N_particles:,} particles, "
              f"grid ({Npsi},{Ntheta},{Nalpha}), "
              f"{cfg.n_steps} steps, dt={cfg.dt}")

    # --- Particle initialization ---
    # Use s-α geometry with same grid dims for Maxwellian sampling,
    # then convert positions to FA/VMEC coords.
    from gyrojax.geometry.salpha import build_salpha_geometry
    r_min  = float(geom.psi_grid[0])
    r_max  = float(geom.psi_grid[-1])
    # Build a thin s-α geometry just for particle sampling
    geom_sa = build_salpha_geometry(
        Npsi, Ntheta, Nalpha,
        R0=geom.R0, a=geom.a, B0=geom.B0,
        q0=float(geom.q_profile[Npsi//2]),
        q1=0.0,
    )
    state_sa = init_maxwellian_particles(
        cfg.N_particles, geom_sa, cfg.vti, cfg.Ti, cfg.mi, key
    )

    # Convert positions to FA coords: α = ζ - q(r)·θ
    psi_p, theta_p, alpha_p = salpha_to_fa_coords(
        state_sa.r, state_sa.theta, state_sa.zeta, geom
    )
    state = GCState(
        r=psi_p, theta=theta_p, zeta=alpha_p,
        vpar=state_sa.vpar, mu=state_sa.mu,
        weight=jnp.zeros(cfg.N_particles, dtype=jnp.float32),
    )

    # Seed ITG perturbation: w = ε·sin(m·θ + n·α)  (m=2, n=1)
    pert_amp = 1e-4
    pert = pert_amp * jnp.sin(2.0 * state.theta + state.zeta)
    state = state._replace(weight=pert)

    phi = jnp.zeros((Npsi, Ntheta, Nalpha))
    grid_shape = (Npsi, Ntheta, Nalpha)
    q_over_m = cfg.e / cfg.mi
    Ln = cfg.R0 / cfg.R0_over_Ln
    LT = cfg.R0 / cfg.R0_over_LT

    # Build global profiles if requested
    global_profiles = None
    if cfg.use_global:
        global_profiles = build_cbc_profiles(
            Npsi=Npsi,
            a=cfg.a,
            R0=cfg.R0,
            q0=cfg.q0,
            q1=cfg.q1,
            R0_over_LT=cfg.R0_over_LT,
            R0_over_Ln=cfg.R0_over_Ln,
            n0_avg=cfg.n0_avg,
            Ti=cfg.Ti,
        )

    diags: List[DiagnosticsFA] = []

    for step in range(cfg.n_steps):

        # 1. Scatter δf weights → δn on grid
        # In δf PIC: δn(x) = Σ_p w_p·f0(X_p)·δ(x-X_p) ≈ n0·(Σ_p w_p·δ(x-X_p)) / N_cell
        # scatter_to_grid_fa accumulates weight values per cell, normalized by N*vol.
        # The result is in units of [weight/volume] ∝ δn/n0.
        # Multiply by n0_avg to get physical δn.
        delta_n = scatter_to_grid_fa(state, geom, grid_shape) * cfg.n0_avg

        # 2. Solve GK Poisson (exact Γ₀(b))
        phi = solve_poisson_fa(
            delta_n, geom,
            cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
        )

        # 3. Gather E to particle positions
        E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state, geom)

        # 4. Push guiding centers (RK4)
        state = push_particles_fa(
            state, E_psi_p, E_theta_p, E_alpha_p,
            geom, q_over_m, cfg.mi, cfg.dt
        )

        # Radial boundary clamp (absorbing wall)
        state = state._replace(
            r=jnp.clip(state.r, geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999)
        )

        # Velocity cap (δf validity)
        state = state._replace(
            vpar=jnp.clip(state.vpar, -cfg.vpar_cap * cfg.vti, cfg.vpar_cap * cfg.vti)
        )

        # 5. Update δf weights
        B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p = interp_fa_to_particles(
            geom, state.r, state.theta, state.zeta
        )
        if cfg.use_global and global_profiles is not None:
            # Global mode: per-particle profiles from radial interpolation
            n0_p, T_p, _Te_p, q_at_r_p, d_ln_n0_dr, d_ln_T_dr = interp_profiles(
                global_profiles, state.r
            )
        else:
            # Flux-tube mode: constant-gradient profiles (original behavior)
            n0_p, T_p = _get_profiles(state.r, cfg)
            d_ln_n0_dr = jnp.full_like(state.r, -1.0 / Ln)
            d_ln_T_dr  = jnp.full_like(state.r, -1.0 / LT)
            q_at_r_p   = _interp_q(state.r, geom)

        state = update_weights(
            state,
            E_psi_p,
            E_theta_p,
            B_p,
            gradB_psi_p,
            gradB_th_p,
            kappa_psi_p,
            kappa_th_p,
            q_at_r_p,
            n0_p, T_p,
            d_ln_n0_dr, d_ln_T_dr,
            q_over_m, cfg.mi, cfg.R0, cfg.dt,
        )

        # Apply Krook damping in buffer zones (global mode only)
        if cfg.use_global and global_profiles is not None:
            state = krook_damping(state, global_profiles, cfg.dt)

        # 6. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi**2))
        phi_max = jnp.max(jnp.abs(phi))
        w_rms   = jnp.sqrt(jnp.mean(state.weight**2))
        diags.append(DiagnosticsFA(phi_rms=phi_rms, phi_max=phi_max, weight_rms=w_rms))

        if verbose and step % 50 == 0:
            print(f"  step {step:4d}/{cfg.n_steps}  |φ|_max={float(phi_max):.3e}  "
                  f"|w|_rms={float(w_rms):.3e}")

    return diags, state, phi, geom


def run_simulation_fa(
    cfg: SimConfigFA,
    key: jax.random.PRNGKey = None,
    verbose: bool = True,
) -> tuple:
    """
    Run a gyrokinetic δf simulation in s-α field-aligned coordinates.

    Builds an analytical s-α FieldAlignedGeometry from cfg parameters,
    then delegates to _run_with_geom.

    Returns (diags, state, phi, geom).
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    geom = build_field_aligned_geometry(
        Npsi=cfg.Npsi, Ntheta=cfg.Ntheta, Nalpha=cfg.Nalpha,
        R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1,
    )
    return _run_with_geom(cfg, geom, key, verbose=verbose)
