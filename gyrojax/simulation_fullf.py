"""
Full-f gyrokinetic PIC simulation — Phase 3 (true Vlasov PIC).

True full-f: each marker carries a CONSTANT weight W_p (characteristic of Vlasov).
Physics comes entirely from particle motion — not weight evolution.

Key differences from δf (simulation_fa.py):
  - NO weight equation (dW/dt = 0)
  - δn(x) = Σ_p W_p · shape(x-X_p) / ΔV  −  n₀(x)
  - Initial perturbation seeded via one-time weight or position perturbation
  - Periodic resampling to control weight variance growth (from phase-space sampling noise)

In the linear phase, full-f and δf give the same growth rate. Differences appear
nonlinearly (zonal flows, profile relaxation).

References:
  Grandgirard et al. (2006) J. Comput. Phys. 217, 395  (GYSELA)
  Idomura et al. (2008) Nucl. Fusion 48, 035002  (GT5D)
  Lin & Lee (1995) Phys. Rev. E 52, 5646
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
from gyrojax.fields.poisson_fa import solve_poisson_fa, filter_single_mode
from gyrojax.geometry.salpha import build_salpha_geometry


@dataclass
class SimConfigFullF:
    """Simulation configuration for full-f (Phase 3) run."""
    # Grid
    Npsi:   int = 16
    Ntheta: int = 32
    Nalpha: int = 32
    # Particles
    N_particles: int = 200_000
    # Time
    n_steps: int = 200
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
    rho_star: float = 1.0 / 180.0
    e:   float = 1000.0
    vti: float = 1.0
    n0_avg: float = 1.0
    # Profiles
    R0_over_LT: float = 6.9
    R0_over_Ln: float = 2.2
    # ITG seed
    pert_amp: float = 1e-4
    # Velocity cap
    vpar_cap: float = 4.0
    # Resampling (0 = never)
    resample_interval: int = 50
    # Single-mode filter (linear benchmark)
    single_mode: bool = False
    k_mode: int = 18
    # Collisions (Krook)
    nu_krook: float = 0.0
    # Minimum alpha wavenumber to keep
    k_alpha_min: int = 0
    # Warmup steps with reduced perturbation
    n_steps_warmup: int = 0
    # Electromagnetic: plasma beta (0.0 = electrostatic)
    beta: float = 0.0


class DiagnosticsFullF(NamedTuple):
    phi_rms:       jnp.ndarray
    phi_max:       jnp.ndarray
    weight_rms:    jnp.ndarray   # should stay constant in true full-f
    n_rms:         jnp.ndarray
    phi_zonal_rms: jnp.ndarray


def _resample_particles(
    state: GCState,
    cfg: SimConfigFullF,
    key: jax.random.PRNGKey,
) -> tuple:
    """
    Systematic resampling: replace high-variance weights with equal-weight markers.

    Standard particle resampling (Gingold & Monaghan style):
      1. Normalize weights → CDF
      2. Draw uniform random offsets
      3. Resample indices proportional to weight
      4. All resampled markers get equal weight W̄ = mean(|W|)

    After resampling: std(W) → 0, positions distributed ∝ |W|.
    """
    N = cfg.N_particles
    W = jnp.abs(state.weight)
    W_total = jnp.sum(W) + 1e-30
    W_norm = W / W_total

    # CDF-based systematic resampling
    cdf = jnp.cumsum(W_norm)
    key, subkey = jax.random.split(key)
    u0 = jax.random.uniform(subkey, shape=()) / N
    u = u0 + jnp.arange(N, dtype=jnp.float32) / N
    # Clamp to [0, 1) for safety
    u = jnp.clip(u, 0.0, 1.0 - 1e-7)
    indices = jnp.searchsorted(cdf, u, side='right')
    indices = jnp.clip(indices, 0, N - 1)

    # All resampled markers get equal weight (mean of original)
    W_mean = W_total / N
    new_w = jnp.full(N, W_mean, dtype=jnp.float32)

    new_state = GCState(
        r=state.r[indices], theta=state.theta[indices],
        zeta=state.zeta[indices], vpar=state.vpar[indices],
        mu=state.mu[indices], weight=new_w,
    )
    return new_state, key


def _interp_q(r: jnp.ndarray, geom: FieldAlignedGeometry) -> jnp.ndarray:
    Nr = geom.psi_grid.shape[0]
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
    ir = jnp.clip((r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
    i0 = jnp.floor(ir).astype(jnp.int32)
    return geom.q_profile[i0]


def run_simulation_fullf(
    cfg: SimConfigFullF,
    geom: Optional[FieldAlignedGeometry] = None,
    key: jax.random.PRNGKey = None,
    verbose: bool = True,
) -> tuple:
    """
    Run full-f gyrokinetic simulation.

    True Vlasov PIC: weights constant, physics from particle motion.

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
        print(f"  True full-f: dW/dt=0, δn=scatter(W)-n0")

    # --- Build geometry for initial particle sampling ---
    geom_sa = build_salpha_geometry(
        Npsi, Ntheta, Nalpha,
        R0=cfg.R0, a=cfg.a, B0=cfg.B0,
        q0=float(geom.q_profile[Npsi // 2]), q1=0.0,
    )

    # --- Initialize particles from Maxwellian f0 ---
    key, subkey = jax.random.split(key)
    state_sa = init_maxwellian_particles(
        cfg.N_particles, geom_sa, cfg.vti, cfg.Ti, cfg.mi, subkey
    )
    # Transform to field-aligned coordinates
    psi_p, theta_p, alpha_p = salpha_to_fa_coords(
        state_sa.r, state_sa.theta, state_sa.zeta, geom
    )

    # --- Full-f: uniform weights W_p = n0_avg ---
    # Since particles are sampled from f0, equal weights → correct n0
    # scatter_to_grid_fa normalizes so W_p=1 → delta_n ≈ 1 everywhere
    # → background n0 = n0_avg, so W_p = n0_avg for consistency
    W0 = jnp.full(cfg.N_particles, cfg.n0_avg, dtype=jnp.float32)

    # --- Seed ITG mode via one-time weight perturbation (t=0 only) ---
    # After this, weights are CONSTANT (true full-f)
    pert = cfg.pert_amp * jnp.sin(2.0 * theta_p + jnp.float32(cfg.k_mode) * alpha_p)
    W_init = W0 * (1.0 + pert)

    state = GCState(
        r=psi_p, theta=theta_p, zeta=alpha_p,
        vpar=state_sa.vpar, mu=state_sa.mu,
        weight=W_init.astype(jnp.float32),
    )

    # Background density: n0_avg (flat in flux-tube approx)
    n0_background = jnp.float32(cfg.n0_avg)

    q_over_m = cfg.e / cfg.mi
    phi = jnp.zeros(grid_shape, dtype=jnp.float32)
    A_par_prev = jnp.zeros(grid_shape, dtype=jnp.float32)
    diags: List[DiagnosticsFullF] = []

    for step in range(cfg.n_steps):

        # 1. Scatter W_p → n(x), compute δn = n - n0
        #    scatter_to_grid_fa normalizes: W_p=n0_avg → output ≈ n0_avg
        n_scatter = scatter_to_grid_fa(state, geom, grid_shape)
        # delta_n: the fluctuation from background
        delta_n = n_scatter - n0_background

        # NaN guard
        if bool(jnp.any(jnp.isnan(delta_n))):
            if verbose:
                print(f"  [WARN] NaN in delta_n at step {step}, resetting to zero")
            delta_n = jnp.zeros_like(delta_n)

        # 2. Solve GK Poisson: δn → φ
        phi, _ = solve_poisson_fa(
            delta_n, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
        )

        # Optional single-mode filter (linear benchmark)
        if cfg.single_mode:
            phi = filter_single_mode(phi, cfg.k_mode)

        # NaN guard on phi
        if bool(jnp.any(jnp.isnan(phi))):
            if verbose:
                print(f"  [WARN] NaN in phi at step {step}, zeroing")
            phi = jnp.zeros_like(phi)

        # 3. Gather E to particles
        E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state, geom)

        # 3b. EM: Ampere solve for A∥, add inductive E∥ = -∂A∥/∂t
        E_par_em_p = None
        if cfg.beta > 0.0:
            from gyrojax.fields.ampere_fa import scatter_jpar_to_grid, solve_ampere_fa
            jpar_grid = scatter_jpar_to_grid(state, geom, grid_shape) * cfg.n0_avg
            A_par = solve_ampere_fa(jpar_grid, geom, cfg.beta)
            dA_dt_grid = (A_par - A_par_prev) / cfg.dt
            dA_dt_p, _, _ = gather_from_grid_fa(dA_dt_grid, state, geom)
            E_theta_p = E_theta_p - dA_dt_p
            E_par_em_p = -dA_dt_p
            A_par_prev = A_par

        # 4. Get B/gradB/curvature at particle positions
        B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p, g_aa_p = \
            interp_fa_to_particles(geom, state.r, state.theta, state.zeta)

        # 5. Push GC (RK4) — same as δf pusher
        q_at_p = _interp_q(state.r, geom)
        state = push_particles_fa(
            state, E_psi_p, E_theta_p, E_alpha_p,
            B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p,
            q_at_p, g_aa_p, q_over_m, cfg.mi, cfg.dt, cfg.R0,
            E_par_em=E_par_em_p,
        )

        # 6. Boundary conditions + velocity clamp
        state = state._replace(
            r=jnp.clip(state.r, geom.psi_grid[0] * 1.001, geom.psi_grid[-1] * 0.999),
            vpar=jnp.clip(state.vpar, -cfg.vpar_cap * cfg.vti, cfg.vpar_cap * cfg.vti),
        )

        # NOTE: NO weight update — true full-f: dW_p/dt = 0
        # The weights remain constant after initialization.

        # 7. Optional Krook collisions (relax towards Maxwellian via vpar damping)
        if cfg.nu_krook > 0.0:
            nu = jnp.float32(cfg.nu_krook * cfg.dt)
            state = state._replace(
                vpar=state.vpar * (1.0 - nu),
            )

        # 8. Periodic resampling to control weight variance
        if cfg.resample_interval > 0 and step > 0 and step % cfg.resample_interval == 0:
            state, key = _resample_particles(state, cfg, key)

        # 9. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi ** 2))
        phi_max = jnp.max(jnp.abs(phi))
        n_rms   = jnp.sqrt(jnp.mean(delta_n ** 2))
        weight_rms = jnp.sqrt(jnp.mean(state.weight ** 2))
        # Zonal phi: average over theta and alpha
        phi_zonal = jnp.mean(phi, axis=(1, 2))  # (Npsi,)
        phi_zonal_rms = jnp.sqrt(jnp.mean(phi_zonal ** 2))

        diags.append(DiagnosticsFullF(
            phi_rms=phi_rms, phi_max=phi_max,
            weight_rms=weight_rms, n_rms=n_rms,
            phi_zonal_rms=phi_zonal_rms,
        ))

        if verbose and step % 50 == 0:
            w_rms = float(weight_rms)
            print(f"  step {step:4d}/{cfg.n_steps}  "
                  f"|φ|_max={float(phi_max):.3e}  |w|_rms={w_rms:.3e}  "
                  f"δn_rms={float(n_rms):.3e}")

    return diags, state, phi, geom
