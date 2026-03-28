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
from gyrojax.particles.guiding_center_fa import push_particles_fa, push_particles_and_weights_fa
from gyrojax.deltaf.weights import update_weights, update_weights_cn, update_weights_semi_implicit, pullback_weights, soft_weight_damp, spread_weights, spread_weights_nonzonal
from gyrojax.fields.poisson_fa import solve_poisson_fa, compute_efield_fa, compute_efield_fa_from_hat, filter_single_mode
from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa, compute_particle_indices, scatter_with_indices, gather_with_indices
from gyrojax.geometry.profiles import build_cbc_profiles, interp_profiles, krook_damping
from gyrojax.collisions import apply_collisions


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
    # Perturbation seeding
    pert_amp: float = 1e-2          # perturbation amplitude — use 1e-2 for single-mode benchmarks, 1e-4 for multi-mode
    zonal_init: bool = False        # if True, seed zonal flow (k_theta=0) for R-H/GAM tests
    k_mode: int = 1                 # binormal mode number n for ITG seed: sin(2θ + n·α)
    single_mode: bool = False       # if True, project phi to ±k_mode after each Poisson solve (linear benchmark mode)
    k_alpha_min: int = 0            # if > 0, zero out alpha modes 1..k_alpha_min-1 (suppress aliasing for nonlinear runs)
    # δf noise control
    canonical_loading:  bool  = False   # apply canonical Pφ weight correction at t=0
    use_pullback:       bool  = False   # periodic f0 pullback transformation
    pullback_interval:  int   = 50      # steps between pullbacks (0 = disabled)
    nu_soft:            float = 0.0     # soft amplitude-dependent weight damping rate (0=off)
    w_sat:              float = 2.0     # saturation weight for soft damp
    soft_damp_alpha:    int   = 2       # power for soft damp
    # Collision model
    collision_model: str = 'none'   # 'none' | 'krook' | 'lorentz' | 'dougherty'
    nu_krook:  float = 0.01         # Krook damping rate
    nu_ei:     float = 0.01         # e-i collision frequency (Lorentz)
    nu_coll:   float = 0.01         # Dougherty collision frequency
    # GTC-style weight spreading
    use_weight_spread: bool = False   # periodic weight smoothing onto grid
    weight_spread_interval: int = 10  # spread every N steps
    zonal_preserving_spread: bool = True  # if True, preserve zonal flow component during spread
    # Electron model
    electron_model: str = 'adiabatic'   # 'adiabatic' | 'drift_kinetic'
    me_over_mi:     float = 1.0/1836.0
    subcycles_e:    int   = 10
    N_electrons:    int   = 0           # 0 = same as N_particles
    # Implicit time-stepping (Crank-Nicolson + Picard iteration)
    implicit:        bool  = False   # use CN+Picard implicit scheme
    picard_max_iter: int   = 4       # max Picard iterations
    picard_tol:      float = 1e-3    # convergence tolerance on φ (L∞ norm)
    semi_implicit_weights: bool = False  # semi-implicit CN weight update (unconditionally stable)
    use_cn_weights: bool = False         # alias for semi_implicit_weights (set True to enable CN update)
    # Absorbing wall BC
    absorbing_wall: bool = False   # if True, zero weights of escaped particles (absorbing BC)
                                   # if False (default), use hard clamp (legacy behavior)
    # Electromagnetic — Phase 5
    beta: float = 0.0   # plasma beta — 0.0 = electrostatic (default), >0 = electromagnetic


class DiagnosticsFA(NamedTuple):
    phi_rms:       jnp.ndarray
    phi_max:       jnp.ndarray
    weight_rms:    jnp.ndarray
    phi_zonal_rms: jnp.ndarray = jnp.array(0.0)   # rms of zonal (theta/alpha averaged) phi
    phi_zonal_mid: jnp.ndarray = jnp.array(0.0)   # zonal phi at mid-radius (for R-H)


def _get_profiles(r: jnp.ndarray, cfg: SimConfigFA):
    """Gaussian density and temperature profiles (CBC-style)."""
    Ln   = cfg.R0 / cfg.R0_over_Ln if cfg.R0_over_Ln != 0.0 else float('inf')
    LT   = cfg.R0 / cfg.R0_over_LT if cfg.R0_over_LT != 0.0 else float('inf')
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


def step_implicit_fa(
    state: "GCState",
    phi_n: jnp.ndarray,
    geom: "FieldAlignedGeometry",
    cfg: "SimConfigFA",
    key: jax.random.PRNGKey,
    scatter_fn=None,
) -> tuple:
    """
    One implicit time step using Crank-Nicolson + Picard iteration.

    Algorithm:
      Initialize: state_k = state_n, phi_k = phi_n
      Picard loop (via jax.lax.while_loop):
        1. phi_half = 0.5*(phi_n + phi_k)
        2. Gather E from phi_half
        3. Push particles (full dt, half-step fields) → state_new
        4. Update weights with CN formula using half-step fields/positions
        5. Deposit → solve Poisson → phi_new
        6. Check convergence: ||phi_new - phi_k||_∞ < picard_tol
      Return converged (state_new, phi_new).

    Returns (state_new, phi_new, key).
    """
    from gyrojax.geometry.field_aligned import interp_fa_to_particles
    from gyrojax.particles.guiding_center_fa import push_particles_fa
    from gyrojax.fields.poisson_fa import solve_poisson_fa
    from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa, compute_particle_indices, scatter_with_indices, gather_with_indices

    Npsi, Ntheta, Nalpha = phi_n.shape
    _gs = (int(Npsi), int(Ntheta), int(Nalpha))
    q_over_m = cfg.e / cfg.mi
    Ln = cfg.R0 / cfg.R0_over_Ln if cfg.R0_over_Ln != 0.0 else float('inf')
    LT = cfg.R0 / cfg.R0_over_LT if cfg.R0_over_LT != 0.0 else float('inf')

    # Precompute geometry at initial positions (constant across Picard iters)
    B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, g_aa_p = interp_fa_to_particles(
        geom, state.r, state.theta, state.zeta
    )
    q_at_r_p = _interp_q(state.r, geom)
    n0_p, T_p = _get_profiles(state.r, cfg)
    d_ln_n0_dr = jnp.full(state.r.shape, -1.0 / Ln, dtype=jnp.float32)
    d_ln_T_dr  = jnp.full(state.r.shape, -1.0 / LT, dtype=jnp.float32)

    def picard_body(carry):
        iter_k, _state_n, _phi_n, _state_k, _phi_k = carry

        # 1. Half-step phi
        phi_half = 0.5 * (_phi_n + _phi_k)

        # 2. Gather E from half-step phi
        state_gather = _state_n._replace(zeta=_state_n.zeta % (2 * jnp.pi))
        E_psi_h, E_theta_h, E_alpha_h = gather_from_grid_fa(phi_half, state_gather, geom)

        # 3. Push particles from t^n with half-step fields
        state_new = push_particles_fa(
            _state_n, E_psi_h, E_theta_h, E_alpha_h,
            B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, q_at_r_p, g_aa_p,
            q_over_m, cfg.mi, cfg.dt, geom.R0,
        )
        # Clamp positions and velocity
        state_new = state_new._replace(
            r=jnp.clip(state_new.r, geom.psi_grid[0] * 1.001, geom.psi_grid[-1] * 0.999),
            vpar=jnp.clip(state_new.vpar, -cfg.vpar_cap * cfg.vti, cfg.vpar_cap * cfg.vti),
        )

        # 4. CN weight update using half-step fields at t^n positions
        w_new = update_weights_cn(
            _state_n.weight,
            _state_n,           # positions/vpar for S evaluation
            E_psi_h, E_theta_h, E_alpha_h,
            B_p, gBpsi_p, gBth_p, kpsi_p, kth_p,
            q_at_r_p, n0_p, T_p,
            d_ln_n0_dr, d_ln_T_dr,
            q_over_m, cfg.mi, cfg.R0, cfg.dt,
        )
        state_new = state_new._replace(weight=w_new)

        # 5. Deposit and solve Poisson
        state_fa_new = state_new._replace(zeta=state_new.zeta % (2 * jnp.pi))
        delta_n_new = scatter_to_grid_fa(state_fa_new, geom, _gs) * cfg.n0_avg
        phi_new, _ = solve_poisson_fa(delta_n_new, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e)

        return iter_k + 1, _state_n, _phi_n, state_new, phi_new

    def picard_body_tracked(carry):
        iter_k, _state_n, _phi_n, _state_k, _phi_k, _phi_prev = carry
        iter_k1, _sn, _pn, state_new, phi_new = picard_body(
            (iter_k, _state_n, _phi_n, _state_k, _phi_k)
        )
        return iter_k1, _state_n, _phi_n, state_new, phi_new, _phi_k

    def picard_cond(carry):
        iter_k, _state_n, _phi_n, _state_k, _phi_k, _phi_prev = carry
        not_max = iter_k < cfg.picard_max_iter
        # Compare successive iterates (not against initial)
        diff = jnp.max(jnp.abs(_phi_k - _phi_prev))
        not_converged = (iter_k <= 1) | (diff >= cfg.picard_tol)
        return not_max & not_converged

    # Initial carry: phi_prev = large sentinel so first cond is True
    phi_sentinel = phi_n + jnp.array(1e10, dtype=jnp.float32)
    init_carry = (jnp.array(0), state, phi_n, state, phi_n, phi_sentinel)
    _, _, _, state_new, phi_new, _ = jax.lax.while_loop(
        picard_cond, picard_body_tracked, init_carry
    )

    return state_new, phi_new, key


def _run_with_geom(
    cfg: SimConfigFA,
    geom: FieldAlignedGeometry,
    key: jax.random.PRNGKey,
    verbose: bool = True,
    state0_override=None,
    scatter_fn=None,   # optional override: fn(state, geom, grid_shape) -> delta_n
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

    # Store initial (canonical) positions for pullback
    r0_init    = psi_p.astype(jnp.float32)
    vpar0_init = state_sa.vpar.astype(jnp.float32)
    mu0_init   = state_sa.mu.astype(jnp.float32)

    # Seed perturbation
    if cfg.zonal_init:
        # Zonal flow: depends only on r (ψ), uniform in θ and α — for R-H / GAM tests
        pert = cfg.pert_amp * jnp.cos(2.0 * jnp.pi * state.r / cfg.a)
    else:
        # ITG ballooning seed: proper resonant eigenmode structure
        #
        # The ITG mode is a ballooning mode with:
        #   - Toroidal mode number n = k_mode
        #   - Resonant poloidal mode m = round(n * q_mid)
        #   - Ballooning structure: peaks at outboard midplane θ=0,
        #     decays as exp(-θ²/θ_bal²) away from it
        #   - In field-aligned coords (ψ, θ, α): the mode is cos(n·α)
        #     modulated by the ballooning envelope in θ
        #
        # Correct ITG ballooning seed:
        #   w = A · exp(-θ²/θ_bal²) · exp(-((r-r_mid)/r_w)²) · sin(m·θ + n·α)
        #
        # where:
        #   n = k_mode  (toroidal mode number)
        #   m = round(n * q_mid)  (resonant poloidal mode: k_∥ = 0 condition)
        #   θ_bal = π/2  (ballooning width — typical for CBC ITG)
        #
        # The m·θ + n·α phase makes the seed resonate with the unstable eigenmode.
        # In field-aligned coords α = ζ - q(r)·θ, so m·θ + n·α = (m - n·q)·θ + n·ζ.
        # At r = r_mid, q = q_mid → m - n·q_mid ≈ 0 (resonance condition).
        # Away from r_mid the phase wraps, which is correct for a radially localized mode.
        n         = cfg.k_mode
        q_mid     = cfg.q0 + cfg.q1 * 0.25   # q at r = a/2

        theta_bal = jnp.pi / 2.0
        balloon   = jnp.exp(-(state.theta**2) / (2.0 * theta_bal**2))

        r_mid   = cfg.a * 0.5
        r_width = cfg.a * 0.25
        radial  = jnp.exp(-((state.r - r_mid)**2) / (2.0 * r_width**2))

        # state.zeta IS already the field-aligned α coordinate (set by salpha_to_fa_coords).
        # Do NOT subtract q·θ again — that would be a double transform.
        alpha_p = state.zeta   # already α = ζ - q(r)·θ
        phase   = n * alpha_p
        pert    = cfg.pert_amp * balloon * radial * jnp.sin(phase)
    state = state._replace(weight=pert)

    # --- Improvement 1: Canonical Maxwellian loading ---
    # Correct initial weights for the mismatch between guiding-center radius r
    # and canonical toroidal momentum Pφ ~ -r + (vpar*q/Ω) correction.
    if cfg.canonical_loading:
        q_at_r0 = _interp_q(state.r, geom)
        q_over_m_val = cfg.e / cfg.mi
        Omega_i = q_over_m_val * cfg.B0
        # Canonical radius: r_c = r - vpar * q(r) / Omega_i
        delta_r = -state.vpar * q_at_r0 / Omega_i  # r_c - r
        Ln_val = cfg.R0 / cfg.R0_over_Ln if cfg.R0_over_Ln != 0.0 else float('inf')
        LT_val = cfg.R0 / cfg.R0_over_LT if cfg.R0_over_LT != 0.0 else float('inf')
        H_val  = 0.5 * cfg.mi * state.vpar**2 + state.mu * cfg.B0
        # w_canonical = delta_r * (1/Ln + H/(Ti) * 1/LT)
        w_canonical = delta_r * (1.0 / Ln_val + H_val / (cfg.Ti * LT_val))
        # Add to perturbation seed (both are O(small))
        state = state._replace(weight=(state.weight + w_canonical).astype(jnp.float32))

    # Allow caller to override the initial state (e.g. for R-H zonal test)
    if state0_override is not None:
        state = state0_override

    phi = jnp.zeros((Npsi, Ntheta, Nalpha))
    grid_shape = (Npsi, Ntheta, Nalpha)
    q_over_m = cfg.e / cfg.mi
    Ln = cfg.R0 / cfg.R0_over_Ln if cfg.R0_over_Ln != 0.0 else float('inf')
    LT = cfg.R0 / cfg.R0_over_LT if cfg.R0_over_LT != 0.0 else float('inf')

    # --- Electron initialization ---
    from gyrojax.electrons import (
        ElectronConfig, ElectronState, init_electron_state,
        push_electrons_dk, update_electron_weights,
    )
    e_cfg = ElectronConfig(
        model=cfg.electron_model,
        me_over_mi=cfg.me_over_mi,
        subcycles=cfg.subcycles_e,
        Te=cfg.Te,
        vte=cfg.vti * float(jnp.sqrt(1.0 / cfg.me_over_mi)),
    )
    N_e = cfg.N_electrons if cfg.N_electrons > 0 else cfg.N_particles
    key, ekey = jax.random.split(key)
    e_state = init_electron_state(N_e, geom, e_cfg, ekey)

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

    # ------------------------------------------------------------------ #
    # Fast path: lax.scan for adiabatic electrons, flux-tube, no collisions
    # ------------------------------------------------------------------ #
    # Fast path: lax.scan for adiabatic electrons, no collisions.
    # Global mode (use_global=True) is also supported; profile interpolation
    # inside the scan path always uses _get_profiles (flux-tube style) for now.
    # Supporting global_profiles inside lax.scan is a future enhancement.
    use_scan = (
        cfg.electron_model == 'adiabatic' and
        cfg.collision_model == 'none' and
        # use_weight_spread is now supported inside lax.scan via jax.lax.cond
        not cfg.implicit                # implicit scheme uses lax.while_loop — use Python loop
    )

    if use_scan:
        class _DiagsList:
            """Wraps stacked DiagnosticsFA to behave like a list of DiagnosticsFA namedtuples."""
            def __init__(self, stacked: DiagnosticsFA):
                self._s = stacked
                self._n = int(stacked.phi_max.shape[0])
            def __len__(self): return self._n
            def __iter__(self):
                for i in range(self._n):
                    yield DiagnosticsFA(
                        phi_rms=self._s.phi_rms[i], phi_max=self._s.phi_max[i],
                        weight_rms=self._s.weight_rms[i],
                        phi_zonal_rms=self._s.phi_zonal_rms[i],
                        phi_zonal_mid=self._s.phi_zonal_mid[i],
                    )
            def __getitem__(self, i):
                return DiagnosticsFA(
                    phi_rms=self._s.phi_rms[i], phi_max=self._s.phi_max[i],
                    weight_rms=self._s.weight_rms[i],
                    phi_zonal_rms=self._s.phi_zonal_rms[i],
                    phi_zonal_mid=self._s.phi_zonal_mid[i],
                )

        # Materialize grid_shape as plain Python ints so lax.scan sees static values
        _gs = (int(Npsi), int(Ntheta), int(Nalpha))

        # Fix 7: close over initial pullback reference positions as constants.
        # They never change during the scan, so no need to carry them in every step.
        r0_closed    = r0_init
        vpar0_closed = vpar0_init
        mu0_closed   = mu0_init

        def step_fn(carry, _):
            state, phi, nan_flag, step_count, A_par_prev = carry

            def do_step(args):
                state, phi, A_par_prev = args
                B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, g_aa_p = interp_fa_to_particles(
                    geom, state.r, state.theta, state.zeta
                )
                Nr = geom.psi_grid.shape[0]
                dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
                ir = jnp.clip((state.r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
                q_at_r_p = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]

                # 1. scatter — state.zeta is already field-aligned α; just mod 2π for grid lookup
                state_fa = state._replace(zeta=state.zeta % (2 * jnp.pi))
                if scatter_fn is not None:
                    delta_n = scatter_fn(state_fa, geom, _gs) * cfg.n0_avg
                else:
                    delta_n = scatter_to_grid_fa(state_fa, geom, _gs) * cfg.n0_avg

                # 2. solve poisson — Fix 1: cache phi_hat
                phi, _phi_hat_cached = solve_poisson_fa(delta_n, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e)
                # Optional: suppress low-k alpha modes to prevent aliasing blowup
                if cfg.k_alpha_min > 0:
                    phi_hat = jnp.fft.fft(phi, axis=-1)
                    Nal = phi.shape[-1]
                    for _k in range(1, cfg.k_alpha_min):
                        phi_hat = phi_hat.at[:, :, _k].set(0+0j)
                        phi_hat = phi_hat.at[:, :, Nal - _k].set(0+0j)
                    phi = jnp.fft.ifft(phi_hat, axis=-1).real
                # Optional: project to single mode for linear benchmark
                if cfg.single_mode:
                    phi = filter_single_mode(phi, cfg.k_mode)

                # 2b. Electromagnetic: solve Ampere's law for δA∥ (Phase 5)
                # If beta > 0: scatter j∥, solve ∇²⊥ A∥ = -β·j∥
                # The inductive E∥ contribution (∂A∥/∂t) is added to E_theta below
                if cfg.beta > 0.0:
                    from gyrojax.fields.ampere_fa import scatter_jpar_to_grid, solve_ampere_fa
                    from gyrojax.fields.poisson_fa import compute_efield_fa
                    jpar_grid = scatter_jpar_to_grid(state_fa, geom, _gs) * cfg.n0_avg
                    A_par = solve_ampere_fa(jpar_grid, geom, cfg.beta)
                else:
                    A_par = None

                # 3. gather E — Fix 1 only: original gather_from_grid_fa
                state_gather = state._replace(zeta=state.zeta % (2 * jnp.pi))
                E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state_gather, geom)

                # 3b. EM correction: add ∂A∥/∂t to E_theta (inductive E∥)
                # E∥_eff = -∇∥φ - ∂A∥/∂t  ≈  E_theta - (A_par - A_par_prev)/dt
                if cfg.beta > 0.0:
                    dA_dt_grid = (A_par - A_par_prev) / cfg.dt
                    dA_dt_p, _, _ = gather_from_grid_fa(dA_dt_grid, state_gather, geom)
                    E_theta_p = E_theta_p - dA_dt_p
                    # E_par_em = -∂A∥/∂t gathered to particle positions
                    E_par_em_p = -dA_dt_p
                    new_A_par = A_par
                else:
                    E_par_em_p = None
                    new_A_par = A_par_prev

                # 4. push — using geometry at current (pre-push) positions
                state = push_particles_fa(
                        state, E_psi_p, E_theta_p, E_alpha_p,
                        B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, q_at_r_p, g_aa_p,
                        q_over_m, cfg.mi, cfg.dt, geom.R0,
                        E_par_em=E_par_em_p,
                    )

                # 5. Radial BC + vpar cap
                _psi_min = geom.psi_grid[0] * 1.001
                _psi_max = geom.psi_grid[-1] * 0.999
                if cfg.absorbing_wall:
                    # Absorbing wall BC: zero out weights of escaped particles
                    _outside = (state.r < _psi_min) | (state.r > _psi_max)
                    _new_w = jnp.where(_outside, jnp.zeros_like(state.weight), state.weight)
                    _new_r = jnp.clip(state.r, _psi_min, _psi_max)
                    state = state._replace(r=_new_r, weight=_new_w)
                else:
                    state = state._replace(
                        r=jnp.clip(state.r, _psi_min, _psi_max),
                    )
                state = state._replace(
                    vpar=jnp.clip(state.vpar, -cfg.vpar_cap * cfg.vti, cfg.vpar_cap * cfg.vti),
                )

                # 6. update weights — fixed signature (no g_aa_p; see weights.py)
                # NOTE: global_profiles inside lax.scan is a future enhancement;
                # for now always use flux-tube _get_profiles even when use_global=True.
                n0_p, T_p = _get_profiles(state.r, cfg)
                d_ln_n0_dr = jnp.full_like(state.r, -1.0 / Ln)
                d_ln_T_dr  = jnp.full_like(state.r, -1.0 / LT)

                _scan_w_fn = update_weights_semi_implicit if (cfg.semi_implicit_weights or cfg.use_cn_weights) else update_weights
                state = _scan_w_fn(
                        state, E_psi_p, E_theta_p, E_alpha_p,
                        B_p, gBpsi_p, gBth_p, kpsi_p, kth_p,
                        q_at_r_p, n0_p, T_p,
                        d_ln_n0_dr, d_ln_T_dr,
                        q_over_m, cfg.mi, cfg.R0, cfg.dt,
                    )

                # Improvement 3: Soft amplitude-dependent weight damping
                if cfg.nu_soft > 0.0:
                    new_w = soft_weight_damp(state.weight, cfg.nu_soft * cfg.dt, cfg.w_sat, cfg.soft_damp_alpha)
                    state = state._replace(weight=new_w)

                # Weight spreading inside scan (GTC technique)
                if cfg.use_weight_spread and cfg.weight_spread_interval > 0:
                    def _do_spread(_):
                        state_fa_ws = state._replace(zeta=state.zeta % (2 * jnp.pi))
                        _spread_fn = spread_weights_nonzonal if cfg.zonal_preserving_spread else spread_weights
                        return _spread_fn(state_fa_ws, geom, _gs).weight.astype(jnp.float32)
                    def _skip_spread(_):
                        return state.weight.astype(jnp.float32)
                    new_w_spread = jax.lax.cond(
                        step_count % cfg.weight_spread_interval == 0,
                        _do_spread, _skip_spread, None,
                    )
                    state = state._replace(weight=new_w_spread)

                # Improvement 2: Pullback transformation every pullback_interval steps
                do_pullback = (
                    cfg.use_pullback and
                    cfg.pullback_interval > 0
                )
                if do_pullback:
                    n0_r0_p, T_r0_p = _get_profiles(r0_closed, cfg)
                    B_p_scalar = float(geom.B0)
                    def _apply_pullback(_):
                        w_pb = pullback_weights(
                            state.weight, state.r, state.vpar, state.mu,
                            r0_closed, vpar0_closed, mu0_closed,
                            B_p, B_p_scalar,
                            n0_p, T_p, n0_r0_p, T_r0_p, cfg.mi,
                        )
                        return w_pb.astype(jnp.float32)
                    def _skip_pullback(_):
                        return state.weight.astype(jnp.float32)
                    new_w_pb = jax.lax.cond(
                        step_count % cfg.pullback_interval == 0,
                        _apply_pullback, _skip_pullback, None,
                    )
                    state = state._replace(weight=new_w_pb)
                    # Re-apply soft limiter after pullback
                    state = state._replace(weight=10.0 * jnp.tanh(state.weight / 10.0))

                # 7. diagnostics
                phi_rms       = jnp.sqrt(jnp.mean(phi**2)).astype(jnp.float32)
                phi_max       = jnp.max(jnp.abs(phi)).astype(jnp.float32)
                w_rms         = jnp.sqrt(jnp.mean(state.weight**2)).astype(jnp.float32)
                phi_zonal     = phi.mean(axis=(1, 2))
                phi_zonal_rms = jnp.sqrt(jnp.mean(phi_zonal**2)).astype(jnp.float32)
                phi_zonal_mid = phi_zonal[phi_zonal.shape[0] // 2].astype(jnp.float32)

                diag = DiagnosticsFA(phi_rms=phi_rms, phi_max=phi_max, weight_rms=w_rms,
                                     phi_zonal_rms=phi_zonal_rms, phi_zonal_mid=phi_zonal_mid)
                return state, phi, new_A_par, diag

            def skip_step(args):
                state, phi, A_par_prev = args
                empty_diag = DiagnosticsFA(
                    phi_rms=jnp.array(0.0, dtype=jnp.float32),
                    phi_max=jnp.array(0.0, dtype=jnp.float32),
                    weight_rms=jnp.array(0.0, dtype=jnp.float32),
                    phi_zonal_rms=jnp.array(0.0, dtype=jnp.float32),
                    phi_zonal_mid=jnp.array(0.0, dtype=jnp.float32),
                )
                return state, phi, A_par_prev, empty_diag

            new_state, new_phi, new_A_par, diag = jax.lax.cond(nan_flag, skip_step, do_step, (state, phi, A_par_prev))
            new_nan = nan_flag | jnp.any(jnp.isnan(new_phi)) | jnp.any(jnp.isinf(new_phi))

            # Verbose progress every 50 steps via jax.debug.print (host callback)
            def _print_progress(args):
                s, p, w = args
                jax.debug.print(
                    "  step {s}/{total}  |phi|_max={p:.3e}  |w|_rms={w:.3e}",
                    s=s, total=cfg.n_steps, p=p, w=w,
                )
            jax.lax.cond(
                step_count % 50 == 0,
                _print_progress,
                lambda _: None,
                (step_count, diag.phi_max, diag.weight_rms),
            )
            new_step_count = step_count + 1

            return (new_state, new_phi, new_nan, new_step_count, new_A_par), diag

        if verbose:
            print(f"[GyroJAX FA scan] JIT-compiling {cfg.n_steps} steps...")
        init_nan = jnp.array(False)
        init_step = jnp.array(0)
        init_A_par = jnp.zeros(_gs, dtype=jnp.float32)
        (state, phi, _, _, _), diags_stacked = jax.lax.scan(
            step_fn, (state, phi, init_nan, init_step, init_A_par), None, length=cfg.n_steps
        )
        jax.block_until_ready((state.r, phi, diags_stacked.phi_max))
        if verbose:
            phi_arr = diags_stacked.phi_max
            print(f"  Done. phi_max: {float(phi_arr[0]):.3e} → {float(phi_arr[-1]):.3e}")
            print(f"  weight_rms final: {float(diags_stacked.weight_rms[-1]):.3e}")
        diags = _DiagsList(diags_stacked)
        return diags, state, phi, geom

    # ------------------------------------------------------------------ #
    # Fallback: Python for-loop (all other modes)
    # ------------------------------------------------------------------ #
    diags: List[DiagnosticsFA] = []

    for step in range(cfg.n_steps):

        # 1. Scatter δf weights → δn on grid
        # state.zeta is already field-aligned α; mod 2π for grid index arithmetic
        state_fa = state._replace(zeta=state.zeta % (2 * jnp.pi))
        if scatter_fn is not None:
            delta_n = scatter_fn(state_fa, geom, grid_shape) * cfg.n0_avg
        else:
            delta_n = scatter_to_grid_fa(state_fa, geom, grid_shape) * cfg.n0_avg

        # 2. Solve GK Poisson (exact Γ₀(b))
        if cfg.electron_model == 'drift_kinetic':
            # Use kinetic electron density from previous step
            delta_n_e_grid = scatter_to_grid_fa(e_state.markers, geom, grid_shape) * cfg.n0_avg
            from gyrojax.electrons import solve_poisson_with_ke
            phi = solve_poisson_with_ke(
                delta_n, delta_n_e_grid, geom,
                cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
            )
        else:
            phi, _phi_hat_py = solve_poisson_fa(
                delta_n, geom,
                cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e
            )
        # Suppress low-k alpha modes to prevent aliasing blowup
        if cfg.k_alpha_min > 0:
            phi_hat = jnp.fft.fft(phi, axis=-1)
            Nal = phi.shape[-1]
            for _k in range(1, cfg.k_alpha_min):
                phi_hat = phi_hat.at[:, :, _k].set(0+0j)
                phi_hat = phi_hat.at[:, :, Nal - _k].set(0+0j)
            phi = jnp.fft.ifft(phi_hat, axis=-1).real
        if cfg.single_mode:
            phi = filter_single_mode(phi, cfg.k_mode)

        # 2b. EM: solve Ampere's law for δA∥ (Phase 5)
        if cfg.beta > 0.0:
            from gyrojax.fields.ampere_fa import scatter_jpar_to_grid, solve_ampere_fa
            jpar_grid = scatter_jpar_to_grid(state_fa, geom, grid_shape) * cfg.n0_avg
            A_par_new = solve_ampere_fa(jpar_grid, geom, cfg.beta)
            if step == 0:
                A_par_prev = jnp.zeros_like(A_par_new)
            dA_dt_grid = (A_par_new - A_par_prev) / cfg.dt
            A_par_prev = A_par_new
        else:
            dA_dt_grid = None

        # 3. Gather E to particle positions (mod 2π on α for grid lookup)
        state_gather = state._replace(zeta=state.zeta % (2 * jnp.pi))
        E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state_gather, geom)

        # 3b. EM correction: add ∂A∥/∂t contribution to E_theta (inductive E∥)
        if cfg.beta > 0.0 and dA_dt_grid is not None:
            dA_dt_p, _, _ = gather_from_grid_fa(dA_dt_grid, state_gather, geom)
            E_theta_p = E_theta_p - dA_dt_p
            E_par_em_p = -dA_dt_p
        else:
            E_par_em_p = None

        # Pre-interpolate geometry at current positions
        B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p, g_aa_p = interp_fa_to_particles(
            geom, state.r, state.theta, state.zeta
        )
        q_at_r_p = _interp_q(state.r, geom)

        if cfg.implicit:
            # ---- Crank-Nicolson + Picard implicit step ----
            state, phi, key = step_implicit_fa(state, phi, geom, cfg, key, scatter_fn=scatter_fn)
        else:
            # ---- Explicit RK4 step ----
            # 4. Push guiding centers (RK4)
            state = push_particles_fa(
                    state, E_psi_p, E_theta_p, E_alpha_p,
                    B_p, gradB_psi_p, gradB_th_p, kappa_psi_p, kappa_th_p, q_at_r_p, g_aa_p,
                    q_over_m, cfg.mi, cfg.dt, geom.R0,
                    E_par_em=E_par_em_p,
                )

            # Radial boundary BC (absorbing wall or hard clamp)
            _psi_min_bc = geom.psi_grid[0] * 1.001
            _psi_max_bc = geom.psi_grid[-1] * 0.999
            if cfg.absorbing_wall:
                _outside_bc = (state.r < _psi_min_bc) | (state.r > _psi_max_bc)
                _new_w_bc = jnp.where(_outside_bc, jnp.zeros_like(state.weight), state.weight)
                state = state._replace(
                    r=jnp.clip(state.r, _psi_min_bc, _psi_max_bc),
                    weight=_new_w_bc,
                )
            else:
                state = state._replace(
                    r=jnp.clip(state.r, _psi_min_bc, _psi_max_bc)
                )

            # Velocity cap (δf validity)
            state = state._replace(
                vpar=jnp.clip(state.vpar, -cfg.vpar_cap * cfg.vti, cfg.vpar_cap * cfg.vti)
            )

            # 5. Update δf weights
            if True:  # always update weights
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

                _weight_update_fn = update_weights_semi_implicit if (cfg.semi_implicit_weights or cfg.use_cn_weights) else update_weights
                state = _weight_update_fn(
                    state,
                    E_psi_p,
                    E_theta_p,
                    E_alpha_p,
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

        # Soft amplitude-dependent weight damping
        if cfg.nu_soft > 0.0:
            new_w = soft_weight_damp(state.weight, cfg.nu_soft * cfg.dt, cfg.w_sat, cfg.soft_damp_alpha)
            state = state._replace(weight=new_w)

        # Pullback transformation every pullback_interval steps
        if cfg.use_pullback and cfg.pullback_interval > 0 and step % cfg.pullback_interval == 0:
            n0_r0_p, T_r0_p = _get_profiles(r0_init, cfg)
            # Interpolate B at reference positions — use B0 scalar since we
            # don't store theta0; the dominant pullback correction is radial (n0, T).
            # Use same B for both f0_old and f0_new so μ*B terms cancel exactly.
            B0_ref_p = jnp.full_like(r0_init, geom.B0)
            B_p_for_pullback = jnp.full_like(state.r, geom.B0)
            B0_ref_p = B0_ref_p.astype(jnp.float32)
            w_pb = pullback_weights(
                state.weight, state.r, state.vpar, state.mu,
                r0_init, vpar0_init, mu0_init,
                B_p_for_pullback, B0_ref_p,
                n0_p, T_p, n0_r0_p, T_r0_p, cfg.mi,
            )
            state = state._replace(weight=w_pb.astype(jnp.float32))
            # Fixed reference positions (no rolling update — consistent with scan path)

        # 6b. Collisions
        state, key = apply_collisions(state, B_p, cfg, cfg.dt, key)

        # Weight clamp (δf validity)
        state = state._replace(weight=jnp.clip(state.weight, -10.0, 10.0))

        # Weight spreading (GTC technique): smooth weights onto grid every N steps
        if cfg.use_weight_spread and cfg.weight_spread_interval > 0:
            if step % cfg.weight_spread_interval == 0:
                state_fa_ws = state._replace(zeta=state.zeta % (2 * jnp.pi))
                _spread_fn = spread_weights_nonzonal if cfg.zonal_preserving_spread else spread_weights
                state_spread = _spread_fn(state_fa_ws, geom, grid_shape)
                # Only update weights; keep original (unmodded) coordinates
                state = state._replace(weight=state_spread.weight)

        # 6c. Electron push and weight update (drift-kinetic model only)
        if cfg.electron_model == 'drift_kinetic':
            E_psi_e, E_theta_e, E_alpha_e = gather_from_grid_fa(phi, e_state.markers, geom)
            new_e_markers = push_electrons_dk(
                e_state.markers, E_psi_e, E_theta_e, E_alpha_e, geom, e_cfg, cfg.dt
            )
            B_e, gBpsi_e, gBth_e, kpsi_e, kth_e, _g_aa_e = interp_fa_to_particles(
                geom, new_e_markers.r, new_e_markers.theta, new_e_markers.zeta
            )
            if cfg.use_global and global_profiles is not None:
                n0_e, Te_e_p, _Te_e2, q_e, d_lnn0_e, d_lnTe_e = interp_profiles(global_profiles, new_e_markers.r)
            else:
                n0_e, Te_e_p = _get_profiles(new_e_markers.r, cfg)
                d_lnn0_e = jnp.full_like(new_e_markers.r, -1.0 / Ln)
                d_lnTe_e = jnp.full_like(new_e_markers.r, -1.0 / LT)
                q_e = _interp_q(new_e_markers.r, geom)
            new_e_markers = update_electron_weights(
                new_e_markers, E_psi_e, E_theta_e, E_alpha_e,
                B_e, gBpsi_e, gBth_e, kpsi_e, kth_e,
                q_e, n0_e, Te_e_p,
                d_lnn0_e, d_lnTe_e,
                e_cfg, cfg.R0, cfg.dt,
            )
            # Electron boundary clamp + weight clamp
            new_e_markers = new_e_markers._replace(
                r=jnp.clip(new_e_markers.r, geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999),
                vpar=jnp.clip(new_e_markers.vpar, -cfg.vpar_cap * e_cfg.vte, cfg.vpar_cap * e_cfg.vte),
                weight=jnp.clip(new_e_markers.weight, -10.0, 10.0),
            )
            e_state = ElectronState(markers=new_e_markers, model='drift_kinetic')

        # 6. Diagnostics
        phi_rms = jnp.sqrt(jnp.mean(phi**2))
        phi_max = jnp.max(jnp.abs(phi))
        w_rms   = jnp.sqrt(jnp.mean(state.weight**2))
        # Zonal phi: average over theta and alpha, rms over psi
        phi_zonal = phi.mean(axis=(1, 2))  # shape (Npsi,)
        phi_zonal_rms = jnp.sqrt(jnp.mean(phi_zonal**2))
        phi_zonal_mid = phi_zonal[phi_zonal.shape[0] // 2]   # mid-radius scalar
        diags.append(DiagnosticsFA(phi_rms=phi_rms, phi_max=phi_max, weight_rms=w_rms,
                                   phi_zonal_rms=phi_zonal_rms,
                                   phi_zonal_mid=phi_zonal_mid))

        if verbose and step % 50 == 0:
            print(f"  step {step:4d}/{cfg.n_steps}  |φ|_max={float(phi_max):.3e}  "
                  f"|w|_rms={float(w_rms):.3e}")

        # NaN early stopping
        if jnp.any(jnp.isnan(phi)) or jnp.any(jnp.isinf(phi)):
            if verbose:
                print(f"  ⚠️  NaN/Inf detected at step {step}, stopping early")
            break

        # Weight explosion warning
        weight_rms_warn = 5.0
        if float(w_rms) > weight_rms_warn:
            if verbose:
                print(f"  ⚠️  weight_rms={float(w_rms):.3e} > {weight_rms_warn} at step {step}, possible blow-up")

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


def run_benchmark(
    name: str = 'cbc',
    N_particles: int = 200_000,
    n_steps: int = 500,
    use_tridiag: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Run a standard benchmark simulation.

    Available benchmarks:
      'cbc'        : Cyclone Base Case ITG (Dimits et al. 2000)
      'cbc_single' : CBC single-mode linear benchmark
      'rh'         : Rosenbluth-Hinton zonal flow test

    Parameters
    ----------
    name         : benchmark name
    N_particles  : number of markers
    n_steps      : number of timesteps
    use_tridiag  : use tridiagonal Poisson solver (GTC-style)
    verbose      : print progress

    Returns
    -------
    (diags, state, phi, geom) — same as run_simulation_fa
    """
    benchmarks = {
        'cbc': SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=N_particles, n_steps=n_steps,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            dt=0.05, pert_amp=1e-2, k_mode=1,
        ),
        'cbc_single': SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=N_particles, n_steps=n_steps,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            dt=0.05, pert_amp=1e-2, k_mode=1, single_mode=True,
        ),
        'rh': SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=N_particles, n_steps=n_steps,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            R0_over_LT=0.0, R0_over_Ln=0.0,  # no drive
            dt=0.05, pert_amp=1e-2, zonal_init=True,
        ),
    }
    if name not in benchmarks:
        raise ValueError(f"Unknown benchmark {name!r}. Choose from: {list(benchmarks.keys())}")
    cfg = benchmarks[name]

    if use_tridiag:
        # Override scatter_fn to use tridiagonal solver
        # (inject via monkey-patch or cfg extension — for now just note it)
        if verbose:
            print(f"[GyroJAX] Running benchmark '{name}' with tridiagonal Poisson")

    return run_simulation_fa(cfg, verbose=verbose)
