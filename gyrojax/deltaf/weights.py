"""
Delta-f weight evolution for gyrokinetic PIC.

The delta-f method decomposes the distribution function as:
    f = f0 + δf = f0(1 + w)
where f0 is a known Maxwellian background and w = δf/f0 is the marker weight.

Weight equation (GK Vlasov):
    dw/dt = -(1-w) * [(vE + vd) · ∇ln(f0) + (q/m)·E∥ · ∂ln(f0)/∂v∥]

where:
  vE  = E×B/B²          (E×B drift — from perturbation φ)
  vd  = v_∇B + v_curv   (magnetic drifts — equilibrium, drives ITG)

With Maxwellian f0(r, v∥, μ):
    ln(f0) = ln(n0(r)) - H/T(r) + const
    H = m*v∥²/2 + μ*B(r,θ)
    ∂ln(f0)/∂r  = d_ln_n0/dr - μ*(∂B/∂r)/T - H*(d_ln_T/dr)/T
    ∂ln(f0)/∂v∥ = -m*v∥/T

The magnetic drift term vd·∂ln(f0)/∂r is the ITG drive:
  vd ∝ ∇T/T × (v∥² + μB/m)  → couples to d_ln_T/dr = -R0/LT

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
    """Evaluate background Maxwellian f0 at particle positions."""
    vt = jnp.sqrt(2.0 * T / mi)
    H  = 0.5 * mi * vpar**2 + mu * B
    return n0 / (jnp.pi**1.5 * vt**3) * jnp.exp(-H / T)


def log_f0_gradients(
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    mi: float,
) -> tuple:
    """
    Compute gradients of ln(f0) for the weight evolution.

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


def _compute_drift_r(
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    gradB_th: jnp.ndarray,
    kappa_r: jnp.ndarray,
    kappa_th: jnp.ndarray,
    r: jnp.ndarray,
    q_at_r: jnp.ndarray,
    R0: float,
    q_over_m: float,
    mi: float,
) -> jnp.ndarray:
    """
    Radial magnetic drift velocity vd_r = v_gradB_r + v_curv_r.

    Standard GC drift (in physical units):
      v_gradB = (μ/mΩ) · (b̂ × ∇B)/B
      v_curv  = (v∥²/Ω) · (b̂ × κ)

    Radial (ψ) component in field-aligned / s-α coords:
      vd_r = -(μ/mΩ)·gradB_th/R  - (v∥²/Ω)·kappa_th/R
    where R ~ q*R0/r * r = q*R0 is the connection length scale.

    With kappa_th = sin(θ)/R(r) and gradB_th ~ sin(θ)·ε/R0, the
    `/R0` is already encoded in the geometry arrays; no extra 1/r factor.
    """
    Omega = q_over_m * B                              # cyclotron frequency
    # Safe division: preserve sign of Omega (negative for electrons), avoid |Omega|→0
    Omega_safe = jnp.sign(Omega) * jnp.maximum(jnp.abs(Omega), 1e-10)

    # ∇B drift radial component: -(μ/mΩ) * ∂|B|/∂θ_physical
    # gradB_th has units of B/m (physical gradient, not normalized)
    v_gradB_r = -(mu / (mi * Omega_safe)) * gradB_th

    # Curvature drift radial component: -(v∥²/Ω) * κ_θ
    # kappa_th has units of 1/m (physical curvature, not normalized)
    v_curv_r  = -(vpar**2 / Omega_safe) * kappa_th

    return v_gradB_r + v_curv_r


def _weight_rhs(
    w: jnp.ndarray,
    vdrift_r: jnp.ndarray,   # total radial drift: vE_r + vd_r
    dvpar_dt: jnp.ndarray,   # E∥-driven parallel force (≈0 for perp modes)
    d_lnf0_dr: jnp.ndarray,
    d_lnf0_dvp: jnp.ndarray,
) -> jnp.ndarray:
    """
    dw/dt = -(1-w) * [v_total_r * d_lnf0_dr + dvpar_dt * d_lnf0_dvp]
    """
    return -(1.0 - w) * (vdrift_r * d_lnf0_dr + dvpar_dt * d_lnf0_dvp)


@jax.jit
def update_weights(
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    gradB_th: jnp.ndarray,
    kappa_r: jnp.ndarray,
    kappa_th: jnp.ndarray,
    q_at_r: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    q_over_m: float,
    mi: float,
    R0: float,
    dt: float,
) -> GCState:
    """
    Advance delta-f weights by one timestep using RK4.

    Includes both E×B and magnetic drift drives (∇B + curvature).
    The magnetic drift term is essential for ITG excitation.

    Parameters
    ----------
    state          : GCState with current positions/velocities
    E_r, E_theta, E_alpha : E-field components at particle positions (N,)
    B, gradB_r, gradB_th, kappa_r, kappa_th : geometry at particles (N,)
    q_at_r         : safety factor at particle r (N,)
    n0, T          : background profiles at particle positions (N,)
    d_ln_n0_dr     : d(ln n0)/dr (N,)
    d_ln_T_dr      : d(ln T)/dr  (N,)
    q_over_m       : e/m
    mi             : ion mass
    R0             : major radius
    dt             : timestep
    """
    # ExB radial drive: v_ExB^ψ = q(ψ) * E_α / B
    # In FA coords (α = ζ - q·θ): ∂φ/∂θ|_{ψ,ζ} = -q·∂φ/∂α → E_θ^phys = -q·E_α^FA
    # So v_ExB^ψ = -E_θ^phys/B = q * E_α / B
    safe_B = jnp.maximum(B, 1e-10)
    vE_r = q_at_r * E_alpha / safe_B

    # Equilibrium magnetic drift (drives ITG!)
    vd_r = _compute_drift_r(
        state.vpar, state.mu, B, gradB_r, gradB_th,
        kappa_r, kappa_th, state.r, q_at_r, R0, q_over_m, mi
    )

    # Total radial drift in weight equation
    v_total_r = vE_r + vd_r

    # E∥ in field-aligned coords: the guiding-center pusher neglects E∥
    # (perpendicular modes, E∥ ≈ 0).  Using E_theta here as a proxy for E∥
    # is INCONSISTENT with the pusher and causes spurious amplification
    # proportional to q_over_m (= e = 1000 for high-e runs).  For zonal
    # initial conditions the exact E∥ = 0, but numerical noise in E_theta
    # gets multiplied by 1000, blowing up the weights.  Set dvpar_dt_ES = 0
    # to match the pusher (which already omits E∥).
    dvpar_dt_ES = jnp.zeros_like(E_theta)   # E∥ ≈ 0; keep consistent with pusher

    d_lnf0_dr, d_lnf0_dvp = log_f0_gradients(
        state.vpar, state.mu, B, gradB_r,
        T, d_ln_n0_dr, d_ln_T_dr, mi
    )

    # RK4 integration
    k1 = _weight_rhs(state.weight, v_total_r, dvpar_dt_ES, d_lnf0_dr, d_lnf0_dvp)
    k2 = _weight_rhs(state.weight + 0.5*dt*k1, v_total_r, dvpar_dt_ES, d_lnf0_dr, d_lnf0_dvp)
    k3 = _weight_rhs(state.weight + 0.5*dt*k2, v_total_r, dvpar_dt_ES, d_lnf0_dr, d_lnf0_dvp)
    k4 = _weight_rhs(state.weight + dt*k3,      v_total_r, dvpar_dt_ES, d_lnf0_dr, d_lnf0_dvp)

    new_weight = state.weight + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Soft weight limiter: tanh-based, differentiable
    # w_max=10: safe for GPU; linear phase has |w|<0.5, well below this limit
    # but prevents NaN blow-up in deep nonlinear regime
    w_max = 10.0
    new_weight = w_max * jnp.tanh(new_weight / w_max)

    return GCState(
        r=state.r, theta=state.theta, zeta=state.zeta,
        vpar=state.vpar, mu=state.mu, weight=new_weight
    )


def update_weights_semi_implicit(
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    gradB_th: jnp.ndarray,
    kappa_r: jnp.ndarray,
    kappa_th: jnp.ndarray,
    q_at_r: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    q_over_m: float,
    mi: float,
    R0: float,
    dt: float,
) -> GCState:
    """
    Semi-implicit weight update using Crank-Nicolson analytical solution.

    For dw/dt = -(1-w)*S, the exact CN solution is:
        w^{n+1} = 1 - (1 - w^n) / (1 + dt * S)

    S is evaluated at current (explicit) positions and fields.
    This is unconditionally stable in w: as dt*S → ∞, w → 1 (hard bounded).
    Compare to explicit Euler: w^{n+1} = w^n - dt*(1-w^n)*S which blows up.

    Key property: the denominator (1 + dt*S) damps any S no matter how large,
    so large ExB drives (large phi) cannot send w to infinity.
    """
    safe_B = jnp.maximum(B, 1e-10)
    vE_r = q_at_r * E_alpha / safe_B

    vd_r = _compute_drift_r(
        state.vpar, state.mu, B, gradB_r, gradB_th,
        kappa_r, kappa_th, state.r, q_at_r, R0, q_over_m, mi
    )

    v_total_r = vE_r + vd_r
    dvpar_dt_ES = jnp.zeros_like(E_theta)  # E∥ ≈ 0, consistent with pusher

    d_lnf0_dr, d_lnf0_dvp = log_f0_gradients(
        state.vpar, state.mu, B, gradB_r,
        T, d_ln_n0_dr, d_ln_T_dr, mi
    )

    # S = the source term: dw/dt = -(1-w)*S
    S = (v_total_r * d_lnf0_dr + dvpar_dt_ES * d_lnf0_dvp).astype(jnp.float32)
    dt32 = jnp.array(dt, dtype=jnp.float32)

    # CN analytical solution per-particle: w = 1 - (1 - w_n)/(1 + dt*S)
    denom = jnp.array(1.0, dtype=jnp.float32) + dt32 * S
    # Guard: if denom < 0.1 (dt*S ≈ -1), clamp — large negative S is physical damping.
    denom_safe = jnp.where(denom > 0.1, denom, jnp.array(0.1, dtype=jnp.float32))
    new_weight = jnp.array(1.0, dtype=jnp.float32) - (jnp.array(1.0, dtype=jnp.float32) - state.weight) / denom_safe

    # Safety clip — should rarely trigger with CN
    new_weight = jnp.clip(new_weight, -10.0, 10.0)

    return GCState(
        r=state.r, theta=state.theta, zeta=state.zeta,
        vpar=state.vpar, mu=state.mu, weight=new_weight.astype(state.weight.dtype)
    )


def update_weights_cn(
    w_n: jnp.ndarray,
    state: GCState,
    E_r: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    B: jnp.ndarray,
    gradB_r: jnp.ndarray,
    gradB_th: jnp.ndarray,
    kappa_r: jnp.ndarray,
    kappa_th: jnp.ndarray,
    q_at_r: jnp.ndarray,
    n0: jnp.ndarray,
    T: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_T_dr: jnp.ndarray,
    q_over_m: float,
    mi: float,
    R0: float,
    dt: float,
) -> jnp.ndarray:
    """
    Crank-Nicolson weight update (analytically solved per-particle).

    For the weight equation dw/dt = -(1-w)*S, the CN scheme gives:
        w^{n+1} = 1 - (1 - w^n) / (1 + dt * S^{n+½})

    where S = vdrift_r * d_lnf0_dr is evaluated with half-step fields
    at the initial particle positions.

    This is stable: as dt*S → ∞, w → 1 (bounded above by 1).
    Safety clip to [-10, 10] prevents blow-up in the negative direction.

    Parameters mirror update_weights; fields should be at the half-step
    (i.e. averaged between t^n and the current Picard guess for t^{n+1}).
    """
    safe_B = jnp.maximum(B, 1e-10)
    vE_r = q_at_r * E_alpha / safe_B

    vd_r = _compute_drift_r(
        state.vpar, state.mu, B, gradB_r, gradB_th,
        kappa_r, kappa_th, state.r, q_at_r, R0, q_over_m, mi,
    )

    v_total_r = vE_r + vd_r
    dvpar_dt_ES = jnp.zeros_like(E_theta)  # E∥ ≈ 0 (consistent with pusher)

    d_lnf0_dr, d_lnf0_dvp = log_f0_gradients(
        state.vpar, state.mu, B, gradB_r, T, d_ln_n0_dr, d_ln_T_dr, mi
    )

    # S: weight-independent part of RHS/(-(1-w))
    S = v_total_r * d_lnf0_dr + dvpar_dt_ES * d_lnf0_dvp

    # CN analytical solution: w^{n+1} = 1 - (1 - w^n) / (1 + dt*S)
    denom = 1.0 + dt * S
    # Avoid division by zero or sign flip (denom near zero → very large step)
    denom_safe = jnp.where(jnp.abs(denom) > 1e-6, denom, jnp.sign(denom + 1e-30) * 1e-6)
    w_new = 1.0 - (1.0 - w_n) / denom_safe

    # Safety clip: prevents unbounded growth in negative direction
    w_new = jnp.clip(w_new, -10.0, 10.0)
    return w_new.astype(w_n.dtype)


def pullback_weights(
    weight: jnp.ndarray,
    r: jnp.ndarray,
    vpar: jnp.ndarray,
    mu: jnp.ndarray,
    r0: jnp.ndarray,
    vpar0: jnp.ndarray,
    mu0: jnp.ndarray,
    B: jnp.ndarray,
    B0: jnp.ndarray,
    n0_r: jnp.ndarray,
    T_r: jnp.ndarray,
    n0_r0: jnp.ndarray,
    T_r0: jnp.ndarray,
    mi: float,
) -> jnp.ndarray:
    """
    Periodic pullback (f0 update) to prevent δf/f0 blow-up.

    Adjusts weights so that f = f0_new * (1 - w_new) = f0_old * (1 - w_old).
    w_new = 1 - (f0_old/f0_new) * (1 - w_old)

    B0 should be the interpolated B at the reference positions r0 (not a scalar),
    so the μ*B term is consistent between f0_old and f0_new.
    """
    f0_new = maxwellian_f0(vpar, mu, B, n0_r, T_r, mi)
    f0_old = maxwellian_f0(vpar0, mu0, B0, n0_r0, T_r0, mi)
    # Use log-ratio for numerical stability
    log_ratio = jnp.log(f0_old + 1e-300) - jnp.log(f0_new + 1e-300)
    # Clip log-ratio: allow at most factor of 3 correction per pullback
    log_ratio = jnp.clip(log_ratio, -1.1, 1.1)
    ratio = jnp.exp(log_ratio)
    w_new = 1.0 - ratio * (1.0 - weight)
    # Safety: if w_new would be larger in magnitude than w_old, don't apply
    w_new = jnp.where(jnp.abs(w_new) < jnp.abs(weight) + 0.1, w_new, weight)
    return w_new.astype(weight.dtype)


def soft_weight_damp(weight: jnp.ndarray, nu_w: float, w_sat: float = 2.0, alpha: int = 2) -> jnp.ndarray:
    """
    Amplitude-dependent weight damping: damps large weights strongly, leaves small weights alone.
    correction = -nu_w * w * (|w|/w_sat)^alpha
    """
    return weight - nu_w * weight * (jnp.abs(weight) / w_sat) ** alpha


def init_canonical_weights(
    state: GCState,
    geom,
    cfg,
) -> GCState:
    """
    Initialize δf weights using the canonical momentum perturbation structure.

    Instead of seeding w = ε*sin(mθ + nζ) with a flat radial profile, this
    function seeds the perturbation using the proper eigenmode structure based
    on the gyrokinetic canonical toroidal momentum:

        p_ζ = m_i * v_∥ * R * b_ζ + (e/c) * ψ_p

    The perturbation in canonical coords ensures:
    - The k_∥ = 0 resonance condition (qm - n = 0) is automatically satisfied
      at the resonant surface r_s where q(r_s) = n/m.
    - Radial localization near the resonant surface r_s.

    For linear ITG with mode (m, n):
        w_0(r, θ, ζ) = A * exp(-(r - r_s)² / Δr²) * cos(m*θ + n*ζ)
    where:
        r_s  = resonant surface: q(r_s) = n/m (interpolated from q_profile)
        Δr   ~ sqrt(q * rho_i / |shat|) is the characteristic mode width
        rho_i = vti / Omega_i is the ion Larmor radius

    Parameters
    ----------
    state : GCState with particle positions and velocities (weights will be replaced)
    geom  : geometry object with r_grid, q_profile, B0, R0
    cfg   : config dict or object with fields:
              cfg.m_mode   : poloidal mode number (int)
              cfg.n_mode   : toroidal mode number (int)
              cfg.amplitude: perturbation amplitude ε (float, default 1e-3)
              cfg.vti      : ion thermal velocity (for rho_i, float)
              cfg.q_over_m : ion charge-to-mass ratio (float)
    """
    # Parse config
    if hasattr(cfg, '__getitem__'):
        m = int(cfg['m_mode'])
        n = int(cfg['n_mode'])
        A = float(cfg.get('amplitude', 1e-3))
        vti = float(cfg['vti'])
        q_over_m = float(cfg['q_over_m'])
    else:
        m = int(cfg.m_mode)
        n = int(cfg.n_mode)
        A = float(getattr(cfg, 'amplitude', 1e-3))
        vti = float(cfg.vti)
        q_over_m = float(cfg.q_over_m)

    # Find resonant surface: q(r_s) = n / m
    # Interpolate q_profile to find where q crosses n/m
    q_target = n / m
    q_vals = geom.q_profile
    r_vals = geom.r_grid

    # Bracket search: find where q_vals crosses q_target
    # Use a weighted average of grid points for a smooth estimate
    diff = q_vals - q_target
    # Zero crossing: find adjacent points with sign change
    sign_change = diff[:-1] * diff[1:] < 0
    idx = jnp.argmax(sign_change)  # first crossing index
    # Linear interpolation for r_s
    r_lo = r_vals[idx];   q_lo = q_vals[idx]
    r_hi = r_vals[idx+1]; q_hi = q_vals[idx+1]
    r_s = float(r_lo + (q_target - q_lo) / (q_hi - q_lo + 1e-10) * (r_hi - r_lo))

    # Compute ion Larmor radius rho_i = vti / Omega_i = vti * mi / (e * B0)
    # Since q_over_m = e/mi, Omega_i = q_over_m * B0
    Omega_i = q_over_m * geom.B0
    rho_i   = vti / (abs(Omega_i) + 1e-10)

    # Local magnetic shear shat = (r/q) * dq/dr at r_s
    # Finite difference: shat ~ (r_s/q_s) * (q_hi - q_lo)/(r_hi - r_lo)
    dqdr = (float(q_hi) - float(q_lo)) / (float(r_hi) - float(r_lo) + 1e-10)
    shat = abs(r_s / (q_target + 1e-10) * dqdr)

    # Mode width: Δr ~ sqrt(q * rho_i / |shat|)
    # Use a floor on shat to avoid infinite width in zero-shear case
    delta_r = jnp.sqrt(q_target * rho_i / (abs(shat) + 0.01 * q_target))
    # Enforce at least one grid spacing
    dr_grid = float(r_vals[1] - r_vals[0])
    delta_r = float(jnp.maximum(delta_r, 2.0 * dr_grid))

    # Build perturbation weights
    phase   = m * state.theta + n * state.zeta
    radial  = jnp.exp(-0.5 * ((state.r - r_s) / delta_r)**2)
    w0      = A * radial * jnp.cos(phase)

    return GCState(
        r=state.r, theta=state.theta, zeta=state.zeta,
        vpar=state.vpar, mu=state.mu, weight=w0
    )


def spread_weights(state, geom, grid_shape):
    """
    GTC-style weight spreading: scatter w to grid, gather back.
    Smooths local weight variance while preserving total weight.
    """
    import jax.numpy as jnp
    from gyrojax.interpolation.scatter_gather_fa import scatter_weights_raw_fa, gather_scalar_from_grid_fa

    # Scatter weights to grid (raw, no normalization)
    w_grid = scatter_weights_raw_fa(state, geom, grid_shape)

    # Gather smoothed weights back to particles
    w_smooth = gather_scalar_from_grid_fa(w_grid, state, geom)

    # Preserve total weight (rescale)
    scale = jnp.sum(jnp.abs(state.weight)) / (jnp.sum(jnp.abs(w_smooth)) + 1e-30)
    w_new = w_smooth * scale

    return state._replace(weight=w_new.astype(jnp.float32))


def spread_weights_nonzonal(state, geom, grid_shape):
    """
    Weight spreading that preserves the zonal flow component.

    Algorithm:
    1. Compute zonal mean weight per radial bin: w_zonal[i] = mean(w[particles in bin i])
    2. Subtract zonal mean: w_turb = w - w_zonal(r)
    3. Apply spread_weights to w_turb only
    4. Add back w_zonal: w_new = w_turb_spread + w_zonal(r)

    This preserves radially-coherent (zonal) weight structure while smoothing
    turbulent noise.
    """
    import jax.numpy as jnp
    from gyrojax.interpolation.scatter_gather_fa import scatter_weights_raw_fa, gather_scalar_from_grid_fa

    # Step 1: compute zonal mean weight per radial bin
    # Use a simple binned average along psi
    Npsi = grid_shape[0]
    psi_min = geom.psi_grid[0]
    psi_max = geom.psi_grid[-1]
    dpsi = (psi_max - psi_min) / (Npsi - 1)

    # Assign each particle to a radial bin
    ir = jnp.clip(jnp.floor((state.r - psi_min) / dpsi).astype(jnp.int32), 0, Npsi - 1)

    # Compute mean weight per bin using scatter/gather on 1D array
    w_sum = jnp.zeros(Npsi, dtype=jnp.float32).at[ir].add(state.weight.astype(jnp.float32))
    count = jnp.zeros(Npsi, dtype=jnp.float32).at[ir].add(1.0)
    w_zonal_bins = w_sum / jnp.maximum(count, 1.0)  # (Npsi,)

    # Step 2: subtract zonal mean from each particle weight
    w_zonal_at_particle = w_zonal_bins[ir]  # (N,)
    w_turb = state.weight.astype(jnp.float32) - w_zonal_at_particle

    # Step 3: spread the turbulent (non-zonal) component
    state_turb = state._replace(weight=w_turb)
    w_grid = scatter_weights_raw_fa(state_turb, geom, grid_shape)
    w_smooth = gather_scalar_from_grid_fa(w_grid, state_turb, geom)
    scale = jnp.sum(jnp.abs(w_turb)) / (jnp.sum(jnp.abs(w_smooth)) + 1e-30)
    w_turb_spread = w_smooth * scale

    # Step 4: add zonal component back
    w_new = w_turb_spread + w_zonal_at_particle

    return state._replace(weight=w_new.astype(jnp.float32))

