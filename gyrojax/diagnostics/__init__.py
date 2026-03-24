"""
Diagnostics for GyroJAX — Phase 4c.

Provides:
  - Zonal flow extraction (phi averaged over θ, α → φ_ZF(ψ))
  - Perpendicular energy spectrum E(k⊥)
  - Parallel spectrum E(k∥)
  - Heat flux Q(ψ,t) = ∫ δf · v_E · ∇T dv
  - Growth rate extraction from time series
  - Mode structure analysis
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, List


# ---------------------------------------------------------------------------
# Zonal flows
# ---------------------------------------------------------------------------

def extract_zonal_flow(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Extract zonal flow component: φ_ZF(ψ) = <φ>_{θ,α}

    The zonal flow is the toroidally and poloidally averaged potential.
    It represents the E×B shear flow that regulates turbulence (Dimits shift).

    Parameters
    ----------
    phi : (Npsi, Ntheta, Nalpha)

    Returns
    -------
    phi_zf : (Npsi,)  — zonal flow profile
    """
    return jnp.mean(phi, axis=(1, 2))


def extract_nonzonal(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Remove zonal component: φ_turb = φ - <φ>_{θ,α}
    """
    phi_zf = extract_zonal_flow(phi)
    return phi - phi_zf[:, None, None]


def zonal_shear(phi: jnp.ndarray, geom) -> jnp.ndarray:
    """
    E×B shear rate: γ_E = d²φ_ZF/dψ²  (second radial derivative of zonal flow)

    This is the quantity that suppresses turbulence in the Dimits regime.
    """
    phi_zf = extract_zonal_flow(phi)   # (Npsi,)
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (len(geom.psi_grid) - 1)
    # Second derivative via finite differences
    d2phi = (jnp.roll(phi_zf, -1) - 2*phi_zf + jnp.roll(phi_zf, 1)) / dpsi**2
    return d2phi


# ---------------------------------------------------------------------------
# Energy spectra
# ---------------------------------------------------------------------------

def perp_spectrum(phi: jnp.ndarray, geom) -> tuple:
    """
    Perpendicular wavenumber spectrum of φ.

    Returns (k_perp_bins, E_k) where E_k = Σ |φ̂(k)|² for modes in each k-bin.

    In field-aligned coords, k_perp is dominated by k_α (binormal direction).
    """
    Npsi, Ntheta, Nalpha = phi.shape
    dal = 2 * np.pi / Nalpha
    kal = np.fft.fftfreq(Nalpha, d=dal) * 2 * np.pi  # (Nalpha,)

    # FFT in α direction (main perpendicular direction for ITG)
    phi_k = jnp.fft.fft(phi, axis=2)   # (Npsi, Ntheta, Nalpha)
    E_k = jnp.mean(jnp.abs(phi_k)**2, axis=(0, 1))  # (Nalpha,), averaged over ψ,θ

    k_rho = np.abs(kal) * float(jnp.sqrt(jnp.mean(geom.galphaalpha))) * 1.0  # k_alpha * rho_i
    # Only positive k
    half = Nalpha // 2
    return k_rho[:half], E_k[:half]


def parallel_spectrum(phi: jnp.ndarray) -> tuple:
    """
    Parallel wavenumber spectrum: E(k∥) from FFT in θ direction.
    """
    Npsi, Ntheta, Nalpha = phi.shape
    dth = 2 * np.pi / Ntheta
    kth = np.fft.fftfreq(Ntheta, d=dth) * 2 * np.pi

    phi_k = jnp.fft.fft(phi, axis=1)
    E_k = jnp.mean(jnp.abs(phi_k)**2, axis=(0, 2))

    half = Ntheta // 2
    return np.abs(kth[:half]), E_k[:half]


# ---------------------------------------------------------------------------
# Heat flux
# ---------------------------------------------------------------------------

def ion_heat_flux(
    state,
    phi: jnp.ndarray,
    geom,
    grid_shape: tuple,
    Ti: float,
    n0_avg: float,
    LT: float,
) -> jnp.ndarray:
    """
    Ion heat flux Q_i(ψ) = <v_E · ∇T · δf> integrated over velocity.

    In δf PIC:
      Q_i(ψ) = n0·Ti · Σ_p w_p · vE_ψ(X_p) · (H_p/Ti - 3/2)  / N_cell(ψ)

    where H_p = v∥²/2 + μ·B is the particle energy.

    This is the normalized ion heat flux (Qgb = n0·Ti·vti·ρi²/R0²).

    Parameters
    ----------
    state      : GCState with weights
    phi        : potential (Npsi, Ntheta, Nalpha)
    geom       : geometry
    grid_shape : (Npsi, Ntheta, Nalpha)
    Ti         : ion temperature
    n0_avg     : background density
    LT         : temperature gradient length scale
    """
    from gyrojax.interpolation.scatter_gather_fa import gather_from_grid_fa
    from gyrojax.geometry.field_aligned import interp_fa_to_particles

    # E×B radial velocity at particle positions
    E_psi, E_theta, E_alpha = gather_from_grid_fa(phi, state, geom)
    B_p, _, _, _, _ = interp_fa_to_particles(geom, state.r, state.theta, state.zeta)

    # vE_ψ = -E_θ / B  (radial ExB drift)
    vE_psi = -E_theta / jnp.maximum(B_p, 1e-10)

    # Normalized energy: (H/Ti - 3/2)
    H = state.vpar**2 / 2.0 + state.mu * B_p
    energy_factor = H / Ti - 1.5

    # Heat flux contribution per particle: w_p * vE_ψ * (H/Ti - 3/2)
    q_p = state.weight * vE_psi * energy_factor

    # Scatter to radial grid (sum over all particles in each ψ-bin)
    Npsi, Ntheta, Nalpha = grid_shape
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    ir = jnp.clip((state.r - geom.psi_grid[0]) / dpsi, 0, Npsi - 1.001)
    i0 = jnp.floor(ir).astype(jnp.int32)

    Q_psi = jnp.zeros(Npsi).at[i0].add(q_p)
    # Normalize by number of cells and n0*Ti
    Q_psi = Q_psi * n0_avg * Ti / (Ntheta * Nalpha)

    return Q_psi


# ---------------------------------------------------------------------------
# Growth rate extraction
# ---------------------------------------------------------------------------

def extract_growth_rate(
    phi_max_series: np.ndarray,
    dt: float,
    window: float = 0.4,
    t_start: float = None,
) -> float:
    """
    Extract linear growth rate from log|φ_max| time series.

    Uses linear regression over the last `window` fraction of the run
    (or from t_start if specified).

    Returns γ in units of vti/R0.
    """
    n = len(phi_max_series)
    if t_start is not None:
        n_start = int(t_start / dt)
    else:
        n_start = int(n * (1 - window))
    n_start = max(n_start, 1)

    t = np.arange(n_start, n) * dt
    log_phi = np.log(np.maximum(phi_max_series[n_start:], 1e-20))

    # Linear regression: log|φ| = γ·t + const
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def extract_growth_rate_smart(phi_max: np.ndarray, dt: float) -> tuple:
    """
    Find the clean linear growth phase automatically.

    Physics constraints for CBC ITG:
    - Transient lasts until t ~ 5 R0/vti  (skip first t < t_skip)
    - Linear phase ends at saturation: phi_max grows beyond ~1e-2
    - Linear phase ends in time at t ~ 25 R0/vti at most
    - Valid growth rates: 0.01 < γ < 0.45 vti/R0

    Method:
    1. Skip initial transient: t < t_skip (default 5 R0/vti)
    2. Cap at saturation onset: first step where phi_max > sat_threshold
       OR t > t_max_linear (default 25 R0/vti)
    3. Within that window, run sliding-window linear fits
    4. Find most stable (lowest CV) segment with positive γ

    Returns: (gamma, step_start, step_end)
    """
    arr = np.array(phi_max, dtype=np.float64)
    arr = np.maximum(arr, 1e-30)
    log_phi = np.log(arr)
    n = len(log_phi)

    if n < 10:
        t = np.arange(n) * dt
        c = np.polyfit(t, log_phi, 1)
        return float(max(c[0], 0.0)), 0, n

    # Physics-based time bounds for CBC ITG:
    # - Transient lasts ~t=1-2 (particle initialization noise)
    # - Linear phase: t=2-8 (phi grows exponentially)
    # - Saturation onset: phi_max > sat_threshold OR t > t_max_linear
    t_skip       = 2.0    # skip initialization noise
    t_max_linear = 15.0   # linear phase ends well before this
    sat_threshold = 3e-2  # phi_max above this → entering nonlinear

    step_skip = max(3, int(t_skip / dt))
    step_max  = min(n, int(t_max_linear / dt))

    # Find saturation onset
    sat_step = step_max
    for i in range(step_skip, n):
        if arr[i] > sat_threshold:
            sat_step = i
            break

    n_end = min(step_max, sat_step)

    if n_end - step_skip < 10:
        # Window too short — relax sat threshold and extend
        for thresh in [0.1, 0.5, 5.0]:
            sat_step = step_max
            for i in range(step_skip, n):
                if arr[i] > thresh:
                    sat_step = i
                    break
            n_end = min(step_max, sat_step)
            if n_end - step_skip >= 10:
                break

    if n_end - step_skip < 5:
        n0, n1 = n // 4, 3 * n // 4
        t = np.arange(n0, n1) * dt
        c = np.polyfit(t, log_phi[n0:n1], 1)
        return float(max(c[0], 0.0)), n0, n1

    log_window = log_phi[step_skip:n_end]
    n_win = len(log_window)

    # Sliding window fit within the linear phase window
    win = max(10, n_win // 3)
    slopes = []
    slope_starts = []
    for i in range(0, n_win - win):
        t_seg = np.arange(win) * dt
        c = np.polyfit(t_seg, log_window[i:i+win], 1)
        slopes.append(c[0])
        slope_starts.append(step_skip + i)

    slopes = np.array(slopes)
    slope_starts = np.array(slope_starts)

    if len(slopes) == 0:
        t_full = np.arange(step_skip, n_end) * dt
        c = np.polyfit(t_full, log_phi[step_skip:n_end], 1)
        return float(max(c[0], 0.0)), step_skip, n_end

    # Filter: physically plausible ITG growth rates
    mask = (slopes > 0.005) & (slopes < 0.5)
    if mask.sum() < 3:
        mask = (slopes > 0.0) & (slopes < 1.0)

    filtered_slopes = slopes[mask]
    filtered_starts = slope_starts[mask]

    if len(filtered_slopes) == 0:
        t_full = np.arange(step_skip, n_end) * dt
        c = np.polyfit(t_full, log_phi[step_skip:n_end], 1)
        return float(max(c[0], 0.0)), step_skip, n_end

    # Find the most stable (lowest CV) segment
    best_gamma = 0.0
    best_start, best_end = int(filtered_starts[0]), n_end
    best_score = float('inf')

    seg_win = max(3, len(filtered_slopes) // 3)
    for i in range(len(filtered_slopes) - seg_win + 1):
        seg = filtered_slopes[i:i+seg_win]
        mean_s = float(seg.mean())
        if mean_s > 0.005:
            cv = float(seg.std()) / (mean_s + 1e-10)
            if cv < best_score:
                best_score = cv
                best_gamma = mean_s
                best_start = int(filtered_starts[i])
                best_end   = min(int(filtered_starts[i]) + seg_win + win, n_end)

    if best_gamma == 0.0:
        t_full = np.arange(step_skip, n_end) * dt
        c = np.polyfit(t_full, log_phi[step_skip:n_end], 1)
        best_gamma = float(max(c[0], 0.0))
        best_start, best_end = step_skip, n_end

    return best_gamma, best_start, best_end


def extract_mode_frequency(
    phi_series: np.ndarray,
    dt: float,
    window: float = 0.3,
) -> float:
    """
    Extract real frequency ω from oscillation of φ(t) in linear phase.

    Uses Hilbert transform to get instantaneous frequency.
    Returns ω in units of vti/R0.
    """
    try:
        from scipy.signal import hilbert
    except ImportError:
        return 0.0

    n = len(phi_series)
    n_start = int(n * (1 - window))
    sig = phi_series[n_start:]
    analytic = hilbert(sig)
    phase = np.unwrap(np.angle(analytic))
    omega = np.mean(np.diff(phase)) / dt
    return float(omega)


# ---------------------------------------------------------------------------
# Diagnostic snapshot
# ---------------------------------------------------------------------------

class DiagSnapshot(NamedTuple):
    """Full diagnostic snapshot at one timestep."""
    t:            float
    phi_rms:      float
    phi_max:      float
    weight_rms:   float
    gamma_inst:   float      # instantaneous growth rate d(log|φ|)/dt
    Q_ion:        np.ndarray  # (Npsi,) ion heat flux profile
    phi_zf_rms:   float      # rms of zonal flow
    phi_turb_rms: float      # rms of turbulent (non-zonal) component


def compute_snapshot(
    t: float,
    state,
    phi: jnp.ndarray,
    geom,
    grid_shape: tuple,
    Ti: float,
    n0_avg: float,
    LT: float,
    phi_prev: jnp.ndarray = None,
    dt: float = None,
) -> DiagSnapshot:
    """Compute full diagnostic snapshot."""
    phi_rms    = float(jnp.sqrt(jnp.mean(phi**2)))
    phi_max    = float(jnp.max(jnp.abs(phi)))
    weight_rms = float(jnp.sqrt(jnp.mean(state.weight**2)))

    phi_zf   = extract_zonal_flow(phi)
    phi_turb = extract_nonzonal(phi)
    phi_zf_rms   = float(jnp.sqrt(jnp.mean(phi_zf**2)))
    phi_turb_rms = float(jnp.sqrt(jnp.mean(phi_turb**2)))

    # Instantaneous growth rate
    if phi_prev is not None and dt is not None:
        phi_prev_max = float(jnp.max(jnp.abs(phi_prev))) + 1e-20
        gamma_inst = (jnp.log(jnp.maximum(phi_max, 1e-20)) - jnp.log(phi_prev_max)) / dt
    else:
        gamma_inst = 0.0

    Q = ion_heat_flux(state, phi, geom, grid_shape, Ti, n0_avg, LT)

    return DiagSnapshot(
        t=t, phi_rms=phi_rms, phi_max=phi_max,
        weight_rms=weight_rms, gamma_inst=float(gamma_inst),
        Q_ion=np.array(Q), phi_zf_rms=phi_zf_rms, phi_turb_rms=phi_turb_rms,
    )


# ---------------------------------------------------------------------------
# Smart growth rate extractor
