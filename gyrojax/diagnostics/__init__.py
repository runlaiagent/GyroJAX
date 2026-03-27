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
    B_p, _, _, _, _, _ = interp_fa_to_particles(geom, state.r, state.theta, state.zeta)

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


def extract_growth_rate_smart(phi_max: np.ndarray, dt: float,
                              phi_rms_series: np.ndarray = None) -> tuple:
    """
    Find the clean linear growth phase automatically.

    Parameters
    ----------
    phi_max : array
        phi_max time series (used for saturation detection).
    dt : float
        Time step.
    phi_rms_series : array, optional
        phi_rms time series.  When provided, the growth rate is fitted
        from phi_rms (cleaner signal, less PIC noise) while phi_max is
        still used only to detect saturation onset.  This typically
        reduces the fitting error by ~10 percentage points for moderate
        particle counts.

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

    # When phi_rms_series is provided, use it for fitting (cleaner signal).
    # phi_max is still used below for saturation detection.
    if phi_rms_series is not None:
        arr_fit = np.array(phi_rms_series, dtype=np.float64)
        arr_fit = np.maximum(arr_fit, 1e-30)
        log_phi_fit = np.log(arr_fit)
    else:
        log_phi_fit = log_phi

    if n < 10:
        t = np.arange(n) * dt
        c = np.polyfit(t, log_phi_fit, 1)
        return float(max(c[0], 0.0)), 0, n

    # Physics-based time bounds for CBC ITG:
    # - With pert_amp=1e-4, initial burst ends at t~3
    # - Clean linear phase: t=3-9 (before nonlinear explosion)
    # - Saturation: phi_max > 4e-2 (mode explosion at t~9-10)
    t_skip       = 3.0    # skip initialization burst
    t_max_linear = 20.0   # hard cap
    sat_threshold = 4e-2  # phi_max above this → nonlinear explosion

    step_skip = max(3, int(t_skip / dt))
    step_max  = min(n, int(t_max_linear / dt))

    # Find saturation onset (always use phi_max for this)
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
        c = np.polyfit(t, log_phi_fit[n0:n1], 1)
        return float(max(c[0], 0.0)), n0, n1

    log_window = log_phi_fit[step_skip:n_end]
    n_win = len(log_window)

    if n_win < 5:
        n0, n1 = n // 4, 3 * n // 4
        t = np.arange(n0, n1) * dt
        c = np.polyfit(t, log_phi_fit[n0:n1], 1)
        return float(max(c[0], 0.0)), n0, n1

    # Primary method: simple linear regression over the full linear window
    # This is most robust when the window is correctly identified
    t_full = np.arange(step_skip, n_end) * dt
    c_full = np.polyfit(t_full, log_phi_fit[step_skip:n_end], 1)
    gamma_full = float(c_full[0])

    # Also try sliding window to find the most stable segment
    win = max(8, n_win // 3)
    slopes = []
    slope_starts = []
    for i in range(0, n_win - win):
        t_seg = np.arange(win) * dt
        c = np.polyfit(t_seg, log_window[i:i+win], 1)
        slopes.append(c[0])
        slope_starts.append(step_skip + i)

    slopes = np.array(slopes) if slopes else np.array([gamma_full])
    slope_starts = np.array(slope_starts) if slope_starts else np.array([step_skip])

    # Filter: physically plausible ITG growth rates
    mask = (slopes > 0.005) & (slopes < 0.5)
    if mask.sum() < 3:
        mask = (slopes > 0.0) & (slopes < 1.0)

    if mask.sum() == 0:
        return float(max(gamma_full, 0.0)), step_skip, n_end

    filtered_slopes = slopes[mask]
    filtered_starts = slope_starts[mask]

    # Take median of filtered slopes — robust to outliers
    best_gamma = float(np.median(filtered_slopes))
    best_start = int(filtered_starts[len(filtered_starts)//2])
    best_end   = min(best_start + win, n_end)

    # Use whichever is more conservative (closer to full-window fit)
    # This prevents picking an outlier peak
    if abs(gamma_full) > 0.005 and abs(best_gamma - gamma_full) > 0.5 * abs(gamma_full):
        best_gamma = gamma_full
        best_start, best_end = step_skip, n_end

    # When phi_rms_series is provided (cleaner signal), the full-window fit is more
    # reliable than the sliding-window median.  Prefer gamma_full in that case.
    if phi_rms_series is not None and abs(gamma_full) > 0.005:
        best_gamma = gamma_full
        best_start, best_end = step_skip, n_end

    return float(max(best_gamma, 0.0)), best_start, best_end


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


# ============================================================
# GTC-style diagnostics (added: mode spectrum, growth rate, zonal flow,
# DiagnosticsRecorder, energy diagnostics)
# ============================================================

import jax
import jax.numpy as jnp
from typing import List, Optional


def compute_mode_spectrum(phi: jnp.ndarray, geom=None) -> dict:
    """
    Compute Fourier mode spectrum like GTC's fieldmode output.
    phi: (Npsi, Ntheta, Nalpha)
    Returns dict with per-mode amplitudes and dominant mode indices.
    """
    mid = phi.shape[0] // 2
    phi_hat = jnp.fft.fft2(phi[mid], axes=(0, 1))   # (Ntheta, Nalpha)
    amps = jnp.abs(phi_hat)

    phi_n = jnp.mean(amps, axis=0)   # (Nalpha,)  — amplitude per toroidal mode
    phi_m = jnp.mean(amps, axis=1)   # (Ntheta,)  — amplitude per poloidal mode

    Ntheta, Nalpha = amps.shape
    flat_idx = int(jnp.argmax(amps.ravel()))
    dominant_m = flat_idx // Nalpha
    dominant_n = flat_idx % Nalpha

    return {
        'phi_n': phi_n,
        'phi_m': phi_m,
        'dominant_n': dominant_n,
        'dominant_m': dominant_m,
        'phi_complex': phi_hat,   # complex (Ntheta, Nalpha) at mid-radius
    }


def compute_growth_rate_from_history(
    phi_rms_history: jnp.ndarray,
    dt: float,
    window: int = 50,
) -> dict:
    """
    Fit linear growth rate gamma from log(phi_rms) time series.
    Like GTC: fit log|phi| = gamma*t + const over last `window` steps.
    Returns {'gamma': float, 't_fit': array, 'phi_fit': array}
    """
    n = phi_rms_history.shape[0]
    t_all = jnp.arange(n, dtype=jnp.float32) * dt
    w = min(window, n)
    t_fit = t_all[-w:]
    amp_fit = jnp.maximum(phi_rms_history[-w:], 1e-30)
    log_amp = jnp.log(amp_fit)

    # Linear regression: log_amp = gamma*t + const
    A = jnp.stack([t_fit, jnp.ones_like(t_fit)], axis=1)
    coeffs, _, _, _ = jnp.linalg.lstsq(A, log_amp, rcond=None)
    gamma = float(coeffs[0])

    # Reconstructed fit for plotting
    phi_fit = jnp.exp(coeffs[0] * t_fit + coeffs[1])

    return {'gamma': gamma, 't_fit': t_fit, 'phi_fit': phi_fit}


def compute_zonal_flow(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Extract zonal flow: average phi over theta and alpha axes.
    Returns shape (Npsi,) — the n=0, m=0 radial profile.
    """
    return phi.mean(axis=(1, 2))


def compute_nonzonal(phi: jnp.ndarray) -> jnp.ndarray:
    """Non-zonal (turbulent) component: phi minus zonal average."""
    zf = compute_zonal_flow(phi)
    return phi - zf[:, None, None]


def compute_particle_moments(state, geom, grid_shape: tuple) -> dict:
    """
    Compute radial profiles of particle moments (like GTC data1d output).
    Returns dict of radial profiles (shape Npsi):
      density, vpar_mean, weight_rms
    """
    from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa

    Npsi = grid_shape[0]

    # Scatter weights → density profile
    delta_n_3d = scatter_to_grid_fa(state, geom, grid_shape)
    density = delta_n_3d.mean(axis=(1, 2))   # (Npsi,)

    # Radial bin particle quantities
    r_min = float(geom.psi_grid[0])
    r_max = float(geom.psi_grid[-1])
    dr = (r_max - r_min) / (Npsi - 1)
    ir = jnp.clip(((state.r - r_min) / dr).astype(jnp.int32), 0, Npsi - 1)

    # vpar mean per radial bin
    vpar_sum   = jnp.zeros(Npsi).at[ir].add(state.vpar)
    count      = jnp.zeros(Npsi).at[ir].add(1.0) + 1e-10
    vpar_mean  = vpar_sum / count

    # weight rms per radial bin
    w2_sum    = jnp.zeros(Npsi).at[ir].add(state.weight**2)
    weight_rms = jnp.sqrt(w2_sum / count)

    return {
        'density':    density,
        'vpar_mean':  vpar_mean,
        'weight_rms': weight_rms,
    }


def compute_energy(state, phi: jnp.ndarray, geom, cfg=None) -> dict:
    """
    Compute free energy components for energy conservation check (like GTC).
    W_field ∝ sum(phi^2), W_kinetic from δf weights.
    """
    # Field energy: integral |phi|^2 dV (proportional)
    W_field = float(jnp.sum(phi**2))

    # Kinetic energy in δf: sum over particles of w^2 * H / f0
    # Simplified: proportional to sum(w^2 * (0.5*vpar^2 + mu*B_avg))
    B_avg = float(geom.B0) if hasattr(geom, 'B0') else 1.0
    H = 0.5 * state.vpar**2 + state.mu * B_avg
    W_kinetic = float(jnp.mean(state.weight**2 * H))

    W_total = W_field + W_kinetic

    return {
        'W_field':   W_field,
        'W_kinetic': W_kinetic,
        'W_total':   W_total,
    }


class DiagnosticsRecorder:
    """
    GTC-style diagnostics recorder. Accumulates history during a simulation run.

    Usage:
        recorder = DiagnosticsRecorder(cfg, geom)
        for step in range(n_steps):
            recorder.record(step, state, phi)
        result = recorder.finalize()
        print(result.summary())
    """
    def __init__(self, cfg, geom):
        self.cfg = cfg
        self.geom = geom
        self.phi_rms_history:    List[float] = []
        self.phi_max_history:    List[float] = []
        self.weight_rms_history: List[float] = []
        self.phi_zonal_history:  List[float] = []
        self.mode_amp_history:   List[complex] = []  # dominant mode complex amp

    def record(self, step: int, state, phi: jnp.ndarray):
        """Record one timestep. Blocks on JAX async to avoid graph accumulation."""
        phi_rms = float(jnp.sqrt(jnp.mean(phi**2)))
        phi_max = float(jnp.max(jnp.abs(phi)))
        w_rms   = float(jnp.sqrt(jnp.mean(state.weight**2)))
        zonal   = float(jnp.sqrt(jnp.mean(compute_zonal_flow(phi)**2)))

        # Dominant mode complex amplitude at mid-psi
        mid = phi.shape[0] // 2
        phi_hat = jnp.fft.fft2(phi[mid], axes=(0, 1))
        amps = jnp.abs(phi_hat)
        flat_idx = int(jnp.argmax(amps.ravel()))
        Nalpha = phi.shape[2]
        dm = flat_idx // Nalpha
        dn = flat_idx % Nalpha
        mode_c = complex(phi_hat[dm, dn])

        self.phi_rms_history.append(phi_rms)
        self.phi_max_history.append(phi_max)
        self.weight_rms_history.append(w_rms)
        self.phi_zonal_history.append(zonal)
        self.mode_amp_history.append(mode_c)

    def finalize(self) -> 'DiagnosticsResult':
        phi_rms  = jnp.array(self.phi_rms_history)
        phi_max  = jnp.array(self.phi_max_history)
        w_rms    = jnp.array(self.weight_rms_history)
        zonal    = jnp.array(self.phi_zonal_history)
        mode_amp = jnp.array(self.mode_amp_history)

        dt = getattr(self.cfg, 'dt', 0.05)
        gr = compute_growth_rate_from_history(phi_rms, dt=dt, window=min(50, len(phi_rms)))

        return DiagnosticsResult(
            phi_rms=phi_rms,
            phi_max=phi_max,
            weight_rms=w_rms,
            phi_zonal_rms=zonal,
            mode_amplitude=mode_amp,
            gamma=gr['gamma'],
            omega=0.0,
            dt=dt,
        )


class DiagnosticsResult:
    """Container for completed simulation diagnostics (GTC-style)."""

    def __init__(self, phi_rms, phi_max, weight_rms, phi_zonal_rms,
                 mode_amplitude, gamma: float, omega: float, dt: float):
        self.phi_rms       = phi_rms
        self.phi_max       = phi_max
        self.weight_rms    = weight_rms
        self.phi_zonal_rms = phi_zonal_rms
        self.mode_amplitude = mode_amplitude
        self.gamma  = gamma
        self.omega  = omega
        self.dt     = dt

    def summary(self) -> str:
        n = len(self.phi_rms)
        return (
            f"=== GyroJAX Diagnostics Summary ===\n"
            f"  Steps:         {n}\n"
            f"  phi_rms final: {float(self.phi_rms[-1]):.3e}\n"
            f"  phi_max final: {float(self.phi_max[-1]):.3e}\n"
            f"  weight_rms:    {float(self.weight_rms[-1]):.3e}\n"
            f"  gamma (fit):   {self.gamma:.4f}\n"
            f"  omega (fit):   {self.omega:.4f}\n"
        )

    def plot_history(self, ax=None):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
            t = np.arange(len(self.phi_rms)) * self.dt
            ax.semilogy(t, np.array(self.phi_rms), label='phi_rms')
            ax.semilogy(t, np.array(self.weight_rms), label='weight_rms', linestyle='--')
            ax.set_xlabel('t [R0/vti]')
            ax.set_ylabel('amplitude')
            ax.legend()
            ax.set_title('GyroJAX History')
            return fig
        except ImportError:
            print("matplotlib not available")

    def plot_growth_rate(self, ax=None, window: int = 50):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
            t = np.arange(len(self.phi_rms)) * self.dt
            ax.semilogy(t, np.array(self.phi_rms), label='phi_rms')
            # Overlay fit
            gr = compute_growth_rate_from_history(self.phi_rms, dt=self.dt, window=window)
            ax.semilogy(np.array(gr['t_fit']), np.array(gr['phi_fit']),
                        'r--', label=f'fit γ={self.gamma:.3f}')
            ax.set_xlabel('t [R0/vti]')
            ax.legend()
            ax.set_title('Growth Rate Fit')
            return fig
        except ImportError:
            print("matplotlib not available")
