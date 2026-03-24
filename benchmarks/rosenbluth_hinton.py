"""
Rosenbluth-Hinton zonal flow residual + GAM frequency benchmark.

Physics:
  Initialize a zonal flow (kθ=kα=0), zero drives, let relax.
  The flow damps to a nonzero residual due to neoclassical orbit averaging.
  Before settling, it oscillates at the GAM frequency.

References:
  Rosenbluth & Hinton, PRL 80, 724 (1998)  — zonal flow residual
  Winsor, Johnson & Dawson, Phys. Fluids 11, 2448 (1968) — GAM frequency

Theory:
  residual = 1 / (1 + 1.6·q²/√ε)    ε = r/R0 (local inverse aspect ratio)
  ω_GAM   = (vti/R0) · √(7/4 + Te/Ti)
"""

import os, sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


# ── Analytic predictions ────────────────────────────────────────────────────

def rh_residual_theory(q: float, eps: float) -> float:
    """Rosenbluth-Hinton residual = 1 / (1 + 1.6·q²/√ε)."""
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))


def gam_frequency_theory(vti: float, R0: float, Te_over_Ti: float = 1.0) -> float:
    """GAM angular frequency ω = (vti/R0)·√(7/4 + Te/Ti)."""
    return (vti / R0) * np.sqrt(7.0 / 4.0 + Te_over_Ti)


# ── Diagnostic helpers ──────────────────────────────────────────────────────

def extract_phi_zonal(phi_flat, grid_shape):
    """Average φ over θ and α → zonal component at each ψ. Shape: (Npsi,)."""
    return np.array(phi_flat).reshape(grid_shape).mean(axis=(1, 2))


def compute_residual(phi_zonal_series, t_series, t_settle: float = 20.0):
    """
    Residual = mean |φ_zonal(t>t_settle)| / |φ_zonal(0)|.
    Uses mid-radius ψ index.
    """
    mid = phi_zonal_series.shape[1] // 2
    phi_mid = phi_zonal_series[:, mid]
    phi0 = abs(phi_mid[0]) + 1e-20
    mask = t_series > t_settle
    if mask.sum() < 5:
        mask = t_series > t_series[-1] * 0.5   # fallback: use last half
    return float(np.abs(phi_mid[mask]).mean() / phi0)


def compute_gam_frequency(phi_zonal_series, dt: float):
    """
    GAM frequency from FFT of zonal φ(t) at mid-radius.
    Returns dominant angular frequency ω in vti/R0 units.
    """
    mid = phi_zonal_series.shape[1] // 2
    signal = phi_zonal_series[:, mid]
    signal -= signal.mean()           # remove DC
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt) * 2.0 * np.pi   # angular freq
    power = np.abs(np.fft.rfft(signal))**2
    power[0] = 0.0                    # zero DC
    # Find peak (exclude very low frequencies = slow drift, not GAM)
    min_freq = 0.3                    # ignore below 0.3 vti/R0
    power[freqs < min_freq] = 0.0
    if power.max() == 0:
        return 0.0
    return float(freqs[np.argmax(power)])


# ── Main benchmark ──────────────────────────────────────────────────────────

def run_rh_benchmark(quick: bool = False, verbose: bool = True):
    print("=" * 65)
    print("  GyroJAX — Rosenbluth-Hinton & GAM Frequency Benchmark")
    print("  Rosenbluth & Hinton, PRL 80, 724 (1998)")
    print("=" * 65)

    # CBC geometry, zero drives
    if quick:
        cfg = SimConfigFA(
            Npsi=24, Ntheta=32, Nalpha=8,
            N_particles=300_000,
            n_steps=600,
            dt=0.05,
            R0_over_LT=0.0,
            R0_over_Ln=0.0,
            pert_amp=1e-2,
            zonal_init=True,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.0,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            vpar_cap=4.0,
        )
        print("  [QUICK MODE: 100k particles, 400 steps]")
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=32, Nalpha=8,
            N_particles=300_000,
            n_steps=800,
            dt=0.05,
            R0_over_LT=0.0,
            R0_over_Ln=0.0,
            pert_amp=1e-2,
            zonal_init=True,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.0,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            vpar_cap=4.0,
        )
        print("  [FULL MODE: 300k particles, 800 steps]")

    print(f"\n  q0={cfg.q0},  a={cfg.a},  R0={cfg.R0},  ε=a/R0={cfg.a/cfg.R0:.3f}")
    print(f"  Grid: ({cfg.Npsi}, {cfg.Ntheta}, {cfg.Nalpha}),  dt={cfg.dt}\n")

    key = jax.random.PRNGKey(0)
    diags, state, phi_final, geom = run_simulation_fa(cfg, key, verbose=verbose)

    grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)
    t_series = np.arange(len(diags)) * cfg.dt

    # Collect zonal φ time series using dedicated phi_zonal_mid diagnostic
    phi_zonal_mid_arr = np.array([float(d.phi_zonal_mid) for d in diags])
    phi_max_arr       = np.array([float(d.phi_max) for d in diags])

    # Use |phi_zonal_mid| for residual measurement
    phi0 = abs(phi_zonal_mid_arr[0]) + 1e-20
    phi_norm = np.abs(phi_zonal_mid_arr) / phi0

    # Settle time: last 40% of run
    t_settle = t_series[int(len(t_series) * 0.60)]
    late_mask = t_series >= t_settle
    residual_measured = float(phi_norm[late_mask].mean())

    # GAM: FFT of zonal phi_mid time series (oscillates before settling)
    signal = phi_zonal_mid_arr.copy()
    signal -= signal[late_mask].mean()    # remove late-time DC offset
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=cfg.dt) * 2.0 * np.pi
    power = np.abs(np.fft.rfft(signal))**2
    power[0] = 0.0
    power[freqs < 0.3] = 0.0
    omega_gam_measured = float(freqs[np.argmax(power)]) if power.max() > 0 else 0.0

    # Theory
    eps_mid = (cfg.a / 2.0) / cfg.R0      # ε at mid-radius r=a/2
    residual_theory = rh_residual_theory(cfg.q0, eps_mid)
    omega_gam_theory = gam_frequency_theory(cfg.vti, cfg.R0, cfg.Te / cfg.Ti)

    # Errors
    rh_err  = abs(residual_measured - residual_theory) / (residual_theory + 1e-10)
    gam_err = abs(omega_gam_measured - omega_gam_theory) / omega_gam_theory if omega_gam_measured > 0 else 1.0

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"\n  ── Rosenbluth-Hinton Residual ──────────────────────────────")
    print(f"  ε (mid-radius)  = {eps_mid:.4f}")
    print(f"  Theory:           {residual_theory:.4f}")
    print(f"  Measured:         {residual_measured:.4f}  (settle t>{t_settle:.1f})")
    print(f"  Error:            {rh_err:.1%}  ", end="")
    if rh_err < 0.10:
        print("✅ PASS (<10%)")
    elif rh_err < 0.30:
        print("⚠️  MARGINAL (<30%)")
    else:
        print("❌ FAIL")

    print(f"\n  ── GAM Frequency ───────────────────────────────────────────")
    print(f"  Theory:   ω_GAM = {omega_gam_theory:.4f} vti/R0")
    print(f"  Measured: ω_GAM = {omega_gam_measured:.4f} vti/R0")
    print(f"  Error:           {gam_err:.1%}  ", end="")
    if omega_gam_measured == 0:
        print("⚠️  No clear oscillation detected")
    elif gam_err < 0.15:
        print("✅ PASS (<15%)")
    elif gam_err < 0.30:
        print("⚠️  MARGINAL (<30%)")
    else:
        print("❌ FAIL")

    print("\n  φ_zonal_mid time series (every 40 steps):")
    for i in range(0, len(phi_zonal_mid_arr), max(1, len(phi_zonal_mid_arr)//10)):
        print(f"    t={t_series[i]:6.1f}  φ_zonal_mid={phi_zonal_mid_arr[i]:.4e}  (norm={phi_norm[i]:.4f})")

    print("=" * 65)
    return residual_measured, residual_theory, omega_gam_measured, omega_gam_theory


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_rh_benchmark(quick=args.quick)
