"""
Cyclone Base Case (CBC) benchmark for GyroJAX.

The CBC is the standard benchmark for gyrokinetic turbulence codes
(Dimits et al. 2000, Phys. Plasmas 7, 969).

Parameters:
    R0/LT = 6.9   (ion temperature gradient drive)
    R0/Ln = 2.2   (density gradient)
    q0 = 1.4, s = 0.78 (magnetic shear)
    ε = r/R0 = 0.18 (inverse aspect ratio)
    Ti/Te = 1.0

Target: linear ITG growth rate γ ≈ 0.17 vti/R0 at peak ky*ρi ≈ 0.3

This script:
1. Initializes a CBC simulation
2. Adds a small φ perturbation at the target ky
3. Runs for n_steps timesteps
4. Extracts the linear growth rate from log|phi_max|(t)
5. Compares to published result
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.geometry.salpha import build_salpha_geometry
from gyrojax.particles.guiding_center import init_maxwellian_particles, GCState
from gyrojax.simulation import SimConfig, run_simulation


def extract_growth_rate(phi_max_series: np.ndarray, dt: float, window: float = 0.5) -> float:
    """
    Extract linear growth rate from log|phi_max|(t) via linear fit.

    Uses the last 'window' fraction of the time series (linear phase).
    """
    n = len(phi_max_series)
    n_start = int(n * (1 - window))
    t = np.arange(n_start, n) * dt
    log_phi = np.log(np.maximum(phi_max_series[n_start:], 1e-20))

    # Linear fit: log|phi| = γ*t + const
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])   # slope = growth rate γ


def run_cbc_benchmark(quick: bool = False):
    """
    Run the Cyclone Base Case benchmark.

    Parameters
    ----------
    quick : bool
        If True, use reduced resolution for fast testing.
    """
    print("=" * 60)
    print("  GyroJAX — Cyclone Base Case Benchmark")
    print("  Dimits et al. (2000) Phys. Plasmas 7, 969")
    print("=" * 60)

    if quick:
        cfg = SimConfig(
            Nr=16, Ntheta=32, Nzeta=16,
            N_particles=50_000,
            n_steps=200,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1.0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
        )
        print("  [QUICK MODE: reduced resolution]")
    else:
        cfg = SimConfig(
            Nr=32, Ntheta=64, Nzeta=32,
            N_particles=500_000,
            n_steps=500,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1.0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
        )

    print(f"\nCBC Parameters:")
    print(f"  R0/LT = {cfg.R0_over_LT},  R0/Ln = {cfg.R0_over_Ln}")
    print(f"  q0 = {cfg.q0},  ε = {cfg.a/cfg.R0}")
    print(f"  Grid: ({cfg.Nr}, {cfg.Ntheta}, {cfg.Nzeta})")
    print(f"  N_particles = {cfg.N_particles:,}")
    print(f"  n_steps = {cfg.n_steps},  dt = {cfg.dt} R0/vti\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi = run_simulation(cfg, key)

    phi_max_series = np.array([float(d.phi_max) for d in diags])
    phi_rms_series = np.array([float(d.phi_rms) for d in diags])
    t_series = np.arange(len(diags)) * cfg.dt

    # Extract growth rate from last 40% of simulation
    gamma = extract_growth_rate(phi_max_series, cfg.dt, window=0.4)

    print("\n" + "=" * 60)
    print(f"  RESULTS:")
    print(f"  Measured growth rate γ = {gamma:.4f} vti/R0")
    print(f"  Target (Dimits 2000):   γ ≈ 0.17 vti/R0")

    target_gamma = 0.17
    rel_err = abs(gamma - target_gamma) / target_gamma

    if gamma > 0:
        print(f"  Mode is UNSTABLE ✓ (positive growth rate)")
        if rel_err < 0.30:
            print(f"  BENCHMARK PASS: γ within 30% of published value (error={rel_err:.1%})")
        else:
            print(f"  BENCHMARK MARGINAL: γ error = {rel_err:.1%} (need more particles/resolution)")
    else:
        print(f"  WARNING: Mode appears STABLE — may need more particles or longer run")
        print(f"  (This can happen with insufficient resolution or particle noise)")

    print("=" * 60)

    # Save results
    results = {
        't': t_series,
        'phi_max': phi_max_series,
        'phi_rms': phi_rms_series,
        'gamma_measured': gamma,
        'gamma_target': target_gamma,
    }

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.semilogy(t_series, phi_max_series + 1e-20, 'b-', label='|φ|_max')
        ax1.semilogy(t_series, phi_rms_series + 1e-20, 'r--', label='|φ|_rms')
        ax1.set_xlabel('t [R0/vti]')
        ax1.set_ylabel('|φ|')
        ax1.set_title('CBC: Potential amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_series, phi_max_series, 'b-')
        ax2.set_xlabel('t [R0/vti]')
        ax2.set_ylabel('|φ|_max')
        ax2.set_title(f'CBC: γ = {gamma:.3f} vti/R0 (target: 0.17)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), 'cbc_result.png')
        plt.savefig(out_path, dpi=120)
        print(f"\n  Plot saved to: {out_path}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GyroJAX Cyclone Base Case benchmark")
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced resolution)')
    args = parser.parse_args()
    run_cbc_benchmark(quick=args.quick)
