"""
Global CBC benchmark — radially varying profiles, Krook buffer BCs.

Uses global geometry (use_global=True) with radially varying n0(r), T(r), q(r).
The growth rate should match the flux-tube result in the linear phase.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


def extract_growth_rate(phi_max_series: np.ndarray, dt: float, window: float = 0.4) -> float:
    n       = len(phi_max_series)
    n_start = int(n * (1 - window))
    t       = np.arange(n_start, n) * dt
    log_phi = np.log(np.maximum(phi_max_series[n_start:], 1e-20))
    if len(t) < 2:
        return float("nan")
    coeffs  = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_global_cbc(quick: bool = False):
    print("=" * 62)
    print("  GyroJAX Global CBC — radially varying profiles + Krook BCs")
    print("  Reference: Dimits et al. (2000) Phys. Plasmas 7, 969")
    print("=" * 62)

    if quick:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=50_000,
            n_steps=200,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
            use_global=True,
        )
        print("  [QUICK MODE: reduced resolution]")
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=100_000,
            n_steps=300,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
            use_global=True,
        )

    print(f"\nGlobal CBC Parameters:")
    print(f"  R0/LT = {cfg.R0_over_LT},  R0/Ln = {cfg.R0_over_Ln}")
    print(f"  q0 = {cfg.q0}, q1 = {cfg.q1}, ε = {cfg.a/cfg.R0:.3f}")
    print(f"  Grid: ({cfg.Npsi}, {cfg.Ntheta}, {cfg.Nalpha})")
    print(f"  N_particles = {cfg.N_particles:,}")
    print(f"  n_steps = {cfg.n_steps},  dt = {cfg.dt} R0/vti")
    print(f"  use_global = {cfg.use_global}\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fa(cfg, key)

    phi_max_series = np.array([float(d.phi_max) for d in diags])
    w_rms_series   = np.array([float(d.weight_rms) for d in diags])

    gamma = extract_growth_rate(phi_max_series, cfg.dt, window=0.4)

    # Check weight boundedness
    w_final_rms = w_rms_series[-1]
    w_max_seen  = np.max(w_rms_series)
    bounded = not np.isnan(w_final_rms) and w_max_seen < 1e3

    print("\n" + "=" * 62)
    print(f"  GLOBAL CBC RESULTS:")
    print(f"  Growth rate γ      = {gamma:.4f} vti/R0")
    print(f"  Final |w|_rms      = {w_final_rms:.3e}")
    print(f"  Max   |w|_rms      = {w_max_seen:.3e}")
    print(f"  Weights bounded    = {bounded}")
    print(f"  Reference γ ≈ 0.17 vti/R0 (Dimits et al. 2000)")
    print("=" * 62)

    return gamma, bounded


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    gamma, bounded = run_global_cbc(quick=quick)
    if not bounded:
        print("WARNING: weights appear unbounded — check Krook damping")
        sys.exit(1)
    print(f"\nGlobal CBC passed. γ = {gamma:.4f}")
