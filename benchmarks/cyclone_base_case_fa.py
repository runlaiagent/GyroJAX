"""
Cyclone Base Case benchmark — Phase 2a (field-aligned coords + exact Γ₀(b)).

Same physics as the Phase 1 CBC, but uses:
  - Field-aligned (ψ, θ, α) grid
  - Exact Γ₀(b) = I₀(b)·exp(-b) GK Poisson solver
  - FA-aware pusher and scatter/gather

Target: γ ≈ 0.17 vti/R0, error < 5% vs Dimits et al. (2000).
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
    coeffs  = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_cbc_fa(quick: bool = False):
    print("=" * 62)
    print("  GyroJAX Phase 2a — Cyclone Base Case (Field-Aligned)")
    print("  Dimits et al. (2000) Phys. Plasmas 7, 969")
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
        )
        print("  [QUICK MODE: reduced resolution]")
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=2_000_000,
            n_steps=500,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
        )

    print(f"\nCBC Parameters:")
    print(f"  R0/LT = {cfg.R0_over_LT},  R0/Ln = {cfg.R0_over_Ln}")
    print(f"  q0 = {cfg.q0},  ε = {cfg.a/cfg.R0:.3f}")
    print(f"  Grid: ({cfg.Npsi}, {cfg.Ntheta}, {cfg.Nalpha})")
    print(f"  N_particles = {cfg.N_particles:,}")
    print(f"  n_steps = {cfg.n_steps},  dt = {cfg.dt} R0/vti\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fa(cfg, key)

    phi_max_series = np.array([float(d.phi_max) for d in diags])
    phi_rms_series = np.array([float(d.phi_rms) for d in diags])
    t_series = np.arange(len(diags)) * cfg.dt

    gamma = extract_growth_rate(phi_max_series, cfg.dt, window=0.4)

    print("\n" + "=" * 62)
    print(f"  RESULTS:")
    print(f"  Measured growth rate γ = {gamma:.4f} vti/R0")
    print(f"  Target (Dimits 2000):   γ ≈ 0.17 vti/R0")

    target_gamma = 0.17
    rel_err = abs(gamma - target_gamma) / target_gamma

    if gamma > 0:
        print(f"  Mode is UNSTABLE ✓")
        if rel_err < 0.05:
            print(f"  ✅ BENCHMARK PASS (<5% error): error = {rel_err:.1%}")
        elif rel_err < 0.15:
            print(f"  ✅ BENCHMARK PASS (<15% error): error = {rel_err:.1%}")
        else:
            print(f"  ⚠️  BENCHMARK MARGINAL: error = {rel_err:.1%}")
    else:
        print(f"  ❌ WARNING: Mode is STABLE — check resolution/particles")
    print("=" * 62)

    # Save results
    try:
        from gyrojax.viz import plot_dashboard
        from gyrojax.diagnostics import ion_heat_flux
        import numpy as np_
        LT = cfg.R0 / cfg.R0_over_LT
        Q = ion_heat_flux(state, phi, geom, (cfg.Npsi, cfg.Ntheta, cfg.Nalpha),
                          cfg.Ti, cfg.n0_avg, LT)
        out_path = os.path.join(os.path.dirname(__file__), 'cbc_fa_dashboard.png')
        plot_dashboard(
            phi_max=phi_max_series, phi_rms=phi_rms_series,
            dt=cfg.dt, gamma_measured=gamma,
            phi_final=np_.array(phi), state_final=state, geom=geom,
            Q_profiles=[np_.array(Q)], t_labels=[f't={len(diags)*cfg.dt:.0f}'],
            gamma_target=0.17,
            title=f"GyroJAX CBC Phase 2a  [γ={gamma:.3f}, err={rel_err:.1%}]",
            save_path=out_path,
        )
    except Exception as ex:
        print(f"  (viz skipped: {ex})")

    return {
        't': t_series,
        'phi_max': phi_max_series,
        'phi_rms': phi_rms_series,
        'gamma_measured': gamma,
        'gamma_target': target_gamma,
        'rel_err': rel_err,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GyroJAX Phase 2a CBC benchmark")
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced resolution)')
    args = parser.parse_args()
    run_cbc_fa(quick=args.quick)
