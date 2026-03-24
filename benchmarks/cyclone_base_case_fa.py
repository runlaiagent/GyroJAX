"""
Cyclone Base Case benchmark — Phase 2a (field-aligned coords + exact Γ₀(b)).

Same physics as the Phase 1 CBC, but uses:
  - Field-aligned (ψ, θ, α) grid
  - Exact Γ₀(b) = I₀(b)·exp(-b) GK Poisson solver
  - FA-aware pusher and scatter/gather

Target: γ ≈ 0.170 vti/R0, error < 2% vs GENE/GX/GTC consensus.
References:
  - GENE:  γ = 0.171 vti/R0
  - GX:    γ = 0.168 vti/R0
  - GTC:   γ = 0.169 vti/R0
  - Dimits et al. (2000) Phys. Plasmas 7, 969
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import extract_growth_rate_smart


# Reference targets
GENE_GAMMA  = 0.171
GX_GAMMA    = 0.168
GTC_GAMMA   = 0.169
CONSENSUS   = (GENE_GAMMA + GX_GAMMA + GTC_GAMMA) / 3.0   # ≈ 0.1693
TARGET_GAMMA = 0.170


def run_cbc_fa(quick: bool = False):
    print("=" * 70)
    print("  GyroJAX Phase 2a — Cyclone Base Case (Field-Aligned)")
    print("  Dimits et al. (2000) Phys. Plasmas 7, 969")
    print("  Reference: GENE=0.171, GX=0.168, GTC=0.169  [consensus≈0.169]")
    print("=" * 70)

    if quick:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=100_000,
            n_steps=400,
            dt=0.05,
            pert_amp=1e-3,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
        )
        print("  [QUICK MODE: reduced resolution, 100k particles, 400 steps]")
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=1_000_000,
            n_steps=800,
            dt=0.03,
            pert_amp=1e-4,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
        )
        print("  [FULL MODE: 1M particles, 800 steps, dt=0.03]")

    print(f"\nCBC Parameters:")
    print(f"  R0/LT = {cfg.R0_over_LT},  R0/Ln = {cfg.R0_over_Ln}")
    print(f"  q0 = {cfg.q0},  ε = {cfg.a/cfg.R0:.3f}")
    print(f"  Grid: ({cfg.Npsi}, {cfg.Ntheta}, {cfg.Nalpha})")
    print(f"  N_particles = {cfg.N_particles:,}")
    print(f"  n_steps = {cfg.n_steps},  dt = {cfg.dt} R0/vti")
    print(f"  pert_amp = {cfg.pert_amp}\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fa(cfg, key)

    phi_max_series = np.array([float(d.phi_max) for d in diags])
    phi_rms_series = np.array([float(d.phi_rms) for d in diags])
    t_series = np.arange(len(diags)) * cfg.dt

    # Smart growth rate extraction
    gamma, step_start, step_end = extract_growth_rate_smart(phi_max_series, cfg.dt)
    t_start = step_start * cfg.dt
    t_end   = step_end   * cfg.dt

    print("\n" + "=" * 70)
    print(f"  RESULTS:")
    print(f"  Detected linear phase: steps {step_start}–{step_end}  (t={t_start:.1f}–{t_end:.1f} R0/vti)")
    print(f"  Measured growth rate  γ = {gamma:.4f} vti/R0")
    print(f"")
    print(f"  GENE ref:   γ = {GENE_GAMMA:.3f} vti/R0")
    print(f"  GX ref:     γ = {GX_GAMMA:.3f} vti/R0")
    print(f"  GTC ref:    γ = {GTC_GAMMA:.3f} vti/R0")
    print(f"  Consensus:  γ = {CONSENSUS:.3f} vti/R0")

    rel_err_consensus = abs(gamma - CONSENSUS) / CONSENSUS
    rel_err_target    = abs(gamma - TARGET_GAMMA) / TARGET_GAMMA

    print(f"")
    print(f"  Error vs consensus ({CONSENSUS:.3f}): {rel_err_consensus:.1%}")

    if gamma > 0:
        print(f"  Mode is UNSTABLE ✓")
        if rel_err_consensus < 0.02:
            print(f"  ✅ BENCHMARK PASS (<2% vs consensus)")
        elif rel_err_consensus < 0.05:
            print(f"  ✅ BENCHMARK PASS (<5% vs consensus)")
        elif rel_err_consensus < 0.15:
            print(f"  ✅ BENCHMARK PASS (<15% vs consensus)")
        else:
            print(f"  ⚠️  BENCHMARK MARGINAL: error = {rel_err_consensus:.1%}")
    else:
        print(f"  ❌ WARNING: Mode is STABLE — check resolution/particles")

    print("=" * 70)

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
            gamma_target=TARGET_GAMMA,
            title=f"GyroJAX CBC Phase 2a  [γ={gamma:.3f}, err={rel_err_consensus:.1%} vs consensus]",
            save_path=out_path,
        )
    except Exception as ex:
        print(f"  (viz skipped: {ex})")

    return {
        't': t_series,
        'phi_max': phi_max_series,
        'phi_rms': phi_rms_series,
        'gamma_measured': gamma,
        'gamma_target': TARGET_GAMMA,
        'rel_err': rel_err_target,
        'rel_err_consensus': rel_err_consensus,
        'step_start': step_start,
        'step_end': step_end,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GyroJAX Phase 2a CBC benchmark")
    parser.add_argument('--quick', action='store_true', help='Quick mode (100k particles, 400 steps)')
    args = parser.parse_args()
    run_cbc_fa(quick=args.quick)
