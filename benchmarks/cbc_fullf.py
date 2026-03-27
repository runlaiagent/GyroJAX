"""
Full-f Cyclone Base Case benchmark — Phase 3.

Compares full-f (true Vlasov PIC, dW/dt=0) result to δf Phase 2a result.
In the linear phase, full-f and δf should give the same growth rate γ.

Target: γ ≈ 0.169–0.172 vti/R0  (GENE/GS2 CBC reference, Cyclone 2000)
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf


def extract_growth_rate_smart(signal: np.ndarray, dt: float, window: float = 0.4) -> float:
    """
    Extract linear growth rate from log-linear fit over the last `window` fraction.
    Robust to initial transient.
    """
    n = len(signal)
    n0 = int(n * (1 - window))
    t = np.arange(n0, n) * dt
    log_phi = np.log(np.maximum(signal[n0:], 1e-30))
    # Linear fit
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_cbc_fullf(quick: bool = False):
    print("=" * 65)
    print("  GyroJAX Phase 3 — Full-f CBC benchmark")
    print("  True Vlasov PIC: dW/dt=0, δn=scatter(W)−n₀")
    print("  Reference: GENE/GS2 γ ≈ 0.169–0.172 vti/R0")
    print("=" * 65)

    if quick:
        cfg = SimConfigFullF(
            Npsi=16, Ntheta=32, Nalpha=32,
            N_particles=300_000,
            n_steps=200, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0, vpar_cap=4.0,
            pert_amp=1e-4,
            resample_interval=50,
            single_mode=True,
            k_mode=18,
        )
        print("  [QUICK MODE — single toroidal mode n=18, 300k particles]")
    else:
        cfg = SimConfigFullF(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=1_000_000,
            n_steps=400, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0, vpar_cap=4.0,
            pert_amp=1e-4,
            resample_interval=100,
            single_mode=True,
            k_mode=18,
        )

    print(f"\n  Grid: ({cfg.Npsi},{cfg.Ntheta},{cfg.Nalpha}), "
          f"N={cfg.N_particles:,}, steps={cfg.n_steps}")
    print(f"  dt={cfg.dt}, single_mode={cfg.single_mode}, k={cfg.k_mode}\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fullf(cfg, key=key, verbose=True)

    phi_max  = np.array([float(d.phi_max)       for d in diags])
    phi_rms  = np.array([float(d.phi_rms)        for d in diags])
    n_rms    = np.array([float(d.n_rms)          for d in diags])
    w_rms    = np.array([float(d.weight_rms)     for d in diags])
    t        = np.arange(len(diags)) * cfg.dt

    gamma_rms = extract_growth_rate_smart(phi_rms, cfg.dt, window=0.4)
    gamma_max = extract_growth_rate_smart(phi_max, cfg.dt, window=0.4)
    gamma     = gamma_rms   # use rms as primary

    gene_ref = 0.169
    df_ref   = 0.172
    rel_err  = abs(gamma - gene_ref) / gene_ref

    # Weight constancy check (full-f property: dW/dt=0)
    w_drift = abs(w_rms[-1] - w_rms[0]) / (w_rms[0] + 1e-30)

    print("\n" + "=" * 65)
    print(f"  RESULTS — Full-f CBC (true Vlasov PIC):")
    print(f"  γ_fullf  (rms) = {gamma_rms:.4f} vti/R0")
    print(f"  γ_fullf  (max) = {gamma_max:.4f} vti/R0")
    print(f"  γ_δf_ref       = {df_ref:.4f} vti/R0  (GyroJAX δf)")
    print(f"  γ_GENE_ref     = {gene_ref:.4f} vti/R0  (GENE, Dimits 2000)")
    print(f"  Error vs GENE  = {rel_err:.1%}")
    print(f"  Weight drift   = {w_drift:.2e}  (should be ~0 in full-f)")
    print()
    if gamma > 0:
        print(f"  ✅ Mode UNSTABLE (γ > 0)")
        if rel_err < 0.25:
            print(f"  ✅ Full-f agrees with GENE within 25%")
        else:
            print(f"  ⚠️  Larger deviation — may need more particles or grid resolution")
    else:
        print(f"  ❌ Mode appears STABLE — check seeding or resolution")
    if w_drift < 1e-3:
        print(f"  ✅ Weights constant (full-f property verified)")
    else:
        print(f"  ⚠️  Weight drift {w_drift:.2e} (weights should be constant)")
    print("=" * 65)

    # Comparison table
    print()
    print("  Comparison table:")
    print(f"  {'Method':<15} {'γ (vti/R0)':<15} {'Error vs GENE':<15}")
    print(f"  {'-'*45}")
    print(f"  {'δf (ref)':<15} {df_ref:<15.4f} {abs(df_ref-gene_ref)/gene_ref:<15.1%}")
    print(f"  {'Full-f':<15} {gamma:<15.4f} {rel_err:<15.1%}")
    print(f"  {'GENE':<15} {gene_ref:<15.4f} {'(reference)':<15}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].semilogy(t, phi_rms + 1e-20, 'b-', label='|φ|_rms')
        axes[0].semilogy(t, phi_max + 1e-20, 'r--', label='|φ|_max')
        axes[0].set_xlabel('t [R0/vti]'); axes[0].set_ylabel('|φ|')
        axes[0].set_title('Full-f CBC potential'); axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        n0 = int(len(t) * 0.6)
        slope_t = t[n0:]
        phi_fit = np.exp(gamma * (slope_t - slope_t[0]) + np.log(phi_rms[n0] + 1e-30))
        axes[1].semilogy(t, phi_rms + 1e-20, 'b-')
        axes[1].semilogy(slope_t, phi_fit, 'r--', label=f'γ={gamma:.3f}')
        axes[1].set_title(f'Growth rate fit: γ={gamma:.3f} (GENE={gene_ref})')
        axes[1].set_xlabel('t'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(t, n_rms + 1e-20, 'g-')
        axes[2].set_xlabel('t'); axes[2].set_ylabel('δn rms')
        axes[2].set_title('Density fluctuation'); axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, w_rms, 'm-')
        axes[3].set_xlabel('t'); axes[3].set_ylabel('weight rms')
        axes[3].set_title('Weight rms (should be constant)')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), 'cbc_fullf_result.png')
        plt.savefig(out, dpi=120); plt.close()
        print(f"\n  Plot saved: {out}")
    except ImportError:
        pass

    return {
        'gamma': gamma, 'gamma_rms': gamma_rms, 'gamma_max': gamma_max,
        't': t, 'phi_rms': phi_rms, 'phi_max': phi_max,
        'w_rms': w_rms, 'gene_ref': gene_ref,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (300k particles, 16×32×32)')
    args = parser.parse_args()
    run_cbc_fullf(quick=args.quick or True)   # default to quick for now
