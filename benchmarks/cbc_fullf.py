"""
Full-f Cyclone Base Case benchmark — Phase 3.

Compares full-f result to δf Phase 2a result.
In the linear phase, full-f and δf should give the same growth rate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf


def extract_growth_rate(phi_max: np.ndarray, dt: float, window: float = 0.4) -> float:
    n = len(phi_max)
    n0 = int(n * (1 - window))
    t = np.arange(n0, n) * dt
    log_phi = np.log(np.maximum(phi_max[n0:], 1e-20))
    return float(np.polyfit(t, log_phi, 1)[0])


def run_cbc_fullf(quick: bool = False):
    print("=" * 62)
    print("  GyroJAX Phase 3 — Full-f CBC benchmark")
    print("  Comparison target: δf Phase 2a γ ≈ 0.17 vti/R0")
    print("=" * 62)

    if quick:
        cfg = SimConfigFullF(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=50_000,
            n_steps=200, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0, vpar_cap=4.0,
            resample_interval=50,
        )
        print("  [QUICK MODE]")
    else:
        cfg = SimConfigFullF(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=1_000_000,
            n_steps=400, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0, vpar_cap=4.0,
            resample_interval=100,
        )

    print(f"\n  Grid: ({cfg.Npsi},{cfg.Ntheta},{cfg.Nalpha}), "
          f"N={cfg.N_particles:,}, steps={cfg.n_steps}\n")

    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fullf(cfg, key=key)

    phi_max = np.array([float(d.phi_max) for d in diags])
    phi_rms = np.array([float(d.phi_rms) for d in diags])
    n_rms   = np.array([float(d.n_rms)   for d in diags])
    t       = np.arange(len(diags)) * cfg.dt

    gamma = extract_growth_rate(phi_max, cfg.dt, window=0.4)
    target = 0.17
    rel_err = abs(gamma - target) / target

    print("\n" + "=" * 62)
    print(f"  RESULTS (Full-f CBC):")
    print(f"  γ_fullf  = {gamma:.4f} vti/R0")
    print(f"  γ_target = {target:.4f} vti/R0  (Dimits 2000)")
    print(f"  Error    = {rel_err:.1%}")
    if gamma > 0:
        print(f"  ✅ Mode UNSTABLE")
        if rel_err < 0.20:
            print(f"  ✅ Full-f agrees with δf within 20%")
        else:
            print(f"  ⚠️  Larger deviation — may need more particles or resampling")
    else:
        print(f"  ❌ Mode STABLE — check seeding or resolution")
    print("=" * 62)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].semilogy(t, phi_max + 1e-20, 'b-', label='|φ|_max')
        axes[0].semilogy(t, phi_rms + 1e-20, 'r--', label='|φ|_rms')
        axes[0].set_xlabel('t'); axes[0].set_ylabel('|φ|')
        axes[0].set_title('Full-f CBC'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, phi_max, 'b-')
        axes[1].set_title(f'γ = {gamma:.3f} (target 0.17, err {rel_err:.1%})')
        axes[1].set_xlabel('t'); axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(t, n_rms + 1e-20, 'g-')
        axes[2].set_xlabel('t'); axes[2].set_ylabel('δn/n0 rms')
        axes[2].set_title('Density fluctuation'); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), 'cbc_fullf_result.png')
        plt.savefig(out, dpi=120); plt.close()
        print(f"\n  Plot: {out}")
    except ImportError:
        pass

    return {'gamma': gamma, 't': t, 'phi_max': phi_max}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_cbc_fullf(quick=args.quick)
