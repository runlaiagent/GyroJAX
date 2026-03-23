"""
Stellarator ITG benchmark — Phase 2b.

Uses VMEC wout geometry (li383 W7-X-like, 3 field periods) with the
full-FA simulation loop (exact Γ₀(b) Poisson, field-aligned coords).

There is no single published reference γ for li383 ITG, so we:
  1. Check that the mode is unstable (γ > 0) for R0/LT = 6.9
  2. Check that γ is in the expected ballpark for stellarator ITG:
     roughly 0.1–0.3 vti/R0 depending on geometry and kρ

Reference: Baumgaertel et al. (2011) PoP 18, 122301 (stellarator ITG benchmarks)
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.geometry.vmec_geometry import load_vmec_geometry
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

WOUT_LI383 = os.path.expanduser(
    "~/wlhx/booz_xform_jax/tests/test_files/wout_li383_1.4m.nc"
)


def extract_growth_rate(phi_max_series: np.ndarray, dt: float, window: float = 0.4) -> float:
    n       = len(phi_max_series)
    n_start = int(n * (1 - window))
    t       = np.arange(n_start, n) * dt
    log_phi = np.log(np.maximum(phi_max_series[n_start:], 1e-20))
    coeffs  = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_stellarator_itg(quick: bool = False):
    print("=" * 64)
    print("  GyroJAX Phase 2b — Stellarator ITG benchmark (li383)")
    print("  Geometry: W7-X-like quasi-isodynamic, nfp=3")
    print("=" * 64)

    if not os.path.exists(WOUT_LI383):
        print(f"ERROR: wout file not found at {WOUT_LI383}")
        return None

    # Load VMEC geometry
    print("\nLoading VMEC geometry...", flush=True)
    if quick:
        geom = load_vmec_geometry(WOUT_LI383, Ntheta=32, Nzeta=16, Ns_out=16)
        N_particles = 50_000
        n_steps = 150
    else:
        geom = load_vmec_geometry(WOUT_LI383, Ntheta=64, Nzeta=32, Ns_out=32)
        N_particles = 500_000
        n_steps = 400

    Ns, Nth, Nze = geom.B_field.shape
    B0   = float(geom.B0)
    R0   = float(geom.R0)
    a    = float(geom.a)
    print(f"  R0 = {R0:.3f} m, a = {a:.3f} m, B0 = {B0:.3f} T")
    print(f"  Grid: ({Ns}, {Nth}, {Nze})")
    print(f"  q range: [{float(geom.q_profile[0]):.2f}, {float(geom.q_profile[-1]):.2f}]")
    print(f"  N_particles = {N_particles:,}, n_steps = {n_steps}")

    dt = 0.05   # R0/vti
    vti = 1.0
    cfg = SimConfigFA(
        Npsi=Ns, Ntheta=Nth, Nalpha=Nze,
        N_particles=N_particles,
        n_steps=n_steps,
        dt=dt,
        R0=R0, a=a, B0=B0,
        q0=float(geom.q_profile[Ns//2]),   # mid-radius q
        q1=0.0,    # VMEC handles q via geometry
        Ti=1.0, Te=1.0, mi=1.0, e=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        vti=vti, n0_avg=1.0,
    )

    print("\nRunning simulation...\n", flush=True)
    key = jax.random.PRNGKey(42)

    # Override geometry in simulation — pass geom directly
    # (run_simulation_fa builds its own geom; we monkey-patch by using
    #  a wrapper that injects the VMEC geom)
    from gyrojax.simulation_fa import _run_with_geom
    diags, state, phi, _ = _run_with_geom(cfg, geom, key)

    phi_max_series = np.array([float(d.phi_max) for d in diags])
    phi_rms_series = np.array([float(d.phi_rms) for d in diags])
    t_series = np.arange(len(diags)) * dt

    gamma = extract_growth_rate(phi_max_series, dt, window=0.4)

    print("\n" + "=" * 64)
    print(f"  RESULTS (Stellarator ITG, li383):")
    print(f"  Measured growth rate γ = {gamma:.4f} vti/R0")
    print(f"  Expected range: ~0.1–0.3 vti/R0")
    if gamma > 0.05:
        print(f"  ✅ Mode is UNSTABLE — ITG present")
    elif gamma > 0:
        print(f"  ⚠️  Weak growth — may need more particles")
    else:
        print(f"  ❌ Mode STABLE — check resolution or profile drive")
    print("=" * 64)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(t_series, phi_max_series + 1e-20, 'b-', label='|φ|_max')
        axes[0].semilogy(t_series, phi_rms_series + 1e-20, 'r--', label='|φ|_rms')
        axes[0].set_xlabel('t [R0/vti]'); axes[0].set_ylabel('|φ|')
        axes[0].set_title('Stellarator ITG (li383)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_series, phi_max_series, 'b-')
        axes[1].set_xlabel('t [R0/vti]'); axes[1].set_ylabel('|φ|_max')
        axes[1].set_title(f'γ = {gamma:.3f} vti/R0')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), 'stellarator_itg_result.png')
        plt.savefig(out, dpi=120); plt.close()
        print(f"\n  Plot saved: {out}")
    except ImportError:
        pass

    return {'gamma': gamma, 't': t_series,
            'phi_max': phi_max_series, 'phi_rms': phi_rms_series}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_stellarator_itg(quick=args.quick)
