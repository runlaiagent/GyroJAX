"""
Rosenbluth-Hinton zonal flow residual test.

Initializes a zonal-flow-only perturbation (no ITG drive), then measures
how the zonal φ decays toward the theoretical neoclassical residual:

    φ_res / φ_0 = 1 / (1 + 1.6 · q² / √ε)

For CBC parameters (q=1.4, ε=0.18):  residual ≈ 0.119

Reference: Rosenbluth & Hinton, Phys. Rev. Lett. 80, 724 (1998)
           Dimits et al., Phys. Plasmas 7, 969 (2000)

Usage:
    python benchmarks/rosenbluth_hinton.py [--quick]
"""

import sys, os
import argparse
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.geometry.field_aligned import build_field_aligned_geometry
from gyrojax.simulation_fa import _run_with_geom


def rh_theory(q: float, epsilon: float) -> float:
    """Rosenbluth-Hinton residual: 1 / (1 + 1.6*q^2/sqrt(eps))"""
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(epsilon))


def extract_zonal_rms(phi: np.ndarray) -> float:
    """
    Extract zonal-flow component of phi.
    Zonal = ky=0, kz=0 → average over theta and alpha (last two dims).
    Returns rms of phi averaged over (theta, alpha).
    """
    phi_zonal = phi.mean(axis=(1, 2))   # shape (Npsi,)
    return float(np.sqrt(np.mean(phi_zonal**2)))


def run_rh_test(quick: bool = True) -> dict:
    q0    = 1.4
    eps   = 0.18       # a/R0
    R0    = 1.0
    a     = R0 * eps

    theory = rh_theory(q0, eps)

    if quick:
        Npsi, Ntheta, Nalpha = 16, 32, 16
        N_particles = 50_000
        n_steps     = 80
    else:
        Npsi, Ntheta, Nalpha = 32, 64, 32
        N_particles = 500_000
        n_steps     = 200

    # No ITG drive — pure zonal flow test
    cfg = SimConfigFA(
        Npsi=Npsi, Ntheta=Ntheta, Nalpha=Nalpha,
        N_particles=N_particles,
        n_steps=n_steps,
        dt=0.05,
        R0=R0, a=a, B0=1.0, q0=q0, q1=0.0,
        Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
        vti=1.0, n0_avg=1.0,
        R0_over_LT=0.0,    # ← NO ITG drive
        R0_over_Ln=0.0,
        vpar_cap=4.0,
        # Zonal seed: initialize with radial weight perturbation
        # (handled below via custom key + weight init)
    )

    # We need to capture phi at each step.
    # Use _run_with_geom internals — the simulation returns diags which
    # has phi_max history but not full phi snapshots.
    # Instead: run and capture phi_rms_history, use it as proxy.
    # The zonal damping shows up as decay in overall phi_rms when only
    # zonal modes are initialized.

    print(f"[R-H test] grid ({Npsi},{Ntheta},{Nalpha}), N={N_particles:,}, {n_steps} steps")
    print(f"[R-H test] Theory residual: {theory:.4f}")

    key = jax.random.PRNGKey(77)
    geom = build_field_aligned_geometry(
        Npsi=Npsi, Ntheta=Ntheta, Nalpha=Nalpha,
        R0=R0, a=a, B0=1.0, q0=q0, q1=0.0,
    )

    # Initialize with zonal perturbation:
    # weight ~ amp * cos(2π * r / a)  so δn is purely radial (zonal)
    from gyrojax.particles.guiding_center import init_maxwellian_particles
    from gyrojax.geometry.salpha import build_salpha_geometry

    geom_sa = build_salpha_geometry(Npsi, Ntheta, Nalpha, R0=R0, a=a,
                                     B0=1.0, q0=q0, q1=0.0)
    key, pkey = jax.random.split(key)
    state0 = init_maxwellian_particles(N_particles, geom_sa, 1.0, 1.0, 1.0, pkey)

    # Perturb weights: zonal cosine in radius
    amp = 0.02
    r_norm = (state0.r - a * 0.0) / a   # r/a ∈ [0,1]
    w_zonal = amp * jnp.cos(2.0 * jnp.pi * r_norm)
    state0 = state0._replace(weight=w_zonal)

    # Run simulation — returns diags with phi_rms per step
    diags, state_f, phi_f, _ = _run_with_geom(cfg, geom, key, verbose=False,
                                                state0_override=state0)

    phi_rms_hist = np.array(diags['phi_rms'])
    phi0 = phi_rms_hist[0] if phi_rms_hist[0] > 1e-14 else phi_rms_hist[1]

    if phi0 < 1e-14:
        print("  WARNING: initial phi is zero — zonal seed may not have excited φ")
        return {'theory': theory, 'measured': float('nan'), 'error_pct': float('nan')}

    # Residual = mean of last 25% of run
    n_tail = max(5, n_steps // 4)
    phi_tail = phi_rms_hist[-n_tail:]
    residual = float(np.mean(phi_tail)) / float(phi0)

    error_pct = abs(residual - theory) / theory * 100.0

    print(f"\n{'='*60}")
    print(f"  Rosenbluth-Hinton Zonal Flow Test")
    print(f"  {'─'*54}")
    print(f"  GyroJAX residual:  {residual:.4f}")
    print(f"  Theory (R-H):      {theory:.4f}")
    print(f"  Error:             {error_pct:.1f}%")
    status = "✅ PASS" if error_pct < 30 else "⚠️  MARGINAL" if error_pct < 60 else "❌ FAIL"
    print(f"  Status:            {status}")
    print(f"{'='*60}\n")

    return {'theory': theory, 'measured': residual, 'error_pct': error_pct,
            'phi_rms_hist': phi_rms_hist.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full',  action='store_true', default=False)
    args = parser.parse_args()
    quick = not args.full
    run_rh_test(quick=quick)
