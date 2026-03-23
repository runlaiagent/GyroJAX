"""
Collision scan benchmark — run CBC at nu=0, 0.01, 0.05, 0.1 for each model.

Verifies:
  - At nu→0: gamma → ~0.17 (CBC target)
  - At nu>0: gamma decreases (collisions stabilize ITG)

Usage:
  python benchmarks/collision_scan.py
  python benchmarks/collision_scan.py --quick
"""

import sys
import os
import argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


def extract_growth_rate(phi_max_series: np.ndarray, dt: float, window: float = 0.4) -> float:
    n = len(phi_max_series)
    n_start = int(n * (1 - window))
    t = np.arange(n_start, n) * dt
    log_phi = np.log(np.maximum(phi_max_series[n_start:], 1e-20))
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_scan(quick: bool = False):
    if quick:
        base_cfg = dict(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=20_000,
            n_steps=150,
            dt=0.05,
        )
    else:
        base_cfg = dict(
            Npsi=24, Ntheta=48, Nalpha=24,
            N_particles=100_000,
            n_steps=300,
            dt=0.05,
        )

    models = ['none', 'krook', 'lorentz', 'dougherty']
    nu_values = [0.0, 0.01, 0.05, 0.1]

    # Header
    print("\n" + "=" * 65)
    print("  GyroJAX Collision Scan — γ vs ν")
    print("=" * 65)
    header = f"{'model':<12} {'nu':>6}  {'gamma (vti/R0)':>16}  {'note'}"
    print(header)
    print("-" * 65)

    results = {}

    for model in models:
        for nu in nu_values:
            # 'none' model only needs nu=0 run
            if model == 'none' and nu > 0.0:
                continue

            cfg = SimConfigFA(
                **base_cfg,
                R0=1.0, a=0.18, B0=1.0,
                q0=1.4, q1=0.5,
                Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
                R0_over_LT=6.9, R0_over_Ln=2.2,
                vti=1.0, n0_avg=1.0,
                collision_model=model if nu > 0.0 or model == 'none' else 'none',
                nu_krook=nu,
                nu_ei=nu,
                nu_coll=nu,
            )

            key = jax.random.PRNGKey(42)
            diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=False)
            phi_max = np.array([float(d.phi_max) for d in diags])
            gamma = extract_growth_rate(phi_max, cfg.dt)

            nu_key = (model, nu)
            results[nu_key] = gamma

            note = ""
            if model == 'none' or nu == 0.0:
                note = "<-- target ~0.17"
            elif gamma < results.get(('none', 0.0), 0.17):
                note = "stabilized"

            label = model if nu > 0.0 or model == 'none' else 'none'
            print(f"  {label:<10} {nu:>6.3f}  {gamma:>16.4f}  {note}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="GyroJAX collision scan benchmark")
    parser.add_argument("--quick", action="store_true", help="Reduced resolution for fast test")
    args = parser.parse_args()
    run_scan(quick=args.quick)


if __name__ == "__main__":
    main()
