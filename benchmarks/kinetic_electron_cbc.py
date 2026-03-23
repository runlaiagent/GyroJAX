"""
Kinetic electron CBC benchmark.

Compares ITG growth rates for:
1. Adiabatic electrons (baseline)
2. Drift-kinetic electrons (DK)

Usage:
    python benchmarks/kinetic_electron_cbc.py
    python benchmarks/kinetic_electron_cbc.py --quick
"""

from __future__ import annotations
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from gyrojax.simulation_fa import run_simulation_fa, SimConfigFA
from gyrojax.diagnostics import extract_growth_rate


def run_cbc(electron_model: str, quick: bool = False) -> dict:
    if quick:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=2000,
            n_steps=80,
            dt=0.1,
            electron_model=electron_model,
            subcycles_e=2 if electron_model == 'drift_kinetic' else 10,
            N_electrons=1000 if electron_model == 'drift_kinetic' else 0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
        )
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=32,
            N_particles=50_000,
            n_steps=500,
            dt=0.05,
            electron_model=electron_model,
            subcycles_e=4 if electron_model == 'drift_kinetic' else 10,
            N_electrons=0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
        )

    print(f"\n{'='*60}")
    print(f"  Electron model: {electron_model.upper()}")
    print(f"  Grid: ({cfg.Npsi},{cfg.Ntheta},{cfg.Nalpha}), N_particles={cfg.N_particles:,}")
    print(f"  n_steps={cfg.n_steps}, dt={cfg.dt}")
    print(f"{'='*60}")

    diags, state, phi, geom = run_simulation_fa(cfg, jax.random.PRNGKey(42), verbose=True)
    phi_rms = np.array([float(d.phi_rms) for d in diags])

    gamma = extract_growth_rate(phi_rms, cfg.dt, window=0.4)
    print(f"\n  → Growth rate γ = {gamma:.4f} [R0/vti]")

    return {
        'model': electron_model,
        'gamma': gamma,
        'phi_rms': phi_rms,
        'cfg': cfg,
    }


def main():
    parser = argparse.ArgumentParser(description='Kinetic electron CBC benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick mode (small grid)')
    args = parser.parse_args()

    results = {}
    for model in ['adiabatic', 'drift_kinetic']:
        results[model] = run_cbc(model, quick=args.quick)

    print('\n' + '='*60)
    print('  SUMMARY: ITG Growth Rates')
    print('='*60)
    print(f"  Adiabatic electrons:    γ = {results['adiabatic']['gamma']:.4f} R0/vti")
    print(f"  Drift-kinetic electrons: γ = {results['drift_kinetic']['gamma']:.4f} R0/vti")
    print()
    print("  Notes:")
    print("  - For pure ITG (no ETG drive), adiabatic and DK should give similar γ")
    print("  - DK electrons capture TEM (trapped electron modes) at higher k_theta*rho_i")
    print("  - Significant γ difference indicates electron physics is important")
    gamma_diff = abs(results['drift_kinetic']['gamma'] - results['adiabatic']['gamma'])
    print(f"  - |γ_DK - γ_ad| = {gamma_diff:.4f} R0/vti")
    print('='*60)


if __name__ == '__main__':
    main()
