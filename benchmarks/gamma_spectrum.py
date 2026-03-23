"""
Linear growth rate spectrum γ(ky·ρi) for CBC.

Sweeps over Nalpha to sample different dominant toroidal mode numbers,
extracting γ at each effective ky.

Reference: Dimits et al. 2000, Phys. Plasmas 7, 969 — Table I.

Usage:
    python benchmarks/gamma_spectrum.py [--quick]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
import jax

# Dimits 2000 reference spectrum
DIMITS_REF = {
    0.1: 0.050, 0.2: 0.130, 0.3: 0.170,
    0.4: 0.160, 0.5: 0.140, 0.6: 0.100,
    0.7: 0.050, 0.8: 0.010,
}


def estimate_ky_rho(Nalpha: int, a: float, rho_i: float) -> float:
    """
    Dominant ky·ρi for a box of width 2π·a in α with Nalpha cells.
    Minimum resolved wavenumber: ky_min = 1 / (a/(2π)) = 2π/a in field-line coords.
    Dominant mode: ky ≈ (Nalpha//4) * (2π / (a * Nalpha)) = π / (2*a)
    In GK normalization: ky·ρi where ρi = a/180
    """
    ky_dominant = 2.0 * np.pi * (Nalpha // 4) / (Nalpha * a)
    return ky_dominant * rho_i


def extract_growth_rate(phi_max: list, dt: float, window: float = 0.35) -> float:
    arr = np.array(phi_max)
    arr = np.maximum(arr, 1e-20)
    n = len(arr)
    n0 = int(n * (1 - window))
    t = np.arange(n0, n) * dt
    log_phi = np.log(arr[n0:])
    if np.all(np.diff(log_phi) < 0):
        return 0.0
    coeffs = np.polyfit(t, log_phi, 1)
    return float(max(coeffs[0], 0.0))


def run_spectrum(quick: bool = True) -> list:
    a   = 0.18
    rho_i = a / 180.0  # ρi = a/180 for CBC

    if quick:
        nalpha_list = [8, 16, 24]
        N_particles = 30_000
        n_steps     = 150
    else:
        nalpha_list = [8, 12, 16, 24, 32, 48]
        N_particles = 200_000
        n_steps     = 300

    results = []
    print(f"\n{'='*65}")
    print(f"  γ(ky·ρi) Spectrum Benchmark  —  CBC Parameters")
    print(f"  Reference: Dimits et al. 2000, Phys. Plasmas 7, 969")
    print(f"{'─'*65}")
    print(f"  {'ky·ρi':>8}  {'GyroJAX γ':>12}  {'Dimits ref':>12}  {'Error':>8}")
    print(f"{'─'*65}")

    for Nalpha in nalpha_list:
        ky_rho = estimate_ky_rho(Nalpha, a, rho_i)
        # Find closest Dimits reference
        closest_ky = min(DIMITS_REF.keys(), key=lambda k: abs(k - ky_rho))
        ref_gamma  = DIMITS_REF[closest_ky]

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=Nalpha,
            N_particles=N_particles, n_steps=n_steps, dt=0.05,
            R0=1.0, a=a, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
        )
        key = jax.random.PRNGKey(42 + Nalpha)
        diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
        gamma = extract_growth_rate(diags['phi_max'], cfg.dt)
        error = abs(gamma - ref_gamma) / ref_gamma * 100 if ref_gamma > 0 else float('nan')
        flag  = '✅' if error < 25 else '⚠️ '
        print(f"  {ky_rho:>8.3f}  {gamma:>12.4f}  {ref_gamma:>12.3f}  {error:>7.1f}%  {flag}")
        results.append({'Nalpha': Nalpha, 'ky_rho': ky_rho, 'gamma': gamma,
                        'ref': ref_gamma, 'error_pct': error})

    print(f"{'='*65}\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full',  action='store_true', default=False)
    args = parser.parse_args()
    run_spectrum(quick=not args.full)
