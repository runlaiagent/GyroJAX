"""
Linear growth rate spectrum γ(ky·ρi) for CBC.

Sweeps over binormal mode number k_mode (n in sin(2θ + n·α)) to sample
different ky·ρi values, extracting γ at each.

The physical perpendicular wavenumber at the reference radius r_ref = a/2:
    k_y · ρ_i = k_mode · q(r_ref) · ρ_star
where ρ_star = ρ_i / a = 1/180 for CBC.

Reference: Dimits et al. 2000, Phys. Plasmas 7, 969 — Table I.

Usage:
    python benchmarks/gamma_spectrum.py [--quick]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
import jax

# Dimits 2000 reference spectrum (ky·ρi → γ in vti/R0)
DIMITS_REF = {
    0.1: 0.050, 0.2: 0.130, 0.3: 0.170,
    0.4: 0.160, 0.5: 0.140, 0.6: 0.100,
    0.7: 0.050, 0.8: 0.010,
}


def estimate_ky_rho(k_mode: int, q_ref: float, rho_star: float) -> float:
    """
    Physical binormal wavenumber for mode number k_mode.

    In field-aligned (ψ, θ, α) coordinates the α-grid spans [0, 2π).
    The physical perpendicular wavenumber at the reference radius is:
        k_y = k_mode · |∇α| = k_mode · q(r_ref) / r_ref
    In gyrokinetic normalization (lengths in units of a):
        k_y · ρ_i = k_mode · q_ref · ρ_star
    where ρ_star = ρ_i / a.
    """
    return k_mode * q_ref * rho_star


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
    a        = 0.18
    q0       = 1.4
    q1       = 0.5
    rho_star = 1.0 / 180.0

    # Reference radius for ky estimate: mid-radius r = a/2
    # q(r_ref) = q0 + q1*(0.5)^2
    q_ref = q0 + q1 * 0.5**2   # ≈ 1.525

    if quick:
        # k_modes chosen to cover ky·ρi ~ 0.1 to 0.6
        # k_mode * q_ref * rho_star ≈ k_mode * 1.525 / 180 ≈ k_mode * 0.00847
        # To hit ky~0.2: k_mode ~ 24; ky~0.3: k_mode~35; ky~0.4: k_mode~47
        k_modes     = [12, 24, 35, 47, 59]
        N_particles = 50_000
        n_steps     = 300
        Nalpha      = 64   # must be >= 2*max(k_mode) to resolve the mode
    else:
        k_modes     = [6, 12, 18, 24, 35, 47, 59, 71]
        N_particles = 500_000
        n_steps     = 600
        Nalpha      = 128

    results = []
    print(f"\n{'='*65}")
    print(f"  γ(ky·ρi) Spectrum Benchmark  —  CBC Parameters")
    print(f"  Reference: Dimits et al. 2000, Phys. Plasmas 7, 969")
    print(f"{'─'*65}")
    print(f"  {'ky·ρi':>8}  {'GyroJAX γ':>12}  {'Dimits ref':>12}  {'Error':>8}")
    print(f"{'─'*65}")

    for k_mode in k_modes:
        ky_rho = estimate_ky_rho(k_mode, q_ref, rho_star)
        # Find closest Dimits reference
        closest_ky = min(DIMITS_REF.keys(), key=lambda k: abs(k - ky_rho))
        ref_gamma  = DIMITS_REF[closest_ky]

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=Nalpha,
            N_particles=N_particles, n_steps=n_steps, dt=0.05,
            R0=1.0, a=a, B0=1.0, q0=q0, q1=q1,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            k_mode=k_mode,
        )
        key = jax.random.PRNGKey(42 + k_mode)
        diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
        gamma = extract_growth_rate([float(d.phi_max) for d in diags], cfg.dt)
        error = abs(gamma - ref_gamma) / ref_gamma * 100 if ref_gamma > 0 else float('nan')
        flag  = '✅' if error < 25 else '⚠️ '
        print(f"  {ky_rho:>8.3f}  {gamma:>12.4f}  {ref_gamma:>12.3f}  {error:>7.1f}%  {flag}")
        results.append({'k_mode': k_mode, 'ky_rho': ky_rho, 'gamma': gamma,
                        'ref': ref_gamma, 'error_pct': error})

    print(f"{'='*65}\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full',  action='store_true', default=False)
    args = parser.parse_args()
    run_spectrum(quick=not args.full)
