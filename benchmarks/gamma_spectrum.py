"""
ITG gamma spectrum benchmark: γ(ky·ρi) sweep across CBC parameters.

Compares measured growth rates to published reference values from
Nevins et al. (2006) and GENE/GX benchmarks.

Usage:
    python benchmarks/gamma_spectrum.py [--quick]
"""

import sys, os, argparse, json, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
import jax

# Dimits 2000 reference (kept for backward compatibility with tests)
DIMITS_REF = {
    0.1: 0.050, 0.2: 0.130, 0.3: 0.170,
    0.4: 0.160, 0.5: 0.140, 0.6: 0.100,
    0.7: 0.050, 0.8: 0.010,
}

# Nevins et al. 2006 / GENE / GX reference spectrum
REFERENCE_GAMMA = {
    0.10: 0.055,
    0.20: 0.110,
    0.30: 0.169,
    0.40: 0.160,
    0.50: 0.125,
    0.60: 0.075,
}

# CBC parameters
_A     = 0.18
_Q0    = 1.4
_Q1    = 0.5
_R_REF = _A * 0.5          # mid-radius = 0.09
_RHO_I = 1.0 / 1000.0      # vti/Omega_i (e=1000, B0=1, mi=1)
_Q_REF = _Q0 + _Q1 * 0.5**2  # ≈ 1.525
_KY_RHO_PER_K = _Q_REF * _RHO_I / _R_REF  # ≈ 0.01694


def estimate_ky_rho(k_mode: int, q_ref: float = None, rho_i_over_r_ref: float = None) -> float:
    """
    Physical binormal wavenumber for mode number k_mode.

    Signature supports both:
      estimate_ky_rho(k_mode)                       # uses CBC defaults
      estimate_ky_rho(k_mode, q_ref, rho_i_over_r_ref)  # explicit params
    """
    if q_ref is None:
        q_ref = _Q_REF
    if rho_i_over_r_ref is None:
        rho_i_over_r_ref = _RHO_I / _R_REF
    return k_mode * q_ref * rho_i_over_r_ref


def extract_growth_rate(phi_max: list, dt: float, window: float = 0.35) -> float:
    """Fit γ from log(phi_max) using trailing window of length `window` fraction."""
    arr = np.array(phi_max, dtype=float)
    arr = np.maximum(arr, 1e-20)
    n = len(arr)
    n0 = int(n * (1 - window))
    t = np.arange(n0, n) * dt
    log_phi = np.log(arr[n0:])
    if np.all(np.diff(log_phi) < 0):
        return 0.0
    coeffs = np.polyfit(t, log_phi, 1)
    return float(max(coeffs[0], 0.0))


def run_spectrum(quick: bool = False) -> list:
    if quick:
        k_modes     = [6, 12, 18, 24, 30, 36]
        N_particles = 50_000
        n_steps     = 100
        Nalpha      = 64
    else:
        k_modes     = [6, 12, 18, 24, 30, 36]
        N_particles = 300_000
        n_steps     = 200
        Nalpha      = 64

    print(f"\nGamma spectrum benchmark — CBC parameters (R/LT=6.9)")
    print("=" * 62)
    print(f"  {'k_mode':>6}  {'ky·ρi':>6}  {'γ_meas':>8}  {'γ_ref':>8}  {'error':>7}  status")
    print("-" * 62)

    results = []
    for k_mode in k_modes:
        ky_rho = estimate_ky_rho(k_mode)
        closest = min(REFERENCE_GAMMA.keys(), key=lambda k: abs(k - ky_rho))
        ref_gamma = REFERENCE_GAMMA[closest]

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=Nalpha,
            N_particles=N_particles, n_steps=n_steps, dt=0.05,
            R0=1.0, a=_A, B0=1.0, q0=_Q0, q1=_Q1,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            single_mode=True, k_mode=k_mode,
        )
        key = jax.random.PRNGKey(42 + k_mode)
        diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
        jax.clear_caches()
        gc.collect()

        # Use 40%-75% window for linear phase
        phi_series = [float(d.phi_max) for d in diags]
        arr = np.maximum(phi_series, 1e-20)
        n = len(arr)
        n0, n1 = int(n * 0.40), int(n * 0.75)
        if n1 <= n0 + 2:
            n0, n1 = 0, n
        t = np.arange(n0, n1) * cfg.dt
        log_phi = np.log(arr[n0:n1])
        coeffs = np.polyfit(t, log_phi, 1)
        gamma = float(max(coeffs[0], 0.0))

        error_pct = abs(gamma - ref_gamma) / ref_gamma * 100 if ref_gamma > 0 else float('nan')
        status = "PASS" if error_pct < 30 else "WARN"

        print(f"  {k_mode:>6}  {ky_rho:>6.3f}  {gamma:>8.4f}  {ref_gamma:>8.3f}  {error_pct:>6.1f}%  {status}")
        results.append({
            'k_mode': k_mode,
            'ky_rho': round(ky_rho, 4),
            'gamma_measured': round(gamma, 5),
            'gamma_reference': ref_gamma,
            'error_pct': round(error_pct, 2),
            'status': status,
        })

    errors = [r['error_pct'] for r in results if not np.isnan(r['error_pct'])]
    print("=" * 62)
    if errors:
        print(f"Mean error: {np.mean(errors):.1f}%   Max error: {max(errors):.1f}%")

    # Save JSON
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gamma_spectrum.json')
    with open(out_path, 'w') as f:
        json.dump({
            'benchmark': 'gamma_spectrum',
            'reference': 'Nevins et al. 2006 / GENE / GX',
            'cbc_params': {'R0_over_LT': 6.9, 'R0_over_Ln': 2.2, 'q0': _Q0, 'q1': _Q1, 'a': _A},
            'quick': quick,
            'N_particles': N_particles,
            'n_steps': n_steps,
            'results': results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Fast check (50k particles, 100 steps)')
    args = parser.parse_args()
    run_spectrum(quick=args.quick)
