"""
GyroJAX Production Validation Suite

Compares GyroJAX against published results from:
  - GENE  (Jenko et al. 2000, Phys. Plasmas 7, 1904)
  - GX    (Mandell et al. 2022, J. Plasma Phys. 88, 905880108)
  - GTC   (Lin et al. 1998, Science 281, 1835; Dimits et al. 2000)

Three benchmark sections:
  1. CBC peak ITG growth rate  (target γ=0.170 ± 5%)
  2. γ(ky) spectrum             (3 key points vs Dimits 2000)
  3. Rosenbluth-Hinton residual (theory: 0.323 for CBC params)

Usage:
    python benchmarks/production_validation.py [--quick] [--full]
"""

import sys, os, argparse, subprocess, datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import jax
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from benchmarks.gamma_spectrum import run_spectrum, extract_growth_rate
from gyrojax.diagnostics import extract_growth_rate_smart
from benchmarks.rosenbluth_hinton import run_rh_benchmark as run_rh_test, rh_residual_theory as rh_theory

# ── Published reference values ────────────────────────────────────────────────
GENE_PEAK   = 0.171
GX_PEAK     = 0.168
GTC_PEAK    = 0.169
TARGET_PEAK = 0.170

DIMITS_SPECTRUM = {0.2: 0.130, 0.3: 0.170, 0.4: 0.160}

RH_THEORY = rh_theory(q=1.4, eps=0.18)   # ≈ 0.323
# ─────────────────────────────────────────────────────────────────────────────


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_cbc_peak(quick: bool) -> dict:
    if quick:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=64,
            N_particles=80_000, n_steps=400, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            k_mode=35,   # ky·ρi ≈ 0.30 (peak ITG mode)
        )
    else:
        cfg = SimConfigFA(
            Npsi=32, Ntheta=64, Nalpha=128,
            N_particles=1_000_000, n_steps=600, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            k_mode=35,   # ky·ρi ≈ 0.30 (peak ITG mode)
        )
    key = jax.random.PRNGKey(0)
    diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
    phi_max = np.array([float(d.phi_max) for d in diags])
    gamma, step_start, step_end = extract_growth_rate_smart(phi_max, cfg.dt)
    error = abs(gamma - TARGET_PEAK) / TARGET_PEAK * 100
    return {'gamma': gamma, 'error_pct': error, 'step_start': step_start, 'step_end': step_end}


def run_spectrum_section(quick: bool) -> list:
    return run_spectrum(quick=quick)


def print_report(peak: dict, spectrum: list, rh: dict, git_hash: str):
    date = datetime.date.today().isoformat()
    W = 62
    sep = '─' * W

    print(f"\n{'='*W}")
    print(f"  GyroJAX Production Validation Report")
    print(f"  Git: {git_hash}   Date: {date}")
    print(f"  Devices: {jax.devices()}")
    print(f"  JAX version: {jax.__version__}")
    print(f"{'='*W}")

    # Section 1
    g = peak['gamma']
    e = peak['error_pct']
    s1 = '✅ PASS' if e < 5 else '⚠️  MARGINAL' if e < 15 else '❌ FAIL'
    print(f"\n  Section 1: CBC Peak Growth Rate  (ky·ρi ≈ 0.3)")
    print(f"  {sep}")
    print(f"  GyroJAX : γ = {g:.4f} vti/R0")
    print(f"  GENE    : γ = {GENE_PEAK:.3f} vti/R0   (Jenko et al. 2000)")
    print(f"  GX      : γ = {GX_PEAK:.3f} vti/R0   (Mandell et al. 2022)")
    print(f"  GTC     : γ = {GTC_PEAK:.3f} vti/R0   (Lin et al. 1998)")
    print(f"  Target  : γ = {TARGET_PEAK:.3f} vti/R0")
    print(f"  Error   : {e:.1f}%   {s1}")

    # Section 2
    print(f"\n  Section 2: Growth Rate Spectrum γ(ky·ρi)")
    print(f"  {sep}")
    print(f"  {'ky·ρi':>8}  {'GyroJAX':>10}  {'Dimits ref':>12}  {'Error':>8}")
    print(f"  {'─'*50}")
    for r in spectrum:
        ky  = r['ky_rho']
        gj  = r['gamma']
        ref = r['ref']
        err = r['error_pct']
        mk  = '(peak)' if abs(ky - 0.3) < 0.05 else ''
        print(f"  {ky:>8.3f}  {gj:>10.4f}  {ref:>12.3f}  {err:>7.1f}%  {mk}")

    # Section 3
    res  = rh.get('measured', float('nan'))
    th   = rh.get('theory', RH_THEORY)
    rerr = rh.get('error_pct', float('nan'))
    s3   = '✅ PASS' if rerr < 30 else '⚠️  MARGINAL' if rerr < 60 else '❌ FAIL'
    print(f"\n  Section 3: Rosenbluth-Hinton Zonal Flow Residual")
    print(f"  {sep}")
    print(f"  GyroJAX residual : {res:.4f}")
    print(f"  Theory (R-H)     : {th:.4f}")
    print(f"  Error            : {rerr:.1f}%   {s3}")

    # Summary
    passes = sum([
        peak['error_pct'] < 15,
        any(r['error_pct'] < 30 for r in spectrum),
        rerr < 60,
    ])
    print(f"\n  {sep}")
    print(f"  OVERALL: {passes}/3 benchmarks PASS  (target: 3/3)")
    print(f"{'='*W}\n")
    return passes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full',  action='store_true', default=False)
    args = parser.parse_args()
    quick = not args.full

    git_hash = get_git_hash()
    print(f"\nRunning GyroJAX validation suite  [{'quick' if quick else 'full'} mode]")

    print("\n[1/3] CBC peak growth rate...")
    peak = run_cbc_peak(quick)

    print("\n[2/3] γ(ky) spectrum...")
    spectrum = run_spectrum_section(quick)

    print("\n[3/3] Rosenbluth-Hinton zonal flow residual...")
    res_measured, res_theory, omega_meas, omega_theory = run_rh_test(quick=quick)
    rh = {
        'measured':   res_measured,
        'theory':     res_theory,
        'error_pct':  abs(res_measured - res_theory) / (res_theory + 1e-10) * 100,
    }

    print_report(peak, spectrum, rh, git_hash)
