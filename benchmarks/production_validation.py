"""
GyroJAX Production Validation Suite

Compares GyroJAX against published results from:
  - GENE  (Jenko et al. 2000, Phys. Plasmas 7, 1904)
  - GX    (Mandell et al. 2022, J. Plasma Phys. 88, 905880108)
  - GTC   (Lin et al. 1998, Science 281, 1835; Dimits et al. 2000)

Three benchmark sections:
  1. CBC peak ITG growth rate  (target γ=0.170 ± 5%)
  2. γ(ky) spectrum             (key points vs Dimits 2000)
  3. Rosenbluth-Hinton residual (theory: ~0.087 for CBC params)

Usage:
    python benchmarks/production_validation.py [--quick] [--full]
"""

import sys, os, argparse, subprocess, datetime, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Published reference values ────────────────────────────────────────────────
GENE_PEAK   = 0.171
GX_PEAK     = 0.168
GTC_PEAK    = 0.169
TARGET_PEAK = 0.170

DIMITS_SPECTRUM = {0.2: 0.130, 0.3: 0.170, 0.4: 0.160}

# ─────────────────────────────────────────────────────────────────────────────

# ── Worker entry-points (run in subprocess) ───────────────────────────────────

def _worker_cbc_peak(quick: bool):
    """Run CBC peak benchmark and print JSON result."""
    import jax
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
    from gyrojax.diagnostics import extract_growth_rate_smart

    if quick:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=64,
            N_particles=300_000, n_steps=600, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            k_mode=35,
        )
    else:
        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=64,
            N_particles=400_000, n_steps=800, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
            k_mode=35,
        )

    diags, _, _, _ = run_simulation_fa(cfg, jax.random.PRNGKey(0), verbose=True)
    phi_max = np.array([float(d.phi_max) for d in diags])
    gamma, step_start, step_end = extract_growth_rate_smart(phi_max, cfg.dt)
    error = abs(gamma - TARGET_PEAK) / TARGET_PEAK * 100
    print("__RESULT__" + json.dumps({'gamma': gamma, 'error_pct': error,
                                      'step_start': int(step_start), 'step_end': int(step_end)}))


def _worker_spectrum(quick: bool):
    """Run γ(ky) spectrum and print JSON result."""
    import jax
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
    from benchmarks.gamma_spectrum import estimate_ky_rho, extract_growth_rate, DIMITS_REF

    a, q0, q1, rho_star = 0.18, 1.4, 0.5, 1.0/180.0
    q_ref = q0 + q1 * 0.5**2

    if quick:
        k_modes = [12, 24, 35, 47, 59]
        N_particles, n_steps, Nalpha = 200_000, 500, 64
    else:
        k_modes = [6, 12, 18, 24, 35, 47, 59, 71]
        N_particles, n_steps, Nalpha = 400_000, 600, 96

    results = []
    for k_mode in k_modes:
        ky_rho = estimate_ky_rho(k_mode, q_ref, rho_star)
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
        diags, _, _, _ = run_simulation_fa(cfg, jax.random.PRNGKey(42 + k_mode), verbose=False)
        gamma = extract_growth_rate([float(d.phi_max) for d in diags], cfg.dt)
        error = abs(gamma - ref_gamma) / ref_gamma * 100 if ref_gamma > 0 else float('nan')
        results.append({'k_mode': k_mode, 'ky_rho': ky_rho, 'gamma': gamma,
                        'ref': ref_gamma, 'error_pct': error})
        print(f"  k_mode={k_mode:3d}  ky·ρi={ky_rho:.3f}  γ={gamma:.4f}  ref={ref_gamma:.3f}  err={error:.1f}%",
              flush=True)

    print("__RESULT__" + json.dumps(results))


def _worker_rh(quick: bool):
    """Run R-H benchmark and print JSON result."""
    from benchmarks.rosenbluth_hinton import run_rh_benchmark
    res_measured, res_theory, omega_meas, omega_theory = run_rh_benchmark(quick=quick)
    print("__RESULT__" + json.dumps({
        'measured': res_measured, 'theory': res_theory,
        'error_pct': abs(res_measured - res_theory) / (res_theory + 1e-10) * 100,
        'omega_measured': omega_meas, 'omega_theory': omega_theory,
    }))


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_in_subprocess(worker: str, quick: bool) -> dict | list:
    """Run a worker function in a fresh Python process, return its JSON result."""
    script = f"""
import sys, os
sys.path.insert(0, {repr(os.path.join(os.path.dirname(__file__), '..'))})
from benchmarks.production_validation import {worker}
{worker}({'True' if quick else 'False'})
"""
    python = sys.executable
    result = subprocess.run([python, '-c', script],
                            capture_output=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Print stderr (simulation progress) live-ish
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"    {line}")
    # Extract __RESULT__ line
    for line in result.stdout.splitlines():
        print(f"    {line}")
        if line.startswith('__RESULT__'):
            return json.loads(line[len('__RESULT__'):])
    raise RuntimeError(f"Worker {worker} failed (exit {result.returncode}):\n{result.stderr[-500:]}")


# ── Report ────────────────────────────────────────────────────────────────────

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return 'unknown'


def print_report(peak: dict, spectrum: list, rh: dict, git_hash: str):
    import jax
    date = datetime.date.today().isoformat()
    W = 62
    sep = '─' * W

    print(f"\n{'='*W}")
    print(f"  GyroJAX Production Validation Report")
    print(f"  Git: {git_hash}   Date: {date}")
    print(f"{'='*W}")

    # Section 1
    g, e = peak['gamma'], peak['error_pct']
    s1 = '✅ PASS' if e < 5 else '⚠️  MARGINAL' if e < 15 else '❌ FAIL'
    print(f"\n  Section 1: CBC Peak Growth Rate  (ky·ρi ≈ 0.3)")
    print(f"  {sep}")
    print(f"  GyroJAX : γ = {g:.4f} vti/R0")
    print(f"  GENE    : γ = {GENE_PEAK:.3f} vti/R0")
    print(f"  GX      : γ = {GX_PEAK:.3f} vti/R0")
    print(f"  GTC     : γ = {GTC_PEAK:.3f} vti/R0")
    print(f"  Target  : γ = {TARGET_PEAK:.3f} vti/R0")
    print(f"  Error   : {e:.1f}%   {s1}")

    # Section 2
    print(f"\n  Section 2: Growth Rate Spectrum γ(ky·ρi)")
    print(f"  {sep}")
    print(f"  {'ky·ρi':>8}  {'GyroJAX':>10}  {'Dimits ref':>12}  {'Error':>8}")
    print(f"  {'─'*50}")
    s2_pass = 0
    for r in spectrum:
        ky, gj, ref, err = r['ky_rho'], r['gamma'], r['ref'], r['error_pct']
        flag = '✅' if err < 25 else '⚠️ '
        if err < 25: s2_pass += 1
        pk = '(peak)' if abs(ky - 0.3) < 0.06 else ''
        print(f"  {ky:>8.3f}  {gj:>10.4f}  {ref:>12.3f}  {err:>7.1f}%  {flag} {pk}")

    # Section 3
    res, th = rh['measured'], rh['theory']
    rerr = rh['error_pct']
    s3 = '✅ PASS' if rerr < 30 else '⚠️  MARGINAL' if rerr < 60 else '❌ FAIL'
    print(f"\n  Section 3: Rosenbluth-Hinton Zonal Flow Residual")
    print(f"  {sep}")
    print(f"  GyroJAX residual : {res:.4f}")
    print(f"  Theory (R-H)     : {th:.4f}")
    print(f"  Error            : {rerr:.1f}%   {s3}")
    omega_m, omega_t = rh.get('omega_measured', 0), rh.get('omega_theory', 0)
    gam_err = abs(omega_m - omega_t) / omega_t * 100 if omega_t > 0 else 100
    sg = '✅' if gam_err < 15 else '⚠️ ' if gam_err < 30 else '❌'
    print(f"  GAM ω measured   : {omega_m:.4f}  (theory: {omega_t:.4f})  err={gam_err:.1f}%  {sg}")

    passes = sum([e < 15, s2_pass >= len(spectrum) // 2, rerr < 60])
    print(f"\n  {sep}")
    print(f"  OVERALL: {passes}/3 sections pass")
    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Worker dispatch (called from subprocess)
    if '--_worker' in sys.argv:
        idx = sys.argv.index('--_worker')
        worker_name = sys.argv[idx + 1]
        quick_flag = '--quick' in sys.argv
        globals()[worker_name](quick_flag)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full',  action='store_true', default=False)
    args = parser.parse_args()
    quick = not args.full

    git_hash = get_git_hash()
    mode = 'quick' if quick else 'full'
    print(f"\nGyroJAX validation suite  [{mode} mode]  git={git_hash}")
    print("Each section runs in an isolated subprocess (fresh CUDA context)\n")

    print("[1/3] CBC peak growth rate...")
    peak = run_in_subprocess('_worker_cbc_peak', quick)

    print("\n[2/3] γ(ky) spectrum...")
    spectrum = run_in_subprocess('_worker_spectrum', quick)

    print("\n[3/3] Rosenbluth-Hinton zonal flow residual...")
    rh = run_in_subprocess('_worker_rh', quick)

    print_report(peak, spectrum, rh, git_hash)
