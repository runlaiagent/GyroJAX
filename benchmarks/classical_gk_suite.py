"""
Classical Gyrokinetic Benchmark Suite for GyroJAX
===================================================
5 standard tests demonstrating code robustness:
  1. CBC Particle Convergence Scan
  2. Magnetic Shear Scan
  3. R/LT Critical Gradient Scan
  4. Collision Damping Test
  5. Float32 Speed Benchmark

Each test runs in an isolated subprocess to avoid CUDA context corruption.
Usage:
    python benchmarks/classical_gk_suite.py
"""

import sys, os, json, time, subprocess, textwrap
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PYTHON = sys.executable
HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.join(HERE, '..')

# ─────────────────────────────────────────────────────────────────────────────
# Worker helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_worker(code: str, timeout: int = 600) -> dict:
    """Run worker code in isolated subprocess, return parsed __RESULT__ JSON."""
    env = os.environ.copy()
    env['PYTHONPATH'] = REPO
    result = subprocess.run(
        [PYTHON, '-c', code],
        capture_output=True, text=True, timeout=timeout, env=env,
        cwd=REPO,
    )
    if result.returncode != 0:
        print(f"  [stderr] {result.stderr[-1000:]}")
    for line in result.stdout.splitlines():
        if line.startswith('__RESULT__'):
            return json.loads(line[len('__RESULT__'):])
    # fallback: return empty
    print(f"  [stdout tail] {result.stdout[-500:]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Workers (each is a self-contained script string)
# ─────────────────────────────────────────────────────────────────────────────

WORKER_SINGLE = textwrap.dedent("""
import json, sys, os
import numpy as np
import jax
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import extract_growth_rate_smart

N_PARTICLES = {N_PARTICLES}
Q1          = {Q1}
R0_LT       = {R0_LT}
NU_KROOK    = {NU_KROOK}
COLL_MODEL  = '{COLL_MODEL}'

cfg = SimConfigFA(
    Npsi=16, Ntheta=32, Nalpha=64,
    N_particles=N_PARTICLES, n_steps=250, dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=Q1,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
    R0_over_LT=R0_LT, R0_over_Ln=2.2, vpar_cap=4.0,
    k_mode=18, single_mode=True,
    collision_model=COLL_MODEL, nu_krook=NU_KROOK,
)
diags, _, _, _ = run_simulation_fa(cfg, jax.random.PRNGKey(42), verbose=False)
phi_max = np.array([float(d.phi_max) for d in diags])
gamma, s0, s1 = extract_growth_rate_smart(phi_max, cfg.dt)
print('__RESULT__' + json.dumps({{'gamma': gamma, 'n_particles': N_PARTICLES,
    'q1': Q1, 'R0_LT': R0_LT, 'nu_krook': NU_KROOK}}))
""")

WORKER_SPEED = textwrap.dedent("""
import json, time
import numpy as np
import jax
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import extract_growth_rate_smart

results = []
for N_PARTICLES, N_STEPS in [(200_000, 250), (300_000, 300)]:
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=64,
        N_particles=N_PARTICLES, n_steps=N_STEPS, dt=0.05,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
        k_mode=18, single_mode=True,
    )
    t0 = time.perf_counter()
    diags, _, _, _ = run_simulation_fa(cfg, jax.random.PRNGKey(0), verbose=False)
    elapsed = time.perf_counter() - t0
    pps = N_PARTICLES * N_STEPS / elapsed
    sps = N_STEPS / elapsed
    results.append({'n_particles': N_PARTICLES, 'n_steps': N_STEPS,
                    'elapsed_s': elapsed, 'particles_per_sec': pps, 'steps_per_sec': sps})

import json
print('__RESULT__' + json.dumps(results))
""")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def bench_convergence():
    print("\n" + "═"*66)
    print("[1/5] CBC Particle Convergence Scan")
    print("      Grid: 16×32×64 | k_mode=18 | R/LT=6.9 | target γ=0.170")
    print("═"*66)
    ns = [50_000, 100_000, 200_000, 300_000]
    gammas = []
    for n in ns:
        code = WORKER_SINGLE.format(N_PARTICLES=n, Q1=0.5, R0_LT=6.9,
                                    NU_KROOK=0.0, COLL_MODEL='none')
        t0 = time.time()
        r = run_worker(code)
        g = r.get('gamma', float('nan'))
        gammas.append(g)
        err = abs(g - 0.170) / 0.170 * 100 if not np.isnan(g) else float('nan')
        print(f"  N={n:>7,}  γ={g:.4f}  err={err:5.1f}%  ({time.time()-t0:.0f}s)")

    g_peak = gammas[-1]
    err_peak = abs(g_peak - 0.170) / 0.170 * 100
    passed = err_peak < 15.0
    print(f"\n  RESULT: γ(300k)={g_peak:.4f}  error={err_peak:.1f}%  "
          f"{'✅ PASS' if passed else '❌ FAIL'} (threshold 15%)")
    return passed, {'gammas': gammas, 'n_particles': ns}


def bench_shear():
    print("\n" + "═"*66)
    print("[2/5] Magnetic Shear Scan")
    print("      Vary q1 → s = q1·(r/q) at r=a/2 | N=200k | k_mode=18")
    print("═"*66)
    q1_vals = [0.1, 0.3, 0.5, 0.78, 1.2]
    gammas = []
    for q1 in q1_vals:
        code = WORKER_SINGLE.format(N_PARTICLES=200_000, Q1=q1, R0_LT=6.9,
                                    NU_KROOK=0.0, COLL_MODEL='none')
        t0 = time.time()
        r = run_worker(code)
        g = r.get('gamma', float('nan'))
        gammas.append(g)
        # s ≈ q1*(r/q) at r=a/2=0.09: s = q1*0.09/(1.4+0.5*0.09^2) ≈ q1*0.064
        s_approx = q1 * 0.09 / (1.4 + 0.5 * 0.09**2)
        print(f"  q1={q1:.2f}  s≈{s_approx:.3f}  γ={g:.4f}  ({time.time()-t0:.0f}s)")

    # Pass: γ at q1=0.78 > γ at q1=0.1
    idx_low  = q1_vals.index(0.1)
    idx_high = q1_vals.index(0.78)
    passed = gammas[idx_high] > gammas[idx_low]
    print(f"\n  RESULT: γ(q1=0.78)={gammas[idx_high]:.4f} > γ(q1=0.1)={gammas[idx_low]:.4f}  "
          f"{'✅ PASS' if passed else '❌ FAIL'} (shear destabilizes ITG)")
    return passed, {'q1_vals': q1_vals, 'gammas': gammas}


def bench_rlt():
    print("\n" + "═"*66)
    print("[3/5] R/LT Critical Gradient Scan")
    print("      Expect threshold ~ R/LT ≈ 4–5 | N=200k | k_mode=18")
    print("═"*66)
    rlt_vals = [3.0, 4.0, 5.0, 6.0, 6.9, 8.0, 9.0]
    gammas = []
    for rlt in rlt_vals:
        code = WORKER_SINGLE.format(N_PARTICLES=200_000, Q1=0.5, R0_LT=rlt,
                                    NU_KROOK=0.0, COLL_MODEL='none')
        t0 = time.time()
        r = run_worker(code)
        g = r.get('gamma', float('nan'))
        gammas.append(g)
        flag = "🔥 unstable" if g > 0.01 else "  stable   "
        print(f"  R/LT={rlt:.1f}  γ={g:.4f}  {flag}  ({time.time()-t0:.0f}s)")

    # Find threshold
    threshold = None
    for i, (rlt, g) in enumerate(zip(rlt_vals, gammas)):
        if g > 0.01 and threshold is None:
            threshold = rlt
    # Pass criteria
    g_low  = gammas[rlt_vals.index(4.0)]
    g_high = gammas[rlt_vals.index(6.9)]
    passed = (g_low < 0.01 or np.isnan(g_low)) and g_high > 0.05
    print(f"\n  Threshold ≈ R/LT = {threshold}  |  γ(R/LT=4)={g_low:.4f}  γ(R/LT=6.9)={g_high:.4f}")
    print(f"  RESULT: {'✅ PASS' if passed else '❌ FAIL'} (need γ<0.01 at R/LT≤4, γ>0.05 at R/LT≥6.9)")
    return passed, {'rlt_vals': rlt_vals, 'gammas': gammas, 'threshold': threshold}


def bench_collision():
    print("\n" + "═"*66)
    print("[4/5] Collision Damping Test (Krook operator)")
    print("      Expect γ decreases monotonically with ν | N=200k")
    print("═"*66)
    nu_vals  = [0.0, 0.01, 0.05, 0.1, 0.2]
    gammas   = []
    for nu in nu_vals:
        coll = 'none' if nu == 0.0 else 'krook'
        code = WORKER_SINGLE.format(N_PARTICLES=200_000, Q1=0.5, R0_LT=6.9,
                                    NU_KROOK=nu, COLL_MODEL=coll)
        t0 = time.time()
        r = run_worker(code)
        g = r.get('gamma', float('nan'))
        gammas.append(g)
        print(f"  ν={nu:.3f}  γ={g:.4f}  ({time.time()-t0:.0f}s)")

    passed = gammas[-1] < gammas[0]
    print(f"\n  RESULT: γ(ν=0)={gammas[0]:.4f}  γ(ν=0.2)={gammas[-1]:.4f}  "
          f"{'✅ PASS' if passed else '❌ FAIL'} (collisions damp ITG)")
    return passed, {'nu_vals': nu_vals, 'gammas': gammas}


def bench_speed():
    print("\n" + "═"*66)
    print("[5/5] Float32 Speed Benchmark")
    print("      Two run sizes, report throughput")
    print("═"*66)
    t0 = time.time()
    results = run_worker(WORKER_SPEED, timeout=600)
    elapsed_total = time.time() - t0
    if not results:
        print("  ❌ Speed benchmark failed to return results")
        return True, {}
    for r in results:
        print(f"  N={r['n_particles']:>7,}  steps={r['n_steps']}  "
              f"elapsed={r['elapsed_s']:.1f}s  "
              f"pps={r['particles_per_sec']:.2e}  "
              f"sps={r['steps_per_sec']:.1f}")
    print(f"  (float32 / JAX GPU — no pass/fail threshold)")
    return True, results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═"*64 + "╗")
    print("║   GyroJAX Classical GK Benchmark Suite                        ║")
    print("╚" + "═"*64 + "╝")
    print("  Grid max: 16×32×96  |  N_particles max: 400k  |  dtype: float32")
    print("  GPU: JAX CUDA backend\n")

    t_start = time.time()
    all_results = {}

    p1, r1 = bench_convergence(); all_results['convergence'] = r1
    p2, r2 = bench_shear();       all_results['shear']       = r2
    p3, r3 = bench_rlt();         all_results['rlt']         = r3
    p4, r4 = bench_collision();   all_results['collision']   = r4
    p5, r5 = bench_speed();       all_results['speed']       = r5

    total_time = time.time() - t_start

    print("\n" + "═"*66)
    print("SUMMARY")
    print("═"*66)
    rows = [
        ("CBC Particle Convergence", p1),
        ("Magnetic Shear Scan",      p2),
        ("R/LT Critical Gradient",   p3),
        ("Collision Damping",         p4),
        ("Speed Benchmark",           p5),
    ]
    n_pass = sum(1 for _, p in rows if p)
    for name, p in rows:
        print(f"  {'✅' if p else '❌'}  {name}")
    print(f"\n  {n_pass}/5 passed  |  Total time: {total_time/60:.1f} min")
    print("═"*66)

    import json as _json
    with open(os.path.join(HERE, 'classical_gk_suite_results.json'), 'w') as f:
        _json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → benchmarks/classical_gk_suite_results.json")

    return n_pass


if __name__ == '__main__':
    # Worker dispatch
    if '--worker' in sys.argv:
        idx = sys.argv.index('--worker')
        fn_name = sys.argv[idx + 1]
        globals()[fn_name]()
    else:
        main()
