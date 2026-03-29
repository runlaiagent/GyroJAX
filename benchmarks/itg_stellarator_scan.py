#!/usr/bin/env python3
"""ITG stellarator scan: growth rate vs q-profile and magnetic shear.

Scans:
  - q0: central safety factor [1.0, 1.4, 2.0, 3.0]
  - q1: shear parameter [0.0, 0.5, 1.0, 2.0]
  - R0_over_LT: temperature gradient drive [5.0, 6.9, 8.0, 10.0]

Uses run_long_simulation_fa with:
  - N_particles = 300,000
  - n_total_steps = 300 (chunked in 100-step pieces)
  - Auto-saves HDF5 every 100 steps per scan point
"""

import os, sys, json, time
os.environ['JAX_PLATFORMS'] = 'cuda'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}", flush=True)

from gyrojax.simulation_fa import SimConfigFA, run_long_simulation_fa

# ── Config ────────────────────────────────────────────────────────────────────
N_PARTICLES  = 300_000
N_STEPS      = 300        # total steps per scan point
CHUNK_SIZE   = 100        # auto-save every 100 steps
DT           = 0.05

os.makedirs("benchmarks/results/itg_scan_h5", exist_ok=True)


def measure_growth_rate(q0, q1, R0_over_LT, key, label=""):
    """Run chunked simulation, save HDF5, return growth rate."""
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=32,
        N_particles=N_PARTICLES,
        n_steps=N_STEPS,           # overridden by run_long per chunk
        dt=DT,
        R0_over_LT=R0_over_LT,
        R0_over_Ln=2.2,
        q0=q0, q1=q1,
        pert_amp=1e-4,
        single_mode=True,
        fused_rk4=True,
    )

    h5_path = f"benchmarks/results/itg_scan_h5/{label}.h5" if label else ""

    result = run_long_simulation_fa(
        cfg,
        n_total_steps=N_STEPS,
        chunk_size=CHUNK_SIZE,
        output_file=h5_path,
        key=key,
        verbose=False,
    )

    phi_vals = result['phi_max']

    # Growth rate from middle third (linear phase)
    i0 = N_STEPS // 3
    i1 = 2 * N_STEPS // 3
    phi_lin = phi_vals[i0:i1]
    phi_lin = phi_lin[phi_lin > 1e-10]

    if len(phi_lin) < 5:
        return float('nan')

    t = np.arange(len(phi_lin)) * DT
    coeffs = np.polyfit(t, np.log(phi_lin + 1e-30), 1)
    return float(coeffs[0])


# ── Scan parameters ───────────────────────────────────────────────────────────
q0_values = [1.0, 1.4, 2.0, 3.0]
q1_values = [0.0, 0.5, 1.0, 2.0]
LT_values = [5.0, 6.9, 8.0, 10.0]

results = []
key = jax.random.PRNGKey(42)

print(f"\nConfig: N_particles={N_PARTICLES:,}  n_steps={N_STEPS}  chunk={CHUNK_SIZE}  dt={DT}")
print(f"HDF5 auto-save every {CHUNK_SIZE} steps → benchmarks/results/itg_scan_h5/")
print(f"\n=== ITG Stellarator Scan ===")
print(f"{'q0':>5} {'q1':>5} {'R0/LT':>7} {'gamma':>10}  status")
print("-" * 50)

# ── Warm-up: trigger JIT compilation once ────────────────────────────────────
print("\n[Warm-up] Compiling JIT kernels...", flush=True)
_wkey = jax.random.PRNGKey(0)
measure_growth_rate(1.4, 0.5, 6.9, _wkey, label="warmup")
print("[Warm-up] Done.\n", flush=True)

t_start = time.time()

# Phase 1: scan q0, q1 at fixed R0/LT=6.9
print("[Phase 1] q-profile & shear scan at R0/LT=6.9", flush=True)
for q0 in q0_values:
    for q1 in q1_values:
        lbl = f"q0{q0:.1f}_q1{q1:.1f}_LT6.9"
        gamma = measure_growth_rate(q0, q1, 6.9, key, label=lbl)
        status = "stable" if gamma < 0 else f"γ={gamma:.3f}"
        print(f"  q0={q0:.1f}  q1={q1:.1f}  R0/LT=6.9  γ={gamma:+.4f}  {status}", flush=True)
        results.append({"q0": q0, "q1": q1, "R0_over_LT": 6.9, "gamma": gamma, "h5": lbl})
        key = jax.random.fold_in(key, len(results))

# Phase 2: R0/LT drive scan
print("\n[Phase 2] R0/LT drive scan at q0=1.4 and q0=2.0", flush=True)
for q0 in [1.4, 2.0]:
    for LT in LT_values:
        lbl = f"q0{q0:.1f}_q1_0.5_LT{LT:.1f}"
        gamma = measure_growth_rate(q0, 0.5, LT, key, label=lbl)
        status = "stable" if gamma < 0 else f"γ={gamma:.3f}"
        print(f"  q0={q0:.1f}  q1=0.5  R0/LT={LT:.1f}  γ={gamma:+.4f}  {status}", flush=True)
        results.append({"q0": q0, "q1": 0.5, "R0_over_LT": LT, "gamma": gamma, "h5": lbl})
        key = jax.random.fold_in(key, len(results))

elapsed = time.time() - t_start
print(f"\nTotal scan time: {elapsed:.1f}s  ({elapsed/len(results):.1f}s/point)")

# ── Save JSON summary ─────────────────────────────────────────────────────────
output = {
    "description": "ITG stellarator scan: growth rate vs q0, q1, R0/LT",
    "config": {"N_particles": N_PARTICLES, "n_steps": N_STEPS, "chunk_size": CHUNK_SIZE, "dt": DT},
    "n_cases": len(results),
    "elapsed_s": elapsed,
    "results": results,
}
with open("benchmarks/results/itg_stellarator_scan.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved → benchmarks/results/itg_stellarator_scan.json")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary: Effect of magnetic shear on ITG (R0/LT=6.9) ===")
cases_69 = [r for r in results if abs(r["R0_over_LT"] - 6.9) < 0.1]
for r in sorted(cases_69, key=lambda x: (x["q0"], x["q1"])):
    bar = "█" * max(0, int(r["gamma"] * 30))
    sign = "▲ UNSTABLE" if r["gamma"] > 0 else "  stable"
    print(f"  q0={r['q0']:.1f} q1={r['q1']:.1f}: γ={r['gamma']:+.4f}  {bar} {sign}")

print("\n=== Summary: ITG threshold vs R0/LT ===")
for q0 in [1.4, 2.0]:
    pts = [(r["R0_over_LT"], r["gamma"]) for r in results
           if abs(r["q0"] - q0) < 0.01 and abs(r.get("q1", 0.5) - 0.5) < 0.01
           and r["R0_over_LT"] in LT_values]
    pts.sort()
    print(f"  q0={q0:.1f}: " + "  ".join(f"R0/LT={lt:.0f}→γ={g:+.3f}" for lt, g in pts))
