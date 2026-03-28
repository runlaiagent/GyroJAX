#!/usr/bin/env python3
"""ITG stellarator scan: growth rate vs q-profile and magnetic shear.

Scans:
  - q0: central safety factor [1.0, 1.4, 2.0, 3.0]
  - q1: shear parameter [0.0, 0.5, 1.0, 2.0]
  - R0_over_LT: temperature gradient drive [5.0, 6.9, 8.0, 10.0]

For each (q0, q1), measures γ(R0/LT) and finds threshold R0/LT_crit.
"""

import os, sys, json, time
os.environ['JAX_PLATFORMS'] = 'cuda'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

def measure_growth_rate(q0, q1, R0_over_LT, key, n_steps=150, N_particles=30_000):
    """Measure ITG growth rate for given q-profile and drive."""
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=32,
        N_particles=N_particles,
        n_steps=n_steps,
        dt=0.05,
        R0_over_LT=R0_over_LT,
        R0_over_Ln=2.2,
        q0=q0, q1=q1,
        pert_amp=1e-4,
        single_mode=True,
        fused_rk4=True,
    )
    
    try:
        diags, _, _, _ = run_simulation_fa(cfg, key=key, verbose=False)
        phi_vals = np.array([float(d.phi_max) for d in diags])
        
        # Growth rate from middle third (linear phase)
        i0 = n_steps // 3
        i1 = 2 * n_steps // 3
        phi_lin = phi_vals[i0:i1]
        phi_lin = phi_lin[phi_lin > 1e-10]
        
        if len(phi_lin) < 5:
            return float('nan')
        
        t = np.arange(len(phi_lin)) * cfg.dt
        coeffs = np.polyfit(t, np.log(phi_lin + 1e-30), 1)
        gamma = float(coeffs[0])
        return gamma
    except Exception as e:
        print(f"  ERROR: {e}")
        return float('nan')


# Scan parameters
q0_values  = [1.0, 1.4, 2.0, 3.0]
q1_values  = [0.0, 0.5, 1.0, 2.0]
LT_values  = [5.0, 6.9, 8.0, 10.0]

results = []
key = jax.random.PRNGKey(42)

print("\n=== ITG Stellarator Scan ===")
print(f"{'q0':>5} {'q1':>5} {'R0/LT':>7} {'gamma':>10} {'stable':>8}")
print("-" * 45)

t_start = time.time()

# First: scan q0, q1 at fixed R0/LT=6.9 to see shear effect
print("\n[Phase 1] q-profile scan at R0/LT=6.9")
for q0 in q0_values:
    for q1 in q1_values:
        gamma = measure_growth_rate(q0, q1, 6.9, key)
        stable = "stable" if gamma < 0 else f"γ={gamma:.3f}"
        print(f"  q0={q0:.1f}  q1={q1:.1f}  R0/LT=6.9  γ={gamma:+.4f}  {stable}")
        results.append({"q0": q0, "q1": q1, "R0_over_LT": 6.9, "gamma": gamma})
        key = jax.random.fold_in(key, len(results))

# Second: scan R0/LT at fixed q0=1.4 (tokamak-like) and q0=2.0 (stellarator-like)
print("\n[Phase 2] R0/LT drive scan at q0=1.4 and q0=2.0")
for q0 in [1.4, 2.0]:
    for LT in LT_values:
        gamma = measure_growth_rate(q0, 0.5, LT, key)
        print(f"  q0={q0:.1f}  q1=0.5  R0/LT={LT:.1f}  γ={gamma:+.4f}")
        results.append({"q0": q0, "q1": 0.5, "R0_over_LT": LT, "gamma": gamma})
        key = jax.random.fold_in(key, len(results))

elapsed = time.time() - t_start
print(f"\nTotal scan time: {elapsed:.1f}s")

# Save results
os.makedirs("benchmarks/results", exist_ok=True)
output = {
    "description": "ITG stellarator scan: growth rate vs q0, q1, R0/LT",
    "n_cases": len(results),
    "elapsed_s": elapsed,
    "results": results,
}
with open("benchmarks/results/itg_stellarator_scan.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to benchmarks/results/itg_stellarator_scan.json")

# Print summary
print("\n=== Summary: Effect of magnetic shear on ITG threshold ===")
# Find cases at R0/LT=6.9
cases_69 = [r for r in results if abs(r["R0_over_LT"] - 6.9) < 0.1]
for r in sorted(cases_69, key=lambda x: (x["q0"], x["q1"])):
    bar = "█" * max(0, int(r["gamma"] * 20))
    print(f"  q0={r['q0']:.1f} q1={r['q1']:.1f}: γ={r['gamma']:+.4f} {bar}")
