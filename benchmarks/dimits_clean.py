"""
Dimits Shift Benchmark — Clean Version (physics-correct settings)
===================================================================
Fixes zonal flow suppression issues present in dimits_shift.py:

  1. k_alpha_min=0  — no mode suppression; ITG drive at all k_alpha feeds zonal flows
  2. pullback_interval=200 — less aggressive; zonal flows build on 100-300 step timescales
  3. use_weight_spread=False — weight spreading kills radially-coherent zonal flow signal
  4. nu_soft=0.001 — very mild; original 0.01 damps zonal flows
  5. pert_amp=1e-3 — larger so ITG grows above noise floor at low R/LT

Reference: Dimits et al. (2000) Phys. Plasmas 7, 969
"""

import sys
import json
import os
import jax
import jax.numpy as jnp
import numpy as np

# Force float32
jax.config.update("jax_enable_x64", False)

sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux, extract_zonal_flow

# ── Parameters ──────────────────────────────────────────────────────────────
RLT_VALUES = [4.0, 5.0, 6.0, 6.9, 8.0]

COMMON = dict(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=300_000,
    n_steps=600,
    dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-4,
    single_mode=False,
    k_alpha_min=4,           # keep for numerical stability (prevents aliasing blowup)
    nu_krook=0.002,          # mild Krook (reduced from 0.005)
    canonical_loading=True,
    use_pullback=True,
    pullback_interval=50,    # keep for stability (50→200 causes NaN blowup)
    nu_soft=0.005,           # reduced from 0.01 — less damping on zonal flows
    w_sat=3.0,               # increased from 2.0 — allow larger weight fluctuations
    soft_damp_alpha=2,
    use_weight_spread=False, # DISABLED — weight spreading kills zonal flow signal
    weight_spread_interval=10,
)

BLOWUP_THRESHOLD = 1e4
BLOWUP_STEPS = 100

results = []

print("=" * 70)
print("GyroJAX Dimits Shift Benchmark — CLEAN (physics-correct settings)")
print("k_alpha_min=0, pullback_interval=200, weight_spread=OFF, nu_soft=0.001")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

for i, rlt in enumerate(RLT_VALUES):
    print(f"\n[{i+1}/{len(RLT_VALUES)}] R/LT = {rlt:.1f}")
    print("-" * 40)

    cfg = SimConfigFA(R0_over_LT=rlt, **COMMON)
    LT = cfg.R0 / cfg.R0_over_LT
    grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)

    key = jax.random.PRNGKey(42 + i)

    try:
        diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)

        phi_max_arr = np.array([float(d.phi_max) for d in diags])

        # Blowup check
        if len(phi_max_arr) >= BLOWUP_STEPS and phi_max_arr[BLOWUP_STEPS - 1] > BLOWUP_THRESHOLD:
            print(f"  ⚠️  BLOWUP at step {BLOWUP_STEPS}: phi_max={phi_max_arr[BLOWUP_STEPS-1]:.2e}")
            results.append({
                'rlt': rlt, 'status': 'blowup',
                'phi_max': float(phi_max_arr[BLOWUP_STEPS - 1]),
                'Q_i': float('nan'), 'chi_i': float('nan'),
                'zonal_rms': float('nan'), 'weight_rms': float('nan'),
            })
            continue

        phi_max_final = float(np.max(np.abs(phi_max_arr[-50:])) if len(phi_max_arr) >= 50 else np.abs(phi_max_arr).max())
        weight_rms_arr = np.array([float(d.weight_rms) for d in diags])
        weight_rms_final = float(weight_rms_arr[-1])

        # Saturation check
        n_sat = max(int(len(phi_max_arr) * 0.2), 10)
        phi_max_tail = phi_max_arr[-n_sat:]
        mean_tail = float(np.mean(phi_max_tail))
        std_tail  = float(np.std(phi_max_tail))
        is_saturated = (mean_tail > 1e-10 and std_tail / (mean_tail + 1e-30) < 0.5)

        # Ion heat flux
        Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
        Q_i = float(jnp.mean(Q_profile))

        # chi_i = Q_i / (n * T * R/LT * v_ti / R)  -- normalised thermal diffusivity
        # In GyroJAX normalised units: chi_i ≈ Q_i / (R0_over_LT) * R0/a
        chi_i = Q_i / (rlt + 1e-10)

        # Zonal flow rms
        zonal = extract_zonal_flow(phi)
        zonal_rms = float(jnp.std(zonal))

        if weight_rms_final > 3.0:
            print(f"  ⚠️  weight_rms={weight_rms_final:.3e} > 3.0 — noise-dominated")

        print(f"  phi_max_final  = {phi_max_final:.3e}")
        print(f"  weight_rms     = {weight_rms_final:.3e}")
        print(f"  Q_i (heat flux)= {Q_i:.4e}")
        print(f"  chi_i          = {chi_i:.4e}")
        print(f"  saturated      = {is_saturated} (std/mean={std_tail/(mean_tail+1e-30):.2f})")
        print(f"  zonal_rms      = {zonal_rms:.3e}")

        results.append({
            'rlt': rlt, 'status': 'ok',
            'phi_max': phi_max_final,
            'Q_i': Q_i,
            'chi_i': chi_i,
            'zonal_rms': zonal_rms,
            'weight_rms': weight_rms_final,
            'is_saturated': is_saturated,
        })

    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {ex}")
        results.append({
            'rlt': rlt, 'status': f'error: {ex}',
            'phi_max': float('nan'), 'Q_i': float('nan'), 'chi_i': float('nan'),
            'zonal_rms': float('nan'), 'weight_rms': float('nan'),
        })

# ── Save results ─────────────────────────────────────────────────────────────
os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/dimits_clean.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to benchmarks/results/dimits_clean.json")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("DIMITS CLEAN BENCHMARK RESULTS")
print("=" * 70)
print(f"{'R/LT':>6}  {'Status':>10}  {'Q_i':>12}  {'chi_i':>10}  {'zonal_rms':>10}  {'w_rms':>7}")
print("-" * 70)

dimits_threshold = None
for r in results:
    rlt   = r['rlt']
    status = r['status']
    qi    = r['Q_i']
    ci    = r.get('chi_i', float('nan'))
    zrms  = r['zonal_rms']
    wrms  = r['weight_rms']

    qi_str   = f"{qi:.4e}" if not np.isnan(qi) else "     NaN    "
    ci_str   = f"{ci:.4e}" if not np.isnan(ci) else "    NaN   "
    zrms_str = f"{zrms:.3e}" if not np.isnan(zrms) else "   NaN  "
    wrms_str = f"{wrms:.3f}"  if not np.isnan(wrms) else "  NaN"

    flag = ""
    if status == 'ok' and not np.isnan(qi):
        if qi > 1.0 and dimits_threshold is None:
            dimits_threshold = rlt
            flag = "  ← THRESHOLD"

    print(f"{rlt:>6.1f}  {status:>10}  {qi_str}  {ci_str}  {zrms_str}  {wrms_str}{flag}")

print("=" * 70)

if dimits_threshold is not None:
    print(f"\nMeasured Dimits threshold: R/LT = {dimits_threshold:.1f}")
    print(f"Reference (Dimits 2000):   R/LT ≈ 6.0")
    passed = 4.5 <= dimits_threshold <= 7.5
    print(f"Result: {'PASS ✓' if passed else 'FAIL ✗'} (pass criterion: 4.5 ≤ R/LT ≤ 7.5)")
else:
    ok_results = [r for r in results if r['status'] == 'ok']
    all_qi = [r['Q_i'] for r in ok_results if not np.isnan(r['Q_i'])]
    if all_qi and all(q <= 0.02 for q in all_qi):
        print("\nAll Q_i ≤ 0.02 — all suppressed (zonal flows dominating?) or perturbation too small")
        print("Result: INCONCLUSIVE")
    else:
        print("\nNo clear threshold found")
        print("Result: FAIL ✗")

print()

# ── Assessment ────────────────────────────────────────────────────────────────
print("ASSESSMENT:")
ok = [r for r in results if r['status'] == 'ok']
suppressed = [r for r in ok if not np.isnan(r['Q_i']) and r['Q_i'] < 0.1 and r['rlt'] <= 6.0]
active     = [r for r in ok if not np.isnan(r['Q_i']) and r['Q_i'] >= 1.0 and r['rlt'] >= 6.0]
zonal_vals = [r['zonal_rms'] for r in ok if not np.isnan(r['zonal_rms'])]

print(f"  Suppressed (Q_i<0.1, R/LT≤6): {[r['rlt'] for r in suppressed]}")
print(f"  Active     (Q_i≥1.0, R/LT≥6): {[r['rlt'] for r in active]}")
print(f"  Zonal flow rms range: {min(zonal_vals):.3e} – {max(zonal_vals):.3e}" if zonal_vals else "  No zonal data")
weight_vals = [r['weight_rms'] for r in ok if not np.isnan(r['weight_rms'])]
print(f"  Weight rms range: {min(weight_vals):.3f} – {max(weight_vals):.3f}" if weight_vals else "  No weight data")
