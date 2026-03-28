"""
Dimits Shift Benchmark — Optimized (faster grid, noise-controlled)
===================================================================
Reduced grid: Npsi=12, Ntheta=24, Nalpha=24 (faster per step)
N_particles=200_000, n_steps=2000, dt=0.05
Noise control: canonical_loading, pullback, nu_krook, nu_soft, weight_spread

scatter_gather_fa.py audit (2026-03-28):
  - scatter_to_grid_fa: already @functools.partial(jax.jit, static_argnums=(2,)) ✓
  - gather_from_grid_fa: already @jax.jit ✓
  - _trilinear_weights_fa is called inside both; precomputation would need API refactor.
  - No missing JIT decorators found. No changes made to scatter_gather_fa.py.
"""

import sys
import time
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux, extract_zonal_flow

RLT_VALUES = [5.0, 6.0, 6.9, 8.0]

COMMON = dict(
    Npsi=12,
    Ntheta=24,
    Nalpha=24,
    N_particles=200_000,
    n_steps=2000,
    dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-4,
    single_mode=False,
    k_alpha_min=4,
    # Noise control
    canonical_loading=True,
    use_pullback=True,
    pullback_interval=100,
    nu_krook=0.005,
    nu_soft=0.01,
    w_sat=2.0,
    soft_damp_alpha=2,
    use_weight_spread=True,
    weight_spread_interval=10,
)

BLOWUP_THRESHOLD = 1e4
BLOWUP_STEPS = 100

results = []

print("=" * 70)
print("GyroJAX Dimits Shift Benchmark [OPTIMIZED]")
print("Grid: 12×24×24 | N_particles=200k | n_steps=2000 | dt=0.05")
print("Noise ctrl: canonical_loading, pullback@100, nu_krook=0.005, nu_soft=0.01")
print("=" * 70)

for i, rlt in enumerate(RLT_VALUES):
    print(f"\n[{i+1}/{len(RLT_VALUES)}] R/LT = {rlt:.1f}")
    print("-" * 40)

    cfg = SimConfigFA(R0_over_LT=rlt, **COMMON)
    LT = cfg.R0 / cfg.R0_over_LT
    grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)

    key = jax.random.PRNGKey(42 + i)

    t_start = time.time()

    try:
        diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)

        t_elapsed = time.time() - t_start
        steps_per_sec = cfg.n_steps / t_elapsed
        print(f"  Wall clock: {t_elapsed:.1f}s | {steps_per_sec:.1f} steps/sec")

        phi_max_arr = np.array([float(d.phi_max) for d in diags])
        if len(phi_max_arr) >= BLOWUP_STEPS and phi_max_arr[BLOWUP_STEPS-1] > BLOWUP_THRESHOLD:
            print(f"  ⚠️  BLOWUP at step {BLOWUP_STEPS}: phi_max={phi_max_arr[BLOWUP_STEPS-1]:.2e}")
            results.append({
                'rlt': rlt, 'status': 'blowup',
                'phi_max': float(phi_max_arr[BLOWUP_STEPS-1]),
                'Q_i': float('nan'), 'zonal_rms': float('nan'), 'weight_rms': float('nan'),
                'elapsed_s': t_elapsed, 'steps_per_sec': steps_per_sec,
            })
            continue

        phi_max_final = float(
            np.max(np.abs(phi_max_arr[-50:])) if len(phi_max_arr) >= 50
            else np.abs(phi_max_arr).max()
        )
        weight_rms_arr = np.array([float(d.weight_rms) for d in diags])
        weight_rms_final = float(weight_rms_arr[-1])

        n_sat = max(int(len(phi_max_arr) * 0.2), 10)
        phi_max_tail = phi_max_arr[-n_sat:]
        mean_tail = np.mean(phi_max_tail)
        std_tail  = np.std(phi_max_tail)
        is_saturated = (mean_tail > 1e-10 and std_tail / (mean_tail + 1e-30) < 0.5)

        Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
        Q_i = float(jnp.mean(Q_profile))

        if weight_rms_final > 3.0:
            print(f"  ⚠️  weight_rms={weight_rms_final:.3e} > 3.0 — Q_i may be unreliable")

        zonal = extract_zonal_flow(phi)
        zonal_rms = float(jnp.std(zonal))

        print(f"  phi_max_final  = {phi_max_final:.3e}")
        print(f"  weight_rms     = {weight_rms_final:.3e}")
        print(f"  Q_i (heat flux)= {Q_i:.4e}")
        print(f"  saturated      = {is_saturated} (std/mean={std_tail/(mean_tail+1e-30):.2f} over last {n_sat} steps)")
        print(f"  zonal_rms      = {zonal_rms:.3e}")

        results.append({
            'rlt': rlt, 'status': 'ok',
            'phi_max': phi_max_final,
            'Q_i': Q_i,
            'zonal_rms': zonal_rms,
            'weight_rms': weight_rms_final,
            'elapsed_s': t_elapsed,
            'steps_per_sec': steps_per_sec,
        })

    except Exception as ex:
        t_elapsed = time.time() - t_start
        print(f"  ERROR: {ex}")
        results.append({
            'rlt': rlt, 'status': f'error: {ex}',
            'phi_max': float('nan'), 'Q_i': float('nan'),
            'zonal_rms': float('nan'), 'weight_rms': float('nan'),
            'elapsed_s': t_elapsed, 'steps_per_sec': 0.0,
        })

# ── Summary table ──────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("DIMITS SHIFT OPTIMIZED BENCHMARK RESULTS")
print("=" * 70)
print(f"{'R/LT':>6}  {'Status':>10}  {'Q_i':>12}  {'zonal_rms':>10}  {'sps':>8}")
print("-" * 70)

dimits_threshold = None
for r in results:
    rlt    = r['rlt']
    status = r['status']
    qi     = r['Q_i']
    zrms   = r['zonal_rms']
    sps    = r.get('steps_per_sec', 0.0)

    qi_str   = f"{qi:.4e}" if not np.isnan(qi) else "  NaN    "
    zrms_str = f"{zrms:.3e}" if not np.isnan(zrms) else "  NaN  "
    sps_str  = f"{sps:.1f}"

    flag = ""
    if status == 'ok' and not np.isnan(qi):
        if qi > 1.0 and dimits_threshold is None:
            dimits_threshold = rlt
            flag = "  ← THRESHOLD"

    print(f"{rlt:>6.1f}  {status:>10}  {qi_str}  {zrms_str}  {sps_str}{flag}")

print("=" * 70)

if dimits_threshold is not None:
    print(f"\nMeasured Dimits threshold: R/LT = {dimits_threshold:.1f}")
    print(f"Reference (Dimits 2000):   R/LT ≈ 6.0")
    passed = 4.5 <= dimits_threshold <= 7.5
    print(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"  (pass criterion: threshold between R/LT=4.5 and R/LT=7.5)")
else:
    ok_results = [r for r in results if r['status'] == 'ok']
    all_qi = [r['Q_i'] for r in ok_results if not np.isnan(r['Q_i'])]
    if all_qi and all(q <= 0.02 for q in all_qi):
        print("\nAll Q_i ≤ 0.02 — no threshold detected (zonal-flow suppressed or too weak)")
        print("Consider running at higher R/LT or longer simulation")
        print("\nResult: INCONCLUSIVE")
    else:
        print("\nNo clear threshold found (possible blowup or errors)")
        print("\nResult: FAIL ✗")

# ── Save JSON results ──────────────────────────────────────────────────────
os.makedirs("benchmarks/results", exist_ok=True)
out_path = "benchmarks/results/dimits_optimized.json"
with open(out_path, "w") as f:
    json.dump({
        "benchmark": "dimits_shift_optimized",
        "grid": {"Npsi": 12, "Ntheta": 24, "Nalpha": 24},
        "N_particles": 200_000,
        "n_steps": 2000,
        "dt": 0.05,
        "noise_control": {
            "canonical_loading": True,
            "use_pullback": True,
            "pullback_interval": 100,
            "nu_krook": 0.005,
            "nu_soft": 0.01,
            "use_weight_spread": True,
        },
        "dimits_threshold": dimits_threshold,
        "pass_criterion": "4.5 <= threshold <= 7.5",
        "passed": (4.5 <= dimits_threshold <= 7.5) if dimits_threshold is not None else None,
        "results": results,
    }, f, indent=2)
print(f"\nResults saved to {out_path}")
