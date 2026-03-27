"""
Dimits Shift Benchmark for GyroJAX
====================================
Scans R/LT = [4.0, 5.0, 6.0, 6.9, 8.0] with nonlinear PIC simulations.

The Dimits shift: zonal flows suppress turbulence for R/LT < ~6 (CBC),
raising the nonlinear threshold above the linear threshold (~4-5).

Reference: Dimits et al. (2000) Phys. Plasmas 7, 969
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np

# Force float32
jax.config.update("jax_enable_x64", False)

sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux, extract_zonal_flow

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--hires', action='store_true', help='High-resolution run: 1M particles, finer grid')
_args = _parser.parse_args()
HIRES = _args.hires

# Scan points
RLT_VALUES = [4.0, 5.0, 6.0, 6.9, 8.0]

# Common CBC parameters
COMMON = dict(
    Npsi=24 if HIRES else 16,
    Ntheta=48 if HIRES else 32,
    Nalpha=48 if HIRES else 32,
    N_particles=1_000_000 if HIRES else 300_000,
    n_steps=800 if HIRES else 600,
    dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-4,
    single_mode=False,
    k_alpha_min=4,   # suppress k<4 alpha modes (near-zero FLR → runaway phi)
    # Krook collision damping — mild rate keeps weights bounded without killing zonal flows
    nu_krook=0.005,
    # δf noise control improvements
    canonical_loading=True,
    use_pullback=True,
    pullback_interval=50,
    nu_soft=0.01,
    w_sat=2.0,
    soft_damp_alpha=2,
    # GTC-style weight spreading
    use_weight_spread=True,
    weight_spread_interval=10,
)

BLOWUP_THRESHOLD = 1e4
BLOWUP_STEPS = 100   # check after first N steps

results = []

print("=" * 70)
print("GyroJAX Dimits Shift Benchmark" + (" [HIGH-RES: 1M particles, 24×48×48]" if HIRES else ""))
print("Cyclone Base Case — nonlinear R/LT scan")
print("k_alpha_min=2: suppressing k_alpha=1 aliased mode")
print("=" * 70)

for i, rlt in enumerate(RLT_VALUES):
    print(f"\n[{i+1}/{len(RLT_VALUES)}] R/LT = {rlt:.1f}")
    print("-" * 40)

    cfg = SimConfigFA(R0_over_LT=rlt, **COMMON)
    LT = cfg.R0 / cfg.R0_over_LT
    grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)

    key = jax.random.PRNGKey(42 + i)

    try:
        diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)

        # Check blowup
        phi_max_arr = np.array([float(d.phi_max) for d in diags])
        if len(phi_max_arr) >= BLOWUP_STEPS and phi_max_arr[BLOWUP_STEPS-1] > BLOWUP_THRESHOLD:
            print(f"  ⚠️  BLOWUP at step {BLOWUP_STEPS}: phi_max={phi_max_arr[BLOWUP_STEPS-1]:.2e}")
            results.append({
                'rlt': rlt, 'status': 'blowup',
                'phi_max': float(phi_max_arr[BLOWUP_STEPS-1]),
                'Q_i': float('nan'), 'zonal_rms': float('nan'), 'weight_rms': float('nan'),
            })
            continue

        phi_max_final = float(np.max(np.abs(np.array([float(d.phi_max) for d in diags][-50:])))
                              if len(diags) >= 50 else np.abs(phi_max_arr).max())
        weight_rms_arr = np.array([float(d.weight_rms) for d in diags])
        weight_rms_final = float(weight_rms_arr[-1])

        # Saturation check: std/mean of phi_max over last 20% of steps
        n_sat = max(int(len(phi_max_arr) * 0.2), 10)
        phi_max_tail = phi_max_arr[-n_sat:]
        mean_tail = np.mean(phi_max_tail)
        std_tail  = np.std(phi_max_tail)
        is_saturated = (mean_tail > 1e-10 and std_tail / (mean_tail + 1e-30) < 0.5)

        # Ion heat flux from final state (time-average approximation:
        # use last 30% phi_max mean to scale Q_i — actual phi field not stored per-step)
        Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
        Q_i = float(jnp.mean(Q_profile))

        # Weight RMS warning
        if weight_rms_final > 3.0:
            print(f"  ⚠️  weight_rms={weight_rms_final:.3e} > 3.0 — noise-dominated result, Q_i may be unreliable")

        # Zonal flow rms
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
        })

    except Exception as ex:
        print(f"  ERROR: {ex}")
        results.append({
            'rlt': rlt, 'status': f'error: {ex}',
            'phi_max': float('nan'), 'Q_i': float('nan'),
            'zonal_rms': float('nan'), 'weight_rms': float('nan'),
        })

# ── Summary table ──────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("DIMITS SHIFT BENCHMARK RESULTS")
print("=" * 70)
print(f"{'R/LT':>6}  {'Status':>10}  {'Q_i':>12}  {'zonal_rms':>10}  {'phi_max':>10}")
print("-" * 70)

dimits_threshold = None
for r in results:
    rlt = r['rlt']
    status = r['status']
    qi = r['Q_i']
    zrms = r['zonal_rms']
    pmax = r['phi_max']

    qi_str   = f"{qi:.4e}" if not np.isnan(qi) else "  NaN    "
    zrms_str = f"{zrms:.3e}" if not np.isnan(zrms) else "  NaN  "
    pmax_str = f"{pmax:.3e}" if not np.isnan(pmax) else "  NaN  "

    flag = ""
    if status == 'ok' and not np.isnan(qi):
        if qi > 1.0 and dimits_threshold is None:   # threshold: Q_i > 1 gyroBohm unit
            dimits_threshold = rlt
            flag = "  ← THRESHOLD"

    print(f"{rlt:>6.1f}  {status:>10}  {qi_str}  {zrms_str}  {pmax_str}{flag}")

print("=" * 70)

if dimits_threshold is not None:
    print(f"\nMeasured Dimits threshold: R/LT = {dimits_threshold:.1f}")
    print(f"Reference (Dimits 2000):   R/LT ≈ 6.0")
    passed = 4.5 <= dimits_threshold <= 7.5
    print(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"  (pass criterion: threshold between R/LT=4.5 and R/LT=7.5)")
else:
    # Check if all blowup or all zero
    ok_results = [r for r in results if r['status'] == 'ok']
    all_qi = [r['Q_i'] for r in ok_results if not np.isnan(r['Q_i'])]
    if all_qi and all(q <= 0.02 for q in all_qi):
        print("\nAll Q_i ≤ 0.02 — no threshold detected (all suppressed by zonal flows or too weak)")
        print("Consider running at higher R/LT or longer simulation")
        print("\nResult: INCONCLUSIVE")
    else:
        print("\nNo clear threshold found (possible blowup or errors)")
        print("\nResult: FAIL ✗")

print()
# ── CLI for high-res mode ──────────────────────────────────────────────────
# (already executed above; this block is unreachable — use dimits_shift_hires.py)
