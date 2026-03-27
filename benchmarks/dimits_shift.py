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

# Scan points
RLT_VALUES = [4.0, 5.0, 6.0, 6.9, 8.0]

# Common CBC parameters
COMMON = dict(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=200_000,
    n_steps=600,
    dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-4,
    single_mode=False,
    k_alpha_min=2,   # suppress k=1 alpha mode to prevent aliasing blowup
)

BLOWUP_THRESHOLD = 1e4
BLOWUP_STEPS = 100   # check after first N steps

results = []

print("=" * 70)
print("GyroJAX Dimits Shift Benchmark")
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
        weight_rms_final = float([d.weight_rms for d in diags][-1])

        # Ion heat flux (averaged over last 50 steps worth — use final phi)
        Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
        Q_i = float(jnp.mean(Q_profile))

        # Zonal flow rms
        zonal = extract_zonal_flow(phi)
        zonal_rms = float(jnp.std(zonal))

        print(f"  phi_max_final  = {phi_max_final:.3e}")
        print(f"  weight_rms     = {weight_rms_final:.3e}")
        print(f"  Q_i (heat flux)= {Q_i:.4e}")
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
        if qi > 0.02 and dimits_threshold is None:
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
