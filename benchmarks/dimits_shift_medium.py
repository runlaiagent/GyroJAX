"""
Dimits Shift Benchmark — Medium Resolution (30 particles/cell)
==============================================================
Grid: 12×24×24 = 6,912 cells, N_particles=200,000 (~29 particles/cell)
Uses semi-implicit CN weight update + full noise control suite.
"""

import sys
import json
import os
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux, extract_zonal_flow

RLT_VALUES = [4.0, 5.0, 6.0, 6.5, 6.9, 7.5, 8.0]

COMMON = dict(
    Npsi=12, Ntheta=24, Nalpha=24,
    N_particles=200_000,
    n_steps=400,
    dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-4,
    single_mode=False,
    k_alpha_min=4,
    nu_krook=0.005,
    canonical_loading=True,
    use_pullback=True,
    pullback_interval=50,
    nu_soft=0.01,
    w_sat=2.0,
    soft_damp_alpha=2,
    use_weight_spread=True,
    weight_spread_interval=10,
    semi_implicit_weights=True,
)

results = []

print("=" * 70)
print("GyroJAX Dimits Shift Benchmark — Medium Resolution")
print("Grid: 12×24×24, N=200k (~29 particles/cell), semi-implicit CN weights")
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

        phi_max_arr = np.array([float(d.phi_max) for d in diags])
        phi_rms_arr = np.array([float(d.phi_rms) for d in diags])
        phi_zonal_arr = np.array([float(d.phi_zonal_rms) for d in diags])
        weight_rms_arr = np.array([float(d.weight_rms) for d in diags])

        w_rms_final = float(weight_rms_arr[-1])

        # Metrics over last 30% of steps
        n_tail = max(int(len(phi_max_arr) * 0.3), 10)
        phi_max_tail = phi_max_arr[-n_tail:]
        phi_rms_tail = phi_rms_arr[-n_tail:]
        phi_zonal_tail = phi_zonal_arr[-n_tail:]

        phi_avg = float(np.mean(phi_max_tail))
        Q_i_proxy = phi_avg  # proxy for heat flux

        # Zonal flow from final phi field
        zonal = extract_zonal_flow(phi)
        zonal_rms = float(jnp.std(zonal))
        zonal_phi_avg = float(np.mean(phi_zonal_tail))

        # Turbulence: phi_rms minus zonal component
        turb_avg = float(np.mean(phi_rms_tail))

        # Try actual heat flux
        try:
            Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
            Q_i_actual = float(jnp.mean(Q_profile))
        except Exception:
            Q_i_actual = float('nan')

        # Status: use phi_avg threshold
        threshold = 0.5  # phi_avg > 0.5 → active turbulence
        if phi_avg < threshold:
            status = "suppressed"
        else:
            status = "active"

        print(f"  w_rms_final  = {w_rms_final:.3f}")
        print(f"  phi_avg      = {phi_avg:.3e}")
        print(f"  Q_i_proxy    = {Q_i_proxy:.3e}")
        print(f"  Q_i_actual   = {Q_i_actual:.4e}")
        print(f"  zonal_rms    = {zonal_rms:.3e}")
        print(f"  zonal_phi_avg= {zonal_phi_avg:.3e}")
        print(f"  turb_avg     = {turb_avg:.3e}")
        print(f"  status       = {status}")

        results.append({
            'rlt': rlt,
            'w_rms': w_rms_final,
            'phi_avg': phi_avg,
            'Q_i_proxy': Q_i_proxy,
            'Q_i_actual': Q_i_actual,
            'zonal_rms': zonal_rms,
            'zonal_phi_avg': zonal_phi_avg,
            'turb_avg': turb_avg,
            'status': status,
        })

    except Exception as ex:
        import traceback
        traceback.print_exc()
        results.append({
            'rlt': rlt, 'w_rms': float('nan'), 'phi_avg': float('nan'),
            'Q_i_proxy': float('nan'), 'Q_i_actual': float('nan'),
            'zonal_rms': float('nan'), 'zonal_phi_avg': float('nan'),
            'turb_avg': float('nan'), 'status': f'error: {ex}',
        })

# Summary table
print("\n")
print("=" * 70)
print("DIMITS SHIFT MEDIUM-RES RESULTS")
print(f"{'R/LT':>6}  {'w_rms':>7}  {'phi_avg':>10}  {'Q_i_proxy':>10}  {'status':>12}")
print("-" * 70)
for r in results:
    rlt = r['rlt']
    w = r['w_rms']
    pa = r['phi_avg']
    qi = r['Q_i_proxy']
    st = r['status']
    print(f"{rlt:>6.1f}  {w:>7.3f}  {pa:>10.3e}  {qi:>10.3e}  {st:>12}")
print("=" * 70)

# Assess physics quality
ok = [r for r in results if r['status'] in ('suppressed', 'active')]
suppressed_rlt = [r['rlt'] for r in ok if r['status'] == 'suppressed']
active_rlt = [r['rlt'] for r in ok if r['status'] == 'active']
print(f"\nSuppressed at R/LT: {suppressed_rlt}")
print(f"Active at R/LT:     {active_rlt}")

# Check if physics is monotonic
phi_avgs = [(r['rlt'], r['phi_avg']) for r in results if not np.isnan(r['phi_avg'])]
if len(phi_avgs) >= 3:
    rlts = [x[0] for x in phi_avgs]
    phis = [x[1] for x in phi_avgs]
    corr = np.corrcoef(rlts, phis)[0, 1]
    print(f"\nMonotonicity (Pearson r between R/LT and phi_avg): {corr:.3f}")
    print(f"(r > 0.7 → monotonically increasing → correct physics)")

# Check w_rms validity
wrms_vals = [r['w_rms'] for r in results if not np.isnan(r['w_rms'])]
print(f"\nw_rms range: {min(wrms_vals):.2f} – {max(wrms_vals):.2f}")
print("NOTE: w_rms < 1.0 is required for valid δf approximation")
if max(wrms_vals) > 1.0:
    print("WARNING: w_rms > 1.0 indicates δf noise dominates — results may be unreliable")

# Save results
os.makedirs("/home/blues/wlhx/GyroJAX/benchmarks/results", exist_ok=True)
out_path = "/home/blues/wlhx/GyroJAX/benchmarks/results/dimits_shift_medium.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {**COMMON, "rlt_values": RLT_VALUES},
        "results": results,
        "notes": [
            "w_rms > 1 indicates delta-f approximation breakdown",
            "Physics signal may still be present if phi_avg is monotonic in R/LT"
        ]
    }, f, indent=2)
print(f"\nResults saved to {out_path}")
