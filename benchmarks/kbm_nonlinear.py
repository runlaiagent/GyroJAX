"""
KBM Nonlinear Saturation Benchmark.

Compares nonlinear saturation at beta=0 (ITG), beta=0.01 (threshold), beta=0.02 (KBM).

Expected physics:
  beta=0.00: ITG with zonal flows, moderate chi_i
  beta=0.01: EM stabilization, reduced phi_sat
  beta=0.02: KBM regime — suppressed zonal flows, higher transport

References:
  Pueschel et al., Phys. Plasmas 15, 102310 (2008)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux


def run_nonlinear(beta, N_particles=50_000, n_steps=200):
    print(f"\n=== beta={beta:.3f} ===")
    t0 = time.time()
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=32,
        N_particles=N_particles,
        n_steps=n_steps,
        dt=0.05,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        pert_amp=1e-4,
        single_mode=False,
        beta=beta,
        canonical_loading=True,
        use_pullback=True, pullback_interval=50,
        nu_soft=0.005, w_sat=3.0, nu_krook=0.002,
        use_weight_spread=True, zonal_preserving_spread=True,
        k_alpha_min=4,
        gyroaverage_scatter=True,
        use_radial_gaa=True,
    )
    try:
        diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(42), verbose=False)
        phi_vals = np.array([float(d.phi_max) for d in diags])
        w_rms_vals = np.array([float(d.weight_rms) for d in diags])
        sat = int(0.75 * n_steps)
        phi_sat = float(np.mean(phi_vals[sat:]))
        w_rms_sat = float(np.mean(w_rms_vals[sat:]))
        phi_zonal = float(phi.mean(axis=2).std())
        grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)
        LT = cfg.R0 / cfg.R0_over_LT
        Q = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
        chi = float(jnp.mean(Q)) * cfg.R0 / (cfg.n0_avg * cfg.R0_over_LT)
        elapsed = time.time() - t0
        print(f"  phi_sat={phi_sat:.3e}  w_rms={w_rms_sat:.3f}  zonal={phi_zonal:.3e}  chi_i={chi:.4e}  [{elapsed:.1f}s]")
        return {"beta": beta, "status": "ok", "phi_sat": phi_sat,
                "w_rms_sat": w_rms_sat, "phi_zonal_rms": phi_zonal,
                "chi_i": chi, "elapsed_s": elapsed}
    except Exception as ex:
        print(f"  ERROR: {ex}")
        return {"beta": beta, "status": str(ex)}


def main():
    print(f"JAX devices: {jax.devices()}")
    results = []
    for beta in [0.0, 0.01, 0.02]:
        results.append(run_nonlinear(beta))

    print("\n" + "="*65)
    print("KBM NONLINEAR SATURATION")
    print("="*65)
    print(f"{'beta':>8}  {'phi_sat':>10}  {'w_rms':>7}  {'zonal':>10}  {'chi_i':>10}")
    print("-"*65)
    for r in results:
        if r.get("status") == "ok":
            print(f"{r['beta']:>8.3f}  {r['phi_sat']:>10.3e}  {r['w_rms_sat']:>7.3f}  "
                  f"{r['phi_zonal_rms']:>10.3e}  {r['chi_i']:>10.4e}")
        else:
            print(f"{r['beta']:>8.3f}  ERROR: {r['status']}")
    print("="*65)

    ok = [r for r in results if r.get("status") == "ok"]
    if len(ok) >= 2:
        r0 = next((r for r in ok if r["beta"] == 0.0), None)
        r2 = next((r for r in ok if r["beta"] >= 0.015), None)
        if r0 and r2:
            zr = r2["phi_zonal_rms"] / max(r0["phi_zonal_rms"], 1e-15)
            cr = r2["chi_i"] / max(abs(r0["chi_i"]), 1e-15)
            print(f"\nZonal ratio (beta=0.02/0.00): {zr:.3f}  {'suppressed ✅' if zr < 0.7 else 'not suppressed'}")
            print(f"Chi_i ratio  (beta=0.02/0.00): {cr:.3f}  {'enhanced ✅' if cr > 1.2 else 'similar/reduced'}")

    # Also run full-f at beta=0 and beta=0.02 for comparison
    print("\n=== Full-f comparison ===")
    try:
        from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf
        for beta in [0.0, 0.02]:
            cfg = SimConfigFullF(
                Npsi=16, Ntheta=32, Nalpha=32,
                N_particles=50_000, n_steps=200, dt=0.05,
                R0_over_LT=6.9, R0_over_Ln=2.2,
                pert_amp=1e-4, beta=beta,
                canonical_loading=True,
                use_pullback=True, pullback_interval=50,
                nu_soft=0.005, w_sat=3.0, nu_krook=0.002,
                use_weight_spread=True, k_alpha_min=4,
            )
            diags, state, phi, geom = run_simulation_fullf(cfg, key=jax.random.PRNGKey(42), verbose=False)
            phi_vals = np.array([float(d.phi_max) for d in diags])
            phi_sat = float(np.mean(phi_vals[150:]))
            phi_zonal = float(phi.mean(axis=2).std())
            grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)
            LT = cfg.R0 / cfg.R0_over_LT
            Q = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
            chi = float(jnp.mean(Q)) * cfg.R0 / (cfg.n0_avg * cfg.R0_over_LT)
            print(f"  full-f beta={beta:.2f}: phi_sat={phi_sat:.3e}  zonal={phi_zonal:.3e}  chi_i={chi:.4e}")
    except Exception as ex:
        print(f"  full-f skipped: {ex}")

    summary = {
        "description": "KBM nonlinear saturation: phi_sat, chi_i, zonal vs beta (fast: 50k×200steps)",
        "parameters": {"N_particles": 50_000, "n_steps": 200, "R_over_LT": 6.9},
        "devices": str(jax.devices()),
        "results": results,
    }
    out = "benchmarks/results/kbm_nonlinear.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
