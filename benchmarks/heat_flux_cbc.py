"""
Nonlinear heat flux benchmark: χᵢ vs R/LT (Cyclone Base Case).

Measures ion thermal diffusivity χᵢ as a function of temperature gradient R/LT
using run_simulation_fa with full noise controls.

Expected result:
  - χᵢ ≈ 0 for R/LT < 6 (Dimits shift regime — transport suppressed by zonal flows)
  - χᵢ > 0 for R/LT > 6.5 (anomalous transport regime)

Usage:
  python benchmarks/heat_flux_cbc.py [--quick] [--output results/heat_flux_cbc.json]
  python benchmarks/heat_flux_cbc.py --N 500000 --steps 600
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
from gyrojax.diagnostics import ion_heat_flux, extract_zonal_flow


def run_heat_flux_scan(
    R_over_LT_values,
    N_particles: int = 500_000,
    n_steps: int = 600,
    dt: float = 0.05,
    verbose: bool = True,
    seed: int = 42,
):
    results = {}

    for i, R0_over_LT in enumerate(R_over_LT_values):
        if verbose:
            print(f"\n=== [{i+1}/{len(R_over_LT_values)}] R/LT = {R0_over_LT} ===")
        t0 = time.time()

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=32,
            N_particles=N_particles,
            n_steps=n_steps,
            dt=dt,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=R0_over_LT,
            R0_over_Ln=2.2,
            pert_amp=1e-4,
            single_mode=False,
            k_alpha_min=4,
            # Full noise controls
            canonical_loading=True,
            use_pullback=True,
            pullback_interval=50,
            nu_krook=0.005,
            nu_soft=0.01,
            w_sat=2.0,
            soft_damp_alpha=2,
            use_weight_spread=True,
            weight_spread_interval=10,
        )

        key = jax.random.PRNGKey(seed + i)
        try:
            diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=verbose)

            phi_arr   = np.array([float(d.phi_max) for d in diags])
            w_rms_arr = np.array([float(d.weight_rms) for d in diags])

            # Check blowup / NaN
            if np.any(np.isnan(phi_arr)):
                first_nan = int(np.argmax(np.isnan(phi_arr)))
                print(f"  ⚠️  NaN at step {first_nan}")
                results[R0_over_LT] = {"R0_over_LT": R0_over_LT, "status": "nan",
                                       "Q_i_avg": float('nan'), "chi_i_avg": float('nan'),
                                       "elapsed_s": time.time() - t0}
                continue

            # Time-average heat flux over last 30% of steps
            grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)
            LT = cfg.R0 / R0_over_LT
            Q_profile = ion_heat_flux(state, phi, geom, grid_shape, cfg.Ti, cfg.n0_avg, LT)
            Q_i = float(jnp.mean(Q_profile))
            chi_i = Q_i * cfg.R0 / (cfg.n0_avg * R0_over_LT)

            w_rms_final = float(w_rms_arr[-1])
            phi_final   = float(phi_arr[-1])
            zonal = extract_zonal_flow(phi)
            zonal_rms = float(jnp.std(zonal))

            if w_rms_final > 3.0:
                print(f"  ⚠️  weight_rms={w_rms_final:.2f} > 3 — result may be noise-dominated")

            elapsed = time.time() - t0
            if verbose:
                print(f"  Q_i={Q_i:.4e}  χᵢ={chi_i:.4e}  phi_max={phi_final:.3e}  "
                      f"w_rms={w_rms_final:.2f}  [{elapsed:.1f}s]")

            results[R0_over_LT] = {
                "R0_over_LT": R0_over_LT,
                "status": "ok",
                "Q_i_avg": Q_i,
                "chi_i_avg": chi_i,
                "phi_max_final": phi_final,
                "weight_rms_final": w_rms_final,
                "zonal_rms": zonal_rms,
                "elapsed_s": elapsed,
            }

        except Exception as ex:
            print(f"  ERROR: {ex}")
            results[R0_over_LT] = {"R0_over_LT": R0_over_LT, "status": f"error: {ex}",
                                   "Q_i_avg": float('nan'), "chi_i_avg": float('nan'),
                                   "elapsed_s": time.time() - t0}

    return results


def main():
    parser = argparse.ArgumentParser(description="Heat flux CBC scan")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 50k particles, 150 steps, 4 scan points")
    parser.add_argument("--N", type=int, default=None, help="Override N_particles")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps")
    parser.add_argument("--output", default="benchmarks/results/heat_flux_cbc.json")
    args = parser.parse_args()

    if args.quick:
        R_over_LT_values = [4.0, 6.0, 7.0, 9.0]
        N_particles = args.N or 50_000
        n_steps     = args.steps or 150
    else:
        R_over_LT_values = [4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]
        N_particles = args.N or 500_000
        n_steps     = args.steps or 600

    print(f"JAX devices: {jax.devices()}")
    print(f"Heat flux scan: R/LT={R_over_LT_values}")
    print(f"N_particles={N_particles}, n_steps={n_steps}")

    results = run_heat_flux_scan(
        R_over_LT_values=R_over_LT_values,
        N_particles=N_particles,
        n_steps=n_steps,
        verbose=True,
    )

    # Summary table
    print("\n" + "=" * 65)
    print("HEAT FLUX RESULTS — χᵢ(R/LT)")
    print("=" * 65)
    print(f"{'R/LT':>6}  {'Q_i':>12}  {'χᵢ':>12}  {'phi_max':>10}  {'w_rms':>6}")
    print("-" * 65)
    for v in results.values():
        rlt = v["R0_over_LT"]
        qi  = v.get("Q_i_avg", float('nan'))
        chi = v.get("chi_i_avg", float('nan'))
        pm  = v.get("phi_max_final", float('nan'))
        wr  = v.get("weight_rms_final", float('nan'))
        st  = v.get("status", "?")
        flag = "  ⚠️" if not np.isnan(wr) and wr > 3.0 else ""
        print(f"{rlt:>6.1f}  {qi:>12.4e}  {chi:>12.4e}  {pm:>10.3e}  {wr:>6.2f}  {st}{flag}")
    print("=" * 65)

    # Dimits shift detection
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    threshold = None
    for rlt in sorted(ok):
        if ok[rlt]["Q_i_avg"] > 0.01 * max((v["Q_i_avg"] for v in ok.values() if v["Q_i_avg"] > 0), default=1):
            threshold = rlt
            break
    if threshold:
        print(f"\nDimits-like onset: R/LT ≈ {threshold}  (ref: ~6.0)")

    # Save
    summary = {
        "description": "Nonlinear CBC heat flux scan: chi_i vs R/LT",
        "parameters": {"N_particles": N_particles, "n_steps": n_steps, "dt": 0.05,
                       "noise_controls": "canonical_loading+pullback+weight_spread+nu_krook"},
        "devices": str(jax.devices()),
        "results": list(results.values()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
