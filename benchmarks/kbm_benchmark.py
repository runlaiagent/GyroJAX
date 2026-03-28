"""
KBM (Kinetic Ballooning Mode) linear benchmark.

Scans β from 0 to 0.02 at fixed R/LT=6.9 (CBC parameters).
Measures γ(β) to identify ITG→KBM transition.

Expected behavior:
  β=0.000: pure ITG, γ ~ 0.1-0.3 (normalized units)
  β=0.005: EM stabilization, γ decreases
  β=0.010: near KBM threshold — γ minimum then rising
  β=0.020: KBM-dominated, γ rising with β

References:
  Pueschel et al., Phys. Plasmas 15, 102310 (2008) — CBC KBM threshold β_crit~0.01
  Candy & Waltz (GYRO), Phys. Rev. Lett. 91, 045001 (2003)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


def measure_growth_rate(diags, dt, fit_start=20, fit_end=100):
    """Fit γ from log(phi_max) slope in linear phase."""
    phi_vals = np.array([float(d.phi_max) for d in diags])
    n = len(phi_vals)
    fit_end = min(fit_end, n - 1)
    if fit_end <= fit_start:
        return float('nan')
    t = np.arange(fit_start, fit_end) * dt
    log_phi = np.log(np.maximum(phi_vals[fit_start:fit_end], 1e-20))
    # Only fit if there's actual growth
    if np.all(log_phi < -10):
        return 0.0
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_kbm_scan(beta_values, N_particles=100_000, n_steps=200, verbose=True):
    results = {}
    for beta in beta_values:
        if verbose:
            print(f"\n=== β = {beta:.4f} ===")
        t0 = time.time()

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=32,
            N_particles=N_particles,
            n_steps=n_steps,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
            pert_amp=1e-5,        # small amplitude — stay in linear regime
            single_mode=True,
            k_mode=5,
            beta=beta,
            # Linear benchmark settings — minimal noise controls
            canonical_loading=False,
            use_pullback=False,
            nu_soft=0.0,
            nu_krook=0.0,
            use_weight_spread=False,
            k_alpha_min=0,
            gyroaverage_scatter=True,
            use_radial_gaa=True,
        )

        key = jax.random.PRNGKey(42)
        try:
            diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=False)
            phi_vals = np.array([float(d.phi_max) for d in diags])
            w_rms_final = float(diags[-1].weight_rms)

            # Fit growth rate in linear phase (steps 20-100)
            gamma = measure_growth_rate(diags, cfg.dt, fit_start=20, fit_end=100)

            # Also try late linear (steps 50-150) in case slow grower
            gamma_late = measure_growth_rate(diags, cfg.dt, fit_start=50, fit_end=150)

            elapsed = time.time() - t0
            status = "ok"
            if np.any(np.isnan(phi_vals)):
                status = "nan"
                gamma = float('nan')

            if verbose:
                print(f"  γ = {gamma:.4f}  γ_late = {gamma_late:.4f}  "
                      f"phi_max = {phi_vals[-1]:.3e}  w_rms = {w_rms_final:.3f}  [{elapsed:.1f}s]")

            results[beta] = {
                "beta": beta,
                "status": status,
                "gamma": gamma,
                "gamma_late": gamma_late,
                "phi_max_final": float(phi_vals[-1]),
                "weight_rms_final": w_rms_final,
                "elapsed_s": elapsed,
            }

        except Exception as ex:
            print(f"  ERROR: {ex}")
            results[beta] = {"beta": beta, "status": f"error: {ex}",
                             "gamma": float('nan'), "elapsed_s": time.time() - t0}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="4-point scan, 50k particles")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", default="benchmarks/results/kbm_deltaf.json")
    args = parser.parse_args()

    if args.quick:
        beta_values = [0.0, 0.005, 0.01, 0.02]
        N = args.N or 50_000
        n_steps = args.steps or 150
    else:
        beta_values = [0.0, 0.002, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
        N = args.N or 100_000
        n_steps = args.steps or 200

    print(f"JAX devices: {jax.devices()}")
    print(f"KBM β scan: {beta_values}")
    print(f"N_particles={N}, n_steps={n_steps}")

    results = run_kbm_scan(beta_values, N_particles=N, n_steps=n_steps)

    # Summary
    print("\n" + "=" * 60)
    print("KBM RESULTS — γ(β)")
    print("=" * 60)
    print(f"{'β':>8}  {'γ':>10}  {'γ_late':>10}  {'phi_max':>10}  {'w_rms':>6}")
    print("-" * 60)
    for v in results.values():
        print(f"{v['beta']:>8.4f}  {v.get('gamma', float('nan')):>10.4f}  "
              f"{v.get('gamma_late', float('nan')):>10.4f}  "
              f"{v.get('phi_max_final', float('nan')):>10.3e}  "
              f"{v.get('weight_rms_final', float('nan')):>6.3f}  {v.get('status','?')}")
    print("=" * 60)

    # Identify KBM threshold: β where γ is minimum (EM stabilization → KBM onset)
    ok = {k: v for k, v in results.items() if v.get("status") == "ok" and not np.isnan(v.get("gamma", float('nan')))}
    if len(ok) >= 3:
        betas = sorted(ok.keys())
        gammas = [ok[b]["gamma"] for b in betas]
        min_idx = int(np.argmin(gammas))
        beta_min = betas[min_idx]
        print(f"\nγ minimum at β={beta_min:.4f} (ITG→KBM transition)")
        if min_idx < len(betas) - 1:
            print(f"KBM onset: β ~ {betas[min_idx]:.4f} – {betas[min(min_idx+1, len(betas)-1)]:.4f}")

    summary = {
        "description": "KBM linear benchmark: gamma(beta), CBC R/LT=6.9, delta-f",
        "parameters": {"N_particles": N, "n_steps": n_steps, "R_over_LT": 6.9,
                       "pert_amp": 1e-5, "single_mode": True, "k_mode": 5},
        "devices": str(jax.devices()),
        "results": list(results.values()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
