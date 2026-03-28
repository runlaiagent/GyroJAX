"""
KBM (Kinetic Ballooning Mode) linear benchmark — full-f path.

Full-f PIC: dW/dt = 0 (constant marker weights).
Higher noise floor than delta-f but no weight blowup constraint.

Expected: same qualitative gamma(beta) curve as delta-f,
possibly higher noise at low beta, but cleaner at high beta.

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
from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf


def measure_growth_rate(diags, dt, fit_start=20, fit_end=100):
    """Fit γ from log(phi_max) slope in linear phase."""
    phi_vals = np.array([float(d.phi_max) for d in diags])
    n = len(phi_vals)
    fit_end = min(fit_end, n - 1)
    if fit_end <= fit_start:
        return float('nan')
    t = np.arange(fit_start, fit_end) * dt
    log_phi = np.log(np.maximum(phi_vals[fit_start:fit_end], 1e-20))
    if np.all(log_phi < -10):
        return 0.0
    coeffs = np.polyfit(t, log_phi, 1)
    return float(coeffs[0])


def run_kbm_scan_fullf(beta_values, N_particles=200_000, n_steps=200, verbose=True):
    results = {}
    for beta in beta_values:
        if verbose:
            print(f"\n=== β = {beta:.4f} ===")
        t0 = time.time()

        cfg = SimConfigFullF(
            Npsi=16, Ntheta=32, Nalpha=32,
            N_particles=N_particles,
            n_steps=n_steps,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=6.9,
            R0_over_Ln=2.2,
            pert_amp=1e-3,        # larger than delta-f: full-f noise floor is higher
            single_mode=True,
            k_mode=5,
            beta=beta,
            nu_krook=0.0,
            resample_interval=0,  # no resampling in linear phase
            k_alpha_min=0,
        )

        key = jax.random.PRNGKey(42)
        try:
            diags, state, phi, geom = run_simulation_fullf(cfg, key=key, verbose=False)
            phi_vals = np.array([float(d.phi_max) for d in diags])
            w_rms_final = float(diags[-1].weight_rms)

            gamma = measure_growth_rate(diags, cfg.dt, fit_start=20, fit_end=100)
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
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {ex}")
            results[beta] = {"beta": beta, "status": f"error: {ex}",
                             "gamma": float('nan'), "elapsed_s": time.time() - t0}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="4-point scan, 50k particles")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", default="benchmarks/results/kbm_fullf.json")
    args = parser.parse_args()

    if args.quick:
        beta_values = [0.0, 0.005, 0.01, 0.02]
        N = args.N or 50_000
        n_steps = args.steps or 150
    else:
        beta_values = [0.0, 0.002, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
        N = args.N or 200_000
        n_steps = args.steps or 200

    print(f"JAX devices: {jax.devices()}")
    print(f"KBM full-f β scan: {beta_values}")
    print(f"N_particles={N}, n_steps={n_steps}, pert_amp=1e-3")

    results = run_kbm_scan_fullf(beta_values, N_particles=N, n_steps=n_steps)

    # Summary
    print("\n" + "=" * 60)
    print("KBM FULL-F RESULTS — γ(β)")
    print("=" * 60)
    print(f"{'β':>8}  {'γ':>10}  {'γ_late':>10}  {'phi_max':>10}  {'status'}")
    print("-" * 60)
    for v in results.values():
        print(f"{v['beta']:>8.4f}  {v.get('gamma', float('nan')):>10.4f}  "
              f"{v.get('gamma_late', float('nan')):>10.4f}  "
              f"{v.get('phi_max_final', float('nan')):>10.3e}  {v.get('status','?')}")
    print("=" * 60)

    # Identify KBM threshold
    ok = {k: v for k, v in results.items()
          if v.get("status") == "ok" and not np.isnan(v.get("gamma", float('nan')))}
    if len(ok) >= 3:
        betas = sorted(ok.keys())
        gammas = [ok[b]["gamma"] for b in betas]
        min_idx = int(np.argmin(gammas))
        beta_min = betas[min_idx]
        print(f"\nγ minimum at β={beta_min:.4f} (ITG→KBM transition)")

    # Compare with delta-f if available
    deltaf_path = Path(args.output).parent / "kbm_deltaf.json"
    if deltaf_path.exists():
        with open(deltaf_path) as f:
            deltaf_data = json.load(f)
        deltaf_results = {r["beta"]: r for r in deltaf_data.get("results", [])}
        print("\n--- Comparison: full-f vs delta-f ---")
        print(f"{'β':>8}  {'γ_fullf':>10}  {'γ_deltaf':>10}  {'diff%':>8}")
        print("-" * 45)
        for v in results.values():
            b = v["beta"]
            gf = v.get("gamma", float('nan'))
            df_v = deltaf_results.get(b, {})
            gd = df_v.get("gamma", float('nan'))
            if not np.isnan(gf) and not np.isnan(gd) and abs(gd) > 1e-6:
                diff_pct = 100.0 * (gf - gd) / abs(gd)
                print(f"{b:>8.4f}  {gf:>10.4f}  {gd:>10.4f}  {diff_pct:>+8.1f}%")
            else:
                print(f"{b:>8.4f}  {gf:>10.4f}  {gd:>10.4f}  {'N/A':>8}")

    summary = {
        "description": "KBM linear benchmark: gamma(beta), CBC R/LT=6.9, full-f path",
        "parameters": {"N_particles": N, "n_steps": n_steps, "R_over_LT": 6.9,
                       "pert_amp": 1e-3, "single_mode": True, "k_mode": 5,
                       "mode": "full-f (dW/dt=0)"},
        "devices": str(jax.devices()),
        "results": list(results.values()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
