"""
Nonlinear heat flux benchmark: χᵢ vs R/LT (Cyclone Base Case).

Measures ion thermal diffusivity χᵢ as a function of temperature gradient R/LT.
Expected result:
  - χᵢ ≈ 0 for R/LT < 6 (Dimits shift regime — transport suppressed by zonal flows)
  - χᵢ > 0 for R/LT > 6.5 (anomalous transport regime)

Physics:
  Q_i = Σ_p w_p · (E_psi_p / B_p) · (ε_p - 1.5)
  where ε_p = (0.5·mi·vpar² + μ·B) / Ti  (normalized particle energy)

  χᵢ = Q_i · LT / (n₀ · Ti)  =  Q_i · R0 / (n₀ · R0_over_LT)

Usage:
  python benchmarks/heat_flux_cbc.py [--quick] [--output results/heat_flux_cbc.json]
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gyrojax.simulation_fa import SimConfigFA
from gyrojax.geometry.field_aligned import build_field_aligned_geometry, interp_fa_to_particles
from gyrojax.particles.guiding_center import init_maxwellian_particles
from gyrojax.particles.guiding_center_fa import push_particles_fa
from gyrojax.fields.poisson_fa import solve_poisson_fa, compute_efield_fa, filter_single_mode
from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa, gather_from_grid_fa
from gyrojax.deltaf.weights import update_weights
from gyrojax.geometry.profiles import build_cbc_profiles


def measure_heat_flux(state, weights, E_psi_p, B_p, cfg):
    """
    Measure ion heat flux Q_i and thermal diffusivity χᵢ.

    Q_i = Σ_p w_p · (E_psi_p / B_p) · (ε_p - 1.5)
    ε_p = (0.5·mi·vpar² + μ·B) / Ti   (normalized particle energy)

    χᵢ = Q_i · LT / (n₀ · Ti) = Q_i · R0 / (n₀ · R0_over_LT)
    """
    energy_p = 0.5 * cfg.mi * state.vpar**2 + state.mu * B_p
    delta_energy = energy_p / cfg.Ti - 1.5   # normalized energy perturbation
    vExB_psi = E_psi_p / B_p                  # radial ExB velocity ∝ E_psi/B

    Q_i = jnp.sum(weights * vExB_psi * delta_energy)
    chi_i = Q_i * cfg.R0 / (cfg.n0_avg * cfg.R0_over_LT)
    return float(Q_i), float(chi_i)


def run_heat_flux_scan(
    R_over_LT_values,
    N_particles: int = 50_000,
    n_steps: int = 200,
    dt: float = 0.05,
    n_avg_frac: float = 0.3,
    verbose: bool = True,
    seed: int = 42,
):
    """
    Run nonlinear CBC R/LT scan and measure χᵢ.

    Parameters
    ----------
    R_over_LT_values : list of R/LT values to scan
    N_particles      : number of marker particles
    n_steps          : number of time steps per run
    dt               : timestep (normalized to R0/vti)
    n_avg_frac       : fraction of steps at end to time-average Q_i
    verbose          : print progress

    Returns
    -------
    dict: {R_over_LT: {Q_i_avg, chi_i_avg, Q_i_time_series}}
    """
    results = {}

    for R0_over_LT in R_over_LT_values:
        if verbose:
            print(f"\n=== R/LT = {R0_over_LT} ===")
        t0 = time.time()

        cfg = SimConfigFA(
            Npsi=16, Ntheta=32, Nalpha=64,
            N_particles=N_particles,
            n_steps=n_steps,
            dt=dt,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            R0_over_LT=R0_over_LT,
            R0_over_Ln=2.2,
            pert_amp=1e-3,
            single_mode=False,  # nonlinear: all modes
            k_alpha_min=0,
        )

        geom = build_field_aligned_geometry(
            Npsi=cfg.Npsi, Ntheta=cfg.Ntheta, Nalpha=cfg.Nalpha,
            R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1,
        )
        grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)
        key = jax.random.PRNGKey(seed)
        state = init_maxwellian_particles(
            cfg.N_particles, geom, vti=cfg.vti, Ti=cfg.Ti, mi=cfg.mi, key=key
        )
        # Seed density perturbation via weights
        key, subkey = jax.random.split(key)
        pert = cfg.pert_amp * jax.random.normal(subkey, shape=(cfg.N_particles,))
        state = state._replace(weight=pert.astype(jnp.float32))

        q_over_m = cfg.e / cfg.mi
        Ln = cfg.R0 / cfg.R0_over_Ln
        LT = cfg.R0 / cfg.R0_over_LT

        Q_series = []
        chi_series = []
        n_avg = max(1, int(n_steps * n_avg_frac))
        avg_start = n_steps - n_avg

        phi = jnp.zeros(grid_shape)

        for step in range(n_steps):
            # 1. Scatter
            state_fa = state._replace(zeta=state.zeta % (2 * jnp.pi))
            delta_n = scatter_to_grid_fa(state_fa, geom, grid_shape) * cfg.n0_avg

            # 2. Solve Poisson
            phi = solve_poisson_fa(delta_n, geom, cfg.n0_avg, cfg.Te, cfg.Ti, cfg.mi, cfg.e)

            # 3. Gather E
            state_gather = state._replace(zeta=state.zeta % (2 * jnp.pi))
            E_psi_p, E_theta_p, E_alpha_p = gather_from_grid_fa(phi, state_gather, geom)

            # 4. Geometry at particle positions
            B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, g_aa_p = interp_fa_to_particles(
                geom, state.r, state.theta, state.zeta
            )
            Nr = geom.psi_grid.shape[0]
            dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
            ir = jnp.clip((state.r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
            q_at_r_p = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]

            # 5. Measure heat flux BEFORE push (using current weights)
            if step >= avg_start:
                Q_i, chi_i = measure_heat_flux(state, state.weight, E_psi_p, B_p, cfg)
                Q_series.append(Q_i)
                chi_series.append(chi_i)

            # 6. Push
            state = push_particles_fa(
                state, E_psi_p, E_theta_p, E_alpha_p,
                B_p, gBpsi_p, gBth_p, kpsi_p, kth_p, q_at_r_p, g_aa_p,
                q_over_m, cfg.mi, cfg.dt, geom.R0,
            )

            # 7. Clamp
            state = state._replace(
                r=jnp.clip(state.r, geom.psi_grid[0]*1.001, geom.psi_grid[-1]*0.999),
                vpar=jnp.clip(state.vpar, -4.0 * cfg.vti, 4.0 * cfg.vti),
            )

            # 8. Update weights
            n0_p = cfg.n0_avg * jnp.exp(-(state.r - cfg.a * 0.5) / Ln)
            T_p  = cfg.Ti * jnp.exp(-(state.r - cfg.a * 0.5) / LT)
            d_ln_n0_dr = jnp.full_like(state.r, -1.0 / Ln)
            d_ln_T_dr  = jnp.full_like(state.r, -1.0 / LT)

            state = update_weights(
                state, E_psi_p, E_theta_p, E_alpha_p,
                B_p, gBpsi_p, gBth_p, kpsi_p, kth_p,
                q_at_r_p, n0_p, T_p,
                d_ln_n0_dr, d_ln_T_dr,
                q_over_m, cfg.mi, cfg.R0, cfg.dt,
            )

            if verbose and (step + 1) % 50 == 0:
                phi_max = float(jnp.max(jnp.abs(phi)))
                print(f"  step {step+1}/{n_steps}  phi_max={phi_max:.3e}")

        Q_avg = float(np.mean(Q_series)) if Q_series else 0.0
        chi_avg = float(np.mean(chi_series)) if chi_series else 0.0
        elapsed = time.time() - t0

        results[R0_over_LT] = {
            "R0_over_LT": R0_over_LT,
            "Q_i_avg": Q_avg,
            "chi_i_avg": chi_avg,
            "Q_i_time_series": Q_series,
            "chi_i_time_series": chi_series,
            "elapsed_s": elapsed,
        }
        if verbose:
            print(f"  Q_i_avg={Q_avg:.4e}  chi_i_avg={chi_avg:.4e}  [{elapsed:.1f}s]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Heat flux CBC scan")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: fewer particles and steps (for testing)")
    parser.add_argument("--output", default="benchmarks/results/heat_flux_cbc.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.quick:
        # Quick test mode: small N, few steps
        R_over_LT_values = [4.0, 6.0, 7.0, 9.0]
        N_particles = 20_000
        n_steps = 80
    else:
        # Full production run
        R_over_LT_values = [4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]
        N_particles = 500_000
        n_steps = 600

    print(f"Running heat flux scan: R/LT = {R_over_LT_values}")
    print(f"N_particles={N_particles}, n_steps={n_steps}")

    results = run_heat_flux_scan(
        R_over_LT_values=R_over_LT_values,
        N_particles=N_particles,
        n_steps=n_steps,
        verbose=True,
    )

    # Build summary
    summary = {
        "description": "Nonlinear CBC heat flux scan: chi_i vs R/LT",
        "parameters": {
            "N_particles": N_particles,
            "n_steps": n_steps,
            "dt": 0.05,
            "R0": 1.0, "a": 0.18, "q0": 1.4, "q1": 0.5,
            "R0_over_Ln": 2.2,
        },
        "results": [
            {
                "R0_over_LT": v["R0_over_LT"],
                "Q_i_avg": v["Q_i_avg"],
                "chi_i_avg": v["chi_i_avg"],
                "elapsed_s": v["elapsed_s"],
            }
            for v in results.values()
        ],
        "expected": {
            "Dimits_shift": "chi_i ~ 0 for R/LT < 6",
            "anomalous_transport": "chi_i > 0 for R/LT > 6.5",
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\nR/LT scan results:")
    print(f"{'R/LT':>8}  {'Q_i_avg':>12}  {'chi_i_avg':>12}")
    print("-" * 36)
    for v in results.values():
        print(f"{v['R0_over_LT']:>8.1f}  {v['Q_i_avg']:>12.4e}  {v['chi_i_avg']:>12.4e}")


if __name__ == "__main__":
    main()
