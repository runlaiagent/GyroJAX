"""
EM CBC Benchmark — tests electromagnetic coupling (beta > 0) in field-aligned gyrokinetics.

Measures ITG growth rate across beta = [0.0, 0.01, 0.05] and tracks A_par amplitude.
"""
import numpy as np
import time

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


def run_em_benchmark(beta: float):
    """Run CBC benchmark with given beta and return (gamma, phi_max_final, A_par_max_final)."""
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=32,
        N_particles=150_000,
        n_steps=400,
        dt=0.05,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        R0_over_LT=6.9,
        R0_over_Ln=2.2,
        single_mode=True,
        beta=beta,
        pert_amp=1e-2,
        k_mode=1,
    )

    print(f"\n{'='*60}")
    print(f"  Running CBC EM benchmark: beta = {beta}")
    print(f"  Grid: {cfg.Npsi}x{cfg.Ntheta}x{cfg.Nalpha}, N_p={cfg.N_particles}, n_steps={cfg.n_steps}")
    print(f"{'='*60}")

    t0 = time.time()
    result = run_simulation_fa(cfg)
    # run_simulation_fa returns (diags, state, phi, geom) or just diags depending on version
    if isinstance(result, tuple):
        diags = result[0]
    else:
        diags = result
    elapsed = time.time() - t0

    dt = cfg.dt
    phi_arr = np.array([float(d.phi_max) for d in diags])

    # Fit log(phi) ~ gamma*t over steps 50-200
    t0_idx, t1_idx = 50, min(200, len(phi_arr))
    if (t1_idx > t0_idx + 10
            and phi_arr[t0_idx] > 0
            and phi_arr[t1_idx - 1] > 0):
        gamma = (np.log(phi_arr[t1_idx - 1]) - np.log(phi_arr[t0_idx])) / ((t1_idx - t0_idx) * dt)
    else:
        gamma = float('nan')

    phi_final = phi_arr[-1]

    # Track A_par_max if beta > 0
    a_par_max = None
    if beta > 0.0:
        try:
            a_par_arr = np.array([float(d.phi_max) for d in diags])  # placeholder
            # Try to get actual A_par from diags if available
            if hasattr(diags[0], 'a_par_max'):
                a_par_arr = np.array([float(d.a_par_max) for d in diags])
                a_par_max = float(np.max(a_par_arr))
        except Exception:
            a_par_max = None

    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  phi_max range: {phi_arr[0]:.3e} → {phi_final:.3e}")
    print(f"  Growth rate gamma (steps {t0_idx}-{t1_idx}): {gamma:.4f} [R0/vti]⁻¹")
    if a_par_max is not None:
        print(f"  A_par_max (peak): {a_par_max:.3e}")

    return gamma, phi_final, a_par_max


def main():
    print("\n" + "="*60)
    print("  GyroJAX EM Ampere CBC Benchmark")
    print("  Tests beta=0 (ES), 0.01 (low-beta EM), 0.05 (mid-beta EM)")
    print("="*60)

    betas = [0.0, 0.01, 0.05]
    results = {}

    for beta in betas:
        gamma, phi_final, a_par_max = run_em_benchmark(beta)
        results[beta] = dict(gamma=gamma, phi_final=phi_final, a_par_max=a_par_max)

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  {'beta':>8}  {'gamma [R0/vti]':>16}  {'phi_final':>12}  {'A_par_max':>12}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*12}  {'-'*12}")
    for beta in betas:
        r = results[beta]
        a_str = f"{r['a_par_max']:.3e}" if r['a_par_max'] is not None else "N/A"
        print(f"  {beta:>8.3f}  {r['gamma']:>16.4f}  {r['phi_final']:>12.3e}  {a_str:>12}")

    print()
    # Physics check: EM stabilization
    g0 = results[0.0]['gamma']
    for beta in [0.01, 0.05]:
        g = results[beta]['gamma']
        if not np.isnan(g) and not np.isnan(g0) and g0 > 0:
            change_pct = 100.0 * (g - g0) / g0
            print(f"  beta={beta}: gamma change vs ES = {change_pct:+.1f}%")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
