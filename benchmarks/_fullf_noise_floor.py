import sys, json, os, inspect
sys.path.insert(0, '/home/blues/wlhx/GyroJAX')
import jax, numpy as np
from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf
from gyrojax.diagnostics import extract_growth_rate_smart

has_rms = 'phi_rms_series' in inspect.signature(extract_growth_rate_smart).parameters

def fit_gamma(diags, dt=0.05):
    phi_max = np.array([float(d.phi_max) for d in diags])
    phi_rms = np.array([float(d.phi_rms) for d in diags])
    result = extract_growth_rate_smart(phi_max, dt, phi_rms_series=phi_rms) if has_rms else extract_growth_rate_smart(phi_max, dt)
    return float(result[0]) if isinstance(result, tuple) else float(result)

COMMON = dict(R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
              Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
              R0_over_LT=6.9, R0_over_Ln=2.2,
              single_mode=True, resample_interval=50,
              Npsi=16, Ntheta=32, Nalpha=64, dt=0.05, n_steps=150)
key = jax.random.PRNGKey(42)
results = {"noise_floor_study": [], "gamma_spectrum": [], "weight_constancy": []}

print("=== Full-f noise floor (k_mode=18) ===")
for N, pa in [(100_000, 1e-3), (300_000, 1e-3), (500_000, 1e-3)]:
    cfg = SimConfigFullF(**COMMON, N_particles=N, k_mode=18, pert_amp=pa)
    diags, _, _, _ = run_simulation_fullf(cfg, key=key, verbose=False)
    g = fit_gamma(diags)
    w = np.array([float(d.weight_rms) for d in diags])
    drift = float(np.std(w) / (np.mean(w) + 1e-30))
    status = "PASS" if drift < 1e-3 else "FAIL"
    print(f"  N={N:>7}  pert={pa:.0e}  gamma={g:.4f}  w_drift={drift:.2e}")
    print(f"  weight_constancy: {status}  drift={drift:.2e}")
    results["noise_floor_study"].append({"N": N, "pert_amp": pa, "gamma": round(g,5), "weight_drift": round(drift,8)})
    results["weight_constancy"].append({"N": N, "k_mode": 18, "drift": round(drift,8), "pass": drift < 1e-3})

print("\n=== Full-f gamma spectrum (N=300k, pert_amp=1e-3) ===")
df_ref = {12: 0.201, 18: 0.172, 24: 0.185, 30: 0.142}
for k in [12, 18, 24, 30]:
    cfg = SimConfigFullF(**COMMON, N_particles=300_000, k_mode=k, pert_amp=1e-3)
    diags, _, _, _ = run_simulation_fullf(cfg, key=key, verbose=False)
    g = fit_gamma(diags)
    gdf = df_ref.get(k, 0)
    w = np.array([float(d.weight_rms) for d in diags])
    drift = float(np.std(w) / (np.mean(w) + 1e-30))
    status = "PASS" if drift < 1e-3 else "FAIL"
    print(f"  k={k:>2}  gamma_ff={g:.4f}  gamma_df={gdf:.4f}  ratio={g/(gdf+1e-9):.2f}")
    print(f"  weight_constancy: {status}  drift={drift:.2e}")
    results["gamma_spectrum"].append({"k_mode": k, "gamma_fullf": round(g,5), "gamma_deltaf": gdf, "ratio": round(g/(gdf+1e-9),3)})
    results["weight_constancy"].append({"N": 300000, "k_mode": k, "drift": round(drift,8), "pass": drift < 1e-3})

os.makedirs("/home/blues/wlhx/GyroJAX/benchmarks/results", exist_ok=True)
with open("/home/blues/wlhx/GyroJAX/benchmarks/results/fullf_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved benchmarks/results/fullf_benchmark.json")
print(json.dumps(results, indent=2))
