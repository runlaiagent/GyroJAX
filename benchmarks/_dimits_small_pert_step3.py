"""
Dimits small perturbation — Step 3: pert_amp=1e-6 with pullback
"""
import sys, json, os
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)
sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

BASE = dict(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=200_000,
    n_steps=600, dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2, vti=1.0, n0_avg=1.0,
    single_mode=False,
    k_alpha_min=4, nu_krook=0.005,
    canonical_loading=True,
    use_weight_spread=True, weight_spread_interval=10,
    semi_implicit_weights=True,
)

CONFIGS = [
    dict(label="1e-6_nopullback", pert_amp=1e-6, use_pullback=False, pullback_interval=50),
    dict(label="1e-6_pullback20", pert_amp=1e-6, use_pullback=True, pullback_interval=20),
    dict(label="1e-5_pullback20", pert_amp=1e-5, use_pullback=True, pullback_interval=20),
]

RLT_LIST = [4.0, 6.9]
all_results = {}

for cfg_extra in CONFIGS:
    label = cfg_extra.pop('label')
    print(f"\n{'#'*60}")
    print(f"CONFIG: {label}")
    print('#'*60)

    results = {}
    for i, rlt in enumerate(RLT_LIST):
        print(f"\n{'='*50}")
        print(f"R/LT = {rlt:.1f}")
        print('='*50)

        cfg = SimConfigFA(R0_over_LT=rlt, **BASE, **cfg_extra)
        key = jax.random.PRNGKey(42 + i)

        try:
            diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)

            phi_max_arr = np.array([float(d.phi_max) for d in diags])
            w_rms_arr   = np.array([float(d.weight_rms) for d in diags])

            print(f"\n--- Periodic report every 50 steps ---")
            for step in range(0, len(diags), 50):
                print(f"  step={step:4d}  w_rms={w_rms_arr[step]:.4f}  phi_max={phi_max_arr[step]:.4e}")
            last = len(diags)-1
            print(f"  step={last:4d}  w_rms={w_rms_arr[last]:.4f}  phi_max={phi_max_arr[last]:.4e}")

            w_rms_200 = float(w_rms_arr[min(200, last)])
            phi_avg = float(np.mean(phi_max_arr[int(0.7*len(phi_max_arr)):]))

            print(f"\nw_rms at step 200 = {w_rms_200:.4f}")
            print(f"phi_avg (last 30%) = {phi_avg:.4e}")

            results[str(rlt)] = {
                'rlt': rlt, 'w_rms_200': w_rms_200,
                'w_rms_final': float(w_rms_arr[-1]),
                'phi_avg': phi_avg,
            }

        except Exception as ex:
            import traceback; traceback.print_exc()
            results[str(rlt)] = {'rlt': rlt, 'error': str(ex)}

    # Evaluate this config
    print(f"\n--- {label} SUMMARY ---")
    r40 = results.get('4.0', {})
    r69 = results.get('6.9', {})
    if 'error' not in r40 and 'error' not in r69:
        w200_ok = r40.get('w_rms_200', 999) < 2.0
        phi_ordered = r40.get('phi_avg', 999) < r69.get('phi_avg', 0)
        ratio = r69.get('phi_avg', 0) / (r40.get('phi_avg', 1) + 1e-30)
        print(f"  R/LT=4.0: w_rms@200={r40.get('w_rms_200','?'):.3f}  phi_avg={r40.get('phi_avg','?'):.4e}")
        print(f"  R/LT=6.9: w_rms@200={r69.get('w_rms_200','?'):.3f}  phi_avg={r69.get('phi_avg','?'):.4e}")
        print(f"  phi_avg ratio (6.9/4.0): {ratio:.2f}")
        print(f"  w_rms<2 at step 200: {w200_ok}")
        print(f"  Correct phi ordering: {phi_ordered}")
    else:
        print("  Errors encountered")

    all_results[label] = results
    # restore label for next iteration
    cfg_extra['label'] = label

# Save
os.makedirs("/home/blues/wlhx/GyroJAX/benchmarks/results", exist_ok=True)
with open("/home/blues/wlhx/GyroJAX/benchmarks/results/dimits_small_pert_step3.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved step3 results.")
