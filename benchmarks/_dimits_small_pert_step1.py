"""
Dimits small perturbation quick check — Step 1
R/LT=4.0 and R/LT=6.9 with pert_amp=1e-5, N_particles=500k
"""
import sys, json, os
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)
sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

COMMON = dict(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=200_000,  # 500k causes GPU OOM; 200k is max stable
    n_steps=400, dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    R0_over_Ln=2.2,
    vti=1.0, n0_avg=1.0,
    pert_amp=1e-5,
    single_mode=False,
    k_alpha_min=4,
    nu_krook=0.005,
    canonical_loading=True,
    use_pullback=False,
    use_weight_spread=True,
    weight_spread_interval=10,
    semi_implicit_weights=True,
)

RLT_LIST = [4.0, 6.9]
results = {}

for i, rlt in enumerate(RLT_LIST):
    print(f"\n{'='*60}")
    print(f"R/LT = {rlt:.1f}  [{i+1}/{len(RLT_LIST)}]")
    print('='*60)

    cfg = SimConfigFA(R0_over_LT=rlt, **COMMON)
    key = jax.random.PRNGKey(42 + i)

    try:
        diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)

        phi_max_arr = np.array([float(d.phi_max) for d in diags])
        w_rms_arr   = np.array([float(d.weight_rms) for d in diags])

        print(f"\n--- Periodic report every 50 steps ---")
        for step in range(0, len(diags), 50):
            print(f"  step={step:4d}  w_rms={w_rms_arr[step]:.4f}  phi_max={phi_max_arr[step]:.4e}")
        # also print final
        last = len(diags)-1
        print(f"  step={last:4d}  w_rms={w_rms_arr[last]:.4f}  phi_max={phi_max_arr[last]:.4e}")

        w_rms_200 = float(w_rms_arr[min(200, last)])
        phi_avg = float(np.mean(phi_max_arr[int(0.7*len(phi_max_arr)):]))

        print(f"\nw_rms at step 200 = {w_rms_200:.4f}")
        print(f"phi_avg (last 30%) = {phi_avg:.4e}")

        results[str(rlt)] = {
            'rlt': rlt,
            'w_rms_200': w_rms_200,
            'w_rms_final': float(w_rms_arr[-1]),
            'phi_avg': phi_avg,
            'phi_max_arr': phi_max_arr.tolist(),
            'w_rms_arr': w_rms_arr.tolist(),
        }

    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {ex}")
        results[str(rlt)] = {'rlt': rlt, 'error': str(ex)}

print("\n\n" + "="*60)
print("STEP 1 SUMMARY")
print("="*60)
for rlt in RLT_LIST:
    r = results.get(str(rlt), {})
    if 'error' in r:
        print(f"  R/LT={rlt}: ERROR {r['error']}")
    else:
        w200 = r.get('w_rms_200', float('nan'))
        phi_avg = r.get('phi_avg', float('nan'))
        print(f"  R/LT={rlt:.1f}: w_rms@200={w200:.3f}  phi_avg={phi_avg:.4e}")

# Check if improvement
ok_keys = [k for k in results if 'error' not in results[k]]
if len(ok_keys) == 2:
    r40 = results['4.0']
    r69 = results['6.9']
    w200_ok = r40.get('w_rms_200', 999) < 2.0
    phi_ordered = r40.get('phi_avg', 999) < r69.get('phi_avg', 0)
    print(f"\nw_rms<2 at step 200 (R/LT=4.0): {w200_ok}")
    print(f"phi_avg(4.0) < phi_avg(6.9) [correct ordering]: {phi_ordered}")
    print(f"  phi_avg ratio (6.9/4.0): {r69.get('phi_avg',0)/(r40.get('phi_avg',1)+1e-30):.2f}")
    print(f"\nPROCEED_TO_STEP2: {w200_ok}")
else:
    print("\nNot enough results to judge improvement.")
    print("PROCEED_TO_STEP2: False")

# Save partial results
os.makedirs("/home/blues/wlhx/GyroJAX/benchmarks/results", exist_ok=True)
with open("/home/blues/wlhx/GyroJAX/benchmarks/results/dimits_small_pert_step1.json", "w") as f:
    # Don't dump full arrays to json to keep it readable; summarize
    summary = {}
    for k, v in results.items():
        s = {kk: vv for kk, vv in v.items() if kk not in ('phi_max_arr', 'w_rms_arr')}
        summary[k] = s
    json.dump(summary, f, indent=2)
print("\nSaved step1 results.")
