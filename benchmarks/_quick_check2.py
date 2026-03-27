"""Quick 2-point check with all noise-control flags"""
import sys
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)
sys.path.insert(0, "/home/blues/wlhx/GyroJAX")

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

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

for i, rlt in enumerate([4.0, 6.9]):
    print(f"\n=== R/LT = {rlt} ===")
    cfg = SimConfigFA(R0_over_LT=rlt, **COMMON)
    key = jax.random.PRNGKey(42 + i)
    diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=True)
    phi_max_arr = np.array([float(d.phi_max) for d in diags])
    weight_rms_arr = np.array([float(d.weight_rms) for d in diags])
    w_rms_final = float(weight_rms_arr[-1])
    n_tail = max(int(len(phi_max_arr) * 0.3), 10)
    phi_avg = float(np.mean(phi_max_arr[-n_tail:]))
    print(f"RESULT: w_rms={w_rms_final:.3f}  phi_avg={phi_avg:.3e}")
    print(f"w_rms < 1: {w_rms_final < 1.0}")
