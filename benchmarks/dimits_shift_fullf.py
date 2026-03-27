"""
Dimits Shift Benchmark — Full-f (Vlasov PIC)
=============================================
Scans R/LT = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0] with full-f nonlinear PIC.

The Dimits shift: ITG turbulence is suppressed by zonal flows below R/LT ≈ 6.0.
Full-f weights are constant (dW/dt=0), so there's no weight blowup and zonal
flows can develop naturally, unlike the δf code.

Reference: Dimits et al. (2000) Phys. Plasmas 7, 969
"""

import sys
import os
import json
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf

# ── Scan parameters ────────────────────────────────────────────────────────
RLT_VALUES = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]

COMMON = dict(
    Npsi=16, Ntheta=32, Nalpha=64,
    N_particles=500_000,
    n_steps=400, dt=0.05,
    R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
    vti=1.0, n0_avg=1.0,
    R0_over_Ln=2.2,
    pert_amp=1e-3,
    resample_interval=50,
    single_mode=False,   # nonlinear — all modes needed
    k_mode=18,
    k_alpha_min=0,       # allow zonal flows (k_alpha=0)
)

TAIL_FRAC = 0.30        # last 30% of steps for time-averaging

# ── Results directory ──────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "dimits_fullf.json")


def tail_mean(arr: np.ndarray, frac: float = TAIL_FRAC) -> float:
    n = max(1, int(len(arr) * frac))
    return float(np.mean(arr[-n:]))


print("=" * 72)
print("GyroJAX Dimits Shift Benchmark — Full-f Vlasov PIC")
print("Cyclone Base Case — nonlinear R/LT scan (k_alpha_min=0, zonal flows on)")
print("=" * 72)

scan_results = []

for idx, rlt in enumerate(RLT_VALUES):
    print(f"\n[{idx+1}/{len(RLT_VALUES)}] R/LT = {rlt:.1f}")
    print("-" * 44)

    cfg = SimConfigFullF(R0_over_LT=rlt, **COMMON)
    key = jax.random.PRNGKey(42 + idx)

    try:
        diags, state, phi, geom = run_simulation_fullf(cfg, key=key, verbose=True)

        phi_rms_arr      = np.array([float(d.phi_rms)       for d in diags])
        phi_max_arr      = np.array([float(d.phi_max)        for d in diags])
        phi_zonal_arr    = np.array([float(d.phi_zonal_rms)  for d in diags])
        weight_rms_arr   = np.array([float(d.weight_rms)     for d in diags])

        # Time averages over last 30%
        phi_rms_mean      = tail_mean(phi_rms_arr)
        phi_zonal_mean    = tail_mean(phi_zonal_arr)
        zonal_ratio       = phi_zonal_mean / (phi_rms_mean + 1e-30)

        # ── Q_i proxy via phi growth / saturation level ──────────────────
        # True Q_i = <vE·∇ψ·δTi> needs per-step fields we don't store.
        # Use phi_rms_mean as a proxy: low phi → suppressed, high phi → active.
        # Normalise by a reference run at R/LT=6.5 (computed post-scan).
        # Store raw phi_rms_mean; Q_i column filled after scan.
        Q_i_proxy = phi_rms_mean  # filled below after normalisation

        w_drift = float(abs(weight_rms_arr[-1] - weight_rms_arr[0])
                        / (weight_rms_arr[0] + 1e-30))

        print(f"  phi_rms (tail)    = {phi_rms_mean:.4e}")
        print(f"  phi_zonal (tail)  = {phi_zonal_mean:.4e}")
        print(f"  zonal_ratio       = {zonal_ratio:.3f}")
        print(f"  weight_drift      = {w_drift:.3e}  (should be ~0)")

        scan_results.append({
            "R_over_LT":      rlt,
            "phi_rms":        float(phi_rms_mean),
            "phi_zonal_rms":  float(phi_zonal_mean),
            "zonal_ratio":    float(zonal_ratio),
            "weight_drift":   float(w_drift),
            "status_raw":     "ok",
        })

    except Exception as ex:
        print(f"  ERROR: {ex}")
        scan_results.append({
            "R_over_LT":      rlt,
            "phi_rms":        float("nan"),
            "phi_zonal_rms":  float("nan"),
            "zonal_ratio":    float("nan"),
            "weight_drift":   float("nan"),
            "status_raw":     f"error: {ex}",
        })

# ── Post-process: compute Q_i (normalised phi_rms) ─────────────────────────
# Use phi_rms at R/LT=6.5 as normalisation baseline (turbulent reference).
ref_rlt = 6.5
ref_entries = [r for r in scan_results if r["R_over_LT"] == ref_rlt and r["status_raw"] == "ok"]
if ref_entries:
    phi_ref = ref_entries[0]["phi_rms"]
else:
    vals = [r["phi_rms"] for r in scan_results if r["status_raw"] == "ok" and not np.isnan(r["phi_rms"])]
    phi_ref = max(vals) if vals else 1.0
if phi_ref < 1e-30:
    phi_ref = 1.0

ZONAL_THRESH = 0.5   # zonal_ratio > 0.5 → suppressed

final_scan = []
threshold_rlt = None

for r in scan_results:
    rlt = r["R_over_LT"]
    ok  = r["status_raw"] == "ok"
    Q_i = r["phi_rms"] / phi_ref if ok else float("nan")

    zonal_ratio = r["zonal_ratio"]
    if ok and not np.isnan(zonal_ratio):
        status = "suppressed" if zonal_ratio > ZONAL_THRESH else "active"
    else:
        status = "error"

    # Find threshold: first R/LT where status flips to active
    if status == "active" and threshold_rlt is None:
        threshold_rlt = rlt

    entry = {
        "R_over_LT":     rlt,
        "Q_i":           round(Q_i, 5) if not np.isnan(Q_i) else None,
        "phi_rms":       round(r["phi_rms"], 6) if not np.isnan(r["phi_rms"]) else None,
        "phi_zonal_rms": round(r["phi_zonal_rms"], 6) if not np.isnan(r["phi_zonal_rms"]) else None,
        "zonal_ratio":   round(zonal_ratio, 4) if not np.isnan(zonal_ratio) else None,
        "status":        status,
    }
    final_scan.append(entry)

output = {
    "scan":            final_scan,
    "threshold_found": threshold_rlt is not None,
    "threshold_RLT":   threshold_rlt,
    "notes":           "Q_i is phi_rms normalised to phi_rms at R/LT=6.5 (proxy for heat flux)",
}

with open(RESULTS_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to {RESULTS_FILE}")

# ── Summary table ──────────────────────────────────────────────────────────
print()
print("=" * 72)
print("DIMITS SHIFT FULL-F BENCHMARK RESULTS")
print("=" * 72)
print(f"{'R/LT':>6}  {'Q_i':>8}  {'phi_rms':>10}  {'zonal_ratio':>11}  status")
print("-" * 72)

for r in final_scan:
    rlt  = r["R_over_LT"]
    qi   = f"{r['Q_i']:.4f}"  if r["Q_i"]          is not None else "   NaN  "
    prms = f"{r['phi_rms']:.4e}" if r["phi_rms"]    is not None else "   NaN    "
    zr   = f"{r['zonal_ratio']:.3f}" if r["zonal_ratio"] is not None else "  NaN "
    st   = r["status"]
    flag = "✅ SUPPRESSED" if st == "suppressed" else ("ACTIVE" if st == "active" else "⚠ ERROR")
    print(f"{rlt:>6.1f}  {qi:>8}  {prms:>10}  {zr:>11}  {flag}")

print("=" * 72)

if threshold_rlt is not None:
    print(f"\nMeasured Dimits threshold (zonal_ratio < {ZONAL_THRESH}): R/LT = {threshold_rlt:.1f}")
    print(f"Reference (Dimits 2000):   R/LT ≈ 6.0")
    passed = 4.5 <= threshold_rlt <= 7.5
    print(f"\nResult: {'PASS ✅' if passed else 'FAIL ✗'}")
    print(f"  (pass criterion: threshold between R/LT=4.5 and 7.5)")
else:
    ok = [r for r in final_scan if r["status"] in ("suppressed", "active")]
    if all(r["status"] == "suppressed" for r in ok):
        print("\nAll R/LT suppressed — try higher R/LT values")
        print("Result: INCONCLUSIVE")
    else:
        print("\nNo clean threshold found")
        print("Result: FAIL ✗")

print()
