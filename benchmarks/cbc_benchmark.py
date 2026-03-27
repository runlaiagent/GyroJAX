"""
CBC (Cyclone Base Case) benchmark — field-aligned Phase 2a.

Wrapper around cyclone_base_case_fa.run_cbc_fa that provides a stable
command-line entry-point and importable run_cbc_benchmark() function.

Target: γ ≈ 0.169 vti/R0  (GENE/GX/GTC consensus), error < 5%.

Usage:
    python benchmarks/cbc_benchmark.py           # full mode
    python benchmarks/cbc_benchmark.py --quick   # quick mode (100k particles, 400 steps)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.cyclone_base_case_fa import run_cbc_fa


# Reference
CONSENSUS_GAMMA = 0.169
TARGET_GAMMA    = 0.170


def run_cbc_benchmark(quick: bool = False) -> dict:
    """
    Run the CBC peak-ITG benchmark.

    Parameters
    ----------
    quick : bool
        If True, use the light 100k-particle / 400-step configuration.
        If False, use the full 1M-particle / 800-step configuration.

    Returns
    -------
    dict with keys:
        gamma_measured, gamma_target, rel_err, rel_err_consensus,
        phi_max, phi_rms, t, step_start, step_end
    """
    return run_cbc_fa(quick=quick)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GyroJAX CBC benchmark (Phase 2a)")
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 100k particles, 400 steps')
    args = parser.parse_args()
    results = run_cbc_benchmark(quick=args.quick)
    gamma = results['gamma_measured']
    err   = results['rel_err_consensus']
    print(f"\n[cbc_benchmark] γ = {gamma:.4f} vti/R0  "
          f"(error vs consensus: {err:.1%})")
    if err < 0.05:
        print("[cbc_benchmark] ✅ PASS: error < 5%")
    else:
        print(f"[cbc_benchmark] ❌ FAIL: error {err:.1%} ≥ 5%")
        sys.exit(1)
