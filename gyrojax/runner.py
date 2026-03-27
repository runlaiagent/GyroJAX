#!/usr/bin/env python3
"""
GyroJAX runner — run a simulation from a TOML input file.

Usage:
    python -m gyrojax.runner inputs/cbc.toml
    python -m gyrojax.runner inputs/cbc.toml --verbose
    python -m gyrojax.runner inputs/cbc.toml --dry-run   # just print the config
    python -m gyrojax.runner inputs/cbc.toml --n-steps 5 # override n_steps
"""

import argparse
import json
from pathlib import Path

from gyrojax.input import load_config


def main():
    parser = argparse.ArgumentParser(description="GyroJAX simulation runner")
    parser.add_argument("input", help="TOML input file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override n_steps from TOML (useful for quick tests)")
    args = parser.parse_args()

    cfg, raw = load_config(args.input)

    # Apply CLI overrides
    if args.n_steps is not None:
        from dataclasses import replace
        cfg = replace(cfg, n_steps=args.n_steps)

    if args.dry_run:
        print("=== GyroJAX Config (dry run) ===")
        print(cfg)
        return

    # Run
    import jax
    key = jax.random.PRNGKey(args.seed)
    method = raw.get("run", {}).get("method", "deltaf").lower()

    if method == "deltaf":
        from gyrojax.simulation_fa import run_simulation_fa
        diags, state, phi, geom = run_simulation_fa(cfg, key, verbose=args.verbose)
    else:
        from gyrojax.simulation_fullf import run_simulation_fullf
        diags, state, phi, geom = run_simulation_fullf(cfg, key=key, verbose=args.verbose)

    # Save results
    out_cfg = raw.get("output", {})
    out_dir = Path(out_cfg.get("output_dir", "results"))
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.input).stem

    results = {
        "input_file": str(args.input),
        "method": method,
        "n_steps": len(diags),
        "phi_max": [float(d.phi_max) for d in diags],
        "phi_rms": [float(d.phi_rms) for d in diags],
    }
    if diags and hasattr(diags[0], "weight_rms"):
        results["weight_rms"] = [float(d.weight_rms) for d in diags]
    if diags and hasattr(diags[0], "n_rms"):
        results["n_rms"] = [float(d.n_rms) for d in diags]

    out_path = out_dir / f"{stem}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Optionally save phi field
    if out_cfg.get("save_phi", False) and phi is not None:
        import numpy as np
        phi_path = out_dir / f"{stem}_phi.npz"
        np.savez(phi_path, phi=phi)
        print(f"Phi field saved to {phi_path}")


if __name__ == "__main__":
    main()
