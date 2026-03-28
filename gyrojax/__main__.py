"""CLI entry point: python -m gyrojax run input.toml"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(prog="gyrojax", description="GyroJAX gyrokinetic PIC simulation")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run simulation from TOML input file")
    run_parser.add_argument("input", help="Path to .toml input file")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # template command
    tmpl_parser = subparsers.add_parser("template", help="Write a default TOML template")
    tmpl_parser.add_argument("output", nargs="?", default="gyrojax_input.toml", help="Output path")

    args = parser.parse_args()

    if args.command == "run":
        import jax
        from gyrojax.io.input_file import load_config
        from gyrojax.simulation_fa import run_simulation_fa
        import time

        print(f"Loading config from {args.input}")
        cfg = load_config(args.input)
        print(f"  Grid: {cfg.Npsi}×{cfg.Ntheta}×{cfg.Nalpha}, {cfg.N_particles} particles, {cfg.n_steps} steps")
        print(f"  Physics: R0/LT={cfg.R0_over_LT}, beta={cfg.beta}")
        if cfg.output_file:
            print(f"  Output: {cfg.output_file}")

        key = jax.random.PRNGKey(args.seed)
        t0 = time.time()
        diags, state, phi, aux = run_simulation_fa(cfg, key=key, verbose=args.verbose)
        elapsed = time.time() - t0

        import numpy as np
        phi_vals = np.array([float(d.phi_max) for d in diags])
        print(f"\nDone in {elapsed:.1f}s ({cfg.n_steps/elapsed:.1f} steps/sec)")
        print(f"Final phi_max: {phi_vals[-1]:.4e}")
        if len(phi_vals) > 10:
            gamma = float(np.log(phi_vals[-1]/phi_vals[len(phi_vals)//2]) / ((len(phi_vals)//2) * cfg.dt))
            print(f"Approx growth rate: {gamma:.4f}")

        if cfg.output_file:
            print(f"Results saved to {cfg.output_file}")

    elif args.command == "template":
        from gyrojax.io.input_file import save_config_template
        save_config_template(args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
