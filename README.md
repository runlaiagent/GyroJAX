# GyroJAX

A gyrokinetic delta-f PIC code for tokamak/stellarator simulation, written in JAX.

## Features

- **Pure functional**: NamedTuples and frozen dataclasses as JAX pytrees
- **Multi-GPU ready**: particle arrays sharded via `jax.sharding.PositionalSharding`
- **Fully JIT-able**: `jax.vmap` + `jax.lax.scan`, no Python loops in hot paths
- **Phase space**: guiding center `(R, Z, φ_tor, v_∥, μ)` + delta-f weight `w`
- **Electrostatic s-α geometry**

## Structure

```
gyrojax/
├── geometry/salpha.py        # s-α magnetic geometry
├── particles/guiding_center.py  # GC state + RK4 pusher
├── fields/poisson.py         # GK Poisson solver (FFT+tridiagonal)
├── deltaf/weights.py         # delta-f weight evolution
├── interpolation/scatter_gather.py  # gyroaveraged scatter/gather
├── sharding.py               # multi-GPU utilities
├── simulation.py             # main time loop (lax.scan)
└── utils.py                  # constants, normalization
```

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v
python benchmarks/cyclone_base_case.py
```

## Cyclone Base Case

Target: ITG linear growth rate γ ≈ 0.17 vti/R0 at kyρi ≈ 0.3.

Parameters: R0/LT = 6.9, R0/Ln = 2.2, q = 1.4, s = 0.78, ε = 0.18, Ti/Te = 1.

## References

- Dimits et al., Phys. Plasmas 7, 969 (2000) — Cyclone Base Case
- Brizard & Hahm, Rev. Mod. Phys. 79, 421 (2007) — Gyrokinetic theory
- Lee, Phys. Fluids 26, 556 (1983) — Gyrokinetic PIC
