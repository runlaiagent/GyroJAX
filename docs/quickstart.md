# Quick Start

## Requirements

- Python 3.11+
- JAX 0.4.x+ (with CUDA support for GPU)
- NVIDIA GPU (8GB+ VRAM recommended for 500k particles)
- NumPy, SciPy, Matplotlib

## Installation

```bash
git clone https://github.com/runlaiagent/GyroJAX.git
cd GyroJAX
pip install -e .
```

For GPU support, install JAX with CUDA:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## First Run: ITG Simulation (CBC Parameters)

The simplest simulation uses the Cyclone Base Case (CBC) parameters for ITG turbulence:

```python
import jax
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

cfg = SimConfigFA(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=50_000,
    n_steps=200,
    dt=0.05,
    R0_over_LT=6.9,   # CBC parameters
)
diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(42))
print(f"Final phi_max: {diags[-1].phi_max:.3e}")
```

This runs ~200 steps at the CBC operating point (R/LT = 6.9, above the Dimits threshold of ~6.0), so you should observe growing ITG fluctuations.

### Plotting Results

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp

phi_rms = jnp.array([d.phi_rms for d in diags])
plt.semilogy(phi_rms)
plt.xlabel("Step")
plt.ylabel("φ_rms")
plt.title("ITG linear growth (CBC)")
plt.show()
```

## KBM (Electromagnetic) Simulation

To enable electromagnetic effects with finite plasma β:

```python
cfg = SimConfigFA(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=100_000,
    n_steps=300,
    dt=0.05,
    R0_over_LT=6.9,
    beta=0.01,              # plasma beta (electromagnetic coupling)
    fused_rk4=True,         # 3.78× speedup
)
diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(0))
```

!!! note "β_crit for KBM"
    KBM onset occurs around β ≈ 0.010–0.012 for CBC-like parameters.
    Below β_crit, ITG dominates; above it, KBM replaces ITG as the fastest-growing mode.

## TEM Simulation (Drift-Kinetic Electrons)

To simulate trapped electron modes, enable the drift-kinetic electron model:

```python
cfg = SimConfigFA(
    Npsi=16, Ntheta=32, Nalpha=32,
    N_particles=100_000,
    n_steps=400,
    dt=0.02,
    R0_over_LT=3.0,          # low ion gradient (below ITG threshold)
    R0_over_LTe=9.0,          # strong electron temperature gradient → TEM
    R0_over_Ln=2.2,
    electron_model='drift_kinetic',
    subcycles_e=10,            # subcycle electron push for stability
)
diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(1))
```

!!! tip "TEM vs ITG"
    TEM is driven by the electron temperature gradient (R₀/LTe). Use `R0_over_LTe` ≥ 9.0
    to see positive growth rates. At R₀/LTe = 5.0 the mode is stable.

## Noise Control Options

For long nonlinear runs, consider enabling noise reduction:

```python
cfg = SimConfigFA(
    ...
    use_pullback=True,          # periodic f₀ pullback (controls weight growth)
    pullback_interval=50,       # pullback every 50 steps
    use_weight_spread=True,     # GTC-style weight spreading
    weight_spread_interval=10,
    zonal_preserving_spread=True,
    semi_implicit_weights=True, # CN weight update (unconditionally stable)
)
```

## Running Benchmarks

```bash
pytest tests/ -q
```

Or run a specific benchmark:

```bash
python benchmarks/dimits_shift.py
python benchmarks/rosenbluth_hinton.py
```
