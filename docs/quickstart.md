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

## Multi-GPU Runs

GyroJAX supports multi-GPU parallelism via JAX `pmap`. The particle array is sharded
across devices; the grid (φ, δn) is replicated and reduced via `AllReduce`.

```python
from gyrojax.parallel import run_simulation_pmap, get_device_count

print(f"Available devices: {get_device_count()}")

cfg = SimConfigFA(
    N_particles=200_000,  # will be split evenly across GPUs
    n_steps=500,
    ...
)

# Automatically uses pmap on multi-GPU, falls back to single-GPU
diags, state, phi, _ = run_simulation_pmap(cfg)
```

### Scaling

| GPUs | Particles | Expected Speedup |
|------|-----------|-----------------|
| 1    | 50k       | 1× (baseline)   |
| 2    | 100k      | ~1.8×           |
| 4    | 200k      | ~3.5×           |
| 8    | 400k      | ~6×             |

> **Note:** Multi-GPU pmap requires identical GPU models for best performance.
> Single-GPU fallback is automatic when only 1 device is detected.

### Implementation Details

- **Particle sharding**: `N_particles` rounded down to multiple of device count
- **Grid**: full (Npsi × Ntheta × Nalpha) replicated on each device
- **Scatter AllReduce**: `jax.lax.psum(delta_n, axis_name='devices')`
- **Poisson solve**: replicated (each device gets identical φ)
- **Memory**: each device holds `N/D` particles + 1 full grid copy

## Running from a TOML Input File

GyroJAX supports TOML input files for reproducible, scriptable runs.

### Generate a template

```bash
python -m gyrojax template myrun.toml
```

### Edit and run

```toml
# myrun.toml
[simulation]
N_particles = 100000
n_steps = 500
output_file = "results.h5"

[physics]
R0_over_LT = 6.9
beta = 0.01

[dtype]
velocity = "bfloat16"   # save memory
phi = "bfloat16"
```

```bash
python -m gyrojax run myrun.toml --verbose
```

### Load in Python

```python
from gyrojax.io.input_file import load_config
cfg = load_config("myrun.toml")
# cfg is a fully-configured SimConfigFA, including DtypeConfig
```

## Post-Processing

Load and analyze simulation output with `PostProcessor`:

```python
from gyrojax.io.postprocess import PostProcessor

pp = PostProcessor("run.h5")

# Print summary
print(pp.summary())
# {'n_steps': 500, 'growth_rate': 0.182, 'chi_i': 0.034, ...}

# Growth rate from linear phase
gamma = pp.growth_rate(t_start=50, t_end=200)
print(f"γ = {gamma:.4f}")

# Heat flux in saturation
chi = pp.heat_flux_chi(saturation_start=300)
print(f"χᵢ = {chi:.4f}")

# Plot diagnostics
pp.plot_growth(save="growth.png")
pp.plot_zonal(save="zonal.png")

# Raw arrays
t = pp.time_axis()
phi = pp.phi_max
weight = pp.weight_rms
```
