# GyroJAX

**GyroJAX** is a GPU-accelerated gyrokinetic particle-in-cell (PIC) simulation code for plasma turbulence, written entirely in [JAX](https://github.com/google/jax). It simulates ion temperature gradient (ITG), kinetic ballooning mode (KBM), and trapped electron mode (TEM) instabilities in tokamak flux-tube geometry — fully JIT-compiled, benchmarked against GENE/GX, and designed for research-grade plasma physics.

---

## Key Features

- **δf PIC gyrokinetic simulation** in field-aligned flux-tube geometry (ψ, θ, α)
- **Electrostatic + electromagnetic** (β > 0) modes via FFT Poisson + Ampere solver
- **Drift-kinetic electron model** for TEM instability simulation
- **Fused RK4 push+weights integrator** — 3.78× GPU speedup over split integration
- **Physics fixes**: symmetric gyroaveraging scatter, radially-resolved g^αα Poisson
- **Stability controls**: absorbing wall BC, zonal-preserving weight spread, pullback transformation
- **float32 throughout** — fits 500k particles comfortably in 8GB VRAM
- **Multi-GPU capable** via JAX `pmap`

## Performance

| Configuration | Throughput |
|---|---|
| 100k–500k particles (RTX 3070 Ti Laptop, 8GB) | ~27 steps/sec |
| Fused RK4 vs split integration | **3.78× speedup** |
| Memory (500k particles, float32) | < 8GB VRAM |

## Quick Install

```bash
git clone https://github.com/runlaiagent/GyroJAX.git
cd GyroJAX
pip install -e .
```

Then run a basic ITG simulation:

```python
import jax
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

cfg = SimConfigFA(N_particles=50_000, n_steps=200, R0_over_LT=6.9)
diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(42))
print(f"Final phi_max: {diags[-1].phi_max:.3e}")
```

## Validated Benchmarks

GyroJAX has been validated against published reference results:

- **Dimits shift** threshold R/LT ≈ 6.0 ([Dimits et al. 2000](https://doi.org/10.1063/1.874242))
- **CBC linear growth rate** γ = 0.172 vti/R₀ (1.9% error vs GENE/GX)
- **KBM β_crit** ≈ 0.010–0.012 ([Pueschel et al. 2008](https://doi.org/10.1063/1.2996358))
- **TEM** γ > 0 at R₀/LTe = 9.0, stable at 5.0

→ [See full benchmark results](benchmarks.md)

---

## Links

- 📚 [Documentation](https://runlaiagent.github.io/GyroJAX)
- 🚀 [Quick Start](quickstart.md)
- 🔧 [Configuration Reference](config.md)
- 🧪 [Benchmarks](benchmarks.md)
- 🤝 [Contributing](contributing.md)
