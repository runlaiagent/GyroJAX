# Contributing

Contributions to GyroJAX are welcome! This guide covers how to set up the development environment, run tests, and submit pull requests.

## Dev Setup

```bash
git clone https://github.com/runlaiagent/GyroJAX.git
cd GyroJAX
pip install -e ".[dev]"
```

Required dependencies:
- JAX 0.4.x+ (with CUDA for GPU)
- NumPy, SciPy, Matplotlib
- pytest (for tests)
- mkdocs-material (for docs)

## Running Tests

```bash
pytest tests/ -q
```

Run a specific test:

```bash
pytest tests/test_simulation_fa.py -v
pytest tests/test_poisson.py -v
```

Run with GPU (default if available):

```bash
JAX_PLATFORM_NAME=gpu pytest tests/ -q
```

Run CPU-only (for CI/testing without GPU):

```bash
JAX_PLATFORM_NAME=cpu pytest tests/ -q
```

## Adding Benchmarks

Benchmarks live in `benchmarks/`. A benchmark should:

1. Define a `SimConfigFA` with the reference parameters
2. Run the simulation using `run_simulation_fa`
3. Extract the relevant diagnostic (growth rate, threshold, etc.)
4. Compare against the published reference value
5. Print a PASS/FAIL summary

Example structure:

```python
# benchmarks/my_benchmark.py
from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
import jax

def run():
    cfg = SimConfigFA(
        N_particles=50_000,
        n_steps=300,
        R0_over_LT=6.9,
        # ...
    )
    diags, state, phi, geom = run_simulation_fa(cfg, key=jax.random.PRNGKey(0))

    # Extract growth rate from log(phi_rms)
    import jax.numpy as jnp
    phi_rms = jnp.array([d.phi_rms for d in diags])
    # ... fit growth rate ...

    ref_value = 0.169  # GENE/GX reference
    tol = 0.05
    assert abs(gamma - ref_value) / ref_value < tol, f"Growth rate {gamma:.3f} outside tolerance"
    print(f"PASS: γ = {gamma:.3f} (ref: {ref_value})")

if __name__ == "__main__":
    run()
```

## Code Style

- **Functional style**: prefer pure functions, avoid mutable state
- **JAX idioms**: use `jax.lax.scan`, `jax.vmap`, `jax.jit` appropriately
- **Type hints**: use for all public functions
- **Docstrings**: Google style, include parameter descriptions
- **No raw loops over particle arrays**: use JAX vectorized ops

## Physics Changes

When changing physics (push equations, Poisson solver, gyroaveraging):

1. Verify against at least one benchmark in `benchmarks/`
2. Add a unit test in `tests/` checking the modified component
3. Update `docs/physics.md` if the model changes
4. Note the change in the PR description with the relevant equations

## PR Guidelines

1. **Fork** the repository and create a feature branch: `git checkout -b feature/my-feature`
2. **Write tests** for any new functionality
3. **Run the full test suite** before submitting: `pytest tests/ -q`
4. **Keep PRs focused** — one feature or fix per PR
5. **Describe the physics** in the PR description if adding/changing a physics model
6. **Reference issues** if the PR closes one: `Closes #42`

PRs are reviewed for:
- Correctness (physics and numerics)
- JAX/JIT compatibility (no Python-side loops over dynamic sizes)
- Test coverage
- Documentation

## Building Docs Locally

```bash
pip install mkdocs-material
mkdocs serve
```

Then open http://localhost:8000 in your browser.

## Contact

Open an issue on GitHub for questions, bugs, or feature requests.
