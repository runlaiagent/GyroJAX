"""
Tests for electromagnetic perturbations (δA∥) — Phase 5.

Tests:
1. beta=0 gives zero A_par
2. Finite beta gives nonzero A_par proportional to current
3. Full EM simulation runs without NaN for 10 steps
4. beta=0 EM simulation gives identical result to ES simulation
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyrojax.geometry.field_aligned import build_field_aligned_geometry
from gyrojax.fields.ampere_fa import scatter_jpar_to_grid, solve_ampere_fa


def _make_geom():
    return build_field_aligned_geometry(
        Npsi=8, Ntheta=16, Nalpha=16,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    )


def _make_state(N=1000, key=None):
    """Create a minimal GCState-like mock for testing."""
    from gyrojax.particles.guiding_center import GCState, init_maxwellian_particles
    geom = _make_geom()
    if key is None:
        key = jax.random.PRNGKey(0)
    state = init_maxwellian_particles(N, geom, vti=1.0, Ti=1.0, mi=1.0, key=key)
    # Set unit weights for clean tests
    state = state._replace(weight=jnp.ones(N, dtype=jnp.float32))
    return state, geom


def test_ampere_solver_zero_beta():
    """beta=0 gives zero A_par regardless of current."""
    geom = _make_geom()
    grid_shape = (8, 16, 16)
    # Some nonzero j_par_grid
    jpar_grid = jnp.ones(grid_shape, dtype=jnp.float32)
    A_par = solve_ampere_fa(jpar_grid, geom, beta=0.0)
    assert jnp.allclose(A_par, 0.0), f"Expected zero A_par for beta=0, got max={jnp.max(jnp.abs(A_par))}"


def test_ampere_solver_finite_beta():
    """Finite beta gives nonzero A_par proportional to current."""
    geom = _make_geom()
    grid_shape = (8, 16, 16)

    # Use a smooth j_par with known Fourier content
    Npsi, Ntheta, Nalpha = grid_shape
    alpha = jnp.linspace(0, 2*jnp.pi, Nalpha, endpoint=False)
    # Single sinusoidal mode in alpha
    jpar_1 = jnp.zeros(grid_shape).at[:, :, :].add(jnp.sin(alpha)[None, None, :])

    A1 = solve_ampere_fa(jpar_1, geom, beta=0.01)
    A2 = solve_ampere_fa(2 * jpar_1, geom, beta=0.01)

    # A should be nonzero
    assert jnp.max(jnp.abs(A1)) > 1e-8, "Expected nonzero A_par for finite beta"
    # A should scale linearly with j
    assert jnp.allclose(A2, 2 * A1, atol=1e-6), "A_par should scale linearly with j_par"

    # A should scale linearly with beta
    A3 = solve_ampere_fa(jpar_1, geom, beta=0.02)
    assert jnp.allclose(A3, 2 * A1, atol=1e-6), "A_par should scale linearly with beta"


def test_em_simulation_runs():
    """Full EM simulation runs without NaN for 10 steps."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    cfg = SimConfigFA(
        Npsi=8, Ntheta=16, Nalpha=16,
        N_particles=5_000,
        n_steps=10,
        dt=0.05,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        pert_amp=1e-3,
        beta=0.01,   # small but nonzero EM coupling
    )
    key = jax.random.PRNGKey(42)
    diags, state, phi, geom = run_simulation_fa(cfg, key=key, verbose=False)

    # Check no NaN
    assert not np.any(np.isnan(np.array(phi))), "phi contains NaN in EM simulation"
    assert not np.any(np.isnan(np.array(state.r))), "particle positions contain NaN in EM simulation"

    # Check diagnostics make sense
    phi_max_vals = np.array([float(d.phi_max) for d in diags])
    assert np.all(np.isfinite(phi_max_vals)), "phi_max diagnostics contain NaN/Inf"
    assert np.all(phi_max_vals >= 0), "phi_max should be non-negative"


def test_electrostatic_limit():
    """beta=0 EM simulation gives same result as ES simulation (beta default)."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    base_cfg = dict(
        Npsi=8, Ntheta=16, Nalpha=16,
        N_particles=2_000,
        n_steps=5,
        dt=0.05,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        pert_amp=1e-3,
    )
    cfg_es = SimConfigFA(**base_cfg)
    cfg_em_zero = SimConfigFA(**base_cfg, beta=0.0)

    key = jax.random.PRNGKey(99)
    diags_es, state_es, phi_es, _ = run_simulation_fa(cfg_es, key=key, verbose=False)
    diags_em, state_em, phi_em, _ = run_simulation_fa(cfg_em_zero, key=key, verbose=False)

    # Results must be near-identical (beta=0 → no EM coupling)
    # atol=1e-5: float32 accumulation over 5 steps through slightly different code paths
    assert jnp.allclose(phi_es, phi_em, atol=1e-5), \
        f"beta=0 EM result differs from ES: max diff = {float(jnp.max(jnp.abs(phi_es - phi_em)))}"
    assert jnp.allclose(state_es.r, state_em.r, atol=1e-5), \
        "Particle positions differ between beta=0 EM and ES runs"
