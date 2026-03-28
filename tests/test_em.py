"""Tests for electromagnetic (Ampere) solver."""
import pytest
import jax
import jax.numpy as jnp
from gyrojax.fields.ampere_fa import solve_ampere_fa
from gyrojax.geometry.field_aligned import build_field_aligned_geometry

@pytest.fixture(scope="module")
def geom():
    return build_field_aligned_geometry(Npsi=8, Ntheta=16, Nalpha=16,
                                         R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)

def test_beta_zero_returns_zeros(geom):
    j = jnp.ones((8, 16, 16), dtype=jnp.float32)
    A = solve_ampere_fa(j, geom, beta=0.0)
    assert jnp.all(A == 0.0)

def test_ampere_zero_jpar(geom):
    j = jnp.zeros((8, 16, 16), dtype=jnp.float32)
    A = solve_ampere_fa(j, geom, beta=0.01)
    assert float(jnp.max(jnp.abs(A))) < 1e-10

def test_ampere_nonzero(geom):
    # Uniform j∥ > 0 → A_par should integrate to zero (k=0 zeroed), but have nonzero structure
    j = jnp.ones((8, 16, 16), dtype=jnp.float32) * 0.0
    # Use sinusoidal j to excite k≠0 modes
    psi = jnp.linspace(0, 1, 8)
    j = jnp.sin(2 * jnp.pi * psi)[:, None, None] * jnp.ones((8, 16, 16))
    j = j.astype(jnp.float32)
    A = solve_ampere_fa(j, geom, beta=0.01)
    assert float(jnp.max(jnp.abs(A))) > 1e-8, "A_par should be nonzero for nonzero j∥"

def test_ampere_scales_with_beta(geom):
    j = jnp.sin(jnp.linspace(0, 2*jnp.pi, 8))[:, None, None] * jnp.ones((8,16,16))
    j = j.astype(jnp.float32)
    A1 = solve_ampere_fa(j, geom, beta=0.01)
    A2 = solve_ampere_fa(j, geom, beta=0.02)
    # A scales linearly with beta
    ratio = float(jnp.max(jnp.abs(A2))) / (float(jnp.max(jnp.abs(A1))) + 1e-30)
    assert abs(ratio - 2.0) < 0.1, f"Expected ratio~2, got {ratio:.3f}"

def test_scatter_jpar_shape():
    from gyrojax.fields.ampere_fa import scatter_jpar_to_grid
    from gyrojax.particles.guiding_center import GCState
    import jax
    key = jax.random.PRNGKey(0)
    geom_ = build_field_aligned_geometry(Npsi=8, Ntheta=16, Nalpha=16,
                                          R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)
    N = 1000
    r   = jax.random.uniform(key, (N,), minval=0.05, maxval=0.31)
    th  = jax.random.uniform(key, (N,), minval=0.0, maxval=6.28)
    ze  = jax.random.uniform(key, (N,), minval=0.0, maxval=6.28)
    vp  = jax.random.normal(key, (N,))
    mu  = jax.random.uniform(key, (N,), minval=0.0, maxval=1.0)
    w   = jax.random.normal(key, (N,))
    state = GCState(r=r, theta=th, zeta=ze, vpar=vp, mu=mu, weight=w)
    grid_shape = (8, 16, 16)
    j = scatter_jpar_to_grid(state, geom_, grid_shape)
    assert j.shape == grid_shape
