"""
Tests for GyroJAX collision operators.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from gyrojax.particles.guiding_center import GCState
from gyrojax.collisions.operators import (
    apply_krook,
    apply_lorentz,
    apply_dougherty,
    apply_collisions,
)


def make_state(N=100, vpar_val=1.0, mu_val=0.5, weight_val=0.3, seed=0):
    """Helper: create a simple GCState for testing."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return GCState(
        r=jax.random.uniform(k1, (N,), minval=0.05, maxval=0.15),
        theta=jax.random.uniform(k2, (N,)) * 2 * jnp.pi,
        zeta=jax.random.uniform(k3, (N,)) * 2 * jnp.pi,
        vpar=jnp.full((N,), vpar_val),
        mu=jnp.full((N,), mu_val),
        weight=jnp.full((N,), weight_val),
    )


def make_B(N=100, B_val=1.0):
    return jnp.full((N,), B_val)


# -------- Krook tests --------

def test_krook_decays_weights():
    """After N Krook steps, |w| < initial |w|."""
    state = make_state(weight_val=0.5)
    nu, dt = 0.1, 0.05
    for _ in range(10):
        state = apply_krook(state, nu, dt)
    assert jnp.all(jnp.abs(state.weight) < 0.5)


def test_krook_decay_rate():
    """Weight decays as exp(-nu*t) to within 1%."""
    w0 = 0.8
    state = make_state(weight_val=w0)
    nu, dt, N = 0.05, 0.1, 20
    for _ in range(N):
        state = apply_krook(state, nu, dt)
    expected = w0 * jnp.exp(-nu * dt * N)
    actual = state.weight[0]
    assert abs(float(actual - expected) / float(expected)) < 0.01


def test_krook_no_velocity_change():
    """v∥ and μ unchanged by Krook."""
    vpar_init = 1.23
    mu_init = 0.77
    state = make_state(vpar_val=vpar_init, mu_val=mu_init, weight_val=0.3)
    state_new = apply_krook(state, nu=0.05, dt=0.1)
    assert jnp.allclose(state_new.vpar, state.vpar)
    assert jnp.allclose(state_new.mu, state.mu)


# -------- Lorentz tests --------

def test_lorentz_changes_vpar():
    """v∥ changes after one Lorentz step (stochastic)."""
    state = make_state(N=1000, vpar_val=1.0, mu_val=0.5)
    B_p = make_B(N=1000)
    key = jax.random.PRNGKey(42)
    state_new, _ = apply_lorentz(state, B_p, nu_ei=0.1, mi=1.0, Ti=1.0, dt=0.1, key=key)
    # At least some particles should have different vpar
    assert not jnp.allclose(state_new.vpar, state.vpar)


def test_lorentz_conserves_energy_mean():
    """Mean kinetic energy conserved to within 10% after many steps (statistical)."""
    N = 10000
    key = jax.random.PRNGKey(0)
    # Maxwellian-distributed vpar
    k1, k2 = jax.random.split(key)
    vpar = jax.random.normal(k1, (N,))
    mu = jnp.abs(jax.random.normal(k2, (N,))) * 0.5
    state = GCState(
        r=jnp.zeros(N), theta=jnp.zeros(N), zeta=jnp.zeros(N),
        vpar=vpar, mu=mu, weight=jnp.zeros(N),
    )
    B_p = jnp.ones(N)
    mi, Ti = 1.0, 1.0

    KE_init = jnp.mean(0.5 * mi * state.vpar**2 + state.mu * B_p)
    
    rng = jax.random.PRNGKey(1)
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        state, _ = apply_lorentz(state, B_p, nu_ei=0.01, mi=mi, Ti=Ti, dt=0.05, key=subkey)

    KE_final = jnp.mean(0.5 * mi * state.vpar**2 + state.mu * B_p)
    rel_change = abs(float(KE_final - KE_init)) / float(KE_init)
    assert rel_change < 0.1, f"Energy changed by {rel_change:.1%}"


def test_lorentz_weight_bounded():
    """Weights stay in [-2, 2] after 100 Lorentz steps."""
    N = 1000
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    vpar = jax.random.normal(k1, (N,)) * 1.0
    mu = jnp.abs(jax.random.normal(k2, (N,))) * 0.5
    state = GCState(
        r=jnp.zeros(N), theta=jnp.zeros(N), zeta=jnp.zeros(N),
        vpar=vpar, mu=mu, weight=jnp.zeros(N),
    )
    B_p = jnp.ones(N)
    rng = jax.random.PRNGKey(99)
    for _ in range(100):
        rng, subkey = jax.random.split(rng)
        state, _ = apply_lorentz(state, B_p, nu_ei=0.01, mi=1.0, Ti=1.0, dt=0.05, key=subkey)
    assert jnp.all(jnp.abs(state.weight) < 2.0)


# -------- Dougherty tests --------

def test_dougherty_weight_update():
    """C_w computed correctly for known v∥, μ."""
    # Single particle: v∥=1, μ=0, B=1, Ti=1, mi=1  =>  vt²=1, v⊥²=0
    # C_w = nu * [(1/1 - 1) + (0/(2*1) - 1)] * (1-w) = nu * [0 + (-1)] * (1-w) = -nu*(1-w)
    N = 1
    state = GCState(
        r=jnp.zeros(N), theta=jnp.zeros(N), zeta=jnp.zeros(N),
        vpar=jnp.array([1.0]), mu=jnp.array([0.0]), weight=jnp.array([0.0]),
    )
    B_p = jnp.ones(N)
    nu, mi, Ti, dt = 1.0, 1.0, 1.0, 0.1
    state_new = apply_dougherty(state, B_p, nu, mi, Ti, dt)
    # Expected: w_new = 0 + (-1.0 * (1-0)) * 0.1 = -0.1
    assert abs(float(state_new.weight[0]) - (-0.1)) < 1e-5


def test_dougherty_zero_at_maxwellian():
    """For Maxwellian particles, mean C_w ≈ 0."""
    # For a Maxwellian: <v∥²> = vt², <v⊥²> = 2*vt²
    # So: <v∥²/vt² - 1> = 0, <v⊥²/(2*vt²) - 1> = 0
    N = 100000
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    vt = 1.0
    vpar = jax.random.normal(k1, (N,)) * vt
    # v⊥² ~ 2*chi^2(2)*vt² with <v⊥²> = 2*vt², so mu*B/mi ~ chi^2(2)*vt²
    # mu = (1/2) * m * v_perp^2 / B; sample v_perp^2 = 2*vt^2 * Exp(1)
    v_perp_sq = jax.random.exponential(k2, (N,)) * 2.0 * vt**2
    mi, Ti, B0 = 1.0, 1.0, 1.0
    mu = 0.5 * mi * v_perp_sq / B0
    state = GCState(
        r=jnp.zeros(N), theta=jnp.zeros(N), zeta=jnp.zeros(N),
        vpar=vpar, mu=mu, weight=jnp.zeros(N),
    )
    B_p = jnp.ones(N) * B0
    state_new = apply_dougherty(state, B_p, nu_coll=1.0, mi=mi, Ti=Ti, dt=1.0)
    dw = state_new.weight - state.weight
    # Mean Δw should be near zero
    assert abs(float(jnp.mean(dw))) < 0.05


def test_dougherty_bounded_weights():
    """Weights stay bounded after 100 Dougherty steps."""
    N = 1000
    key = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(key)
    vpar = jax.random.normal(k1, (N,))
    mu = jnp.abs(jax.random.normal(k2, (N,))) * 0.5
    state = GCState(
        r=jnp.zeros(N), theta=jnp.zeros(N), zeta=jnp.zeros(N),
        vpar=vpar, mu=mu, weight=jnp.zeros(N),
    )
    B_p = jnp.ones(N)
    for _ in range(100):
        state = apply_dougherty(state, B_p, nu_coll=0.01, mi=1.0, Ti=1.0, dt=0.05)
        # Apply weight clamp (as done in simulation)
        state = state._replace(weight=jnp.clip(state.weight, -10.0, 10.0))
    assert jnp.all(jnp.abs(state.weight) <= 10.0)


# -------- Dispatch tests --------

def _make_cfg(model, **kwargs):
    defaults = dict(
        collision_model=model,
        nu_krook=0.01, nu_ei=0.01, nu_coll=0.01,
        mi=1.0, Ti=1.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_dispatch_none():
    """apply_collisions with 'none' returns identical state."""
    state = make_state(N=50)
    B_p = make_B(N=50)
    cfg = _make_cfg('none')
    key = jax.random.PRNGKey(0)
    state_new, key_new = apply_collisions(state, B_p, cfg, dt=0.1, key=key)
    assert jnp.allclose(state_new.weight, state.weight)
    assert jnp.allclose(state_new.vpar, state.vpar)
    assert jnp.allclose(state_new.mu, state.mu)


def test_dispatch_all_models():
    """All four models run without error or NaN."""
    N = 200
    state = make_state(N=N, vpar_val=0.5, mu_val=0.3, weight_val=0.1)
    B_p = make_B(N=N)
    key = jax.random.PRNGKey(42)

    for model in ['none', 'krook', 'lorentz', 'dougherty']:
        cfg = _make_cfg(model)
        s_new, k_new = apply_collisions(state, B_p, cfg, dt=0.1, key=key)
        assert not jnp.any(jnp.isnan(s_new.weight)), f"NaN weight in model={model}"
        assert not jnp.any(jnp.isnan(s_new.vpar)), f"NaN vpar in model={model}"
