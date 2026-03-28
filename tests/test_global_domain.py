"""Tests for global radial domain with shape-controlled profiles and Krook buffers."""
import pytest
import jax
import jax.numpy as jnp


def test_lt_profile_flat():
    from gyrojax.geometry.profiles import make_LT_profile
    psi = jnp.linspace(0.0, 0.18, 16)
    profile = make_LT_profile(psi, 6.9, "flat")
    assert profile.shape == (16,)
    assert jnp.allclose(profile, 6.9)


def test_lt_profile_gaussian():
    from gyrojax.geometry.profiles import make_LT_profile
    psi = jnp.linspace(0.0, 0.18, 16)
    profile = make_LT_profile(psi, 6.9, "gaussian")
    assert profile.shape == (16,)
    assert float(profile[8]) == pytest.approx(6.9, rel=0.1)  # peak at center
    assert float(profile[0]) < float(profile[8])  # falls off at boundaries


def test_lt_profile_tanh():
    from gyrojax.geometry.profiles import make_LT_profile
    psi = jnp.linspace(0.0, 0.18, 32)
    profile = make_LT_profile(psi, 6.9, "tanh")
    assert profile.shape == (32,)
    # tanh profile: high at inner (psi[0]), low at outer (psi[-1])
    assert float(profile[0]) > float(profile[-1])


def test_krook_mask():
    from gyrojax.geometry.profiles import make_krook_mask
    psi = jnp.linspace(0.0, 0.18, 32)
    nu = make_krook_mask(psi, buffer_width=0.1, buffer_rate=1.0)
    assert nu.shape == (32,)
    assert float(nu[16]) == pytest.approx(0.0, abs=1e-5)  # zero in bulk
    assert float(nu[0]) > 0.5  # nonzero at inner boundary
    assert float(nu[-1]) > 0.5  # nonzero at outer boundary


def test_krook_mask_zero_rate():
    from gyrojax.geometry.profiles import make_krook_mask
    psi = jnp.linspace(0.0, 0.18, 32)
    nu = make_krook_mask(psi, buffer_width=0.1, buffer_rate=0.0)
    assert jnp.allclose(nu, 0.0)


def test_apply_radial_profile_to_particles():
    from gyrojax.geometry.profiles import apply_radial_profile_to_particles
    psi_grid = jnp.linspace(0.0, 0.18, 16)
    profile = psi_grid * 10.0  # linear profile
    r = jnp.array([0.0, 0.09, 0.18])
    result = apply_radial_profile_to_particles(r, psi_grid, profile)
    assert result.shape == (3,)
    assert float(result[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(result[1]) == pytest.approx(0.9, rel=0.01)


def test_global_domain_run():
    """Global domain simulation runs without error."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
    cfg = SimConfigFA(
        Npsi=8, Ntheta=16, Nalpha=16,
        N_particles=500, n_steps=5, dt=0.05,
        R0_over_LT=6.9,
        global_domain=True,
        LT_profile="gaussian",
        krook_buffer_width=0.15,
        krook_buffer_rate=1.0,
    )
    diags, state, phi, _ = run_simulation_fa(cfg, key=jax.random.PRNGKey(0), verbose=False)
    assert jnp.isfinite(phi).all(), "phi should be finite in global domain run"
    assert jnp.isfinite(state.weight).all(), "weights should be finite"


def test_global_domain_flat_vs_fluxtube():
    """Global domain with flat profile and zero Krook closely matches flux-tube result."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    base = dict(Npsi=8, Ntheta=16, Nalpha=16, N_particles=500, n_steps=5,
                dt=0.05, R0_over_LT=6.9, krook_buffer_rate=0.0)

    cfg_ft = SimConfigFA(**base, global_domain=False, use_global=False)
    cfg_gl = SimConfigFA(**base, global_domain=True, LT_profile="flat")

    key = jax.random.PRNGKey(0)
    _, _, phi_ft, _ = run_simulation_fa(cfg_ft, key=key, verbose=False)
    _, _, phi_gl, _ = run_simulation_fa(cfg_gl, key=key, verbose=False)

    # With flat profile and zero Krook, results should be close
    diff = float(jnp.max(jnp.abs(phi_ft - phi_gl)))
    assert diff < 1e-2, f"Global flat profile should approximately match flux-tube, diff={diff}"


def test_global_domain_gaussian_differs_from_flat():
    """Gaussian LT profile produces different results than flat profile."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    base = dict(Npsi=8, Ntheta=16, Nalpha=16, N_particles=1000, n_steps=10,
                dt=0.05, R0_over_LT=6.9, krook_buffer_rate=0.0)

    cfg_flat = SimConfigFA(**base, global_domain=True, LT_profile="flat")
    cfg_gauss = SimConfigFA(**base, global_domain=True, LT_profile="gaussian")

    key = jax.random.PRNGKey(42)
    _, _, phi_flat, _ = run_simulation_fa(cfg_flat, key=key, verbose=False)
    _, _, phi_gauss, _ = run_simulation_fa(cfg_gauss, key=key, verbose=False)

    diff = float(jnp.max(jnp.abs(phi_flat - phi_gauss)))
    # gaussian profile should produce measurably different phi
    assert diff > 0.0, "Gaussian and flat profiles should produce different results"
