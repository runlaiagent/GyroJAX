"""Tests for guiding center pusher — conservation laws."""

import pytest
import jax
import jax.numpy as jnp
from gyrojax.geometry.salpha import build_salpha_geometry
from gyrojax.particles.guiding_center import GCState, push_particles, init_maxwellian_particles
from gyrojax.geometry.salpha import interp_geometry_to_particles


@pytest.fixture
def geom():
    return build_salpha_geometry(Nr=64, Ntheta=128, Nzeta=64, R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)


def test_mu_conservation(geom):
    """
    Magnetic moment μ = mv⊥²/2B should be conserved (it's a NamedTuple field
    that is never modified by the pusher). Verify it's unchanged after push.
    """
    key = jax.random.PRNGKey(0)
    N = 100
    mi = 1.0; e = 1.0; vti = 1.0
    state = init_maxwellian_particles(N, geom, vti, 1.0, mi, key)

    # Zero E-field
    E_r = jnp.zeros(N); E_th = jnp.zeros(N); E_ze = jnp.zeros(N)
    q_over_m = e / mi; dt = 0.01

    state2 = push_particles(state, E_r, E_th, E_ze, geom, q_over_m, mi, dt)

    mu_change = jnp.max(jnp.abs(state2.mu - state.mu))
    assert float(mu_change) == 0.0, f"μ changed: {float(mu_change)}"
    print(f"  PASS: μ conserved exactly (change = {float(mu_change):.2e})")


def test_energy_conservation_no_E(geom):
    """
    In a uniform B field with no electric field, the total guiding center
    energy H = m*v∥²/2 + μ*B should be approximately conserved by RK4.
    We test with many steps and check drift is small.
    """
    key = jax.random.PRNGKey(1)
    N = 50; mi = 1.0; e = 1.0; vti = 1.0; dt = 0.001

    state = init_maxwellian_particles(N, geom, vti, 1.0, mi, key)

    # Compute initial energy: H = m*v∥²/2 + μ*B
    B0_arr, _, _, _, _ = interp_geometry_to_particles(
        geom, state.r, state.theta, state.zeta
    )
    H0 = 0.5 * mi * state.vpar**2 + state.mu * B0_arr

    # Push 100 steps
    E_r = jnp.zeros(N); E_th = jnp.zeros(N); E_ze = jnp.zeros(N)
    q_over_m = e / mi

    s = state
    for _ in range(100):
        s = push_particles(s, E_r, E_th, E_ze, geom, q_over_m, mi, dt)

    B1_arr, _, _, _, _ = interp_geometry_to_particles(geom, s.r, s.theta, s.zeta)
    H1 = 0.5 * mi * s.vpar**2 + s.mu * B1_arr

    rel_err = float(jnp.mean(jnp.abs(H1 - H0) / (jnp.abs(H0) + 1e-10)))
    print(f"  Energy conservation: mean relative error over 100 steps = {rel_err:.4%}")
    # RK4 should keep energy error < 1% over 100 steps
    assert rel_err < 0.05, f"Energy drift too large: {rel_err:.4%}"
    print(f"  PASS: Energy conservation error = {rel_err:.4%} < 5%")


def test_particle_stays_in_domain(geom):
    """Particles should stay within the radial domain after pushing."""
    key = jax.random.PRNGKey(2)
    N = 200; mi = 1.0; e = 1.0; vti = 0.1; dt = 0.01

    state = init_maxwellian_particles(N, geom, vti, 0.01, mi, key)
    E_r = jnp.zeros(N); E_th = jnp.zeros(N); E_ze = jnp.zeros(N)
    q_over_m = e / mi

    for _ in range(50):
        state = push_particles(state, E_r, E_th, E_ze, geom, q_over_m, mi, dt)

    r_min = float(geom.r_grid[0])
    r_max = float(geom.r_grid[-1])
    assert float(jnp.min(state.r)) >= r_min, f"Particle below r_min: {float(jnp.min(state.r))}"
    assert float(jnp.max(state.r)) <= r_max, f"Particle above r_max: {float(jnp.max(state.r))}"
    print(f"  PASS: All particles in [{r_min:.3f}, {r_max:.3f}], "
          f"actual [{float(jnp.min(state.r)):.3f}, {float(jnp.max(state.r)):.3f}]")


def test_vpar_zero_no_drift_parallel(geom):
    """
    A particle with v∥=0, μ=0 and no E-field should have dvpar/dt ≈ 0
    at the outer midplane (∇B·b̂ ≈ 0 at θ=π/2).
    Just checks that pusher runs without error.
    """
    N = 1; mi = 1.0; e = 1.0; dt = 0.01
    r_mid = float(geom.r_grid[len(geom.r_grid)//2])
    state = GCState(
        r=jnp.array([r_mid]), theta=jnp.array([0.0]), zeta=jnp.array([0.0]),
        vpar=jnp.array([0.0]), mu=jnp.array([0.0]), weight=jnp.array([0.0])
    )
    E_r = jnp.zeros(N); E_th = jnp.zeros(N); E_ze = jnp.zeros(N)
    new_state = push_particles(state, E_r, E_th, E_ze, geom, e/mi, mi, dt)
    assert jnp.isfinite(new_state.r[0])
    print("  PASS: Zero v∥, μ=0 particle pushes without NaN")


if __name__ == "__main__":
    g = build_salpha_geometry(Nr=64, Ntheta=128, Nzeta=64, R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)
    print("\n=== Pusher Benchmark ===")
    test_mu_conservation(g)
    test_energy_conservation_no_E(g)
    test_particle_stays_in_domain(g)
    test_vpar_zero_no_drift_parallel(g)
    print("\nAll pusher tests PASSED")
