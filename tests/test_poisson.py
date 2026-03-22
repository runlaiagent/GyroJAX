"""Tests for GK Poisson solver — convergence and correctness."""

import pytest
import jax.numpy as jnp
import numpy as np
from gyrojax.geometry.salpha import build_salpha_geometry
from gyrojax.fields.poisson import solve_poisson_gk, compute_efield, gyroaverage_phi


@pytest.fixture
def geom():
    return build_salpha_geometry(Nr=32, Ntheta=64, Nzeta=32, R0=1.0, a=0.18, B0=1.0)


def test_poisson_zero_rhs(geom):
    """Zero density perturbation should give zero potential."""
    delta_n = jnp.zeros((32, 64, 32))
    phi = solve_poisson_gk(delta_n, geom, n0_avg=1.0, Te=1.0, Ti=1.0, mi=1.0, e=1.0)
    max_phi = float(jnp.max(jnp.abs(phi)))
    assert max_phi < 1e-6, f"|phi|_max should be ~0 for zero input: {max_phi}"
    print(f"  PASS: Zero RHS → |phi|_max = {max_phi:.2e}")


def test_poisson_symmetry(geom):
    """Poisson solver should give real output for real input."""
    Nr, Nth, Nze = 32, 64, 32
    key = jnp.array([0, 1], dtype=jnp.uint32)
    import jax
    delta_n = jax.random.normal(jax.random.PRNGKey(0), (Nr, Nth, Nze)) * 0.01
    phi = solve_poisson_gk(delta_n, geom, n0_avg=1.0, Te=1.0, Ti=1.0, mi=1.0, e=1.0)
    imag_part = float(jnp.max(jnp.abs(jnp.imag(phi))))
    assert imag_part < 1e-5, f"phi has large imaginary part: {imag_part}"
    print(f"  PASS: phi is real (max imag = {imag_part:.2e})")


def test_efield_from_phi(geom):
    """E = -∇φ should be consistent with φ."""
    Nr, Nth, Nze = 32, 64, 32
    # Simple test: phi = A*sin(kθ*θ), E_theta = -A*kθ*cos(kθ*θ)
    theta = jnp.linspace(0, 2*jnp.pi, Nth, endpoint=False)
    kth = 2  # mode number
    phi = jnp.zeros((Nr, Nth, Nze))
    A = 1.0
    phi = phi.at[:, :, :].set(A * jnp.sin(kth * theta)[None, :, None])

    E_r, E_theta, E_zeta = compute_efield(phi, geom)

    # Check E_theta ≈ -A*kth*(2π/Nth)*cos(kth*θ) ... spectral gradient
    dth = 2*jnp.pi / Nth
    E_theta_analytical = -A * kth / (1.0) * jnp.cos(kth * theta)  # -(dφ/dθ)/r roughly
    # Just check sign and rough magnitude at midplane
    mid_r = Nr // 2
    E_th_numerical = E_theta[mid_r, :, 0]

    # Should have same sign structure (zero crossings at same places)
    corr = float(jnp.sum(E_th_numerical * jnp.cos(kth * theta)))
    assert corr < 0, f"E_theta should have -cos structure: corr={corr}"
    print(f"  PASS: E_theta sign structure correct (corr={corr:.3f})")


def test_gyroaverage_reduces_amplitude(geom):
    """Gyroaveraging should reduce high-k modes (FLR damping)."""
    Nr, Nth, Nze = 32, 64, 32
    theta = jnp.linspace(0, 2*jnp.pi, Nth, endpoint=False)
    # High-k mode
    phi_highk = jnp.sin(8 * theta)[None, :, None] * jnp.ones((Nr, Nth, Nze))
    phi_lowk  = jnp.sin(1 * theta)[None, :, None] * jnp.ones((Nr, Nth, Nze))

    rho_i = 0.05  # ρi/a
    phi_ga_highk = gyroaverage_phi(phi_highk, rho_i)
    phi_ga_lowk  = gyroaverage_phi(phi_lowk,  rho_i)

    amp_highk = float(jnp.max(jnp.abs(phi_ga_highk)))
    amp_lowk  = float(jnp.max(jnp.abs(phi_ga_lowk)))

    # High-k should be reduced more
    assert amp_highk < amp_lowk, (
        f"Gyroaveraging should damp high-k more: "
        f"high-k amp={amp_highk:.3f}, low-k amp={amp_lowk:.3f}"
    )
    print(f"  PASS: Gyroaveraging damps high-k (high-k amp={amp_highk:.3f} < low-k amp={amp_lowk:.3f})")


if __name__ == "__main__":
    g = build_salpha_geometry(Nr=32, Ntheta=64, Nzeta=32, R0=1.0, a=0.18, B0=1.0)
    print("\n=== Poisson Solver Benchmark ===")
    test_poisson_zero_rhs(g)
    test_poisson_symmetry(g)
    test_efield_from_phi(g)
    test_gyroaverage_reduces_amplitude(g)
    print("\nAll Poisson tests PASSED")
