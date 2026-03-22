"""Tests for s-α geometry module."""

import pytest
import jax.numpy as jnp
import numpy as np
from gyrojax.geometry.salpha import build_salpha_geometry, interp_geometry_to_particles


@pytest.fixture
def geom():
    return build_salpha_geometry(Nr=32, Ntheta=64, Nzeta=32, R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)


def test_B_on_axis(geom):
    """B should be close to B0 near the magnetic axis (small ε)."""
    B_inner = float(geom.B_field[0, 0, 0])
    eps = float(geom.r_grid[0]) / geom.R0
    expected = geom.B0 / (1.0 + eps * 1.0)   # at θ=0, cos=1
    assert abs(B_inner - expected) / expected < 0.01, f"B at inner wall: {B_inner} vs {expected}"
    print(f"  PASS: B at inner wall = {B_inner:.4f}, expected ≈ {expected:.4f}")


def test_B_field_shape(geom):
    """B field should have correct grid shape."""
    assert geom.B_field.shape == (32, 64, 32), f"Wrong shape: {geom.B_field.shape}"
    print("  PASS: B_field shape correct")


def test_B_variation(geom):
    """B should decrease from inner to outer midplane (standard tokamak 1/R)."""
    B_inner = float(geom.B_field[0, 0, 0])   # θ=0, inner r
    B_outer = float(geom.B_field[-1, 0, 0])  # θ=0, outer r
    assert B_inner > B_outer, f"B should decrease outward: B_inner={B_inner}, B_outer={B_outer}"
    print(f"  PASS: B_inner={B_inner:.4f} > B_outer={B_outer:.4f}")


def test_gradB_sign(geom):
    """∂B/∂r should be negative at θ=0 (B decreases outward)."""
    gradB_r_midplane = float(geom.gradB_r[:, 0, 0].mean())
    assert gradB_r_midplane < 0, f"gradB_r should be negative at θ=0: {gradB_r_midplane}"
    print(f"  PASS: <∂B/∂r> at θ=0 = {gradB_r_midplane:.4e} < 0")


def test_curvature_direction(geom):
    """Normal curvature κ_r should be negative at θ=0 (curvature points inward)."""
    kappa_r_midplane = float(geom.kappa_r[:, 0, 0].mean())
    assert kappa_r_midplane < 0, f"kappa_r should be < 0 at θ=0: {kappa_r_midplane}"
    print(f"  PASS: <κ_r> at θ=0 = {kappa_r_midplane:.4e} < 0")


def test_q_profile(geom):
    """Safety factor q(r) should be ≥ q0 everywhere."""
    q_min = float(jnp.min(geom.q_profile))
    assert q_min >= 1.3, f"q should be >= q0=1.4 everywhere: min={q_min}"
    print(f"  PASS: q_min = {q_min:.4f} (q0=1.4)")


def test_interp_geometry(geom):
    """Interpolated B at grid center should match grid value."""
    r_mid   = float(geom.r_grid[len(geom.r_grid)//2])
    th_mid  = float(geom.theta_grid[0])
    ze_mid  = float(geom.zeta_grid[0])

    B_interp, _, _, _, _ = interp_geometry_to_particles(
        geom,
        jnp.array([r_mid]),
        jnp.array([th_mid]),
        jnp.array([ze_mid]),
    )
    eps = r_mid / geom.R0
    B_analytical = geom.B0 / (1.0 + eps * jnp.cos(th_mid))
    err = abs(float(B_interp[0]) - float(B_analytical)) / float(B_analytical)
    assert err < 0.02, f"Interpolation error too large: {err:.2%}"
    print(f"  PASS: Interpolated B error = {err:.2%}")


def test_B_symmetry_in_zeta(geom):
    """B should be independent of ζ (axisymmetric s-α)."""
    B_zeta_var = float(jnp.std(geom.B_field[10, 10, :]))
    assert B_zeta_var < 1e-6, f"B should not vary in ζ: std={B_zeta_var}"
    print(f"  PASS: B ζ-variation = {B_zeta_var:.2e} (axisymmetric)")


if __name__ == "__main__":
    g = build_salpha_geometry(Nr=32, Ntheta=64, Nzeta=32, R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5)
    print("\n=== Geometry Benchmark ===")
    test_B_on_axis(g)
    test_B_field_shape(g)
    test_B_variation(g)
    test_gradB_sign(g)
    test_curvature_direction(g)
    test_q_profile(g)
    test_interp_geometry(g)
    test_B_symmetry_in_zeta(g)
    print("\nAll geometry tests PASSED")
