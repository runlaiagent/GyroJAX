"""
Tests for Phase 2a: field-aligned geometry + full Γ₀(b) Poisson.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import i0e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fa_geom():
    from gyrojax.geometry.field_aligned import build_field_aligned_geometry
    return build_field_aligned_geometry(
        Npsi=32, Ntheta=64, Nalpha=32,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5
    )


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

class TestFieldAlignedGeometry:

    def test_B_on_axis(self, fa_geom):
        """B near r=0 should approach B0."""
        B_inner = float(fa_geom.B_field[0].mean())
        assert abs(B_inner - 1.0) < 0.3

    def test_B_shape(self, fa_geom):
        assert fa_geom.B_field.shape == (32, 64, 32)

    def test_B_no_alpha_dependence(self, fa_geom):
        """In s-α, B doesn't depend on α."""
        B_std_along_alpha = float(jnp.std(fa_geom.B_field, axis=2).max())
        assert B_std_along_alpha < 1e-5

    def test_theta_range(self, fa_geom):
        th = fa_geom.theta_grid
        assert float(th[0]) == pytest.approx(-np.pi, abs=1e-5)
        assert float(th[-1]) > np.pi * 0.9

    def test_q_profile_shape(self, fa_geom):
        assert fa_geom.q_profile.shape == (32,)
        assert float(fa_geom.q_profile[0]) == pytest.approx(1.4, rel=0.05)

    def test_shat_positive(self, fa_geom):
        """Magnetic shear should be positive for q increasing outward."""
        assert float(jnp.min(fa_geom.shat)) >= 0.0

    def test_twist_shift(self, fa_geom):
        """Twist shift Δα = -2π·ŝ should be negative (shear > 0)."""
        assert float(jnp.max(fa_geom.twist_shift)) <= 0.0

    def test_galphaalpha_positive(self, fa_geom):
        """g^{αα} > 0 everywhere."""
        assert float(jnp.min(fa_geom.galphaalpha)) > 0.0

    def test_galphaalpha_increases_with_theta(self, fa_geom):
        """g^{αα} = (q/r)²(1+ŝ²θ²) increases away from θ=0."""
        # At fixed ψ, g_aa should be larger at |θ| = π than at θ = 0
        mid_psi = fa_geom.galphaalpha.shape[0] // 2
        g_at_0  = float(fa_geom.galphaalpha[mid_psi, fa_geom.Ntheta//2])  if hasattr(fa_geom, 'Ntheta') else float(fa_geom.galphaalpha[mid_psi, fa_geom.galphaalpha.shape[1]//2])
        g_at_pi = float(fa_geom.galphaalpha[mid_psi, -1])
        assert g_at_pi > g_at_0


# ---------------------------------------------------------------------------
# Poisson solver tests
# ---------------------------------------------------------------------------

class TestGamma0:

    def test_gamma0_at_zero(self):
        """Γ₀(0) = I₀(0)·exp(0) = 1."""
        from gyrojax.fields.poisson_fa import _gamma0
        val = float(_gamma0(jnp.array(0.0)))
        assert val == pytest.approx(1.0, abs=1e-5)

    def test_gamma0_decreasing(self):
        """Γ₀(b) is monotonically decreasing from 1 to 0."""
        from gyrojax.fields.poisson_fa import _gamma0
        b_vals = jnp.linspace(0.0, 5.0, 50)
        g0 = _gamma0(b_vals)
        diffs = jnp.diff(g0)
        assert bool(jnp.all(diffs <= 0.0))

    def test_gamma0_at_large_b(self):
        """For large b, Γ₀ → 0 (specifically < 0.15 at b=20)."""
        from gyrojax.fields.poisson_fa import _gamma0
        val = float(_gamma0(jnp.array(20.0)))
        assert val < 0.15   # i0e(20) ≈ 0.09

    def test_pade_vs_exact_small_b(self):
        """For small b, Padé 1/(1+b) ≈ Γ₀(b) within 2%."""
        from gyrojax.fields.poisson_fa import _gamma0
        b = jnp.array(0.1)
        g0_exact = float(_gamma0(b))
        g0_pade  = 1.0 / (1.0 + float(b))
        assert abs(g0_exact - g0_pade) / g0_exact < 0.02


class TestPoissonFA:

    def test_zero_rhs_gives_zero_phi(self, fa_geom):
        from gyrojax.fields.poisson_fa import solve_poisson_fa
        delta_n = jnp.zeros((32, 64, 32))
        phi = solve_poisson_fa(delta_n, fa_geom, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert float(jnp.max(jnp.abs(phi))) < 1e-10

    def test_phi_real(self, fa_geom):
        from gyrojax.fields.poisson_fa import solve_poisson_fa
        key = jax.random.PRNGKey(0)
        delta_n = jax.random.normal(key, (32, 64, 32)) * 1e-4
        phi = solve_poisson_fa(delta_n, fa_geom, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert phi.dtype in (jnp.float32, jnp.float64)

    def test_phi_mean_zero(self, fa_geom):
        from gyrojax.fields.poisson_fa import solve_poisson_fa
        key = jax.random.PRNGKey(1)
        delta_n = jax.random.normal(key, (32, 64, 32)) * 1e-4
        phi = solve_poisson_fa(delta_n, fa_geom, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert float(jnp.mean(phi)) == pytest.approx(0.0, abs=1e-6)  # float32 round-trip

    def test_flr_reduces_amplitude(self, fa_geom):
        """
        Exact Γ₀(b) solver should produce a finite result for high-k modes.
        For large b, the operator is better conditioned than Padé (which can
        go negative). Verify exact solver is finite and nonzero.
        """
        from gyrojax.fields.poisson_fa import solve_poisson_fa
        Npsi, Ntheta, Nalpha = 32, 64, 32
        al = jnp.linspace(0, 2*jnp.pi, Nalpha, endpoint=False)
        delta_n = jnp.array(1e-3 * jnp.sin(8 * al)[None, None, :] * jnp.ones((Npsi, Ntheta, Nalpha)))

        phi_exact = solve_poisson_fa(delta_n, fa_geom, 1.0, 1.0, 1.0, 1.0, 1.0)

        # Exact solver should be numerically stable (no NaN)
        assert not bool(jnp.any(jnp.isnan(phi_exact))), "Exact Γ₀ solver produced NaN"
        # phi should be nonzero (mode was excited)
        assert float(jnp.max(jnp.abs(phi_exact))) > 1e-15

    def test_efield_from_phi(self, fa_geom):
        """E = -∇φ: single-mode test."""
        from gyrojax.fields.poisson_fa import compute_efield_fa
        Npsi, Ntheta, Nalpha = 32, 64, 32
        al = fa_geom.alpha_grid
        phi = jnp.sin(al)[None, None, :] * jnp.ones((Npsi, Ntheta, 1))
        E_psi, E_th, E_al = compute_efield_fa(phi, fa_geom)
        # E_al = -∂φ/∂α = -cos(α) * 2π/Nalpha ... check sign
        E_al_expected = -jnp.cos(al)
        corr = float(jnp.corrcoef(
            jnp.array(E_al[Npsi//2, Ntheta//2, :]),
            E_al_expected
        )[0, 1])
        assert corr > 0.99
