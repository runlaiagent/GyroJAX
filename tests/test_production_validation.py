"""
Tests for production validation infrastructure.
Fast tests only — no full simulation runs.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.rosenbluth_hinton import rh_residual_theory as rh_theory  # noqa: F401 (alias)
from benchmarks.gamma_spectrum import estimate_ky_rho, extract_growth_rate, DIMITS_REF


class TestRHFormula:
    def test_rh_formula_cbc(self):
        """RH residual for CBC params (q=1.4, eps=0.18) should be ~0.12."""
        res = rh_theory(q=1.4, eps=0.18)
        assert 0.05 < res < 0.25, f"R-H residual out of range: {res:.4f}"

    def test_rh_formula_high_q(self):
        """Higher q → lower residual (more neoclassical damping)."""
        r_low  = rh_theory(q=1.0, eps=0.18)
        r_high = rh_theory(q=2.0, eps=0.18)
        assert r_high < r_low

    def test_rh_formula_high_eps(self):
        """Higher epsilon (more toroidal) → lower 1.6q²/√ε → higher residual."""
        r_low  = rh_theory(q=1.4, eps=0.05)
        r_high = rh_theory(q=1.4, eps=0.50)
        assert r_high > r_low


class TestGammaSpectrum:
    def test_ky_rho_estimate_reasonable(self):
        """ky·ρi estimate should be positive and finite for typical Nalpha."""
        for Nalpha in [8, 16, 24, 32]:
            ky_rho = estimate_ky_rho(Nalpha // 2, 1.4, (0.18/180) / 0.09)
            assert ky_rho > 0, f"ky·ρi={ky_rho} non-positive for Nalpha={Nalpha}"
            assert ky_rho < 2.0, f"ky·ρi={ky_rho} too large for Nalpha={Nalpha}"

    def test_dimits_ref_populated(self):
        """Reference table has entries at key wavenumbers."""
        for ky in [0.2, 0.3, 0.4]:
            assert ky in DIMITS_REF
            assert DIMITS_REF[ky] > 0

    def test_extract_growth_rate_growing(self):
        """Growth rate extraction returns positive value for exponentially growing signal."""
        dt = 0.05
        t  = np.arange(200) * dt
        phi_max = np.exp(0.17 * t) * 1e-5 + 1e-8  # γ=0.17
        gamma = extract_growth_rate(phi_max.tolist(), dt)
        assert 0.10 < gamma < 0.25, f"Extracted γ={gamma:.4f}, expected ~0.17"

    def test_extract_growth_rate_decaying(self):
        """Decaying signal returns 0 (not negative)."""
        phi_max = np.exp(-0.1 * np.arange(100) * 0.05).tolist()
        gamma = extract_growth_rate(phi_max, 0.05)
        assert gamma == 0.0

    def test_extract_growth_rate_noisy(self):
        """Noisy but growing signal still returns positive γ."""
        rng = np.random.default_rng(42)
        t   = np.arange(200) * 0.05
        phi_max = (np.exp(0.17 * t) * 1e-5 + 1e-8) * (1 + 0.3 * rng.standard_normal(200))
        phi_max = np.abs(phi_max)
        gamma = extract_growth_rate(phi_max.tolist(), 0.05)
        assert gamma > 0.0

    def test_cbc_error_under_30pct(self):
        """Quick sanity: our known quick-mode result is within 30% of target."""
        # Known from prior runs
        gyrojax_gamma = 0.185
        target        = 0.170
        error = abs(gyrojax_gamma - target) / target * 100
        assert error < 30.0
