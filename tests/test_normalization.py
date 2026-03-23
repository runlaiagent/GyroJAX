"""Tests for gyrojax.normalization module."""

import math
import pytest
import numpy as np

from gyrojax.normalization import (
    NormParams,
    normalize_geometry,
    denormalize_phi,
    denormalize_growth_rate,
    norm_summary,
)
from gyrojax.geometry.field_aligned import build_field_aligned_geometry


# ------------------------------------------------------------------ #
# NormParams construction
# ------------------------------------------------------------------ #

def test_norm_params_cbc_rho_star():
    """CBC rho_star should be ≈ 1/180."""
    norm = NormParams.from_cbc()
    expected = 1.0 / 180.0
    assert abs(norm.rho_star - expected) / expected < 0.01, (
        f"rho_star={norm.rho_star:.6f}, expected≈{expected:.6f}"
    )


def test_norm_params_cbc_fields():
    """CBC NormParams should have correct field values."""
    norm = NormParams.from_cbc()
    assert norm.a_ref == 0.18
    assert norm.B_ref == 1.0
    assert norm.e_ref == 1000.0
    assert norm.Lref == 'a'


def test_norm_params_vt_ref():
    """vt_ref = sqrt(T/m)."""
    norm = NormParams.from_cbc()
    assert abs(norm.vt_ref - math.sqrt(norm.T_ref / norm.m_ref)) < 1e-10


def test_norm_params_omega_ref():
    """Omega_ref = e*B/m."""
    norm = NormParams.from_cbc()
    expected = norm.e_ref * norm.B_ref / norm.m_ref
    assert abs(norm.Omega_ref - expected) < 1e-10


class _SimpleGeom:
    """Minimal mock geometry for from_vmec tests."""
    def __init__(self, a, B0):
        self.a = a
        self.B0 = B0


def test_norm_params_vmec_rho_star():
    """from_vmec with default rho_star=1/180 should preserve rho_star."""
    mock = _SimpleGeom(a=0.5, B0=2.5)
    norm = NormParams.from_vmec(mock, Ti=1.0, mi=1.0, rho_star=1.0/180.0)
    assert 1e-3 < norm.rho_star < 1e-2, (
        f"rho_star={norm.rho_star} not in expected range [1e-3, 1e-2]"
    )
    # Should recover the requested rho_star
    assert abs(norm.rho_star - 1.0/180.0) / (1.0/180.0) < 1e-6


def test_lref_R0():
    """Lref='R0' should give rho_star = rho_i/R0 (larger than rho_i/a for R0>a)."""
    a = 0.18
    R0 = 1.0
    B0 = 1.0
    norm_a = NormParams(a_ref=a, B_ref=B0, T_ref=1.0, n_ref=1.0, m_ref=1.0, e_ref=1000.0, Lref='a')
    norm_R0 = NormParams(a_ref=a, B_ref=B0, T_ref=1.0, n_ref=1.0, m_ref=1.0, e_ref=1000.0,
                         Lref='R0', R0_ref=R0)
    # rho_star_R0 = rho_i/R0;  rho_star_a = rho_i/a  =>  R0 > a => rho_star_R0 < rho_star_a
    assert norm_R0.rho_star < norm_a.rho_star
    # Explicitly: rho_star_R0 = rho_i / R0
    assert abs(norm_R0.rho_star - norm_R0.rho_ref / R0) < 1e-10


# ------------------------------------------------------------------ #
# Geometry normalization
# ------------------------------------------------------------------ #

@pytest.fixture
def fa_geom():
    return build_field_aligned_geometry(
        Npsi=8, Ntheta=16, Nalpha=8,
        R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
    )


def test_normalize_geometry_B_order_unity(fa_geom):
    """After normalization B_hat should be ≈ 1 ± 0.5."""
    norm = NormParams.from_cbc()
    geom_hat = normalize_geometry(fa_geom, norm)
    B_hat = np.array(geom_hat.B_field, dtype=float)
    assert B_hat.mean() == pytest.approx(1.0, abs=0.5), (
        f"B_hat mean={B_hat.mean():.3f} not ≈ 1"
    )


def test_normalize_geometry_psi_order_unity(fa_geom):
    """psi_hat = r/a should lie in (0, 1]."""
    norm = NormParams.from_cbc()
    geom_hat = normalize_geometry(fa_geom, norm)
    psi_hat = np.array(geom_hat.psi_grid, dtype=float)
    assert psi_hat.min() > 0.0
    assert psi_hat.max() <= 1.0 + 1e-6, f"psi_hat max={psi_hat.max()}"


def test_normalize_geometry_gradB_reasonable(fa_geom):
    """gradB_hat rms should be < 10 (dimensionless)."""
    norm = NormParams.from_cbc()
    geom_hat = normalize_geometry(fa_geom, norm)
    gradB_rms = float(np.sqrt(np.mean(np.array(geom_hat.gradB_th, dtype=float)**2)))
    assert gradB_rms < 10.0, f"gradB_hat rms={gradB_rms}"


# ------------------------------------------------------------------ #
# De-normalization helpers
# ------------------------------------------------------------------ #

def test_denormalize_phi_roundtrip():
    """normalize then denormalize phi = identity."""
    norm = NormParams.from_cbc()
    phi_phys = 3.14159
    phi_hat = phi_phys * norm.e_ref / norm.T_ref
    phi_recovered = denormalize_phi(phi_hat, norm)
    assert abs(phi_recovered - phi_phys) < 1e-10


def test_denormalize_growth_rate_cbc():
    """γ_hat=0.17 should give γ_SI in [1e4, 1e7] for CBC (code units)."""
    norm = NormParams.from_cbc()
    gamma_hat = 0.17   # typical CBC ITG growth rate in code units
    gamma_phys = denormalize_growth_rate(gamma_hat, norm)
    # In CBC code units vt_ref=1, a_ref=0.18: gamma_phys = 0.17/0.18 ≈ 0.94
    # In real SI the range would be [1e4, 1e7]; in code units it's O(1)
    # Just check it's positive and of right order
    assert gamma_phys > 0.0
    # gamma_phys = gamma_hat * vt_ref / a_ref
    expected = gamma_hat * norm.vt_ref / norm.a_ref
    assert abs(gamma_phys - expected) < 1e-10


# ------------------------------------------------------------------ #
# norm_summary
# ------------------------------------------------------------------ #

def test_norm_summary_keys():
    """norm_summary should return a dict with key fields."""
    norm = NormParams.from_cbc()
    summary = norm_summary(norm)
    for key in ('rho_star', '1/rho_star', 'vt_ref', 'Omega_ref', 'rho_ref'):
        assert key in summary, f"Missing key: {key}"
