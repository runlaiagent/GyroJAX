"""Tests for post-processing module."""
import pytest
import numpy as np


def _make_test_h5(tmp_path):
    """Create a minimal HDF5 file for testing."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "test_run.h5"
    n = 100
    with h5py.File(path, "w") as f:
        diags = f.create_group("diags")
        # Simulate linear growth then saturation
        t = np.arange(n) * 0.05
        phi_max = 1e-3 * np.exp(0.2 * t)
        phi_max[50:] = phi_max[50] * (1 + 0.1 * np.random.default_rng(0).normal(size=50))
        diags.create_dataset("phi_rms", data=phi_max * 0.7)
        diags.create_dataset("phi_max", data=phi_max)
        diags.create_dataset("weight_rms", data=np.ones(n) * 0.05)
        cfg = f.create_group("config")
        cfg.attrs["dt"] = 0.05
        cfg.attrs["R0_over_LT"] = 6.9
    return path


def test_postprocessor_loads(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _make_test_h5(tmp_path)
    from gyrojax.io.postprocess import PostProcessor
    pp = PostProcessor(path)
    assert pp.n_steps == 100
    assert len(pp.phi_max) == 100


def test_growth_rate(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _make_test_h5(tmp_path)
    from gyrojax.io.postprocess import PostProcessor
    pp = PostProcessor(path)
    gamma = pp.growth_rate(t_start=5, t_end=45)
    # Synthetic data has growth rate 0.2 — allow 20% error
    assert abs(gamma - 0.2) < 0.05, f"Expected ~0.2, got {gamma:.4f}"


def test_heat_flux(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _make_test_h5(tmp_path)
    from gyrojax.io.postprocess import PostProcessor
    pp = PostProcessor(path)
    chi = pp.heat_flux_chi(saturation_start=50)
    assert chi > 0


def test_summary(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _make_test_h5(tmp_path)
    from gyrojax.io.postprocess import PostProcessor
    pp = PostProcessor(path)
    s = pp.summary()
    assert "growth_rate" in s
    assert "chi_i" in s
    assert s["n_steps"] == 100


def test_load_results_convenience(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _make_test_h5(tmp_path)
    from gyrojax.io.postprocess import load_results
    pp = load_results(path)
    assert isinstance(pp.phi_max, np.ndarray)
