"""Tests for run_long_simulation_fa chunked runner."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa, run_long_simulation_fa


@pytest.fixture
def small_cfg():
    return SimConfigFA(
        Npsi=8, Ntheta=16, Nalpha=16,
        N_particles=1000, dt=0.05, R0_over_LT=6.9,
    )


def test_long_run_chunked(small_cfg):
    """run_long_simulation_fa first chunk matches single run_simulation_fa."""
    cfg = small_cfg
    key = jax.random.PRNGKey(42)

    # Single run: 20 steps
    cfg_20 = SimConfigFA(**{**vars(cfg), 'n_steps': 20})
    diags_single, _, _, _ = run_simulation_fa(cfg_20, key=key, verbose=False)
    phi_single = np.array([float(d.phi_max) for d in diags_single])

    # Chunked run: 20 steps in chunks of 5
    result = run_long_simulation_fa(cfg, n_total_steps=20, chunk_size=5, key=key, verbose=False)
    phi_chunked = result['phi_max']

    assert len(phi_chunked) == 20
    # First chunk should match exactly (same init, same key)
    np.testing.assert_allclose(phi_single[:5], phi_chunked[:5], rtol=1e-4)


def test_long_run_returns_correct_shape(small_cfg):
    """run_long_simulation_fa returns arrays of correct length."""
    result = run_long_simulation_fa(
        small_cfg, n_total_steps=10, chunk_size=3, verbose=False
    )
    assert result['n_steps'] == 10
    assert len(result['phi_rms']) == 10
    assert len(result['phi_max']) == 10
    assert len(result['weight_rms']) == 10
    assert isinstance(result['chi_i'], float)


def test_long_run_hdf5(tmp_path, small_cfg):
    """run_long_simulation_fa saves to HDF5 correctly."""
    pytest.importorskip("h5py")
    import h5py

    output = str(tmp_path / "long_run.h5")
    result = run_long_simulation_fa(
        small_cfg, n_total_steps=10, chunk_size=5,
        output_file=output, key=jax.random.PRNGKey(0), verbose=False
    )

    assert result['n_steps'] == 10
    assert result['output_file'] == output

    import os
    assert os.path.exists(output)

    with h5py.File(output, "r") as f:
        assert "diags" in f
        assert len(f["diags/phi_max"][:]) == 10
