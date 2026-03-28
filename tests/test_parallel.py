"""Tests for multi-GPU pmap parallelism infrastructure."""
import pytest
import jax
import jax.numpy as jnp


def test_get_device_count():
    """get_device_count returns a positive integer."""
    from gyrojax.parallel import get_device_count
    n = get_device_count()
    assert isinstance(n, int)
    assert n >= 1


def test_shard_particles_shape():
    """shard_particles produces correct shape."""
    from gyrojax.parallel import shard_particles
    arr = jnp.ones((100, 6))
    sharded = shard_particles(arr, n_devices=4)
    assert sharded.shape == (4, 25, 6)


def test_shard_particles_trims():
    """shard_particles trims array to be evenly divisible."""
    from gyrojax.parallel import shard_particles
    arr = jnp.ones((101, 6))
    sharded = shard_particles(arr, n_devices=4)
    assert sharded.shape == (4, 25, 6)  # 100 particles, not 101


def test_pmap_single_device_fallback():
    """On single GPU, run_simulation_pmap falls back to run_simulation_fa."""
    from gyrojax.parallel import run_simulation_pmap, get_device_count
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    if get_device_count() > 1:
        pytest.skip("This test is for single-device fallback only")

    cfg = SimConfigFA(
        Npsi=8, Ntheta=16, Nalpha=16,
        N_particles=1000, n_steps=3, dt=0.05,
    )
    key = jax.random.PRNGKey(0)

    # pmap should give same result as direct run on single device
    diags_pmap, _, phi_pmap, _ = run_simulation_pmap(cfg, key=key, verbose=False)
    diags_fa,   _, phi_fa,   _ = run_simulation_fa(cfg,  key=key, verbose=False)

    assert jnp.allclose(phi_pmap, phi_fa, atol=1e-5), (
        "pmap single-device fallback must match run_simulation_fa"
    )


def test_pmap_multi_gpu_raises():
    """On multi-GPU, run_simulation_pmap raises NotImplementedError (pending hardware)."""
    from gyrojax.parallel import run_simulation_pmap, get_device_count
    from gyrojax.simulation_fa import SimConfigFA

    if get_device_count() == 1:
        pytest.skip("This test requires multiple devices")

    cfg = SimConfigFA(Npsi=8, Ntheta=16, Nalpha=16, N_particles=1000, n_steps=2)
    with pytest.raises(NotImplementedError):
        run_simulation_pmap(cfg)
