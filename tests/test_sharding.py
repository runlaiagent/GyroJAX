"""Tests for multi-device sharding."""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from gyrojax.particles.guiding_center import GCState
from gyrojax.sharding import (
    create_sharding, shard_gc_state, replicate_array, sharded_scatter,
    ShardingConfig,
)


def _make_test_state(N: int) -> GCState:
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)
    return GCState(
        r=jax.random.uniform(keys[0], (N,), minval=0.05, maxval=0.18),
        theta=jax.random.uniform(keys[1], (N,), minval=-jnp.pi, maxval=jnp.pi),
        zeta=jax.random.uniform(keys[2], (N,), minval=0.0, maxval=2*jnp.pi),
        vpar=jax.random.normal(keys[3], (N,)),
        mu=jax.random.uniform(keys[4], (N,), minval=0.0, maxval=1.0),
        weight=jax.random.normal(keys[5], (N,)) * 1e-4,
    )


class TestShardingConfig:
    def test_create_sharding_single(self):
        """Single device sharding works."""
        shard_cfg = create_sharding(num_devices=1)
        assert shard_cfg.num_devices == 1
        assert len(shard_cfg.devices) == 1

    def test_create_sharding_max(self):
        """Max device sharding uses all available."""
        shard_cfg = create_sharding()
        assert shard_cfg.num_devices == len(jax.devices())

    def test_sharding_config_has_mesh(self):
        """ShardingConfig has mesh and sharding objects."""
        shard_cfg = create_sharding()
        assert shard_cfg.mesh is not None
        assert shard_cfg.particle_sharding is not None
        assert shard_cfg.grid_sharding is not None

    def test_shard_gc_state(self):
        """GCState sharding preserves shape."""
        shard_cfg = create_sharding()
        N = 256
        state = _make_test_state(N)
        sharded = shard_gc_state(state, shard_cfg)
        assert sharded.r.shape == (N,)
        assert sharded.theta.shape == (N,)
        assert sharded.weight.shape == (N,)

    def test_replicate_array(self):
        """Array replication preserves shape."""
        shard_cfg = create_sharding()
        arr = jnp.ones((16, 32, 16))
        rep = replicate_array(arr, shard_cfg)
        assert rep.shape == arr.shape

    def test_replicate_array_values(self):
        """Replicated array values match original."""
        shard_cfg = create_sharding()
        arr = jnp.arange(48.0).reshape(3, 4, 4)
        rep = replicate_array(arr, shard_cfg)
        np.testing.assert_allclose(np.array(rep), np.array(arr))


class TestShardedScatter:
    def test_sharded_scatter_shape(self):
        """Sharded scatter produces correct output shape."""
        from gyrojax.geometry.field_aligned import build_field_aligned_geometry

        shard_cfg = create_sharding()
        Npsi, Nth, Nal = 8, 16, 8
        geom = build_field_aligned_geometry(
            Npsi=Npsi, Ntheta=Nth, Nalpha=Nal,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        )
        grid_shape = (Npsi, Nth, Nal)
        N = 512
        state = _make_test_state(N)
        sharded_state = shard_gc_state(state, shard_cfg)

        result = sharded_scatter(sharded_state, geom, grid_shape, shard_cfg)
        assert result.shape == grid_shape
        assert np.all(np.isfinite(np.array(result)))

    def test_sharded_scatter_matches_local(self):
        """Sharded scatter gives same result as single-device scatter."""
        from gyrojax.geometry.field_aligned import build_field_aligned_geometry
        from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa

        shard_cfg = create_sharding(num_devices=1)
        Npsi, Nth, Nal = 8, 16, 8
        geom = build_field_aligned_geometry(
            Npsi=Npsi, Ntheta=Nth, Nalpha=Nal,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
        )
        grid_shape = (Npsi, Nth, Nal)
        N = 512
        state = _make_test_state(N)

        local_result = scatter_to_grid_fa(state, geom, grid_shape)
        sharded_state = shard_gc_state(state, shard_cfg)
        sharded_result = sharded_scatter(sharded_state, geom, grid_shape, shard_cfg)

        np.testing.assert_allclose(
            np.array(local_result), np.array(sharded_result), rtol=1e-4, atol=1e-7
        )


class TestShardedSimulation:
    def test_sharded_sim_no_nan(self):
        """Sharded sim runs 40 steps without NaN."""
        from gyrojax.simulation_fa import SimConfigFA
        from gyrojax.simulation_sharded import run_simulation_sharded

        cfg = SimConfigFA(
            Npsi=8, Ntheta=16, Nalpha=8,
            N_particles=2_000, n_steps=40, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            n0_avg=1.0, R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
        )
        diags, _, phi, _ = run_simulation_sharded(
            cfg, key=jax.random.PRNGKey(0), verbose=False
        )
        assert len(diags) > 0
        phi_max_vals = [float(d.phi_max) for d in diags]
        assert all(np.isfinite(v) for v in phi_max_vals), \
            f"NaN found in phi_max: {phi_max_vals}"

    def test_sharded_sim_single_device(self):
        """Single-device sharded sim returns same structure as run_simulation_fa."""
        from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
        from gyrojax.simulation_sharded import run_simulation_sharded

        cfg = SimConfigFA(
            Npsi=8, Ntheta=16, Nalpha=8,
            N_particles=1_000, n_steps=10, dt=0.05,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            n0_avg=1.0, R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
        )
        key = jax.random.PRNGKey(7)
        diags, state, phi, geom = run_simulation_sharded(
            cfg, key=key, num_devices=1, verbose=False
        )
        assert len(diags) == 10
        assert phi.shape == (8, 16, 8)
        assert state.r.shape == (1_000,)

    def test_nan_early_stopping(self):
        """NaN early stopping: if phi goes NaN, simulation stops cleanly."""
        from gyrojax.simulation_fa import SimConfigFA
        from gyrojax.simulation_sharded import run_simulation_sharded

        # Very large dt forces blow-up and NaN
        cfg = SimConfigFA(
            Npsi=8, Ntheta=16, Nalpha=8,
            N_particles=500, n_steps=20, dt=500.0,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0,
            n0_avg=1.0, R0_over_LT=6.9, R0_over_Ln=2.2, vpar_cap=4.0,
        )
        # Should not raise
        diags, _, _, _ = run_simulation_sharded(
            cfg, key=jax.random.PRNGKey(0), nan_stop=True, verbose=False
        )
        # Must return at least 1 diagnostic (even if stopped early)
        assert len(diags) >= 1
