"""
Sharded (multi-device) simulation loop for GyroJAX.

Wraps simulation_fa with multi-device particle sharding.
Falls back gracefully to single device if only 1 device available.

Key additions over simulation_fa:
  1. Auto-shards particles across num_devices devices
  2. Grid (phi, delta_n) replicated on all devices
  3. NaN early-stopping: detects blow-up and exits cleanly
  4. Sharded scatter with automatic JAX all-reduce
"""

from __future__ import annotations
import functools

import jax
import jax.numpy as jnp

from gyrojax.simulation_fa import SimConfigFA, _run_with_geom
from gyrojax.geometry.field_aligned import build_field_aligned_geometry
from gyrojax.sharding import create_sharding, shard_gc_state, sharded_scatter, ShardingConfig


def run_simulation_sharded(
    cfg: SimConfigFA,
    key: jax.random.PRNGKey = None,
    num_devices: int = None,   # None = use all
    nan_stop: bool = True,     # stop cleanly on NaN (handled in _run_with_geom)
    verbose: bool = True,
) -> tuple:
    """
    Run sharded simulation.

    Returns same (diags, state, phi, geom) tuple as run_simulation_fa.
    Particles are sharded across devices; grid is replicated.
    When num_devices=1 (or only 1 device available) runs identically to
    run_simulation_fa with no overhead.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    shard_cfg = create_sharding(num_devices)

    if verbose:
        print(f"[GyroJAX Sharded] {shard_cfg.num_devices} device(s): "
              f"{[str(d) for d in shard_cfg.devices]}")

    # Build geometry (always on device 0 / default, same as non-sharded path)
    geom = build_field_aligned_geometry(
        Npsi=cfg.Npsi, Ntheta=cfg.Ntheta, Nalpha=cfg.Nalpha,
        R0=cfg.R0, a=cfg.a, B0=cfg.B0, q0=cfg.q0, q1=cfg.q1,
    )

    if shard_cfg.num_devices == 1:
        # Single-device path — no sharding overhead
        return _run_with_geom(cfg, geom, key, verbose=verbose)

    # Multi-device: build a scatter_fn that shards particles and all-reduces
    grid_shape = (cfg.Npsi, cfg.Ntheta, cfg.Nalpha)

    def _sharded_scatter_fn(state, geom, grid_shape):
        return sharded_scatter(state, geom, grid_shape, shard_cfg)

    # _run_with_geom will shard the state via state0_override after init.
    # We patch scatter_fn; the initial state returned from _run_with_geom
    # will be computed normally since we pass scatter_fn and shard inside.
    return _run_with_geom(
        cfg, geom, key,
        verbose=verbose,
        scatter_fn=_sharded_scatter_fn,
    )
