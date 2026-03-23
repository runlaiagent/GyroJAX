"""
Multi-device sharding for GyroJAX using JAX sharding API.

Strategy:
  - Particles: sharded across devices along axis 0
    Each device owns N//num_devices particles
  - Grid (phi, delta_n): replicated on all devices
  - Scatter (particles->grid): local scatter per device + all-reduce
  - Gather (grid->particles): purely local (grid replicated)
  - Poisson solve: runs on all devices identically (cheap, no comm)
  - Weight update: local (no comm)
  - GC push: local (no comm)

Communication pattern:
  Only one all-reduce per step (scatter -> delta_n)
  Everything else is embarrassingly parallel.
"""

from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from gyrojax.particles.guiding_center import GCState


@dataclass
class ShardingConfig:
    """Sharding configuration for multi-device GyroJAX runs."""
    num_devices: int
    devices: list
    mesh: Mesh             # (num_devices,) mesh with axis 'particles'
    particle_sharding: NamedSharding  # shards along 'particles' axis
    grid_sharding: NamedSharding      # replicated


def create_sharding(num_devices: int = None) -> ShardingConfig:
    """Set up sharding config. num_devices=None uses all available."""
    # Try GPU first, fall back to CPU
    try:
        all_devices = jax.devices('gpu')
    except RuntimeError:
        all_devices = jax.devices('cpu')

    if num_devices is not None:
        all_devices = all_devices[:num_devices]

    num_devices = len(all_devices)
    devices_arr = np.array(all_devices)
    mesh = Mesh(devices_arr, axis_names=('particles',))
    particle_sharding = NamedSharding(mesh, P('particles'))
    grid_sharding = NamedSharding(mesh, P())  # replicated

    return ShardingConfig(
        num_devices=num_devices,
        devices=list(all_devices),
        mesh=mesh,
        particle_sharding=particle_sharding,
        grid_sharding=grid_sharding,
    )


def shard_gc_state(state: GCState, shard_cfg: ShardingConfig) -> GCState:
    """Distribute particle arrays in GCState across devices."""
    ps = shard_cfg.particle_sharding

    def _shard1d(x):
        return jax.device_put(x, ps)

    return GCState(
        r=_shard1d(state.r),
        theta=_shard1d(state.theta),
        zeta=_shard1d(state.zeta),
        vpar=_shard1d(state.vpar),
        mu=_shard1d(state.mu),
        weight=_shard1d(state.weight),
    )


def replicate_array(arr: jnp.ndarray, shard_cfg: ShardingConfig) -> jnp.ndarray:
    """Replicate an array to all devices."""
    return jax.device_put(arr, shard_cfg.grid_sharding)


def sharded_scatter(
    state: GCState,
    geom,
    grid_shape: tuple,
    shard_cfg: ShardingConfig,
) -> jnp.ndarray:
    """
    Multi-device scatter: each device scatters its local particles,
    then JAX automatically inserts the all-reduce when it sees sharded
    inputs with a replicated output sharding.

    Returns delta_n on all devices (replicated).
    """
    from gyrojax.interpolation.scatter_gather_fa import scatter_to_grid_fa

    @functools.partial(jax.jit, out_shardings=shard_cfg.grid_sharding)
    def _scatter(state, geom):
        return scatter_to_grid_fa(state, geom, grid_shape)

    return _scatter(state, geom)


def sharded_gather(
    phi: jnp.ndarray,
    state: GCState,
    geom,
    shard_cfg: ShardingConfig,
):
    """
    Multi-device gather: grid is replicated, particles are sharded.
    Each device gathers E at its local particle positions.

    Returns (E_psi, E_theta, E_alpha) — all sharded along particles axis.
    """
    from gyrojax.interpolation.scatter_gather_fa import gather_from_grid_fa

    in_shardings = (shard_cfg.grid_sharding, shard_cfg.particle_sharding)
    out_shardings = (
        shard_cfg.particle_sharding,
        shard_cfg.particle_sharding,
        shard_cfg.particle_sharding,
    )

    @functools.partial(jax.jit, out_shardings=out_shardings)
    def _gather(phi, state, geom):
        return gather_from_grid_fa(phi, state, geom)

    return _gather(phi, state, geom)


# ── Legacy helpers (keep backward compat) ─────────────────────────────────────

def setup_devices(n_gpus: int = None):
    """Return available GPU devices (falls back to CPU)."""
    try:
        devices = jax.devices('gpu')
    except RuntimeError:
        devices = jax.devices()
    if n_gpus:
        devices = devices[:n_gpus]
    print(f"[GyroJAX] Using {len(devices)} device(s): {[str(d) for d in devices]}")
    return devices


def replicate_field(arr: jnp.ndarray, devices) -> jnp.ndarray:
    """Replicate a field array to all devices (legacy API)."""
    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(np.array(devices))
    return jax.device_put(arr, sharding.replicate())
