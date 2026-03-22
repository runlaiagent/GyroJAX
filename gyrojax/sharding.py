# FILE: gyrojax/sharding.py
"""Multi-GPU sharding utilities for GyroJAX."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from gyrojax.particles.guiding_center import GCState


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


def shard_array(arr: jnp.ndarray, devices) -> jnp.ndarray:
    """Shard a 1D particle array across devices along axis 0."""
    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(np.array(devices)).reshape(len(devices), 1)
    return jax.device_put(arr[:, None], sharding).reshape(-1)


def shard_gc_state(state: GCState, devices) -> GCState:
    """Shard all particle arrays in GCState across devices."""
    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(np.array(devices))
    n = len(devices)

    def shard1d(x):
        return jax.device_put(x, sharding.reshape(n))

    return GCState(
        r=shard1d(state.r), theta=shard1d(state.theta), zeta=shard1d(state.zeta),
        vpar=shard1d(state.vpar), mu=shard1d(state.mu), weight=shard1d(state.weight),
    )


def replicate_field(arr: jnp.ndarray, devices) -> jnp.ndarray:
    """Replicate a field array to all devices."""
    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(np.array(devices))
    return jax.device_put(arr, sharding.replicate())
