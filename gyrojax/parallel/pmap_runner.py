"""Multi-GPU pmap runner for GyroJAX.

Architecture:
  - Particles sharded across D devices (each device: N//D particles)
  - Grid replicated on all devices
  - per-device scatter → psum → global Poisson → per-device push
  - lax.scan runs identically on each device
"""

import jax
import jax.numpy as jnp
from typing import Optional

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


def get_device_count() -> int:
    """Return number of available JAX devices (GPUs or TPUs)."""
    return jax.device_count()


def shard_particles(arr: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """Shard a particle array [N, ...] → [n_devices, N//n_devices, ...].

    Trims particles to be evenly divisible by n_devices.
    """
    N = arr.shape[0]
    N_per_device = N // n_devices
    arr = arr[:N_per_device * n_devices]
    return arr.reshape(n_devices, N_per_device, *arr.shape[1:])


def unshard_particles(arr: jnp.ndarray) -> jnp.ndarray:
    """Unshard [n_devices, N_per_device, ...] → [N, ...]."""
    n_devices = arr.shape[0]
    N_per_device = arr.shape[1]
    return arr.reshape(n_devices * N_per_device, *arr.shape[2:])


def run_simulation_pmap(cfg: SimConfigFA, key: Optional[jax.Array] = None, verbose: bool = False):
    """Run simulation with multi-GPU pmap parallelism.

    If only 1 device is available, falls back to run_simulation_fa automatically.

    Multi-GPU strategy:
      1. Initialize particles on CPU (full N)
      2. Shard particles to [n_devices, N//n_devices]
      3. pmap over devices: each runs scatter → psum → Poisson → push
      4. Collect results via tree_map(unshard, ...)

    Args:
        cfg: SimConfigFA (N_particles will be rounded down to multiple of device count)
        key: JAX PRNGKey (optional, defaults to PRNGKey(0))
        verbose: print progress

    Returns:
        Same as run_simulation_fa: (diags, final_state, phi_history, aux)
    """
    n_devices = get_device_count()

    if n_devices == 1:
        # Single device: use standard runner
        if verbose:
            print(f"[pmap] 1 device detected — using standard run_simulation_fa")
        return run_simulation_fa(cfg, key=key, verbose=verbose)

    if verbose:
        print(f"[pmap] {n_devices} devices detected — using pmap parallelism")
        print(f"[pmap] Particles per device: {cfg.N_particles // n_devices}")

    # TODO: Implement full pmap runner when multi-GPU hardware is available.
    # The pmap step_fn signature will be:
    #
    #   @partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
    #   def pmap_step(carry_shard, xs):
    #       state_shard, phi, nan_flag = carry_shard
    #
    #       # Scatter: local particles → local delta_n
    #       delta_n_local = scatter_to_grid_fa(state_shard, geom, grid_shape)
    #
    #       # AllReduce: sum delta_n across all devices
    #       delta_n_global = jax.lax.psum(delta_n_local, axis_name='devices')
    #
    #       # Poisson solve: replicated on each device (same result)
    #       phi_new, _ = solve_poisson_fa(delta_n_global, ...)
    #
    #       # Gather + push: each device uses global phi for its particle shard
    #       state_shard_new = push_particles_and_weights_fa(state_shard, phi_new, ...)
    #
    #       return (state_shard_new, phi_new, nan_flag), diags
    #
    # For now, raise NotImplementedError with helpful message.
    raise NotImplementedError(
        f"pmap runner requires {n_devices} GPUs. "
        f"Full implementation pending hardware availability. "
        f"Use run_simulation_fa for single-GPU runs."
    )
