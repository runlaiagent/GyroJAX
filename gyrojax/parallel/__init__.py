"""Multi-GPU parallelism for GyroJAX using JAX pmap.

Particle sharding strategy:
  - N_particles split evenly across D = jax.device_count() GPUs
  - Each device owns a contiguous shard of particles
  - Grid (phi, delta_n) is replicated across all devices
  - AllReduce (psum) collects delta_n after scatter

Usage (multi-GPU):
    from gyrojax.parallel import run_simulation_pmap

    cfg = SimConfigFA(N_particles=200_000, ...)
    diags, state, phi, _ = run_simulation_pmap(cfg)

Single-GPU fallback:
    If jax.device_count() == 1, automatically falls back to run_simulation_fa.
"""

from gyrojax.parallel.pmap_runner import run_simulation_pmap, get_device_count, shard_particles

__all__ = ["run_simulation_pmap", "get_device_count", "shard_particles"]
