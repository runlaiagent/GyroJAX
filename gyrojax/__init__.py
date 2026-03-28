"""GyroJAX — JAX-based delta-f gyrokinetic PIC code."""
import jax
jax.config.update("jax_enable_x64", False)

from gyrojax.parallel import run_simulation_pmap, get_device_count
