"""Physical constants and normalization utilities for GyroJAX."""

import jax
# Default float dtype — float32 for GPU performance
DTYPE = jax.numpy.float32

from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalConstants:
    """SI physical constants."""
    e: float = 1.602176634e-19    # elementary charge [C]
    mi: float = 1.6726219e-27     # proton mass [kg]
    me: float = 9.1093837e-31     # electron mass [kg]
    eps0: float = 8.8541878e-12   # vacuum permittivity [F/m]
    c: float = 2.99792458e8       # speed of light [m/s]


PHYS = PhysicalConstants()


@dataclass(frozen=True)
class NormalizationParams:
    """Normalization parameters for a gyrokinetic simulation.

    All quantities normalized to ion gyroradius ρ_i = v_ti / Ω_i
    and thermal velocity v_ti = sqrt(T_i / m_i).

    Attributes
    ----------
    R0 : float  Major radius [m]
    B0 : float  On-axis field [T]
    Ti0 : float Ion temperature on axis [eV]
    ni0 : float Ion density on axis [m^-3]
    """
    R0: float
    B0: float
    Ti0: float          # [eV]
    ni0: float          # [m^-3]
    mi: float = PHYS.mi
    e: float = PHYS.e

    @property
    def vti(self) -> float:
        """Ion thermal velocity sqrt(Ti/mi) [m/s]."""
        return float(jnp.sqrt(self.Ti0 * self.e / self.mi))

    @property
    def Omega_i(self) -> float:
        """Ion cyclotron frequency [rad/s]."""
        return float(self.e * self.B0 / self.mi)

    @property
    def rho_i(self) -> float:
        """Ion gyroradius vti/Omega_i [m]."""
        return self.vti / self.Omega_i

    @property
    def time_unit(self) -> float:
        """Time unit R0/vti [s]."""
        return self.R0 / self.vti


def gyroradius(mu: float, B: float, mi: float, e: float) -> float:
    """Compute thermal Larmor radius sqrt(2*mu*mi) / (e*B)."""
    return float(jnp.sqrt(2.0 * mu * mi) / (e * B))


def estimate_gpu_memory_mb(cfg) -> dict:
    """Estimate GPU memory usage for a simulation configuration.

    Args:
        cfg: SimConfigFA instance

    Returns:
        dict with memory estimates in MB for each component
    """
    N = cfg.N_particles
    G = cfg.Npsi * cfg.Ntheta * cfg.Nalpha
    bytes_per_float = 4  # float32

    # Particle state: r, theta, zeta, vpar, mu, weight
    state_mb = 6 * N * bytes_per_float / 1e6

    # RK4 intermediates: 4 stages × 5 equations (psi, theta, alpha, vpar, weight)
    rk4_mb = 4 * 5 * N * bytes_per_float / 1e6

    # Scatter peak: 1 corner at a time (after fix) + grid
    scatter_peak_mb = N * bytes_per_float / 1e6  # single (N,) corner array

    # Geometry interpolated at particles: B, gBpsi, gBth, kpsi, kth, g_aa
    geom_interp_mb = 6 * N * bytes_per_float / 1e6

    # Grid arrays: phi, delta_n, E_psi, E_theta, E_alpha, gyroavg buffer
    grid_mb = 6 * G * bytes_per_float / 1e6

    # lax.scan output stacking: DiagnosticsFA has 5 scalars per step
    scan_diag_mb = 5 * cfg.n_steps * bytes_per_float / 1e6

    peak_per_particle = (state_mb + rk4_mb + geom_interp_mb + scatter_peak_mb) / N

    peak_mb = state_mb + rk4_mb + geom_interp_mb + scatter_peak_mb + grid_mb

    def max_particles_for_budget(budget_mb: float) -> int:
        """Maximum N_particles that fits in given GPU memory budget (MB)."""
        available = budget_mb - grid_mb - scan_diag_mb
        return max(0, int(available / peak_per_particle))

    return {
        'state_mb': round(state_mb, 1),
        'rk4_peak_mb': round(rk4_mb, 1),
        'scatter_peak_mb': round(scatter_peak_mb, 1),
        'geom_interp_mb': round(geom_interp_mb, 1),
        'grid_mb': round(grid_mb, 1),
        'scan_diag_mb': round(scan_diag_mb, 3),
        'peak_estimate_mb': round(peak_mb, 1),
        'max_particles_for_budget': max_particles_for_budget,
    }


def configure_gpu(mem_fraction: float = 0.90, disable_tf32: bool = True):
    """Apply recommended JAX/XLA settings for GPU PIC simulation performance.

    Call this BEFORE importing JAX or at the top of simulation scripts:
        from gyrojax.utils import configure_gpu
        configure_gpu()

    Args:
        mem_fraction: fraction of GPU memory JAX can pre-allocate (default 0.90)
        disable_tf32: disable TF32 on Ampere+ GPUs for better numerical precision (default True)
    """
    import os
    import jax

    # Pre-allocate GPU memory pool: reduces fragmentation during long runs
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))

    # Threefry partitionable: enables pmap-compatible PRNG (better GPU parallelism)
    try:
        jax.config.update("jax_threefry_partitionable", True)
    except Exception:
        pass  # older JAX versions may not support this

    # Disable TF32: Ampere GPUs use TF32 by default (10-bit mantissa vs float32's 23-bit)
    # For PIC simulations, weight accumulation needs full float32 precision
    if disable_tf32:
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
