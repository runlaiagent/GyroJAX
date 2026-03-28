"""Post-processing for GyroJAX simulation output.

Reads HDF5 checkpoint files and computes derived physics quantities.
Fully decoupled from simulation code — no JAX required.

Example usage:
    from gyrojax.io.postprocess import PostProcessor

    pp = PostProcessor("run.h5")
    gamma = pp.growth_rate(t_start=50, t_end=150)
    chi_i = pp.heat_flux_chi()
    pp.plot_growth(save="growth.png")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class PostProcessor:
    """Load and analyze a GyroJAX HDF5 output file.

    Args:
        path: path to .h5 file produced by run_simulation_fa or run_long_simulation_fa
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        """Load all diagnostics from HDF5."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for post-processing: pip install h5py")

        data = {}
        with h5py.File(self.path, "r") as f:
            # Load diagnostic time series
            if "diags" in f:
                for key in f["diags"]:
                    data[key] = np.array(f["diags"][key])
            # Load config if present
            if "config" in f:
                data["config"] = dict(f["config"].attrs)
            # Load phi snapshots if present
            if "phi" in f:
                data["phi"] = np.array(f["phi"])
        return data

    @property
    def phi_rms(self) -> np.ndarray:
        return self._data.get("phi_rms", np.array([]))

    @property
    def phi_max(self) -> np.ndarray:
        return self._data.get("phi_max", np.array([]))

    @property
    def weight_rms(self) -> np.ndarray:
        return self._data.get("weight_rms", np.array([]))

    @property
    def n_steps(self) -> int:
        return len(self.phi_max)

    def time_axis(self, dt: float = None) -> np.ndarray:
        """Return time array. Uses dt from config if not specified."""
        if dt is None:
            dt = self._data.get("config", {}).get("dt", 0.05)
        return np.arange(self.n_steps) * float(dt)

    def growth_rate(self, t_start: int = None, t_end: int = None) -> float:
        """Estimate linear growth rate γ from phi_max time series.

        Uses log-linear fit: ln(phi_max) = γ*t + const

        Args:
            t_start: start step index (default: first quarter)
            t_end: end step index (default: third quarter, before saturation)

        Returns:
            γ in simulation time units
        """
        n = self.n_steps
        if t_start is None:
            t_start = n // 4
        if t_end is None:
            t_end = 3 * n // 4

        phi = self.phi_max[t_start:t_end]
        phi = phi[phi > 0]
        if len(phi) < 3:
            return float("nan")

        dt = float(self._data.get("config", {}).get("dt", 0.05))
        t = np.arange(len(phi)) * dt

        # Linear fit to log(phi)
        coeffs = np.polyfit(t, np.log(phi), 1)
        return float(coeffs[0])

    def heat_flux_chi(self, saturation_start: int = None) -> float:
        """Estimate ion heat flux χᵢ from saturated phi_max.

        Uses chi_i ~ <phi_max²> in the saturated phase.

        Args:
            saturation_start: step index where saturation begins (default: last half)

        Returns:
            χᵢ estimate (normalized)
        """
        n = self.n_steps
        if saturation_start is None:
            saturation_start = n // 2

        phi_sat = self.phi_max[saturation_start:]
        if len(phi_sat) == 0:
            return float("nan")
        return float(np.mean(phi_sat**2))

    def zonal_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute zonal flow energy spectrum from phi snapshots.

        Returns:
            (k_alpha, E_k): wavenumber array and energy spectrum
            Returns (None, None) if no phi snapshots available.
        """
        if "phi" not in self._data:
            return None, None

        phi = self._data["phi"]  # shape: (n_snapshots, Npsi, Ntheta, Nalpha)
        if phi.ndim < 4:
            return None, None

        # Average over r, theta → zonal phi(alpha)
        phi_zonal = phi.mean(axis=(1, 2))  # (n_snapshots, Nalpha)
        phi_zonal_mean = phi_zonal.mean(axis=0)  # (Nalpha,)

        # FFT over alpha
        Nal = phi_zonal_mean.shape[0]
        fft = np.abs(np.fft.rfft(phi_zonal_mean))**2
        k_alpha = np.fft.rfftfreq(Nal) * Nal

        return k_alpha, fft

    def weight_pdf(self, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return histogram of weight_rms evolution.

        High weight_rms signals δf validity breakdown (weights too large).

        Returns:
            (time, weight_rms) arrays
        """
        return self.time_axis(), self.weight_rms

    def summary(self) -> dict:
        """Return a summary dict of key physics quantities."""
        return {
            "n_steps": self.n_steps,
            "growth_rate": self.growth_rate(),
            "chi_i": self.heat_flux_chi(),
            "phi_max_final": float(self.phi_max[-1]) if self.n_steps > 0 else None,
            "weight_rms_final": float(self.weight_rms[-1]) if self.n_steps > 0 else None,
            "has_phi_snapshots": "phi" in self._data,
        }

    def plot_growth(self, save: str = None, show: bool = False):
        """Plot phi_max vs time with growth rate annotation."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        t = self.time_axis()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # phi_max
        ax = axes[0]
        ax.semilogy(t, self.phi_max, 'b-', lw=1.5, label=r'$|\phi|_{max}$')
        gamma = self.growth_rate()
        if not np.isnan(gamma):
            ax.set_title(f"Growth rate γ = {gamma:.3f}")
        ax.set_xlabel("Time (a/vti)")
        ax.set_ylabel(r"$|\phi|_{max}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # weight_rms
        ax = axes[1]
        ax.plot(t, self.weight_rms, 'r-', lw=1.5, label=r'$\langle w^2\rangle^{1/2}$')
        ax.set_xlabel("Time (a/vti)")
        ax.set_ylabel("Weight RMS")
        ax.set_title("δf validity monitor")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save}")
        if show:
            plt.show()
        plt.close()

    def plot_zonal(self, save: str = None, show: bool = False):
        """Plot zonal flow energy spectrum."""
        k, E = self.zonal_spectrum()
        if k is None:
            print("No phi snapshots available for zonal spectrum")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(k[1:], E[1:], 'g-o', ms=4)
        ax.set_xlabel(r"$k_\alpha$")
        ax.set_ylabel(r"$E(k_\alpha)$")
        ax.set_title("Zonal flow energy spectrum")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


def load_results(path: Union[str, Path]) -> PostProcessor:
    """Convenience function: load a GyroJAX HDF5 file."""
    return PostProcessor(path)
