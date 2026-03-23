"""
GyroJAX Visualization — Phase 4

Rich plots for simulation analysis:
  - Growth rate panel (log|φ| vs t with fit line)
  - Mode structure (φ in poloidal cross-section)
  - Zonal flow profile and shear
  - Perpendicular energy spectrum k⊥ρi
  - Ion heat flux radial profile
  - Particle phase space (r, v∥)
  - Full dashboard (all panels combined)
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from typing import List, Optional


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f8f8",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
    })
    return plt, gridspec


# ---------------------------------------------------------------------------
# 1. Growth rate panel
# ---------------------------------------------------------------------------

def plot_growth_rate(
    phi_max: np.ndarray,
    phi_rms: np.ndarray,
    dt: float,
    gamma_measured: float,
    gamma_target: float = 0.17,
    title: str = "CBC Growth Rate",
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    t = np.arange(len(phi_max)) * dt
    ax.semilogy(t, phi_max + 1e-20, "b-",  lw=1.5, label=r"$|\phi|_\mathrm{max}$")
    ax.semilogy(t, phi_rms + 1e-20, "r--", lw=1.2, label=r"$|\phi|_\mathrm{rms}$", alpha=0.8)

    # Growth rate fit line over last 40%
    n = len(phi_max)
    n0 = int(n * 0.6)
    t_fit = t[n0:]
    phi_ref = phi_max[n0]
    fit_line = phi_ref * np.exp(gamma_measured * (t_fit - t_fit[0]))
    ax.semilogy(t_fit, fit_line, "g-", lw=2.5, alpha=0.7,
                label=rf"fit $\gamma={gamma_measured:.3f}$ (target {gamma_target:.3f})")

    err = abs(gamma_measured - gamma_target) / gamma_target * 100
    color = "#2ecc71" if err < 15 else "#e74c3c"
    ax.set_title(f"{title}  [error={err:.1f}%]", color=color, fontweight="bold")
    ax.set_xlabel(r"$t\,[R_0/v_{ti}]$")
    ax.set_ylabel(r"$|\phi|$")
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        return fig
    return ax


# ---------------------------------------------------------------------------
# 2. Mode structure (poloidal cross-section)
# ---------------------------------------------------------------------------

def plot_mode_structure(
    phi: np.ndarray,
    geom,
    psi_idx: int = None,
    title: str = r"$\phi(\theta, \alpha)$",
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    if psi_idx is None:
        psi_idx = phi.shape[0] // 2

    slice_2d = np.array(phi[psi_idx, :, :])   # (Ntheta, Nalpha)
    vmax = np.max(np.abs(slice_2d)) + 1e-20

    im = ax.imshow(slice_2d, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0, 2*np.pi, 0, 2*np.pi])
    ax.set_xlabel(r"$\alpha$ (binormal)")
    ax.set_ylabel(r"$\theta$ (poloidal)")
    r_val = float(geom.psi_grid[psi_idx])
    ax.set_title(f"{title}  [r={r_val:.3f}]")
    if standalone:
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return ax, im


# ---------------------------------------------------------------------------
# 3. Zonal flow profile
# ---------------------------------------------------------------------------

def plot_zonal_flow(
    phi: np.ndarray,
    geom,
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    from gyrojax.diagnostics import extract_zonal_flow, extract_nonzonal

    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        axes = ax

    psi = np.array(geom.psi_grid)
    phi_zf = np.array(extract_zonal_flow(jnp.array(phi)))
    phi_turb_rms = np.array(jnp.sqrt(jnp.mean(
        jnp.array(extract_nonzonal(jnp.array(phi)))**2, axis=(1, 2)
    )))

    axes[0].plot(psi, phi_zf, "b-", lw=2, label="Zonal flow")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel(r"$\psi$ (r)")
    axes[0].set_ylabel(r"$\langle\phi\rangle_{\theta,\alpha}$")
    axes[0].set_title("Zonal Flow Profile")
    axes[0].legend()

    axes[1].plot(psi, phi_turb_rms, "r-", lw=2, label="Turbulent rms")
    axes[1].set_xlabel(r"$\psi$ (r)")
    axes[1].set_ylabel(r"$|\phi_\mathrm{turb}|_\mathrm{rms}(\psi)$")
    axes[1].set_title("Turbulent Amplitude Profile")
    axes[1].legend()

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ---------------------------------------------------------------------------
# 4. Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    phi: np.ndarray,
    geom,
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    from gyrojax.diagnostics import perp_spectrum, parallel_spectrum

    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        axes = ax

    k_perp, E_perp = perp_spectrum(jnp.array(phi), geom)
    k_par,  E_par  = parallel_spectrum(jnp.array(phi))

    k_perp, E_perp = np.array(k_perp), np.array(E_perp) + 1e-20
    k_par,  E_par  = np.array(k_par),  np.array(E_par)  + 1e-20

    axes[0].semilogy(k_perp[1:], E_perp[1:], "b-o", ms=4, lw=1.5)
    axes[0].set_xlabel(r"$k_\perp \rho_i$")
    axes[0].set_ylabel(r"$E(k_\perp)$")
    axes[0].set_title(r"Perpendicular Spectrum $|\hat\phi(k_\perp)|^2$")

    axes[1].semilogy(k_par[1:], E_par[1:], "r-o", ms=4, lw=1.5)
    axes[1].set_xlabel(r"$k_\parallel R_0$")
    axes[1].set_ylabel(r"$E(k_\parallel)$")
    axes[1].set_title(r"Parallel Spectrum $|\hat\phi(k_\parallel)|^2$")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ---------------------------------------------------------------------------
# 5. Heat flux
# ---------------------------------------------------------------------------

def plot_heat_flux(
    Q_profiles: List[np.ndarray],
    geom,
    t_labels: List[str] = None,
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    psi = np.array(geom.psi_grid)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Q_profiles)))

    for i, Q in enumerate(Q_profiles):
        label = t_labels[i] if t_labels else f"t={i}"
        ax.plot(psi, Q, color=colors[i], lw=1.5, label=label)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\psi$ (r)")
    ax.set_ylabel(r"$Q_i(\psi)$")
    ax.set_title("Ion Heat Flux Profile")
    if len(Q_profiles) <= 6:
        ax.legend(fontsize=8)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ---------------------------------------------------------------------------
# 6. Particle phase space
# ---------------------------------------------------------------------------

def plot_phase_space(
    state,
    geom,
    n_sample: int = 5000,
    ax=None,
    save_path: str = None,
):
    plt, _ = _get_plt()
    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        axes = ax

    N = len(state.r)
    idx = np.random.choice(N, min(n_sample, N), replace=False)

    r    = np.array(state.r[idx])
    vpar = np.array(state.vpar[idx])
    w    = np.array(state.weight[idx])

    sc = axes[0].scatter(r, vpar, c=w, cmap="RdBu_r",
                         s=1, alpha=0.4,
                         vmin=-np.percentile(np.abs(w), 95),
                         vmax= np.percentile(np.abs(w), 95))
    axes[0].set_xlabel(r"$\psi$ (r)")
    axes[0].set_ylabel(r"$v_\parallel / v_{ti}$")
    axes[0].set_title("Phase Space  (colored by weight)")
    plt.colorbar(sc, ax=axes[0], shrink=0.8, label="w")

    axes[1].hist(vpar, bins=50, color="steelblue", edgecolor="white", lw=0.3)
    axes[1].set_xlabel(r"$v_\parallel / v_{ti}$")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Parallel Velocity Distribution")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ---------------------------------------------------------------------------
# 7. Full dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    phi_max: np.ndarray,
    phi_rms: np.ndarray,
    dt: float,
    gamma_measured: float,
    phi_final: np.ndarray,
    state_final,
    geom,
    Q_profiles: List[np.ndarray] = None,
    t_labels: List[str] = None,
    gamma_target: float = 0.17,
    title: str = "GyroJAX Simulation Dashboard",
    save_path: str = "gyrojax_dashboard.png",
):
    """
    Full 6-panel dashboard:
      [growth rate]  [mode structure]
      [zonal flow]   [spectrum]
      [heat flux]    [phase space]
    """
    plt, gridspec = _get_plt()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    # Panel 1: growth rate
    ax1 = fig.add_subplot(gs[0, 0])
    plot_growth_rate(phi_max, phi_rms, dt, gamma_measured, gamma_target, ax=ax1)

    # Panel 2: mode structure
    ax2 = fig.add_subplot(gs[0, 1])
    plot_mode_structure(np.array(phi_final), geom, ax=ax2)

    # Panel 3: zonal flow
    ax3a = fig.add_subplot(gs[1, 0])
    ax3b = fig.add_subplot(gs[1, 1])
    plot_zonal_flow(np.array(phi_final), geom, ax=[ax3a, ax3b])

    # Panel 4: spectrum (reuse row 1 right)
    ax4a = fig.add_subplot(gs[2, 0])
    ax4b = fig.add_subplot(gs[2, 1])
    plot_spectrum(np.array(phi_final), geom, ax=[ax4a, ax4b])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Dashboard saved: {save_path}")
    return fig
