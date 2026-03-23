"""
Kinetic electron model for GyroJAX — Phase 4a.

Three electron models, selectable via config:

1. ADIABATIC (default, Phase 1-3):
   δn_e = (e·φ/Te) · n0    [Boltzmann response]
   Fast, no electron particles needed.

2. HYBRID (drift-kinetic electrons, no gyroaverage):
   Electrons treated as drift-kinetic (k⊥ρe → 0).
   Separate electron marker population, GC equations with
   me << mi (stiff — requires implicit or subcycling).

3. GYROKINETIC (full GK electrons):
   Full GK treatment for electrons. Not yet implemented.

The adiabatic model modifies the GK Poisson equation:
   [n0/Te · (1 - Γ0_i(b_i)) + n0/Ti · Γ0_i(b_i)] · eφ = δn_i(gyro)

Adiabatic electrons add the n0/Te term on the LHS:
   [n0/Te + n0/Ti · (Γ0_i - 1)] · eφ = δn_i

Which is exactly what solve_poisson_fa already implements! The "1 - Γ0"
term comes from the electron adiabatic response, and "ρi²k²" from ion FLR.

For drift-kinetic (DK) electrons, the electron weight equation is:
   dw_e/dt = -(1-w_e) · [vE · ∇ln(f0e) + (e_e/me)·E∥ · ∂ln(f0e)/∂v∥e]

Key difference from ions: electrons are much faster → need subcycling
(typical ratio v_te/v_ti = sqrt(mi/me) ~ 60 for hydrogen).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import jax
import jax.numpy as jnp

from gyrojax.particles.guiding_center import GCState


# ---------------------------------------------------------------------------
# Electron model selector
# ---------------------------------------------------------------------------

ElectronModel = Literal["adiabatic", "drift_kinetic"]


@dataclass
class ElectronConfig:
    """Configuration for electron physics."""
    model:      ElectronModel = "adiabatic"
    # For drift_kinetic:
    me_over_mi: float = 1.0 / 1836.0   # hydrogen mass ratio
    subcycles:  int   = 10              # electron subcycling steps per ion step
    Te:         float = 1.0
    vte:        float = 42.72           # vte/vti = sqrt(mi/me) ≈ 42.72 (H)
    N_electrons: int  = 0               # 0 = use same as ions


# ---------------------------------------------------------------------------
# Adiabatic electron density response (already in Poisson, provided for clarity)
# ---------------------------------------------------------------------------

def adiabatic_electron_density(phi: jnp.ndarray, n0: float, Te: float, e: float) -> jnp.ndarray:
    """
    Adiabatic (Boltzmann) electron density response.

    δn_e(x) = (e/Te) · n0 · φ(x)

    This is the density perturbation that goes into quasineutrality:
      δn_i = δn_e   →   solve_poisson_fa already uses this.

    Provided here for diagnostics and hybrid model coupling.
    """
    return (e / Te) * n0 * phi


# ---------------------------------------------------------------------------
# Drift-kinetic electron pusher (subcycled)
# ---------------------------------------------------------------------------

def push_electrons_dk(
    state_e: GCState,
    phi: jnp.ndarray,
    geom,
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    e_cfg: ElectronConfig,
    dt_ion: float,
) -> GCState:
    """
    Push drift-kinetic electrons for one ion timestep using subcycling.

    Electrons are pushed N_subcycles times per ion step with dt_e = dt_ion/N_subcycles.
    Only E×B drift (no gyroaverage for electrons since k⊥ρe ~ 0).

    Parameters
    ----------
    state_e   : electron GCState
    phi       : electrostatic potential (Npsi, Ntheta, Nalpha)
    geom      : FieldAlignedGeometry
    E_*       : electric field components at electron positions
    e_cfg     : ElectronConfig
    dt_ion    : ion timestep
    """
    from gyrojax.particles.guiding_center_fa import push_particles_fa

    dt_e = dt_ion / e_cfg.subcycles
    q_over_m_e = -1.0 / e_cfg.me_over_mi   # e_charge/me = -e/(me/mi * mi) = -1/me_over_mi

    state = state_e
    for _ in range(e_cfg.subcycles):
        state = push_particles_fa(
            state, E_psi, E_theta, E_alpha,
            geom, q_over_m_e, e_cfg.me_over_mi, dt_e,
        )
    return state


def init_electron_markers(
    N: int,
    geom,
    e_cfg: ElectronConfig,
    key: jax.random.PRNGKey,
) -> GCState:
    """
    Initialize drift-kinetic electron markers from Maxwellian.

    Uses same spatial distribution as ions, but electron thermal velocity.
    Works with both SAlphaGeometry and FieldAlignedGeometry.
    """
    from gyrojax.particles.guiding_center import init_maxwellian_particles
    from gyrojax.geometry.salpha import build_salpha_geometry

    # For FA geometry, build a thin s-α proxy just for particle sampling
    if hasattr(geom, 'psi_grid') and not hasattr(geom, 'r_grid'):
        Npsi   = geom.B_field.shape[0]
        Ntheta = geom.B_field.shape[1]
        Nalpha = geom.B_field.shape[2]
        geom_sa = build_salpha_geometry(
            Npsi, Ntheta, Nalpha,
            R0=geom.R0, a=geom.a, B0=geom.B0,
            q0=float(geom.q_profile[Npsi//2]), q1=0.0,
        )
        return init_maxwellian_particles(N, geom_sa, e_cfg.vte, e_cfg.Te, e_cfg.me_over_mi, key)
    return init_maxwellian_particles(N, geom, e_cfg.vte, e_cfg.Te, e_cfg.me_over_mi, key)


# ---------------------------------------------------------------------------
# Modified Poisson for drift-kinetic electrons
# ---------------------------------------------------------------------------

def solve_poisson_with_ke(
    delta_n_i: jnp.ndarray,
    delta_n_e: jnp.ndarray,
    geom,
    n0_avg: float,
    Te: float,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    GK Poisson with kinetic electron density.

    Quasineutrality: δn_i(gyro) = δn_e(drift-kinetic)

    In k-space:
      Γ0_i(b_i) · φ̂ · n0/Ti - φ̂ · n0/Te = (δn_i - δn_e) / e

    i.e. the electron density is subtracted from the RHS.
    This replaces the adiabatic approximation δn_e = (e/Te)·n0·φ.
    """
    from gyrojax.fields.poisson_fa import solve_poisson_fa
    # Effective δn for Poisson: δn_i - δn_e
    # (δn_e is computed from electron markers, not from Boltzmann)
    delta_n_eff = delta_n_i - delta_n_e
    return solve_poisson_fa(delta_n_eff, geom, n0_avg, Te, Ti, mi, e)
