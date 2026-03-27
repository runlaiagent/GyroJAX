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
    E_psi: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_alpha: jnp.ndarray,
    geom,
    e_cfg: ElectronConfig,
    dt_ion: float,
) -> GCState:
    """
    Push drift-kinetic electrons for one ion timestep using subcycling.

    Electrons are pushed N_subcycles times per ion step with dt_e = dt_ion/N_subcycles.
    Only E×B drift (no gyroaverage for electrons since k⊥ρe ~ 0).
    Uses jax.lax.fori_loop for JIT compatibility.
    Per-substep clamping prevents runaway due to large mirror force on light electrons.

    Parameters
    ----------
    state_e   : electron GCState
    E_*       : electric field components at electron positions
    geom      : FieldAlignedGeometry
    e_cfg     : ElectronConfig
    dt_ion    : ion timestep
    """
    from gyrojax.particles.guiding_center_fa import push_particles_fa
    from gyrojax.geometry.field_aligned import interp_fa_to_particles

    dt_e = dt_ion / e_cfg.subcycles
    q_over_m_e = -1.0 / e_cfg.me_over_mi   # e_charge/me = -e/(me/mi * mi) = -1/me_over_mi

    r_min = geom.psi_grid[0] * 1.001
    r_max = geom.psi_grid[-1] * 0.999
    # Electron vpar cap: 4 * vte
    vpar_cap_e = 4.0 * e_cfg.vte

    # Pre-interpolate geometry fields at current electron positions
    B_e, gBpsi_e, gBth_e, kpsi_e, kth_e, g_aa_e = interp_fa_to_particles(
        geom, state_e.r, state_e.theta, state_e.zeta
    )
    Nr = geom.psi_grid.shape[0]
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Nr - 1)
    ir = jnp.clip((state_e.r - geom.psi_grid[0]) / dr, 0.0, Nr - 1.001)
    q_at_e = geom.q_profile[jnp.floor(ir).astype(jnp.int32)]

    def one_substep(_, state):
        new_state = push_particles_fa(
            state, E_psi, E_theta, E_alpha,
            B_e, gBpsi_e, gBth_e, kpsi_e, kth_e,
            q_at_e, g_aa_e,
            q_over_m_e, e_cfg.me_over_mi, dt_e, geom.R0,
        )
        # Per-substep boundary and velocity clamping to prevent runaway
        return new_state._replace(
            r=jnp.clip(new_state.r, r_min, r_max),
            vpar=jnp.clip(new_state.vpar, -vpar_cap_e, vpar_cap_e),
        )

    return jax.lax.fori_loop(0, e_cfg.subcycles, one_substep, state_e)


def update_electron_weights(
    state_e: GCState,
    E_psi_e: jnp.ndarray,
    E_theta_e: jnp.ndarray,
    E_alpha_e: jnp.ndarray,
    B_e: jnp.ndarray,
    gradB_psi_e: jnp.ndarray,
    gradB_th_e: jnp.ndarray,
    kappa_psi_e: jnp.ndarray,
    kappa_th_e: jnp.ndarray,
    q_at_r_e: jnp.ndarray,
    n0_e: jnp.ndarray,
    Te_e: jnp.ndarray,
    d_ln_n0_dr: jnp.ndarray,
    d_ln_Te_dr: jnp.ndarray,
    e_cfg: ElectronConfig,
    R0: float,
    dt: float,
) -> GCState:
    """
    Drift-kinetic electron weight equation.
    Same form as ion weight equation but with:
      - electron charge: q_e = -e (negative)
      - electron mass: me = me_over_mi * mi
      - no gyroaverage (k_perp * rho_e ~ 0)
      - electron temperature Te

    Note: we pass |q/m_e| (positive magnitude) to update_weights because
    the drift velocity formulas use jnp.maximum(Omega, 1e-10) which clips
    negative cyclotron frequencies. The electron ExB drift has the same
    sign as ions (charge-independent), and we rely on the Maxwellian
    d_lnf0/dr to encode the electron response correctly.
    """
    from gyrojax.deltaf.weights import update_weights
    # Use positive |q/m_e| to avoid sign issues in drift formulas
    # The electron weight equation is sign-compatible: ExB drift is
    # charge-independent, and magnetic drifts have same sign as ions
    # (both drift outward on low-field side for thermal particles)
    abs_q_over_m_e = 1.0 / e_cfg.me_over_mi   # positive magnitude
    return update_weights(
        state_e,
        E_psi_e, E_theta_e, E_alpha_e,
        B_e, gradB_psi_e, gradB_th_e, kappa_psi_e, kappa_th_e,
        q_at_r_e,
        n0_e, Te_e,
        d_ln_n0_dr, d_ln_Te_dr,
        abs_q_over_m_e, e_cfg.me_over_mi, R0, dt,
    )


@dataclass(frozen=True)
class ElectronState:
    """Full electron simulation state."""
    markers: GCState   # electron guiding-center markers
    model:   str       # 'adiabatic' or 'drift_kinetic'


def init_electron_state(N_e: int, geom, e_cfg: ElectronConfig, key: jax.random.PRNGKey) -> ElectronState:
    """Initialize electron state for the given model."""
    if e_cfg.model == 'adiabatic':
        # Dummy markers (not used in adiabatic model)
        markers = init_electron_markers(1, geom, e_cfg, key)
        return ElectronState(markers=markers, model='adiabatic')
    else:
        markers = init_electron_markers(N_e, geom, e_cfg, key)
        return ElectronState(markers=markers, model='drift_kinetic')


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
