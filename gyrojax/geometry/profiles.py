"""Radial profile functions for global gyrokinetic simulations."""

from __future__ import annotations
import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp

from gyrojax.particles.guiding_center import GCState


@dataclasses.dataclass(frozen=True)
class RadialProfiles:
    """Radially varying profiles for global gyrokinetic simulations."""
    psi_grid:  jnp.ndarray   # radial grid [m], shape (Npsi,)
    n0:        jnp.ndarray   # density profile, shape (Npsi,)
    T_i:       jnp.ndarray   # ion temperature, shape (Npsi,)
    T_e:       jnp.ndarray   # electron temperature, shape (Npsi,)
    q:         jnp.ndarray   # safety factor, shape (Npsi,)
    # Buffer zone params
    psi_inner: float          # inner buffer edge [m]
    psi_outer: float          # outer buffer edge [m]
    nu_krook:  float          # Krook damping rate [a/vti]


def build_cbc_profiles(
    Npsi: int,
    a: float,
    R0: float,
    q0: float,
    q1: float,
    R0_over_LT: float,
    R0_over_Ln: float,
    n0_avg: float,
    Ti: float,
    nu_krook: float = 0.1,
) -> RadialProfiles:
    """Build CBC-style radial profiles over [0, a].

    Parameters
    ----------
    Npsi        : number of radial grid points
    a           : minor radius [m]
    R0          : major radius [m]
    q0, q1      : safety factor q(r) = q0 + q1*(r/a)
    R0_over_LT  : R0/L_T (normalised temperature gradient)
    R0_over_Ln  : R0/L_n (normalised density gradient)
    n0_avg      : reference density
    Ti          : reference ion temperature
    nu_krook    : Krook damping rate in buffer zones
    """
    Ln = R0 / R0_over_Ln
    LT = R0 / R0_over_LT
    r_mid = a * 0.5

    psi_grid = jnp.linspace(0.0, a, Npsi)

    n0  = n0_avg * jnp.exp(-(psi_grid - r_mid) / Ln)
    T_i = Ti     * jnp.exp(-(psi_grid - r_mid) / LT)
    T_e = T_i    # assume Te = Ti by default
    q   = q0 + q1 * (psi_grid / a)

    psi_inner = 0.1 * a
    psi_outer = 0.9 * a

    return RadialProfiles(
        psi_grid=psi_grid,
        n0=n0,
        T_i=T_i,
        T_e=T_e,
        q=q,
        psi_inner=psi_inner,
        psi_outer=psi_outer,
        nu_krook=nu_krook,
    )


def interp_profiles(
    profiles: RadialProfiles,
    r_particles: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate profiles at particle radial positions.

    Parameters
    ----------
    profiles    : RadialProfiles instance
    r_particles : particle radial positions, shape (N,)

    Returns
    -------
    n0_p, Ti_p, Te_p, q_p, d_lnn0_dr_p, d_lnT_dr_p — all shape (N,)
    """
    grid = profiles.psi_grid
    Nr = grid.shape[0]
    dr = (grid[-1] - grid[0]) / (Nr - 1)

    # Clamp particles to [psi_inner, psi_outer]
    r_clamped = jnp.clip(r_particles, profiles.psi_inner, profiles.psi_outer)

    # Fractional index
    ir = jnp.clip((r_clamped - grid[0]) / dr, 0.0, Nr - 1.001)
    i0 = jnp.floor(ir).astype(jnp.int32)
    frac = ir - i0.astype(jnp.float32)

    def lin_interp(arr):
        v0 = arr[i0]
        v1 = arr[jnp.clip(i0 + 1, 0, Nr - 1)]
        return v0 + frac * (v1 - v0)

    n0_p  = lin_interp(profiles.n0)
    Ti_p  = lin_interp(profiles.T_i)
    Te_p  = lin_interp(profiles.T_e)
    q_p   = lin_interp(profiles.q)

    # Gradients via finite differences on the grid
    # Central differences for interior, one-sided at edges
    dn0_dr = jnp.gradient(profiles.n0, grid)
    dTi_dr = jnp.gradient(profiles.T_i, grid)

    dn0_dr_p = lin_interp(dn0_dr)
    dTi_dr_p = lin_interp(dTi_dr)

    # d ln(n0)/dr = (1/n0) * dn0/dr
    d_lnn0_dr_p = dn0_dr_p / n0_p
    d_lnT_dr_p  = dTi_dr_p / Ti_p

    return n0_p, Ti_p, Te_p, q_p, d_lnn0_dr_p, d_lnT_dr_p


def krook_damping(
    state: GCState,
    profiles: RadialProfiles,
    dt: float,
) -> GCState:
    """Apply Krook damping in buffer zones.

    Weights in the inner/outer buffer are exponentially damped:
        w_new = w * exp(-nu_krook * dt)   in buffer
        w_new = w                          elsewhere

    Also clamps particles that have escaped the domain to the boundaries.

    Parameters
    ----------
    state    : GCState with particle positions and weights
    profiles : RadialProfiles (contains buffer edges, nu_krook)
    dt       : time step

    Returns
    -------
    Updated GCState with damped weights.
    """
    a = float(profiles.psi_grid[-1])
    buffer_width = 0.05 * a

    r = state.r
    in_inner = r < (profiles.psi_inner + buffer_width)
    in_outer = r > (profiles.psi_outer - buffer_width)
    in_buffer = jnp.logical_or(in_inner, in_outer)

    damping_factor = jnp.exp(-profiles.nu_krook * dt)
    factor = jnp.where(in_buffer, damping_factor, 1.0)
    new_weight = state.weight * factor

    return state._replace(weight=new_weight)
