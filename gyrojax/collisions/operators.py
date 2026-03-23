"""
Collision operator implementations for GyroJAX δf PIC.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.particles.guiding_center import GCState


def apply_krook(state: GCState, nu: float, dt: float) -> GCState:
    """BGK/Krook: exponential weight relaxation.
    
    C_w[w] = -nu * w  =>  w_new = w * exp(-nu * dt)
    """
    new_weight = state.weight * jnp.exp(-nu * dt)
    return state._replace(weight=new_weight)


def apply_lorentz(
    state: GCState,
    B_p: jnp.ndarray,
    nu_ei: float,
    mi: float,
    Ti: float,
    dt: float,
    key: jax.random.PRNGKey,
) -> tuple:
    """Pitch-angle scattering via stochastic v∥ kicks.
    
    Implements the Lorentz (pitch-angle) collision operator in δf PIC
    via stochastic kicks to v∥, with deterministic weight correction.
    """
    vt_sq = Ti / mi
    v_perp_sq = 2.0 * state.mu * B_p / mi          # v⊥²
    v_sq = state.vpar**2 + v_perp_sq                # v²
    v = jnp.sqrt(jnp.maximum(v_sq, 1e-10))
    # Velocity-dependent collision rate: nu_D(v) = nu_ei * vt_ref^3 / v^3
    nu_D = nu_ei * vt_sq**1.5 / jnp.maximum(v**3, 1e-10)

    # Deterministic friction
    dv_friction = -nu_D * state.vpar * dt

    # Stochastic diffusion
    noise = jax.random.normal(key, shape=state.vpar.shape)
    dv_stoch = jnp.sqrt(jnp.maximum(2.0 * nu_D * v_perp_sq * dt, 0.0)) * noise

    # Weight correction for deterministic (friction) part only
    # dw = -(1-w) * (d ln f0 / d v∥) * dv_friction
    #    = -(1-w) * (-v∥ / vt²) * (-nu_D * v∥ * dt)
    #    = -(1-w) * nu_D * v∥² / vt² * dt
    dw = -(1.0 - state.weight) * (state.vpar / vt_sq) * dv_friction

    new_vpar = state.vpar + dv_friction + dv_stoch
    new_weight = state.weight + dw

    return state._replace(vpar=new_vpar, weight=new_weight), key


def apply_dougherty(
    state: GCState,
    B_p: jnp.ndarray,
    nu_coll: float,
    mi: float,
    Ti: float,
    dt: float,
) -> GCState:
    """Dougherty model Fokker-Planck — deterministic weight update.
    
    GX-compatible, conserves particles and energy.
    C_w[w] = nu * [(v∥²/vt² - 1) + (v⊥²/(2*vt²) - 1)] * (1 - w)
    """
    vt_sq = Ti / mi
    v_perp_sq = 2.0 * state.mu * B_p / mi

    # Dougherty weight increment (per-particle, via array ops)
    C_w = nu_coll * (
        (state.vpar**2 / vt_sq - 1.0) +          # parallel part
        (v_perp_sq / (2.0 * vt_sq) - 1.0)        # perpendicular part
    ) * (1.0 - state.weight)

    new_weight = state.weight + C_w * dt
    return state._replace(weight=new_weight)


def apply_collisions(
    state: GCState,
    B_p: jnp.ndarray,
    cfg,
    dt: float,
    key: jax.random.PRNGKey,
) -> tuple:
    """Dispatch to correct collision operator based on cfg.collision_model.
    
    Returns (new_state, new_key).
    """
    model = cfg.collision_model
    if model == 'none':
        return state, key
    elif model == 'krook':
        return apply_krook(state, cfg.nu_krook, dt), key
    elif model == 'lorentz':
        key, subkey = jax.random.split(key)
        new_state, _ = apply_lorentz(state, B_p, cfg.nu_ei, cfg.mi, cfg.Ti, dt, subkey)
        return new_state, key
    elif model == 'dougherty':
        return apply_dougherty(state, B_p, cfg.nu_coll, cfg.mi, cfg.Ti, dt), key
    else:
        raise ValueError(f"Unknown collision model: {model!r}")
