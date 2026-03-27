"""
TOML input file parser for GyroJAX.

Converts a .toml input file to the appropriate SimConfig dataclass.
Usage:
    from gyrojax.input import load_config
    cfg, raw = load_config("inputs/cbc.toml")
"""

import tomllib  # Python 3.11+ stdlib
from pathlib import Path

from gyrojax.simulation_fa import SimConfigFA
from gyrojax.simulation_fullf import SimConfigFullF


def load_config(path: str | Path) -> tuple[object, dict]:
    """
    Parse a TOML input file and return (SimConfig, raw_dict).

    Returns SimConfigFA for method="deltaf", SimConfigFullF for method="fullf".
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    method = raw.get("run", {}).get("method", "deltaf").lower()
    integrator = raw.get("run", {}).get("time_integrator", "explicit").lower()

    if method == "deltaf":
        cfg = _build_deltaf_config(raw, integrator)
    elif method == "fullf":
        cfg = _build_fullf_config(raw)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'deltaf' or 'fullf'.")

    return cfg, raw


def _build_deltaf_config(raw: dict, integrator: str) -> SimConfigFA:
    """Map TOML sections → SimConfigFA fields."""
    grid    = raw.get("grid", {})
    parts   = raw.get("particles", {})
    time    = raw.get("time", {})
    geom    = raw.get("geometry", {})
    phys    = raw.get("physics", {})
    species = raw.get("species", {})
    domain  = raw.get("domain", {})
    init    = raw.get("init", {})
    num     = raw.get("numerics", {})

    # time integrator → boolean flags
    semi_implicit = (integrator == "semi_implicit")
    implicit      = (integrator == "implicit")

    # rho_star / e mapping
    if "rho_star" in phys:
        rho_star = phys["rho_star"]
        e = 1.0 / rho_star
    else:
        rho_star = SimConfigFA.rho_star if hasattr(SimConfigFA, "rho_star") else 1.0 / 180.0
        e = phys.get("e", 1000.0)

    return SimConfigFA(
        # Grid
        Npsi   = grid.get("Npsi",   SimConfigFA.__dataclass_fields__["Npsi"].default),
        Ntheta = grid.get("Ntheta", SimConfigFA.__dataclass_fields__["Ntheta"].default),
        Nalpha = grid.get("Nalpha", SimConfigFA.__dataclass_fields__["Nalpha"].default),
        # Particles
        N_particles       = parts.get("N_particles", 200_000),
        canonical_loading = (parts.get("loading", "maxwellian") == "canonical"),
        vpar_cap          = parts.get("vpar_cap", 4.0),
        # Time
        dt      = time.get("dt",      0.05),
        n_steps = time.get("n_steps", 200),
        # Geometry
        R0 = geom.get("R0", 1.0),
        a  = geom.get("a",  0.18),
        B0 = geom.get("B0", 1.0),
        q0 = geom.get("q0", 1.4),
        q1 = geom.get("q1", 0.5),
        # Physics
        Ti         = phys.get("Ti",         1.0),
        Te         = phys.get("Te",         1.0),
        mi         = phys.get("mi",         1.0),
        rho_star   = rho_star,
        e          = e,
        vti        = phys.get("vti",        1.0),
        n0_avg     = phys.get("n0_avg",     1.0),
        R0_over_LT = phys.get("R0_over_LT", 6.9),
        R0_over_Ln = phys.get("R0_over_Ln", 2.2),
        # Domain
        use_global = not domain.get("flux_tube", True),
        # Init
        pert_amp    = init.get("pert_amp",    1e-4),
        single_mode = init.get("single_mode", False),
        k_mode      = init.get("k_mode",      1),
        zonal_init  = init.get("zonal_init",  False),
        # Numerics — collisions
        collision_model        = num.get("collision_model",    "none"),
        nu_krook               = num.get("nu_krook",           0.005),
        # Numerics — noise control
        use_weight_spread      = num.get("use_weight_spread",  False),
        weight_spread_interval = num.get("weight_spread_interval", 10),
        use_pullback           = num.get("use_pullback",       False),
        pullback_interval      = num.get("pullback_interval",  50),
        nu_soft                = num.get("nu_soft",            0.0),
        k_alpha_min            = num.get("k_alpha_min",        0),
        # Implicit solver
        semi_implicit_weights  = semi_implicit,
        implicit               = implicit,
        picard_max_iter        = num.get("picard_max_iter",    4),
        picard_tol             = num.get("picard_tol",         1e-3),
        # Electrons
        electron_model         = species.get("electrons", "adiabatic"),
    )


def _build_fullf_config(raw: dict) -> SimConfigFullF:
    """Map TOML sections → SimConfigFullF fields."""
    grid   = raw.get("grid", {})
    parts  = raw.get("particles", {})
    time   = raw.get("time", {})
    geom   = raw.get("geometry", {})
    phys   = raw.get("physics", {})
    init   = raw.get("init", {})
    num    = raw.get("numerics", {})

    if "rho_star" in phys:
        rho_star = phys["rho_star"]
        e = 1.0 / rho_star
    else:
        rho_star = 1.0 / 180.0
        e = phys.get("e", 1000.0)

    return SimConfigFullF(
        # Grid
        Npsi   = grid.get("Npsi",   16),
        Ntheta = grid.get("Ntheta", 32),
        Nalpha = grid.get("Nalpha", 32),
        # Particles
        N_particles = parts.get("N_particles", 200_000),
        vpar_cap    = parts.get("vpar_cap", 4.0),
        # Time
        dt      = time.get("dt",      0.05),
        n_steps = time.get("n_steps", 200),
        # Geometry
        R0 = geom.get("R0", 1.0),
        a  = geom.get("a",  0.18),
        B0 = geom.get("B0", 1.0),
        q0 = geom.get("q0", 1.4),
        q1 = geom.get("q1", 0.5),
        # Physics
        Ti         = phys.get("Ti",         1.0),
        Te         = phys.get("Te",         1.0),
        mi         = phys.get("mi",         1.0),
        rho_star   = rho_star,
        e          = e,
        vti        = phys.get("vti",        1.0),
        n0_avg     = phys.get("n0_avg",     1.0),
        R0_over_LT = phys.get("R0_over_LT", 6.9),
        R0_over_Ln = phys.get("R0_over_Ln", 2.2),
        # Init
        pert_amp    = init.get("pert_amp",    1e-4),
        single_mode = init.get("single_mode", False),
        k_mode      = init.get("k_mode",      18),
        # Numerics
        nu_krook    = num.get("nu_krook",     0.0),
        k_alpha_min = num.get("k_alpha_min",  0),
        resample_interval = num.get("resample_interval", 50),
        n_steps_warmup    = num.get("n_steps_warmup",    0),
    )
