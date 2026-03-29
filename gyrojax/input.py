"""
TOML input file parser for GyroJAX.

Converts a .toml input file to the appropriate SimConfig dataclass.
Usage:
    from gyrojax.input import load_config
    cfg, raw = load_config("inputs/cbc.toml")

All SimConfigFA and SimConfigFullF fields are exposed via TOML sections:

  [run]          method, time_integrator
  [grid]         Npsi, Ntheta, Nalpha
  [particles]    N_particles, loading, vpar_cap, N_electrons
  [time]         dt, n_steps, checkpoint_interval
  [geometry]     R0, a, B0, q0, q1
  [physics]      Ti, Te, mi, rho_star (or e), vti, n0_avg,
                 R0_over_LT, R0_over_LTe, R0_over_Ln, beta
  [profiles]     LT_profile, Ln_profile, LT_profile_width,
                 krook_buffer_width, krook_buffer_rate
  [domain]       flux_tube (true=local, false=global), global_domain
  [init]         pert_amp, single_mode, k_mode, k_alpha_min, zonal_init
  [numerics]     collision_model, nu_krook, nu_ei, nu_coll,
                 use_weight_spread, weight_spread_interval, zonal_preserving_spread,
                 use_pullback, pullback_interval,
                 nu_soft, w_sat, soft_damp_alpha,
                 absorbing_wall, gyroaverage_scatter, use_radial_gaa,
                 fused_rk4, scatter_block_size, particle_shape,
                 picard_max_iter, picard_tol,
                 resample_interval (full-f), n_steps_warmup (full-f)
  [species]      electrons, me_over_mi, subcycles_e
  [output]       output_file

See inputs/cbc.toml for a fully-annotated example.
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

    method     = raw.get("run", {}).get("method",           "deltaf").lower()
    integrator = raw.get("run", {}).get("time_integrator",  "explicit").lower()

    if method == "deltaf":
        cfg = _build_deltaf_config(raw, integrator)
    elif method == "fullf":
        cfg = _build_fullf_config(raw)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'deltaf' or 'fullf'.")

    return cfg, raw


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(section: dict, key: str, default):
    """Return section[key] if present, else default."""
    return section[key] if key in section else default


def _rho_star_e(phys: dict):
    """Resolve rho_star / e from [physics] section."""
    if "rho_star" in phys:
        rho_star = float(phys["rho_star"])
        e        = 1.0 / rho_star
    elif "e" in phys:
        e        = float(phys["e"])
        rho_star = 1.0 / e
    else:
        rho_star = 1.0 / 180.0   # CBC default
        e        = 180.0
    return rho_star, e


# ── δf builder ────────────────────────────────────────────────────────────────

def _build_deltaf_config(raw: dict, integrator: str) -> SimConfigFA:
    """Map TOML sections → SimConfigFA fields (all fields covered)."""
    grid     = raw.get("grid",     {})
    parts    = raw.get("particles",{})
    time     = raw.get("time",     {})
    geom     = raw.get("geometry", {})
    phys     = raw.get("physics",  {})
    prof     = raw.get("profiles", {})
    domain   = raw.get("domain",   {})
    init     = raw.get("init",     {})
    num      = raw.get("numerics", {})
    species  = raw.get("species",  {})
    output   = raw.get("output",   {})

    # time integrator flags
    semi_implicit = (integrator == "semi_implicit")
    implicit      = (integrator == "implicit")
    use_cn        = semi_implicit or _get(num, "use_cn_weights", False)

    rho_star, e = _rho_star_e(phys)

    return SimConfigFA(
        # ── Grid ──────────────────────────────────────────────────────────────
        Npsi   = _get(grid, "Npsi",   32),
        Ntheta = _get(grid, "Ntheta", 64),
        Nalpha = _get(grid, "Nalpha", 32),

        # ── Particles ─────────────────────────────────────────────────────────
        N_particles       = _get(parts, "N_particles",       200_000),
        canonical_loading = (_get(parts, "loading", "maxwellian") == "canonical"),
        vpar_cap          = _get(parts, "vpar_cap",          4.0),
        N_electrons       = _get(parts, "N_electrons",       0),

        # ── Time ──────────────────────────────────────────────────────────────
        dt                  = _get(time, "dt",                  0.05),
        n_steps             = _get(time, "n_steps",             500),
        checkpoint_interval = _get(time, "checkpoint_interval", 0),

        # ── Geometry ──────────────────────────────────────────────────────────
        R0 = _get(geom, "R0", 1.0),
        a  = _get(geom, "a",  0.18),
        B0 = _get(geom, "B0", 1.0),
        q0 = _get(geom, "q0", 1.4),
        q1 = _get(geom, "q1", 0.5),

        # ── Physics ───────────────────────────────────────────────────────────
        Ti          = _get(phys, "Ti",          1.0),
        Te          = _get(phys, "Te",          1.0),
        mi          = _get(phys, "mi",          1.0),
        rho_star    = rho_star,
        e           = e,
        vti         = _get(phys, "vti",         1.0),
        n0_avg      = _get(phys, "n0_avg",      1.0),
        R0_over_LT  = _get(phys, "R0_over_LT",  6.9),
        R0_over_LTe = _get(phys, "R0_over_LTe", 6.9),
        R0_over_Ln  = _get(phys, "R0_over_Ln",  2.2),
        beta        = _get(phys, "beta",         0.0),

        # ── Profiles ──────────────────────────────────────────────────────────
        LT_profile           = _get(prof, "LT_profile",           "flat"),
        Ln_profile           = _get(prof, "Ln_profile",           "flat"),
        LT_profile_width     = _get(prof, "LT_profile_width",     0.5),
        krook_buffer_width   = _get(prof, "krook_buffer_width",   0.1),
        krook_buffer_rate    = _get(prof, "krook_buffer_rate",    1.0),

        # ── Domain ────────────────────────────────────────────────────────────
        # flux_tube=true (default) → local; flux_tube=false → global
        use_global    = not _get(domain, "flux_tube",    True),
        global_domain = _get(domain, "global_domain",    False),

        # ── Initialisation ────────────────────────────────────────────────────
        pert_amp    = _get(init, "pert_amp",    0.01),
        single_mode = _get(init, "single_mode", False),
        k_mode      = _get(init, "k_mode",      1),
        k_alpha_min = _get(init, "k_alpha_min", _get(num, "k_alpha_min", 0)),
        zonal_init  = _get(init, "zonal_init",  False),

        # ── Collisions ────────────────────────────────────────────────────────
        collision_model = _get(num, "collision_model", "none"),
        nu_krook        = _get(num, "nu_krook",        0.01),
        nu_ei           = _get(num, "nu_ei",           0.01),
        nu_coll         = _get(num, "nu_coll",         0.01),

        # ── Noise control ─────────────────────────────────────────────────────
        use_weight_spread          = _get(num, "use_weight_spread",          False),
        weight_spread_interval     = _get(num, "weight_spread_interval",     10),
        zonal_preserving_spread    = _get(num, "zonal_preserving_spread",    True),
        use_pullback               = _get(num, "use_pullback",               False),
        pullback_interval          = _get(num, "pullback_interval",          50),
        nu_soft                    = _get(num, "nu_soft",                    0.0),
        w_sat                      = _get(num, "w_sat",                      2.0),
        soft_damp_alpha            = _get(num, "soft_damp_alpha",            2),
        absorbing_wall             = _get(num, "absorbing_wall",             False),

        # ── Poisson / interpolation ────────────────────────────────────────────
        gyroaverage_scatter = _get(num, "gyroaverage_scatter", True),
        use_radial_gaa      = _get(num, "use_radial_gaa",      True),
        particle_shape      = _get(num, "particle_shape",      "cic"),
        scatter_block_size  = _get(num, "scatter_block_size",  0),

        # ── Performance ───────────────────────────────────────────────────────
        fused_rk4 = _get(num, "fused_rk4", True),

        # ── Implicit solver ───────────────────────────────────────────────────
        semi_implicit_weights = semi_implicit,
        use_cn_weights        = use_cn,
        implicit              = implicit,
        picard_max_iter       = _get(num, "picard_max_iter", 4),
        picard_tol            = _get(num, "picard_tol",      1e-3),

        # ── Electrons ─────────────────────────────────────────────────────────
        electron_model = _get(species, "electrons",   "adiabatic"),
        me_over_mi     = _get(species, "me_over_mi",  0.0005446623093681918),
        subcycles_e    = _get(species, "subcycles_e", 10),

        # ── Output ────────────────────────────────────────────────────────────
        output_file = _get(output, "output_file", ""),
    )


# ── full-f builder ────────────────────────────────────────────────────────────

def _build_fullf_config(raw: dict) -> SimConfigFullF:
    """Map TOML sections → SimConfigFullF fields (all fields covered)."""
    grid   = raw.get("grid",     {})
    parts  = raw.get("particles",{})
    time   = raw.get("time",     {})
    geom   = raw.get("geometry", {})
    phys   = raw.get("physics",  {})
    init   = raw.get("init",     {})
    num    = raw.get("numerics", {})

    rho_star, e = _rho_star_e(phys)

    return SimConfigFullF(
        # ── Grid ──────────────────────────────────────────────────────────────
        Npsi   = _get(grid, "Npsi",   16),
        Ntheta = _get(grid, "Ntheta", 32),
        Nalpha = _get(grid, "Nalpha", 32),

        # ── Particles ─────────────────────────────────────────────────────────
        N_particles = _get(parts, "N_particles", 200_000),
        vpar_cap    = _get(parts, "vpar_cap",    4.0),

        # ── Time ──────────────────────────────────────────────────────────────
        dt      = _get(time, "dt",      0.05),
        n_steps = _get(time, "n_steps", 200),

        # ── Geometry ──────────────────────────────────────────────────────────
        R0 = _get(geom, "R0", 1.0),
        a  = _get(geom, "a",  0.18),
        B0 = _get(geom, "B0", 1.0),
        q0 = _get(geom, "q0", 1.4),
        q1 = _get(geom, "q1", 0.5),

        # ── Physics ───────────────────────────────────────────────────────────
        Ti          = _get(phys, "Ti",          1.0),
        Te          = _get(phys, "Te",          1.0),
        mi          = _get(phys, "mi",          1.0),
        rho_star    = rho_star,
        e           = e,
        vti         = _get(phys, "vti",         1.0),
        n0_avg      = _get(phys, "n0_avg",      1.0),
        R0_over_LT  = _get(phys, "R0_over_LT",  6.9),
        R0_over_Ln  = _get(phys, "R0_over_Ln",  2.2),
        beta        = _get(phys, "beta",         0.0),

        # ── Initialisation ────────────────────────────────────────────────────
        pert_amp    = _get(init, "pert_amp",    1e-4),
        single_mode = _get(init, "single_mode", False),
        k_mode      = _get(init, "k_mode",      18),
        k_alpha_min = _get(init, "k_alpha_min", 0),

        # ── Numerics ──────────────────────────────────────────────────────────
        nu_krook          = _get(num, "nu_krook",          0.0),
        resample_interval = _get(num, "resample_interval", 50),
        n_steps_warmup    = _get(num, "n_steps_warmup",    0),
    )
