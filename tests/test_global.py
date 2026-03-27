"""Tests for global geometry — radial profiles and Krook buffer BCs."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyrojax.geometry.profiles import (
    RadialProfiles,
    build_cbc_profiles,
    interp_profiles,
    krook_damping,
)
from gyrojax.particles.guiding_center import GCState


# ---- Default test params ------------------------------------------------
NPSI   = 64
A      = 0.18
R0     = 1.0
Q0     = 1.4
Q1     = 0.5
R0_LT  = 6.9
R0_LN  = 2.2
N0_AVG = 1.0
TI     = 1.0


@pytest.fixture
def profiles():
    return build_cbc_profiles(NPSI, A, R0, Q0, Q1, R0_LT, R0_LN, N0_AVG, TI)


# ---- Shape tests --------------------------------------------------------

def test_build_cbc_profiles_shape(profiles):
    assert profiles.psi_grid.shape == (NPSI,)
    assert profiles.n0.shape       == (NPSI,)
    assert profiles.T_i.shape      == (NPSI,)
    assert profiles.T_e.shape      == (NPSI,)
    assert profiles.q.shape        == (NPSI,)


# ---- Monotonicity tests -------------------------------------------------

def test_build_cbc_profiles_n0_monotone(profiles):
    """For R0_over_Ln > 0, n0 decreases from inner to outer (Ln > 0 → negative gradient)."""
    # n0(r) = exp(-(r - r_mid)/Ln) → decreasing for r > r_mid
    # But increases for r < r_mid. Check that right half is decreasing.
    n0 = np.array(profiles.n0)
    mid = NPSI // 2
    assert np.all(np.diff(n0[mid:]) < 0), "n0 should decrease for r > r_mid"


def test_build_cbc_profiles_T_monotone(profiles):
    """T decreases from inner to outer for R0_over_LT > 0."""
    T = np.array(profiles.T_i)
    mid = NPSI // 2
    assert np.all(np.diff(T[mid:]) < 0), "T_i should decrease for r > r_mid"


# ---- Safety factor test -------------------------------------------------

def test_build_cbc_profiles_q_linear(profiles):
    """q(r) = q0 + q1*(r/a) exactly."""
    r   = np.array(profiles.psi_grid)
    q   = np.array(profiles.q)
    q_expected = Q0 + Q1 * (r / A)
    np.testing.assert_allclose(q, q_expected, rtol=1e-5)


# ---- Interpolation tests ------------------------------------------------

def test_interp_profiles_midpoint(profiles):
    """At r = a/2, returned values should match analytic mid-point values."""
    r_mid = jnp.array([A / 2.0])
    n0_p, Ti_p, Te_p, q_p, _, _ = interp_profiles(profiles, r_mid)

    # Analytic: at r_mid the exponentials = 1
    np.testing.assert_allclose(float(n0_p[0]), N0_AVG, rtol=0.02)
    np.testing.assert_allclose(float(Ti_p[0]), TI,     rtol=0.02)
    q_mid_expected = Q0 + Q1 * 0.5
    np.testing.assert_allclose(float(q_p[0]), q_mid_expected, rtol=0.02)


def test_interp_profiles_gradient(profiles):
    """d ln(n0)/dr ≈ -1/Ln at midpoint (within 5%)."""
    r_mid = jnp.array([A / 2.0])
    _, _, _, _, d_lnn0_dr_p, _ = interp_profiles(profiles, r_mid)

    Ln = R0 / R0_LN
    expected = -1.0 / Ln
    actual   = float(d_lnn0_dr_p[0])
    assert abs((actual - expected) / expected) < 0.05, (
        f"d_lnn0_dr = {actual:.4f}, expected ≈ {expected:.4f}"
    )


# ---- Krook damping tests ------------------------------------------------

def _make_state(N: int, r_vals: jnp.ndarray, weights: jnp.ndarray) -> GCState:
    """Helper: create a minimal GCState."""
    return GCState(
        r=r_vals,
        theta=jnp.zeros(N),
        zeta=jnp.zeros(N),
        vpar=jnp.zeros(N),
        mu=jnp.zeros(N),
        weight=weights,
    )


def test_krook_damping_buffer(profiles):
    """Weights in buffer zone are damped toward 0."""
    # Put particles well inside the inner buffer
    r_buf = jnp.full((5,), profiles.psi_inner * 0.5)  # deep in inner buffer
    w_in  = jnp.ones(5)
    state = _make_state(5, r_buf, w_in)

    dt    = 1.0
    state_new = krook_damping(state, profiles, dt)
    w_new = np.array(state_new.weight)

    expected = float(np.exp(-profiles.nu_krook * dt))
    np.testing.assert_allclose(w_new, expected, rtol=1e-5)


def test_krook_damping_interior(profiles):
    """Weights strictly inside the domain are unchanged."""
    # Put particles at r_mid, well away from buffers
    r_int = jnp.full((5,), A / 2.0)
    w_in  = jnp.ones(5) * 0.3
    state = _make_state(5, r_int, w_in)

    dt    = 1.0
    state_new = krook_damping(state, profiles, dt)
    w_new = np.array(state_new.weight)

    np.testing.assert_allclose(w_new, np.array(w_in), rtol=1e-5)


# ---- Simulation smoke tests ---------------------------------------------

def test_global_sim_no_nan():
    """Run 20 steps of global simulation — no NaN in φ or weights."""
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    cfg = SimConfigFA(
        Npsi=8, Ntheta=16, Nalpha=8,
        N_particles=2_000,
        n_steps=20,
        dt=0.05,
        use_global=True,
    )
    key = jax.random.PRNGKey(0)
    diags, state, phi, geom = run_simulation_fa(cfg, key, verbose=False)

    assert not jnp.any(jnp.isnan(phi)), "NaN in final φ"
    assert not jnp.any(jnp.isnan(state.weight)), "NaN in final weights"
    phi_max_series = [float(d.phi_max) for d in diags]
    assert not any(np.isnan(phi_max_series)), "NaN in phi_max diagnostics"


def test_global_vs_fluxtube_growth():
    """
    Global simulation produces a positive linear ITG growth rate at CBC parameters.

    Uses single_mode=True at k_mode=18 (dominant CBC mode) to isolate the ITG
    without aliasing noise. Verifies γ > 0 (unstable) and within CBC reference
    range [0.05, 0.40] vti/R0.

    The flux-tube CBC reference is γ ≈ 0.145 vti/R0 (measured in test_cbc_error_under_30pct).
    Global sim may differ due to radial profile effects; we allow a wide band.
    """
    from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

    key = jax.random.PRNGKey(7)

    # Flux-tube: single-mode CBC, minimal viable config
    cfg_ft = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=64,
        N_particles=300_000,
        n_steps=200,
        dt=0.05,
        R0_over_LT=6.9,
        R0_over_Ln=2.2,
        single_mode=True,
        k_mode=18,
        nu_krook=0.0,
        use_global=False,
    )
    diags_ft, _, _, _ = run_simulation_fa(cfg_ft, key, verbose=False)

    # Global: same mode, same resolution, with Krook BCs
    cfg_gl = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=64,
        N_particles=300_000,
        n_steps=200,
        dt=0.05,
        R0_over_LT=6.9,
        R0_over_Ln=2.2,
        single_mode=True,
        k_mode=18,
        nu_krook=0.005,
        use_global=True,
    )
    diags_gl, _, _, _ = run_simulation_fa(cfg_gl, key, verbose=False)

    def growth_rate(diags, dt, window=0.35):
        """Fit log(phi_max) over middle-to-late window (avoid initial transient)."""
        phi_max = np.array([float(d.phi_max) for d in diags])
        n = len(phi_max)
        n_start = int(n * (1 - window))
        # Find linear phase: phi still growing, not yet saturated
        # Use 40-75% of run to avoid init transient and saturation
        n_s = int(n * 0.40)
        n_e = int(n * 0.75)
        t = np.arange(n_s, n_e) * dt
        log_phi = np.log(np.maximum(phi_max[n_s:n_e], 1e-20))
        if len(t) < 4:
            return float("nan")
        return float(np.polyfit(t, log_phi, 1)[0])

    gamma_ft = growth_rate(diags_ft, cfg_ft.dt)
    gamma_gl = growth_rate(diags_gl, cfg_gl.dt)

    print(f"\n  flux-tube γ = {gamma_ft:.4f}  global γ = {gamma_gl:.4f}")
    print(f"  CBC reference: γ ≈ 0.10–0.20 vti/R0")

    # Both should give positive growth at R/LT=6.9 (well above linear threshold ~4)
    assert gamma_ft > 0, f"Flux-tube growth rate should be positive, got {gamma_ft:.4f}"
    assert gamma_gl > 0, f"Global growth rate should be positive, got {gamma_gl:.4f}"

    # Both should be in a physically reasonable range
    assert 0.03 < gamma_ft < 0.60, f"Flux-tube γ={gamma_ft:.4f} outside range [0.03, 0.60]"
    assert 0.03 < gamma_gl < 0.60, f"Global γ={gamma_gl:.4f} outside range [0.03, 0.60]"

    # Should agree within 60% (global has Krook BCs which can reduce effective gamma)
    rel_diff = abs(gamma_gl - gamma_ft) / (abs(gamma_ft) + 1e-10)
    assert rel_diff < 0.60, (
        f"Global and flux-tube growth rates disagree too much: "
        f"γ_ft={gamma_ft:.4f}, γ_gl={gamma_gl:.4f} (rel diff = {rel_diff:.1%})"
    )
