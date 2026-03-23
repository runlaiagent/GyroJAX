"""
Tests for Phase 3: full-f simulation.

Smoke tests — verify no NaN, proper density behavior, and that
phi grows (ITG mode seeded by position perturbation).
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from gyrojax.simulation_fullf import SimConfigFullF, run_simulation_fullf


@pytest.fixture(scope="module")
def fullf_result():
    cfg = SimConfigFullF(
        Npsi=16, Ntheta=32, Nalpha=16,
        N_particles=10_000,
        n_steps=30,
        dt=0.05,
        R0=1.0, a=0.18, B0=1.0,
        q0=1.4, q1=0.5,
        Ti=1.0, Te=1.0, mi=1.0, e=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        vti=1.0, n0_avg=1.0,
        resample_interval=0,
    )
    key = jax.random.PRNGKey(7)
    return run_simulation_fullf(cfg, key=key, verbose=False)


class TestFullFSmoke:

    def test_no_nan_phi(self, fullf_result):
        _, _, phi, _ = fullf_result
        assert not bool(jnp.any(jnp.isnan(phi))), "NaN in phi"

    def test_no_inf_phi(self, fullf_result):
        _, _, phi, _ = fullf_result
        assert not bool(jnp.any(jnp.isinf(phi))), "Inf in phi"

    def test_no_nan_positions(self, fullf_result):
        _, state, _, _ = fullf_result
        for f in (state.r, state.theta, state.zeta, state.vpar):
            assert not bool(jnp.any(jnp.isnan(f))), "NaN in particle state"

    def test_particles_in_bounds(self, fullf_result):
        _, state, _, geom = fullf_result
        assert bool(jnp.all(state.r >= float(geom.psi_grid[0]) * 0.99))
        assert bool(jnp.all(state.r <= float(geom.psi_grid[-1]) * 1.01))

    def test_weights_constant(self, fullf_result):
        """Full-f: weights should remain constant (no weight equation)."""
        _, state, _, _ = fullf_result
        W = state.weight
        W_std = float(jnp.std(W))
        W_mean = float(jnp.mean(W))
        # All weights should be equal (unit weight) — std/mean << 1
        assert W_std / (W_mean + 1e-30) < 1e-3, (
            f"Weights changed! std/mean = {W_std/W_mean:.3e} (should be ~0)"
        )

    def test_diag_length(self, fullf_result):
        diags, _, _, _ = fullf_result
        assert len(diags) == 30

    def test_phi_nonzero(self, fullf_result):
        diags, _, _, _ = fullf_result
        assert float(diags[-1].phi_max) > 1e-20

    def test_n_rms_finite(self, fullf_result):
        diags, _, _, _ = fullf_result
        for d in diags:
            assert not np.isnan(float(d.n_rms))


class TestFullFResampling:

    def test_resample_runs(self):
        """Verify resampling doesn't crash."""
        cfg = SimConfigFullF(
            Npsi=16, Ntheta=32, Nalpha=16,
            N_particles=5_000,
            n_steps=15,
            dt=0.05,
            R0=1.0, a=0.18, B0=1.0,
            q0=1.4, q1=0.5,
            Ti=1.0, Te=1.0, mi=1.0, e=1.0,
            R0_over_LT=6.9, R0_over_Ln=2.2,
            vti=1.0, n0_avg=1.0,
            resample_interval=5,
        )
        key = jax.random.PRNGKey(0)
        diags, state, phi, _ = run_simulation_fullf(cfg, key=key, verbose=False)
        assert len(diags) == 15
        assert not bool(jnp.any(jnp.isnan(phi)))

    def test_resample_equalizes_weights(self):
        """After resampling, all weights should be equal."""
        from gyrojax.simulation_fullf import _resample_particles
        from gyrojax.geometry.field_aligned import build_field_aligned_geometry
        from gyrojax.particles.guiding_center import GCState

        key = jax.random.PRNGKey(1)
        N = 1000
        # Unequal weights (simulate variance growth)
        key, k1, k2 = jax.random.split(key, 3)
        W = jax.random.uniform(k1, (N,), minval=0.1, maxval=2.0)
        state = GCState(
            r=jax.random.uniform(k2, (N,), minval=0.1, maxval=0.18),
            theta=jnp.zeros(N), zeta=jnp.zeros(N),
            vpar=jnp.zeros(N), mu=jnp.zeros(N), weight=W,
        )
        cfg = SimConfigFullF(N_particles=N)
        geom = build_field_aligned_geometry(16, 32, 16, 1.0, 0.18, 1.0)
        new_state, _ = _resample_particles(state, geom, cfg, key)

        W_new = new_state.weight
        assert float(jnp.std(W_new)) < 1e-5, "Resampled weights should be equal"
        # Total weight conserved
        assert abs(float(jnp.sum(W_new)) - float(jnp.sum(W))) / float(jnp.sum(W)) < 1e-5
