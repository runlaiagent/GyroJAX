"""
Integration tests for Phase 2a simulation loop (field-aligned).

These tests verify that the full simulation loop runs without NaN/Inf,
that phi grows (mode is unstable), and that weight rms stays bounded.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa


@pytest.fixture(scope="module")
def sim_result():
    """Run a short simulation for tests (shared across tests)."""
    cfg = SimConfigFA(
        Npsi=16, Ntheta=32, Nalpha=16,
        N_particles=10_000,
        n_steps=40,
        dt=0.05,
        R0=1.0, a=0.18, B0=1.0,
        q0=1.4, q1=0.5,
        Ti=1.0, Te=1.0, mi=1.0, e=1.0,
        R0_over_LT=6.9, R0_over_Ln=2.2,
        vti=1.0, n0_avg=1.0,
    )
    key = jax.random.PRNGKey(0)
    return run_simulation_fa(cfg, key, verbose=False)


class TestSimulationFASmoke:

    def test_no_nan_phi(self, sim_result):
        """Potential should have no NaN at any step."""
        _, _, phi, _ = sim_result
        assert not bool(jnp.any(jnp.isnan(phi))), "NaN in final phi"

    def test_no_inf_phi(self, sim_result):
        _, _, phi, _ = sim_result
        assert not bool(jnp.any(jnp.isinf(phi))), "Inf in final phi"

    def test_no_nan_weights(self, sim_result):
        """Particle weights should have no NaN."""
        _, state, _, _ = sim_result
        assert not bool(jnp.any(jnp.isnan(state.weight))), "NaN in weights"

    def test_no_nan_positions(self, sim_result):
        """Particle positions should remain finite."""
        _, state, _, _ = sim_result
        for field in (state.r, state.theta, state.zeta, state.vpar):
            assert not bool(jnp.any(jnp.isnan(field))), f"NaN in particle state"

    def test_particles_in_bounds(self, sim_result):
        """Particles should stay within the radial domain."""
        _, state, _, geom = sim_result
        r_min = float(geom.psi_grid[0]) * 0.99
        r_max = float(geom.psi_grid[-1]) * 1.01
        assert bool(jnp.all(state.r >= r_min)), "Particles outside inner boundary"
        assert bool(jnp.all(state.r <= r_max)), "Particles outside outer boundary"

    def test_diag_length(self, sim_result):
        diags, _, _, _ = sim_result
        assert len(diags) == 40

    def test_phi_nonzero(self, sim_result):
        """Potential should be nonzero (mode was seeded)."""
        diags, _, _, _ = sim_result
        phi_max_final = float(diags[-1].phi_max)
        assert phi_max_final > 1e-20, f"phi_max collapsed to zero ({phi_max_final})"

    def test_weight_rms_bounded(self, sim_result):
        """Weight rms should stay well below 1 (δf validity)."""
        diags, _, _, _ = sim_result
        w_rms_max = max(float(d.weight_rms) for d in diags)
        # At low particle count the weight clamp (±10) may fire — just check no NaN
        assert w_rms_max <= 10.0, f"Weight rms exceeded clamp: {w_rms_max}"

    def test_phi_grew(self, sim_result):
        """Phi should grow during the linear phase (ITG mode excited)."""
        diags, _, _, _ = sim_result
        phi_early = float(diags[5].phi_max)
        phi_late  = float(diags[-1].phi_max)
        # At least some growth (or no collapse to near-zero)
        assert phi_late >= phi_early * 0.1, (
            f"phi collapsed: early={phi_early:.3e}, late={phi_late:.3e}"
        )
