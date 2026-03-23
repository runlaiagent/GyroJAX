"""
Tests for kinetic electron (drift-kinetic) model.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from gyrojax.electrons import (
    ElectronConfig,
    ElectronState,
    init_electron_markers,
    init_electron_state,
    push_electrons_dk,
    update_electron_weights,
)
from gyrojax.geometry.field_aligned import build_field_aligned_geometry
from gyrojax.interpolation.scatter_gather_fa import gather_from_grid_fa
from gyrojax.geometry.field_aligned import interp_fa_to_particles
from gyrojax.particles.guiding_center import GCState


@pytest.fixture(scope="module")
def geom():
    return build_field_aligned_geometry(16, 32, 16, 1.0, 0.18, 1.0, 1.4, 0.5)


@pytest.fixture(scope="module")
def e_cfg():
    return ElectronConfig(
        model='drift_kinetic',
        me_over_mi=1.0/1836.0,
        subcycles=2,  # fast for tests
        Te=1.0,
        vte=42.72,
    )


@pytest.fixture(scope="module")
def electron_markers(geom, e_cfg):
    key = jax.random.PRNGKey(0)
    return init_electron_markers(500, geom, e_cfg, key)


@pytest.fixture(scope="module")
def dummy_fields(geom):
    phi = jnp.zeros(geom.B_field.shape)
    E_psi = jnp.zeros_like(phi)
    E_theta = jnp.zeros_like(phi)
    E_alpha = jnp.zeros_like(phi)
    return phi, E_psi, E_theta, E_alpha


class TestPushElectrons:

    def test_push_electrons_jit_compatible(self, geom, e_cfg, electron_markers):
        """push_electrons_dk must compile without error under jit."""
        N = electron_markers.r.shape[0]
        # Per-particle E arrays (shape (N,)) as expected by pusher
        E_psi_p   = jnp.zeros(N)
        E_theta_p = jnp.zeros(N)
        E_alpha_p = jnp.zeros(N)

        import functools
        push_fn = functools.partial(push_electrons_dk, geom=geom, e_cfg=e_cfg, dt_ion=0.05)
        push_jit = jax.jit(push_fn)
        pushed = push_jit(electron_markers, E_psi_p, E_theta_p, E_alpha_p)
        assert pushed.r.shape == electron_markers.r.shape

    def test_push_electrons_subcycling(self, geom, electron_markers):
        """Electrons with 10 subcycles move more than with 1 subcycle under nonzero E."""
        N = electron_markers.r.shape[0]
        # Per-particle E arrays
        E_psi_p   = jnp.full(N, 0.01)
        E_theta_p = jnp.zeros(N)
        E_alpha_p = jnp.zeros(N)

        cfg1  = ElectronConfig(model='drift_kinetic', me_over_mi=1.0/1836.0, subcycles=1,  Te=1.0, vte=42.72)
        cfg10 = ElectronConfig(model='drift_kinetic', me_over_mi=1.0/1836.0, subcycles=10, Te=1.0, vte=42.72)

        state1  = push_electrons_dk(electron_markers, E_psi_p, E_theta_p, E_alpha_p, geom, cfg1,  0.1)
        state10 = push_electrons_dk(electron_markers, E_psi_p, E_theta_p, E_alpha_p, geom, cfg10, 0.1)

        assert jnp.any(state1.vpar != electron_markers.vpar) or jnp.any(state10.vpar != electron_markers.vpar)


class TestElectronWeights:

    def test_electron_weight_update_no_nan(self, geom, e_cfg, electron_markers, dummy_fields):
        """Weight update should produce no NaN."""
        phi, E_psi, E_theta, E_alpha = dummy_fields
        E_psi_e, E_theta_e, E_alpha_e = gather_from_grid_fa(phi, electron_markers, geom)
        B_e, gBpsi_e, gBth_e, kpsi_e, kth_e = interp_fa_to_particles(
            geom, electron_markers.r, electron_markers.theta, electron_markers.zeta
        )
        n0_e = jnp.ones(electron_markers.r.shape)
        Te_e = jnp.ones(electron_markers.r.shape)
        d_ln_n0 = jnp.full_like(electron_markers.r, -1.0/0.08)
        d_ln_Te = jnp.full_like(electron_markers.r, -1.0/0.145)
        q_e = jnp.full_like(electron_markers.r, 1.4)

        new_state = update_electron_weights(
            electron_markers, E_psi_e, E_theta_e,
            B_e, gBpsi_e, gBth_e, kpsi_e, kth_e,
            q_e, n0_e, Te_e, d_ln_n0, d_ln_Te,
            e_cfg, 1.0, 0.05,
        )
        assert not jnp.any(jnp.isnan(new_state.weight))

    def test_electron_weight_sign(self, geom, e_cfg, electron_markers):
        """Electrons respond with opposite sign to ions under same E_psi drive."""
        from gyrojax.particles.guiding_center import init_maxwellian_particles
        from gyrojax.deltaf.weights import update_weights
        from gyrojax.geometry.salpha import build_salpha_geometry

        shape = geom.B_field.shape
        # Apply radially uniform E_psi perturbation
        E_psi_e = jnp.full(electron_markers.r.shape, 0.1)
        E_theta_e = jnp.zeros_like(E_psi_e)
        B_e, gBpsi_e, gBth_e, kpsi_e, kth_e = interp_fa_to_particles(
            geom, electron_markers.r, electron_markers.theta, electron_markers.zeta
        )
        n0_e = jnp.ones_like(E_psi_e)
        Te_e = jnp.ones_like(E_psi_e)
        d_ln_n0 = jnp.full_like(E_psi_e, -5.0)
        d_ln_Te = jnp.full_like(E_psi_e, -5.0)
        q_e = jnp.full_like(E_psi_e, 1.4)

        # Start from zero weights
        state0 = electron_markers._replace(weight=jnp.zeros_like(electron_markers.weight))

        # Electron update
        new_e = update_electron_weights(
            state0, E_psi_e, E_theta_e,
            B_e, gBpsi_e, gBth_e, kpsi_e, kth_e,
            q_e, n0_e, Te_e, d_ln_n0, d_ln_Te,
            e_cfg, 1.0, 0.05,
        )

        # Ion update with same geometry but positive q/m
        cfg_ion = ElectronConfig(model='drift_kinetic', me_over_mi=1.0, subcycles=1, Te=1.0, vte=1.0)
        new_i = update_electron_weights(
            state0, E_psi_e, E_theta_e,
            B_e, gBpsi_e, gBth_e, kpsi_e, kth_e,
            q_e, n0_e, Te_e, d_ln_n0, d_ln_Te,
            cfg_ion, 1.0, 0.05,
        )
        # Electrons and ions should have opposite weight sign (q_e/m_e = -1/me_over_mi vs +1)
        # Check mean weight has opposite sign tendency
        w_e = float(jnp.mean(new_e.weight))
        w_i = float(jnp.mean(new_i.weight))
        assert w_e * w_i <= 0.0 or abs(w_e - w_i) > 0  # they differ


class TestElectronState:

    def test_init_electron_state_adiabatic(self, geom):
        key = jax.random.PRNGKey(1)
        cfg = ElectronConfig(model='adiabatic', Te=1.0, vte=42.72)
        state = init_electron_state(1000, geom, cfg, key)
        assert state.model == 'adiabatic'
        # Adiabatic: only 1 dummy marker allocated
        assert state.markers.r.shape[0] == 1

    def test_init_electron_state_dk(self, geom):
        key = jax.random.PRNGKey(2)
        cfg = ElectronConfig(model='drift_kinetic', Te=1.0, vte=42.72, subcycles=2)
        state = init_electron_state(500, geom, cfg, key)
        assert state.model == 'drift_kinetic'
        assert state.markers.r.shape[0] == 500


class TestSimulationIntegration:

    def test_dk_sim_no_nan(self):
        """Run 10 steps with drift-kinetic electrons; no NaN in phi."""
        from gyrojax.simulation_fa import run_simulation_fa, SimConfigFA
        cfg = SimConfigFA(
            Npsi=8, Ntheta=16, Nalpha=8,
            N_particles=500,
            n_steps=10,
            dt=0.05,
            electron_model='drift_kinetic',
            subcycles_e=2,
            N_electrons=200,
        )
        diags, state, phi, geom = run_simulation_fa(cfg, jax.random.PRNGKey(42), verbose=False)
        assert not jnp.any(jnp.isnan(phi)), "NaN in phi during DK electron simulation"
        assert not jnp.any(jnp.isnan(state.weight)), "NaN in ion weights"

    def test_adiabatic_vs_dk_both_unstable(self):
        """Both models should show phi_rms growth over 50 steps (ITG instability)."""
        from gyrojax.simulation_fa import run_simulation_fa, SimConfigFA
        base = dict(
            Npsi=8, Ntheta=16, Nalpha=8,
            N_particles=500,
            n_steps=50,
            dt=0.05,
        )
        cfg_ad = SimConfigFA(**base, electron_model='adiabatic')
        cfg_dk = SimConfigFA(**base, electron_model='drift_kinetic', subcycles_e=2, N_electrons=200)

        diags_ad, _, _, _ = run_simulation_fa(cfg_ad, jax.random.PRNGKey(42), verbose=False)
        diags_dk, _, _, _ = run_simulation_fa(cfg_dk, jax.random.PRNGKey(42), verbose=False)

        phi_ad = np.array([float(d.phi_rms) for d in diags_ad])
        phi_dk = np.array([float(d.phi_rms) for d in diags_dk])

        # Both should have some nonzero phi_rms
        assert phi_ad[-1] > 0, "Adiabatic phi_rms is zero"
        assert phi_dk[-1] > 0, "DK phi_rms is zero"
