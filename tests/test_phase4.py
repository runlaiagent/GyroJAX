"""
Tests for Phase 4: diagnostics and electron model.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from gyrojax.diagnostics import (
    extract_zonal_flow, extract_nonzonal, zonal_shear,
    perp_spectrum, parallel_spectrum, extract_growth_rate,
    compute_snapshot,
)
from gyrojax.electrons import (
    adiabatic_electron_density, ElectronConfig, init_electron_markers,
)
from gyrojax.geometry.field_aligned import build_field_aligned_geometry


@pytest.fixture(scope="module")
def geom():
    return build_field_aligned_geometry(16, 32, 16, 1.0, 0.18, 1.0, 1.4, 0.5)


@pytest.fixture(scope="module")
def sample_phi(geom):
    """Synthetic phi: zonal + ITG mode."""
    psi = geom.psi_grid
    th  = jnp.linspace(0, 2*jnp.pi, 32, endpoint=False)
    al  = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
    PSI, TH, AL = jnp.meshgrid(psi, th, al, indexing='ij')
    zonal = 0.1 * jnp.sin(2*jnp.pi*(PSI - psi[0])/(psi[-1]-psi[0]))[:,:,:]
    turb  = 0.01 * jnp.sin(TH + 2*AL)
    return zonal + turb


class TestDiagnostics:

    def test_zonal_flow_shape(self, geom, sample_phi):
        zf = extract_zonal_flow(sample_phi)
        assert zf.shape == (16,)

    def test_zonal_flow_removes_mean(self, geom, sample_phi):
        zf = extract_zonal_flow(sample_phi)
        turb = extract_nonzonal(sample_phi)
        # Non-zonal should have near-zero mean over theta,alpha
        assert float(jnp.mean(jnp.abs(jnp.mean(turb, axis=(1,2))))) < 1e-5

    def test_zonal_plus_nonzonal_equals_phi(self, geom, sample_phi):
        zf   = extract_zonal_flow(sample_phi)
        turb = extract_nonzonal(sample_phi)
        reconstructed = turb + zf[:, None, None]
        assert float(jnp.max(jnp.abs(reconstructed - sample_phi))) < 1e-5

    def test_zonal_shear_shape(self, geom, sample_phi):
        shear = zonal_shear(sample_phi, geom)
        assert shear.shape == (16,)

    def test_perp_spectrum_shape(self, geom, sample_phi):
        k, E = perp_spectrum(sample_phi, geom)
        assert len(k) == 8   # Nalpha//2
        assert len(E) == 8
        assert bool(jnp.all(E >= 0))

    def test_parallel_spectrum_shape(self, sample_phi):
        k, E = parallel_spectrum(sample_phi)
        assert len(k) == 16  # Ntheta//2
        assert bool(jnp.all(E >= 0))

    def test_growth_rate_extraction(self):
        # Synthetic growing signal
        dt = 0.05
        t = np.arange(200) * dt
        gamma_true = 0.17
        phi = np.exp(gamma_true * t) * (1 + 0.01*np.random.randn(200))
        gamma = extract_growth_rate(phi, dt, window=0.4)
        assert abs(gamma - gamma_true) / gamma_true < 0.05

    def test_snapshot_no_nan(self, geom, sample_phi):
        from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
        cfg = SimConfigFA(Npsi=16, Ntheta=32, Nalpha=16,
                         N_particles=5000, n_steps=5, dt=0.05,
                         R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.5,
                         Ti=1.0, Te=1.0, mi=1.0, e=1000.0,
                         R0_over_LT=6.9, R0_over_Ln=2.2, vti=1.0, n0_avg=1.0)
        diags, state, phi, geom2 = run_simulation_fa(cfg, verbose=False)
        LT = cfg.R0 / cfg.R0_over_LT
        snap = compute_snapshot(0.25, state, phi, geom2, (16,32,16),
                                cfg.Ti, cfg.n0_avg, LT)
        assert not np.isnan(snap.phi_rms)
        assert not np.isnan(snap.Q_ion).any()
        assert snap.phi_zf_rms >= 0
        assert snap.phi_turb_rms >= 0


class TestElectrons:

    def test_adiabatic_density(self, geom, sample_phi):
        dn_e = adiabatic_electron_density(sample_phi, 1.0, 1.0, 1000.0)
        assert dn_e.shape == sample_phi.shape
        # δn_e = (e/Te)*n0*phi → proportional to phi
        ratio = dn_e / (sample_phi + 1e-30)
        interior = ratio[1:-1, 1:-1, 1:-1]
        assert float(jnp.std(interior)) < 1.0  # roughly constant ratio

    def test_electron_marker_init(self, geom):
        e_cfg = ElectronConfig(model="drift_kinetic", me_over_mi=1/1836,
                               vte=42.72, Te=1.0)
        key = jax.random.PRNGKey(99)
        state_e = init_electron_markers(1000, geom, e_cfg, key)
        assert state_e.r.shape == (1000,)
        assert not bool(jnp.any(jnp.isnan(state_e.vpar)))
        # Electron thermal speed should be much higher than ion (vte/vti = 42.72)
        assert float(jnp.sqrt(jnp.mean(state_e.vpar**2))) > 10.0
