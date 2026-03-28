"""
Tests for Phase 2b: VMEC geometry loading.

Uses the test wout files from booz_xform_jax.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import os

WOUT_LI383   = os.path.expanduser(
    "~/wlhx/booz_xform_jax/tests/test_files/wout_li383_1.4m.nc"
)
WOUT_CIRCULAR = os.path.expanduser(
    "~/wlhx/booz_xform_jax/tests/test_files/wout_circular_tokamak.nc"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(WOUT_LI383),
    reason="wout_li383_1.4m.nc not found"
)


@pytest.fixture(scope="module")
def geom_li383():
    from gyrojax.geometry.vmec_geometry import load_vmec_geometry
    return load_vmec_geometry(WOUT_LI383, Ntheta=32, Nzeta=16, Ns_out=24)


@pytest.fixture(scope="module")
def geom_circular():
    from gyrojax.geometry.vmec_geometry import load_vmec_geometry
    return load_vmec_geometry(WOUT_CIRCULAR, Ntheta=32, Nzeta=16, Ns_out=24)


class TestVMECGeometryLI383:

    def test_B_shape(self, geom_li383):
        assert geom_li383.B_field.shape == (24, 32, 16)

    def test_B_positive(self, geom_li383):
        assert float(jnp.min(geom_li383.B_field)) > 0.0, "B must be positive everywhere"

    def test_B_near_axis_value(self, geom_li383):
        """B near s~0 should be close to b0 (axis field)."""
        B_inner = float(geom_li383.B_field[0].mean())
        B0 = geom_li383.B0
        assert abs(B_inner - B0) / B0 < 0.3, (
            f"B_inner={B_inner:.3f} too far from B0={B0:.3f}"
        )

    def test_q_profile_shape(self, geom_li383):
        assert geom_li383.q_profile.shape == (24,)

    def test_q_positive(self, geom_li383):
        """Safety factor should be positive (forward-going field)."""
        assert float(jnp.min(jnp.abs(geom_li383.q_profile))) > 0.1

    def test_shat_shape(self, geom_li383):
        assert geom_li383.shat.shape == (24,)

    def test_gradB_no_nan(self, geom_li383):
        assert not bool(jnp.any(jnp.isnan(geom_li383.gradB_psi)))
        assert not bool(jnp.any(jnp.isnan(geom_li383.gradB_th)))

    def test_galphaalpha_positive(self, geom_li383):
        assert float(jnp.min(geom_li383.galphaalpha)) > 0.0

    def test_twist_shift_shape(self, geom_li383):
        assert geom_li383.twist_shift.shape == (24,)

    def test_theta_grid_range(self, geom_li383):
        th = geom_li383.theta_grid
        assert float(th[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(th[-1]) < 2 * np.pi

    def test_zeta_grid_range(self, geom_li383):
        """Zeta covers one field period: [0, 2π/nfp)."""
        # nfp=3 for li383 → zeta_max ≈ 2π/3
        ze = geom_li383.alpha_grid
        assert float(ze[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(ze[-1]) < 2 * np.pi / 3 + 0.5   # some slack


class TestVMECGeometryCircular:
    """Tokamak (circular, up-down symmetric) — should look like s-α."""

    def test_B_shape(self, geom_circular):
        assert geom_circular.B_field.shape == (24, 32, 16)

    def test_B_helical_variation(self, geom_circular):
        """B should vary in theta (toroidal mirror effect 1/R)."""
        B_std = float(jnp.std(geom_circular.B_field[12, :, 0]))
        assert B_std > 0.0, "B has no theta variation — unexpected"

    def test_no_nan_in_geometry(self, geom_circular):
        for field in (geom_circular.B_field, geom_circular.gradB_psi,
                      geom_circular.gradB_th, geom_circular.kappa_th,
                      geom_circular.q_profile):
            assert not bool(jnp.any(jnp.isnan(field)))


class TestVMECCompatibilityWithFA:
    """Check that VMEC geometry can be passed to FA simulation components."""

    def test_interp_to_particles(self, geom_li383):
        from gyrojax.geometry.field_aligned import interp_fa_to_particles
        N = 100
        key = jax.random.PRNGKey(0)
        s_p = jax.random.uniform(key, (N,),
                                  minval=float(geom_li383.psi_grid[0]),
                                  maxval=float(geom_li383.psi_grid[-1]))
        th_p = jax.random.uniform(key, (N,), minval=0.0, maxval=2*np.pi)
        al_p = jax.random.uniform(key, (N,), minval=0.0, maxval=float(geom_li383.alpha_grid[-1]))

        B_p, gBs, gBt, kaps, kapt, _ = interp_fa_to_particles(geom_li383, s_p, th_p, al_p)
        assert B_p.shape == (N,)
        assert not bool(jnp.any(jnp.isnan(B_p)))
        assert float(jnp.min(B_p)) > 0.0

    def test_poisson_fa_accepts_vmec_geom(self, geom_li383):
        from gyrojax.fields.poisson_fa import solve_poisson_fa
        Ns, Nth, Nze = geom_li383.B_field.shape
        delta_n = jnp.zeros((Ns, Nth, Nze))
        phi, _ = solve_poisson_fa(delta_n, geom_li383, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert phi.shape == (Ns, Nth, Nze)
        assert float(jnp.max(jnp.abs(phi))) < 1e-10   # zero rhs → zero phi
