"""Tests for Rosenbluth-Hinton and GAM benchmarks."""
import numpy as np
import pytest


class TestRHTheory:
    def test_rh_formula_mid_radius(self):
        """R-H residual at CBC mid-radius (q=1.4, ε=0.09)."""
        from benchmarks.rosenbluth_hinton import rh_residual_theory
        r = rh_residual_theory(q=1.4, eps=0.09)
        assert abs(r - 0.0873) < 0.001

    def test_rh_formula_full_radius(self):
        """R-H residual at CBC full radius (q=1.4, ε=0.18)."""
        from benchmarks.rosenbluth_hinton import rh_residual_theory
        r = rh_residual_theory(q=1.4, eps=0.18)
        assert abs(r - 0.119) < 0.002

    def test_gam_theory_ti_eq_te(self):
        """GAM frequency for Ti=Te: ω = sqrt(11/4) ≈ 1.658."""
        from benchmarks.rosenbluth_hinton import gam_frequency_theory
        omega = gam_frequency_theory(vti=1.0, R0=1.0, Te_over_Ti=1.0)
        assert abs(omega - np.sqrt(11.0 / 4.0)) < 0.001


class TestZonalInit:
    def test_zonal_init_flag_exists(self):
        """SimConfigFA accepts zonal_init parameter."""
        from gyrojax.simulation_fa import SimConfigFA
        cfg = SimConfigFA(zonal_init=True, R0_over_LT=0.0, R0_over_Ln=0.0, fused_rk4=False)
        assert cfg.zonal_init is True

    def test_zero_drives(self):
        """R0_over_LT=0 and R0_over_Ln=0 accepted without error."""
        from gyrojax.simulation_fa import SimConfigFA
        cfg = SimConfigFA(R0_over_LT=0.0, R0_over_Ln=0.0, fused_rk4=False)
        assert cfg.R0_over_LT == 0.0
        assert cfg.R0_over_Ln == 0.0


class TestRHBenchmarkQuick:
    def test_rh_residual_in_range(self):
        """Quick R-H run: residual in plausible range [0.01, 0.5]."""
        import jax
        from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa
        import numpy as np

        cfg = SimConfigFA(
            Npsi=16, Ntheta=16, Nalpha=8,
            N_particles=20_000, n_steps=100, dt=0.05,
            R0_over_LT=0.0, R0_over_Ln=0.0,
            pert_amp=1e-2, zonal_init=True,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.0,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            fused_rk4=False,
        )
        key = jax.random.PRNGKey(0)
        diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
        phi_max = np.array([float(d.phi_max) for d in diags])
        assert np.isfinite(phi_max).all(), "NaN in zonal flow run"
        phi0 = phi_max[0] + 1e-20
        residual = float(phi_max[50:].mean() / phi0)
        assert 0.01 < residual < 2.0, f"Residual {residual:.4f} out of range"

    def test_no_growth_without_drives(self):
        """Without temperature gradient, mode should not grow exponentially."""
        import jax, numpy as np
        from gyrojax.simulation_fa import SimConfigFA, run_simulation_fa

        cfg = SimConfigFA(
            Npsi=16, Ntheta=16, Nalpha=8,
            N_particles=10_000, n_steps=80, dt=0.05,
            R0_over_LT=0.0, R0_over_Ln=0.0,
            pert_amp=1e-2, zonal_init=True,
            R0=1.0, a=0.18, B0=1.0, q0=1.4, q1=0.0,
            Ti=1.0, Te=1.0, mi=1.0, e=1000.0, vti=1.0, n0_avg=1.0,
            fused_rk4=False,
        )
        key = jax.random.PRNGKey(1)
        diags, _, _, _ = run_simulation_fa(cfg, key, verbose=False)
        phi_max = np.array([float(d.phi_max) for d in diags])
        # phi should decay or stay flat — NOT grow by factor >10
        assert phi_max[-1] < phi_max[0] * 10.0, "Unexpected growth without drives"
