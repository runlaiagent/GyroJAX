# Configuration Reference

All simulation parameters are set via the `SimConfigFA` dataclass in `gyrojax/simulation_fa.py`.

```python
from gyrojax.simulation_fa import SimConfigFA
cfg = SimConfigFA(Npsi=32, N_particles=200_000, beta=0.01)
```

## Grid Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Npsi` | int | 32 | Radial (ψ) grid points |
| `Ntheta` | int | 64 | Poloidal (θ) grid points |
| `Nalpha` | int | 32 | Binormal (α) grid points |

## Particle Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `N_particles` | int | 200000 | Number of ion markers |

## Time Integration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_steps` | int | 500 | Number of time steps |
| `dt` | float | 0.05 | Time step size (normalized to R₀/vti) |

## Geometry

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `R0` | float | 1.0 | Major radius (normalized) |
| `a` | float | 0.18 | Minor radius |
| `B0` | float | 1.0 | On-axis magnetic field |
| `q0` | float | 1.4 | Safety factor at axis: q(r) = q₀ + q₁·(r/a) |
| `q1` | float | 0.5 | Safety factor shear coefficient |

## Physics Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Ti` | float | 1.0 | Ion temperature (normalized) |
| `Te` | float | 1.0 | Electron temperature (normalized) |
| `mi` | float | 1.0 | Ion mass (normalized) |
| `rho_star` | float | 1/180 | ρᵢ/a — gyroradius-to-machine-size ratio (CBC: 1/180) |
| `e` | float | 1000.0 | Charge-to-mass ratio factor (CBC: Ωᵢ = 1000) |
| `vti` | float | 1.0 | Ion thermal velocity (normalized) |
| `n0_avg` | float | 1.0 | Average background density |

## Profile Gradients (CBC Parameters)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `R0_over_LT` | float | 6.9 | Ion temperature gradient R₀/LT (CBC: 6.9) |
| `R0_over_LTe` | float | 6.9 | Electron temperature gradient R₀/LTe (TEM driver) |
| `R0_over_Ln` | float | 2.2 | Density gradient R₀/Ln (CBC: 2.2) |

## Particle Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vpar_cap` | float | 4.0 | Parallel velocity cap (multiples of vti) |
| `use_global` | bool | False | Global radial profiles (True) vs flux-tube (False) |

## Perturbation Seeding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pert_amp` | float | 1e-2 | Initial perturbation amplitude (use 1e-4 for multi-mode runs) |
| `zonal_init` | bool | False | Seed zonal flow (k_θ=0) for Rosenbluth-Hinton/GAM tests |
| `k_mode` | int | 1 | Binormal mode number n for ITG seed: sin(2θ + n·α) |
| `single_mode` | bool | False | Project φ to ±k_mode after each Poisson solve (linear benchmark) |
| `k_alpha_min` | int | 0 | Zero out α modes 1..k_alpha_min−1 (suppress aliasing) |

## δf Noise Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `canonical_loading` | bool | False | Apply canonical Pφ weight correction at t=0 |
| `use_pullback` | bool | False | Periodic f₀ pullback transformation |
| `pullback_interval` | int | 50 | Steps between pullbacks (0 = disabled) |
| `nu_soft` | float | 0.0 | Soft amplitude-dependent weight damping rate (0 = off) |
| `w_sat` | float | 2.0 | Saturation weight for soft damping |
| `soft_damp_alpha` | int | 2 | Power exponent for soft damping |

## Collision Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collision_model` | str | `'none'` | Collision operator: `'none'` \| `'krook'` \| `'lorentz'` \| `'dougherty'` |
| `nu_krook` | float | 0.01 | Krook (BGK) damping rate |
| `nu_ei` | float | 0.01 | Electron-ion collision frequency (Lorentz pitch-angle scattering) |
| `nu_coll` | float | 0.01 | Dougherty collision frequency |

## Weight Spreading (GTC-Style)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_weight_spread` | bool | False | Enable periodic weight smoothing onto grid |
| `weight_spread_interval` | int | 10 | Spread every N steps |
| `zonal_preserving_spread` | bool | True | Preserve zonal flow component during spreading |

## Electron Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `electron_model` | str | `'adiabatic'` | `'adiabatic'` (Boltzmann) \| `'drift_kinetic'` (kinetic TEM) |
| `me_over_mi` | float | 1/1836 | Electron-to-ion mass ratio |
| `subcycles_e` | int | 10 | Electron subcycles per ion step |
| `N_electrons` | int | 0 | Number of electron markers (0 = same as N_particles) |

## Implicit Time-Stepping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `implicit` | bool | False | Use Crank-Nicolson + Picard iteration |
| `picard_max_iter` | int | 4 | Maximum Picard iterations per step |
| `picard_tol` | float | 1e-3 | Convergence tolerance on φ (L∞ norm) |
| `semi_implicit_weights` | bool | False | Semi-implicit CN weight update (unconditionally stable) |
| `use_cn_weights` | bool | False | Alias for `semi_implicit_weights` |

## Boundary Conditions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `absorbing_wall` | bool | False | Absorbing BC: zero weights of escaped particles. False = hard clamp (legacy) |

## Electromagnetic

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beta` | float | 0.0 | Plasma β: 0.0 = electrostatic, >0 = electromagnetic (KBM) |
| `gyroaverage_scatter` | bool | True | Apply √Γ₀(b) to δn before Poisson solve (symmetric gyroaveraging) |
| `use_radial_gaa` | bool | True | Use radially-resolved g^αα(ψ) in Γ₀ operator (clamped+tapered) |
| `fused_rk4` | bool | True | Fuse particle push + weight update into single RK4 (3.78× speedup) |
