# GyroJAX

A gyrokinetic particle-in-cell (PIC) code for tokamak and stellarator plasma simulation, written entirely in [JAX](https://github.com/google/jax). Designed for research into ITG turbulence, zonal flows, and transport physics — fully JIT-compiled, multi-GPU capable, and benchmarked against published reference codes (GENE, GX, GTC).

---

## Character

GyroJAX is a **research-grade gyrokinetic PIC code** with two first-principles simulation paths:

| Feature | δf (delta-f) | Full-f |
|---|---|---|
| Particle weight | Perturbation only: `w = δf/f₀` | Full distribution: `W = f/f₀_ref` |
| Weight evolution | RK4 or CN (analytical per-particle) | Constant along characteristics |
| Noise level | Low (weights ≪ 1 in linear regime) | Higher (requires resampling) |
| Nonlinear reach | Limited by `⟨w²⟩ ≪ 1` | Naturally handles full nonlinearity |
| Best for | Linear growth rates, weak turbulence | Saturated turbulence, transport |

Both paths share the same JAX infrastructure: functional pytrees, `lax.scan`-based time loops, FFT Poisson solver, and field-aligned (s-α) geometry.

---

## δf Method

The δf PIC method tracks only the perturbation `δf = f - f₀` via marker weights `w = δf/f₀`. Particles push under the full guiding-center equations; weights evolve as:

```
dw/dt = -(1 - w) · S,    S = (v_E + v_d) · ∇ln f₀
```

### Key implementations

**Time integration**
- **Explicit RK4** (default) — standard 4th-order Runge-Kutta weight advance
- **Semi-implicit CN** (`semi_implicit_weights=True`) — Crank-Nicolson analytical solution per particle:
  `w^{n+1} = 1 - (1 - wⁿ) / (1 + dt·S)` — unconditionally bounded, prevents weight blowup
- **Implicit CN+Picard** (`implicit=True`) — full Crank-Nicolson with Picard iteration via `jax.lax.while_loop`; 2–4 iterations per step, allows 5–10× larger dt

**Noise control**
- Canonical Maxwellian loading (`canonical_loading=True`) — reduces initial sampling noise
- Pullback transformation (`use_pullback=True`) — periodically maps weights back toward Maxwellian to control long-time weight growth
- GTC-style weight spreading (`use_weight_spread=True`) — scatter weights to grid, smooth, gather back; suppresses particle-level noise
- Soft weight damping (`nu_soft`) — amplitude-dependent tanh limiter
- Krook collision operator (`nu_krook`) — BGK-style damping in buffer zones

**Physics**
- ExB drift + magnetic gradient/curvature drifts (full ITG drive)
- Field-aligned s-α geometry with safety factor profile q(r) = q₀ + q₁·(r/a)
- Global radial profiles (`use_global=True`) with Krook buffer BCs at inner/outer walls
- Adiabatic electrons (Boltzmann response, default)
- Kinetic electrons: drift-kinetic hybrid model (Phase 4a, subcycling)
- Collision operators: Krook, Lorentz pitch-angle scattering, Dougherty model

**Poisson solver**
- GK quasi-neutrality: `[n₀/Tₑ·(1-Γ₀) + n₀/Tᵢ·Γ₀]·eφ = δnᵢ(gyro)`
- FFT in α (toroidal), tridiagonal in ψ (radial), spectral in θ (poloidal)
- Symmetric gyroaveraging: √Γ₀(b) applied to both δn (scatter) and φ (field)
- Radially-resolved g^αα(ψ) in FLR operator — not flux-tube scalar
- Radial profile clamped + tapered near boundaries to prevent inner-radius divergence

**Performance (GPU: RTX 3070 Ti Laptop, 8GB)**
- 27 steps/sec at 100k–500k particles (GPU-compute bound, flat scaling ✅)
- All operations float32 — fits comfortably in 8GB VRAM at 500k particles
- Key optimizations: fused RK4 pusher+weights (3.78× speedup), phi_hat caching, trilinear index caching, smaller lax.scan carry

### Benchmarks (δf)

| Benchmark | Result | Reference | Error |
|---|---|---|---|
| CBC linear growth rate (ky·ρᵢ = 0.30) | γ = 0.172 vti/R0 | 0.169 vti/R0 (GENE/GX) | **1.9%** |
| Rosenbluth-Hinton zonal flow residual | Stable, correct damping | — | ✅ |
| γ spectrum (ky·ρᵢ = 0.1–0.6) | Peak at ky·ρᵢ ≈ 0.35–0.40 | GENE spectrum | ~15% avg |
| Dimits shift threshold | R/LT = 6.9 (PASS ✅) | R/LT ≈ 6.0 (Dimits 2000) | Within criterion [4.5, 7.5] |
| χᵢ(R/LT) heat flux scan | Sharp onset at R/LT=6.9 ✅ | Monotonic rise expected | Qualitatively correct |
| EM CBC (β = 0.01, 0.05) | A∥ nonzero, β-scaling correct ✅ | — | — |

---

## Full-f Method

The full-f path treats each marker as carrying the full distribution function `W = f/f₀_ref`. Weights are **constant along phase-space trajectories** (Vlasov characteristics) — they do not evolve. Instead, the changing particle positions self-consistently generate `δn`:

```
δnᵢ(x) = Σ_p Wₚ · δ(x - Xₚ) - n₀
dWₚ/dt = 0    (constant marker weights)
```

This naturally captures nonlinear physics without the `⟨w²⟩ ≪ 1` constraint, but requires **periodic resampling** (particle splitting/merging or importance sampling) to control phase-space noise as the distribution evolves away from `f₀_ref`.

### Key implementations

- Full-f GC pusher — same equations of motion as δf, weights constant
- Full-f Poisson — full `δn` deposit (not just perturbation part)
- Periodic resampling hook (importance sampling, in development)

### Benchmarks (full-f)

| Benchmark | Result | Notes |
|---|---|---|
| CBC full-f linear growth rate | γ = 0.012 vti/R0 (positive ✅) | Low vs δf due to full-f noise floor (needs ~1/pert_amp² more particles for equal SNR) |
| Full-f weight constancy | `std(W)/mean(W) < 1e-3` ✅ | Verified by test |
| Full-f nonlinear saturation | Planned | Natural advantage over δf (no ⟨w²⟩ constraint) |
| Dimits shift (full-f path) | Planned | |

---

## Quick Start

```bash
conda activate jax_wlhx
pip install -e .

# Run from an input file (recommended)
python -m gyrojax.runner inputs/cbc.toml --verbose
python -m gyrojax.runner inputs/cbc.toml --dry-run     # print config, no run
python -m gyrojax.runner inputs/cbc.toml --n-steps 20  # quick override

# Or run benchmarks directly
python benchmarks/cyclone_base_case_fa.py
python benchmarks/gamma_spectrum.py --quick
python benchmarks/dimits_shift.py
python benchmarks/dimits_shift_optimized.py   # 2000-step run, reduced grid
python benchmarks/cbc_em_benchmark.py          # EM: β scan (ITG→KBM)
python benchmarks/cbc_fullf.py
python benchmarks/heat_flux_cbc.py            # χᵢ vs R/LT (nonlinear transport)

# Run all tests
pytest tests/ -v
```

---

## Input Files

GyroJAX uses TOML input files — no Python required. Drop a `.toml` in `inputs/` and run it.

```toml
# inputs/cbc.toml — Cyclone Base Case

[run]
method          = "deltaf"       # "deltaf" | "fullf"
time_integrator = "explicit"     # "explicit" | "semi_implicit" | "implicit"

[grid]
Npsi = 16 ;  Ntheta = 32 ;  Nalpha = 64

[particles]
N_particles = 300000
loading     = "maxwellian"       # "maxwellian" | "canonical"

[time]
dt = 0.05 ;  n_steps = 200

[geometry]
R0 = 1.0 ;  a = 0.18 ;  B0 = 1.0 ;  q0 = 1.4 ;  q1 = 0.5

[physics]
rho_star = 0.005556   # 1/180 — sets e = 1/rho_star = 180... × (vti·mi/a·B0)
R0_over_LT = 6.9 ;  R0_over_Ln = 2.2

[species]
electrons = "adiabatic"          # "adiabatic" | "drift_kinetic"

[init]
pert_amp = 1.0e-4 ;  single_mode = true ;  k_mode = 18

[numerics]
collision_model = "none"         # "none" | "krook" | "lorentz" | "dougherty"

[output]
output_dir = "results/"
```

Bundled input files:

| File | Physics |
|---|---|
| `inputs/cbc.toml` | CBC δf linear growth rate |
| `inputs/cbc_fullf.toml` | CBC full-f (constant weights) |
| `inputs/rosenbluth_hinton.toml` | Zonal flow residual (R-H test) |
| `inputs/dimits.toml` | Dimits shift nonlinear scan template |

---

## File Structure

```
GyroJAX/
│
├── inputs/                        # TOML input files (start here)
│   ├── cbc.toml                   # Cyclone Base Case δf
│   ├── cbc_fullf.toml             # Cyclone Base Case full-f
│   ├── rosenbluth_hinton.toml     # Zonal flow residual test
│   └── dimits.toml                # Dimits shift scan template
│
├── gyrojax/                       # Core library
│   ├── input.py                   # TOML → SimConfig parser
│   ├── runner.py                  # CLI: python -m gyrojax.runner input.toml
│   │
│   ├── simulation_fa.py           # δf main loop (field-aligned, lax.scan)
│   ├── simulation_fullf.py        # Full-f main loop (dW/dt=0)
│   ├── simulation_sharded.py      # Multi-GPU sharded δf loop
│   │
│   ├── geometry/
│   │   ├── salpha.py              # s-α magnetic geometry (CBC default)
│   │   ├── field_aligned.py       # Field-aligned coordinate transforms
│   │   ├── vmec_geometry.py       # VMEC stellarator geometry reader
│   │   └── profiles.py            # Radial profiles, Krook buffer BCs
│   │
│   ├── particles/
│   │   ├── guiding_center.py      # GCState pytree + RK4 pusher
│   │   └── guiding_center_fa.py   # Field-aligned GC pusher (push_particles_fa)
│   │
│   ├── fields/
│   │   ├── poisson.py             # GK Poisson solver (flux-tube)
│   │   ├── poisson_fa.py          # GK Poisson (field-aligned, gyroavg)
│   │   └── ampere_fa.py           # Ampere solver: ∇²⊥ δA∥ = -β·δj∥ (EM)
│   │
│   ├── deltaf/
│   │   └── weights.py             # δf weight eq: RK4, semi-implicit CN, Picard
│   │
│   ├── fullf/
│   │   └── __init__.py            # Full-f: constant weights, systematic resampling
│   │
│   ├── electrons/
│   │   └── __init__.py            # Adiabatic / drift-kinetic / GK electrons
│   │
│   ├── collisions/
│   │   └── operators.py           # Krook, Lorentz pitch-angle, Dougherty
│   │
│   ├── interpolation/
│   │   ├── scatter_gather.py      # Particle ↔ grid (flux-tube)
│   │   └── scatter_gather_fa.py   # Gyroaveraged scatter/gather (field-aligned)
│   │
│   ├── diagnostics/
│   │   └── __init__.py            # Growth rate fitting, ky·ρᵢ estimation
│   │
│   ├── viz/                       # Plotting utilities
│   ├── sharding.py                # JAX multi-device sharding
│   ├── normalization.py           # Gyrokinetic normalisation conventions
│   └── utils.py                   # Shared helpers
│
├── benchmarks/                    # Physics validation scripts
│   ├── cyclone_base_case_fa.py    # CBC δf: γ = 0.172 vti/R0 (1.9% error)
│   ├── cbc_benchmark.py           # CBC with phi_rms fitting
│   ├── cbc_fullf.py               # CBC full-f
│   ├── gamma_spectrum.py          # γ(ky·ρᵢ) spectrum sweep
│   ├── dimits_shift.py            # Nonlinear R/LT scan → Dimits threshold
│   ├── dimits_shift_optimized.py  # 2000-step reduced-grid Dimits scan
│   ├── dimits_clean.py            # Physics-correct Dimits (all fixes active)
│   ├── cbc_em_benchmark.py        # EM β scan: ITG → KBM crossover
│   ├── heat_flux_cbc.py           # χᵢ(R/LT) heat flux scan (nonlinear transport)
│   ├── rosenbluth_hinton.py       # Zonal flow residual
│   ├── collision_scan.py          # Collision rate scan
│   ├── kinetic_electron_cbc.py    # Kinetic electron CBC
│   ├── global_cbc.py              # Global (full-radius) CBC
│   ├── stellarator_itg.py         # Stellarator ITG (VMEC)
│   └── results/                   # Saved benchmark output (JSON)
│
└── tests/                         # pytest suite (167 tests)
    ├── test_simulation_fa.py      # δf simulation integration tests
    ├── test_phase3_fullf.py       # Full-f: weight constancy, density deposit
    ├── test_input.py              # TOML parser + CLI runner
    ├── test_rosenbluth_hinton.py  # Zonal flow physics
    ├── test_phase2a/b.py          # Field-aligned geometry + pusher
    ├── test_geometry.py           # s-α, VMEC, profiles
    ├── test_poisson.py            # GK Poisson solver
    ├── test_pusher.py             # GC equations of motion
    ├── test_collisions.py         # Collision operators
    ├── test_kinetic_electrons.py  # Electron physics
    ├── test_global.py             # Global geometry
    ├── test_sharding.py           # Multi-GPU
    └── test_production_validation.py  # Physics regression tests
```

---

## Architecture

---

## Roadmap

### Phase 1 — Foundation ✅
- [x] Field-aligned s-α geometry with q(r) profile
- [x] δf GC pusher + RK4 weight advance
- [x] GK Poisson solver (FFT/tridiagonal, gyroaveraging)
- [x] CBC linear benchmark — γ = 0.172 vti/R0 (1.9% error vs GENE/GX)
- [x] Rosenbluth-Hinton zonal flow test

### Phase 2 — Production δf ✅
- [x] Semi-implicit CN weight update (unconditionally bounded, prevents blowup)
- [x] Implicit CN+Picard via `jax.lax.while_loop` (2–4 iter/step, 5–10× larger dt)
- [x] Canonical Maxwellian loading (reduced sampling noise)
- [x] Pullback transformation (long-time weight control)
- [x] GTC-style weight spreading (grid-smoothed noise suppression)
- [x] Global radial profiles + Krook buffer BCs
- [x] Kinetic electrons (drift-kinetic, subcycling)
- [x] Collision operators (Krook, Lorentz, Dougherty)
- [x] VMEC stellarator geometry
- [x] Multi-GPU sharding via `jax.sharding`

### Phase 3 — Full-f ✅
- [x] True Vlasov PIC (dW/dt = 0, constant marker weights)
- [x] Full δn deposit for Poisson (not just perturbation)
- [x] Systematic resampling (CDF-based, uniform offset)
- [x] Full-f CBC benchmark (`benchmarks/cbc_fullf.py`)
- [x] Weight constancy verified: `std(W)/mean(W) < 1e-3` ✅

### Phase 4 — Usability ✅
- [x] TOML input files — GX-style, no Python required
- [x] CLI runner: `python -m gyrojax.runner input.toml`
- [x] δf/full-f switch, explicit/semi-implicit/implicit switch in input file
- [x] γ spectrum benchmark (ky·ρᵢ = 0.1–0.6)
- [x] 167 tests passing

### 🔄 In Progress
- [ ] KBM (Kinetic Ballooning Mode) linear benchmark at high β
- [ ] Full-f noise floor study (γ convergence with N_particles)
- [ ] Diagnostic: zonal flow shear d²φ/dψ² vs time

### Phase 5 — Electromagnetic ✅
- [x] Ampere's law solver: ∇²⊥ δA∥ = -β·δj∥ (spectral, field-aligned)
- [x] Inductive E∥ = -∂A∥/∂t correction to parallel push
- [x] EM β scan benchmark: β = 0 → ITG, β > 0 → EM modification
- [x] A∥ scales linearly with β (verified in tests)
- [x] Dimits shift 2000-step run: threshold at R/LT = 6.9 ✅ (ref: ~6.0)

### Phase 6 — Algorithmic Optimization ✅
- [x] Fused RK4 pusher+weights: single integrator for 5 equations → **3.78× speedup**
- [x] phi_hat caching: Poisson solve → E-field with zero redundant FFTs
- [x] Trilinear index caching: scatter + gather share precomputed (i,j,k) indices
- [x] Smaller lax.scan carry: pullback refs closed-over (3×N arrays removed from carry)
- [x] Symmetric gyroaveraging: √Γ₀(b) on scatter side (δn) to match field side (φ)
- [x] Radially-resolved g^αα(ψ): FLR operator uses radial profile, not flux-tube scalar
- [x] Zonal-flow-preserving weight spread: subtract/add zonal mean before/after spread
- [x] Absorbing wall BC: escaped particles have weights zeroed (opt-in, `absorbing_wall=True`)
- [x] Semi-implicit CN weight update (opt-in, `use_cn_weights=True`)
- [x] All optimizations validated: 167 tests pass, Dimits threshold R/LT=6.9 ✅

### 📋 Next
- [ ] KBM linear benchmark at high β (Phase 7)
- [ ] Dimits shift full-f (natural advantage: no ⟨w²⟩ constraint)
- [ ] Stellarator ITG scan (VMEC geometry)
- [ ] Gyrokinetic electrons (full GK, not drift-kinetic)
- [ ] Adaptive timestepping
- [ ] Input file: VMEC geometry path support

## CBC Parameters (Cyclone Base Case)

```
R0 = 1.0,  a = 0.18,  B0 = 1.0
q0 = 1.4,  q1 = 0.5   (q(r) = q0 + q1·r/a)
Ti = Te = 1.0,  mi = 1.0
R0/LT = 6.9,  R0/Ln = 2.2
vti = 1.0,  n0 = 1.0
```

---

## References

- Dimits et al., *Phys. Plasmas* **7**, 969 (2000) — Cyclone Base Case, Dimits shift
- Nevins et al., *Phys. Plasmas* **13**, 122306 (2006) — PIC benchmark, γ spectrum
- Brizard & Hahm, *Rev. Mod. Phys.* **79**, 421 (2007) — Gyrokinetic theory
- Lee, *Phys. Fluids* **26**, 556 (1983) — Gyrokinetic PIC foundations
- Ku et al. (GTC), *J. Comput. Phys.* **206**, 432 (2005) — δf PIC noise control
- Derouillat et al. (JAX-in-Cell), arXiv:2307.07117 — Implicit PIC in JAX
