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
- Gyroaveraging via Bessel function `Γ₀(b) = I₀(b)e^{-b}`, b = k²⊥ρᵢ²

### Benchmarks (δf)

| Benchmark | Result | Reference | Error |
|---|---|---|---|
| CBC linear growth rate (ky·ρᵢ = 0.30) | γ = 0.172 vti/R0 | 0.169 vti/R0 (GENE/GX) | **1.9%** |
| Rosenbluth-Hinton zonal flow residual | Stable, correct damping | — | ✅ |
| γ spectrum (ky·ρᵢ = 0.1–0.6) | Peak at ky·ρᵢ ≈ 0.35–0.40 | GENE spectrum | ~15% avg |
| Dimits shift threshold | In progress | R/LT ≈ 6.0 (Dimits 2000) | — |

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

| Benchmark | Status |
|---|---|
| CBC full-f growth rate | ✅ Implemented (`benchmarks/cbc_fullf.py`) |
| Full-f nonlinear saturation | In development |
| Dimits shift (full-f path) | Planned |

---

## Architecture

```
gyrojax/
├── geometry/
│   ├── salpha.py          # s-α field-aligned geometry
│   ├── vmec.py            # VMEC stellarator geometry
│   └── profiles.py        # radial profiles, Krook BCs
├── particles/
│   └── guiding_center.py  # GCState pytree + RK4 pusher
├── fields/
│   └── poisson.py         # GK Poisson solver (FFT + tridiagonal)
├── deltaf/
│   └── weights.py         # δf weight evolution (RK4, CN, semi-implicit)
├── fullf/
│   └── __init__.py        # Full-f marker weights + resampling
├── electrons/
│   └── __init__.py        # Adiabatic / drift-kinetic / GK electrons
├── collisions/
│   └── operators.py       # Krook, Lorentz, Dougherty collision operators
├── interpolation/
│   └── scatter_gather_fa.py  # Gyroaveraged scatter/gather (field-aligned)
├── diagnostics/
│   └── __init__.py        # Growth rate fitting, ky·ρᵢ estimation
├── simulation_fa.py        # δf field-aligned main loop (lax.scan)
├── simulation_fullf.py     # Full-f main loop
├── simulation_sharded.py   # Multi-GPU sharded δf loop
└── sharding.py             # JAX device sharding utilities
```

---

## Roadmap

### ✅ Done
- [x] Field-aligned s-α geometry with q(r) profile
- [x] δf GC pusher + RK4 weight advance
- [x] GK Poisson solver (FFT/tridiagonal, gyroaveraging)
- [x] CBC linear benchmark — γ = 0.172 (1.9% error vs GENE)
- [x] Rosenbluth-Hinton zonal flow test
- [x] Semi-implicit CN weight update (unconditionally bounded)
- [x] Implicit CN+Picard via `lax.while_loop`
- [x] Canonical Maxwellian loading
- [x] Pullback transformation
- [x] GTC-style weight spreading
- [x] Global radial profiles + Krook buffer BCs
- [x] Kinetic electrons (drift-kinetic, subcycling)
- [x] Collision operators (Krook, Lorentz, Dougherty)
- [x] VMEC stellarator geometry
- [x] γ spectrum benchmark (ky·ρᵢ = 0.1–0.6)
- [x] Multi-GPU sharding via `jax.sharding`
- [x] Full-f simulation path

### 🔄 In Progress
- [ ] Dimits shift nonlinear benchmark (zonal flow suppression at R/LT < 6)
- [ ] Full-f resampling (importance sampling / particle splitting)

### 📋 Planned
- [ ] Electromagnetic perturbations (δA∥)
- [ ] Nonlinear CBC heat flux benchmark (χᵢ vs R/LT)
- [ ] Stellarator ITG scan (VMEC geometry)
- [ ] Gyrokinetic electrons (full GK, not drift-kinetic)
- [ ] Adaptive timestepping

---

## Quick Start

```bash
conda activate jax_wlhx
pip install -e .

# Run tests (147 tests)
pytest tests/ -v

# CBC linear growth rate benchmark
python benchmarks/cyclone_base_case_fa.py

# γ(ky·ρᵢ) spectrum
python benchmarks/gamma_spectrum.py --quick

# Dimits shift scan
python benchmarks/dimits_shift.py

# Full-f CBC
python benchmarks/cbc_fullf.py
```

---

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
