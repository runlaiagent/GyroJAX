# Physics Model

## Gyrokinetic δf PIC Formulation

GyroJAX implements the **δf particle-in-cell** method for gyrokinetic simulation. Rather than tracking the full distribution function f, we track only the perturbation δf = f − f₀, where f₀ is a fixed Maxwellian background. Each simulation marker carries a weight w = δf/f₀, which evolves along guiding-center trajectories according to:

$$\frac{dw}{dt} = -(1 - w) \cdot S, \qquad S = (\mathbf{v}_E + \mathbf{v}_d) \cdot \nabla \ln f_0$$

where **v**_E is the E×B drift and **v**_d includes magnetic gradient and curvature drifts. This formulation greatly reduces statistical noise compared to full-f PIC, since weights remain small (|w| ≪ 1) in the linear regime. The guiding-center equations of motion are integrated with 4th-order Runge-Kutta (RK4), with an optional fused variant that integrates positions and weights in a single kernel for 3.78× GPU speedup.

## Flux-Tube Geometry (ψ, θ, α Coordinates)

The simulation domain is a **flux-tube** — a thin volume following a single magnetic field line, sufficient to capture local turbulence at a given minor radius. Coordinates are:

- **ψ** (radial): labels magnetic flux surfaces, discretized on a uniform grid with Npsi points
- **θ** (poloidal): follows the field line around the torus, discretized with Ntheta points
- **α = φ − q(ψ)θ** (field-line label / binormal): the toroidal angle minus the safety-factor-weighted poloidal angle, discretized with Nalpha points

The safety factor profile q(r) = q₀ + q₁·(r/a) determines field-line pitch. Metric coefficients g^αα(ψ) vary radially and are computed from the s-α equilibrium model, capturing magnetic shear effects on FLR operators. The geometry is set up via the `FieldAlignedGeometry` object, which precomputes all necessary metric tensors and is passed to each simulation function.

## Poisson Equation with Gyroaveraging (Γ₀ Operator)

The electrostatic potential φ is determined by the gyrokinetic quasi-neutrality condition:

$$\left[\frac{n_0}{T_e}(1 - \Gamma_0) + \frac{n_0}{T_i}\Gamma_0\right] e\phi = \delta n_i^{(\text{gyro})}$$

where Γ₀(b) = I₀(b)e^{−b} is the gyroaveraging operator, b = k⊥²ρᵢ²/2, and δnᵢ^(gyro) is the gyro-averaged ion density perturbation. The Poisson solve is implemented spectrally: FFT in α (toroidal), tridiagonal solve in ψ (radial), and spectral decomposition in θ (poloidal). GyroJAX applies **symmetric gyroaveraging** — the √Γ₀(b) operator is applied to both the scatter (δn) and gather (φ) steps, ensuring exact self-adjointness. The radially-resolved g^αα(ψ) profile is used in the FLR operator rather than a flux-tube-averaged scalar, correcting a common approximation that leads to errors at finite ρ*.

## Electromagnetic Extension: Ampere's Law for δA∥

When β > 0, GyroJAX solves a coupled electrostatic-electromagnetic system. In addition to Poisson's equation for φ, the parallel vector potential δA∥ is determined by Ampere's law:

$$-\nabla_\perp^2 \delta A_\parallel = \mu_0 \delta j_\parallel$$

where δj∥ is the parallel current perturbation from the gyrokinetic distribution. The particle push is modified to include the **v∥ × δB** force from the perturbed magnetic field δB = ∇ × (δA∥ ê∥). This allows simulation of kinetic ballooning modes (KBM), which are driven by pressure gradients and stabilized by magnetic tension. The β_crit for KBM onset is ≈ 0.010–0.012 for CBC parameters, consistent with gyrofluid predictions.

## Drift-Kinetic Electrons for TEM

Trapped electron modes (TEM) are driven by the electron temperature gradient and require kinetic (not adiabatic) electron treatment. GyroJAX implements a **drift-kinetic hybrid model**: electrons are pushed with the drift-kinetic equations (no gyroaveraging, since ρe ≪ ρi) using a subcycling scheme with `subcycles_e` inner steps per ion step to handle the faster electron dynamics. The electron distribution contributes to the quasi-neutrality condition alongside the ion gyrokinetic response. TEM is observed for R₀/LTe ≳ 7–9 at CBC-like parameters; below ~5 the electron drive is insufficient for instability.

## Weight Equation and Pullback Transformation

The δf weight equation `dw/dt = -(1-w)·S` is exact only for w ≪ 1. In long nonlinear simulations, weight growth can lead to numerical blow-up. GyroJAX includes several stabilization mechanisms:

- **Semi-implicit CN weights**: The weight update is solved analytically as w^{n+1} = 1 − (1 − wⁿ)/(1 + dt·S), which is unconditionally bounded and prevents weight divergence
- **Pullback transformation** (`use_pullback=True`): Periodically maps the perturbed distribution back toward f₀ by subtracting the coarse-grained δf from the weights. This controls secular weight growth without altering the physical content
- **Weight spreading**: GTC-style scatter of weights to a grid, smoothing, and re-gathering, which suppresses particle-level noise while optionally preserving the zonal flow component

These mechanisms together allow stable nonlinear simulations over hundreds of turbulence correlation times.
