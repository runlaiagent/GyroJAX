"""
Gyrokinetic Poisson solver with full Γ₀(b) FLR operator in field-aligned coords.

Solves the GK quasineutrality equation:
    -∇·(n₀·ρ²ᵢ·∇⊥φ) + (n₀·e/Tₑ)·(1 - Γ₀(b))·φ = (e/Tₑ)·⟨δnᵢ⟩

In the long-wavelength limit this reduces to the Padé form; here we use the
exact FLR operator:

    Γ₀(b) = I₀(b)·exp(-b)     b = k²⊥·ρ²ᵢ/2

Applied spectrally (exact in Fourier space):
    [n₀e/Tₑ · (1 - Γ₀(b)) + n₀·ρ²ᵢ·k²⊥] · φ̂ = (e/Tₑ)·δn̂ᵢ

In field-aligned (ψ, θ, α) coordinates:
    k²⊥ = kψ²·g^{ψψ} + kα²·g^{αα}(ψ,θ)

Note on the Padé vs Γ₀ difference:
    Padé:  Γ₀(b) ≈ 1/(1+b)   — accurate only for b << 1
    Exact: Γ₀(b) = I₀(b)e^{-b} — correct for all b
    At ky·ρᵢ = 0.3, b ~ 0.045 → error ~2% from Padé
    At ky·ρᵢ = 1.0, b ~ 0.5  → error ~20% from Padé

References:
    Hatzky et al. (2002) Phys. Plasmas 9, 898
    Lapillonne et al. (2009) Phys. Plasmas 16, 032308
    Jolliet et al. (2007) Comput. Phys. Commun. 177, 409
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import i0e   # I₀(x)·exp(-x)  — numerically stable
from gyrojax.geometry.field_aligned import FieldAlignedGeometry


def _gamma0(b: jnp.ndarray) -> jnp.ndarray:
    """
    Exact FLR operator: Γ₀(b) = I₀(b)·exp(-b).

    Uses JAX's i0e(b) = I₀(b)·exp(-|b|), which is numerically stable for
    large b.  For b >= 0 this equals Γ₀(b) exactly.

    b = k⊥²·ρᵢ²/2  (dimensionless, non-negative)
    """
    return i0e(b)   # i0e(b) = I₀(b)·exp(-b) for b ≥ 0


@jax.jit
def solve_poisson_fa(
    delta_n: jnp.ndarray,
    geom: FieldAlignedGeometry,
    n0_avg: float,
    Te: float,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    Solve GK Poisson in field-aligned coordinates with exact Γ₀(b).

    GK quasineutrality:
        [n₀e/Tₑ · (1 - Γ₀(b)) + n₀·ρᵢ²·k²⊥] · φ̂ = (e/Tₑ)·⟨δnᵢ⟩̂

    This simplifies to the adiabatic-electron GK Poisson:
        (1 - Γ₀(b) + ρᵢ²·k²⊥ · Tₑ/Tᵢ) · φ̂ = (Tₑ/n₀e) · δn̂

    Parameters
    ----------
    delta_n : (Npsi, Ntheta, Nalpha)  perturbed ion density (gyroaveraged)
    geom    : FieldAlignedGeometry
    n0_avg  : background density
    Te, Ti  : electron / ion temperatures
    mi      : ion mass
    e       : elementary charge

    Returns
    -------
    phi : (Npsi, Ntheta, Nalpha)  electrostatic potential
    """
    Npsi, Ntheta, Nalpha = delta_n.shape

    # Grid spacings
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dth  = 2.0 * jnp.pi / Ntheta
    dal  = 2.0 * jnp.pi / Nalpha

    # Ion thermal gyroradius:  ρᵢ² = Tᵢ / (mᵢ·Ωᵢ²) = Tᵢ·mᵢ / (e·B₀)²
    Omega_i = e * geom.B0 / mi
    rho_i_sq = Ti / (mi * Omega_i**2)

    # Wavenumbers
    kpsi = jnp.fft.fftfreq(Npsi,  d=dpsi) * 2 * jnp.pi    # (Npsi,)
    kth  = jnp.fft.fftfreq(Ntheta, d=dth) * 2 * jnp.pi    # (Ntheta,)
    kal  = jnp.fft.fftfreq(Nalpha, d=dal) * 2 * jnp.pi    # (Nalpha,)

    KPSI, KTH, KAL = jnp.meshgrid(kpsi, kth, kal, indexing='ij')  # (Npsi,Ntheta,Nalpha)

    # g^{αα}(ψ,θ) = (q/r)²·(1 + ŝ²θ²)
    # Broadcast to (Npsi, Ntheta, Nalpha)
    g_aa = geom.galphaalpha[:, :, None]   # (Npsi, Ntheta, 1) → broadcasts

    # Perpendicular wavenumber squared in field-aligned coords:
    #   k²⊥ = kψ²·g^{ψψ} + kα²·g^{αα}
    # (θ is the parallel direction so we exclude k_θ from k⊥)
    kperp_sq = KPSI**2 * 1.0 + KAL**2 * g_aa   # (Npsi, Ntheta, Nalpha)

    # b = k⊥²·ρᵢ²/2
    b = kperp_sq * rho_i_sq / 2.0

    # Exact FLR operator
    G0 = _gamma0(b)   # Γ₀(b) = I₀(b)·e^{-b}

    # GK Poisson operator in Fourier space:
    #   op = (Te/Ti)·(1 - Γ₀) + ρᵢ²·k²⊥
    # (We divide out n₀·e/Te which cancels in (Te/e)·δn/n₀ form)
    op = (Te / Ti) * (1.0 - G0) + rho_i_sq * kperp_sq

    # Zero-out all purely parallel modes (kperp=0, kth≠0) — these have op≈0
    # and cannot be constrained by GK Poisson. Set phi=0 for these modes.
    # Also zero the (0,0,0) mean mode to fix gauge.
    kperp_zero = (kperp_sq < 1e-10)   # (Npsi, Ntheta, Nalpha)
    op = jnp.where(kperp_zero, 1.0, op)   # avoid /0; will zero phi_hat below

    # FFT of RHS: (Te/n₀e) * δn
    delta_n_hat = jnp.fft.fftn(delta_n.astype(jnp.complex64))
    rhs_hat = (Te / (n0_avg * e)) * delta_n_hat

    # Solve
    phi_hat = rhs_hat / op
    # Zero out all kperp=0 modes in phi (parallel modes undetermined by GK Poisson)
    phi_hat = jnp.where(kperp_zero, 0.0 + 0j, phi_hat)

    phi = jnp.fft.ifftn(phi_hat).real
    return phi


@jax.jit
def compute_efield_fa(
    phi: jnp.ndarray,
    geom: FieldAlignedGeometry,
) -> tuple:
    """
    Compute electric field E = -∇φ in field-aligned coordinates via spectral diff.

    Returns
    -------
    E_psi, E_theta, E_alpha : each shape (Npsi, Ntheta, Nalpha)
    """
    Npsi, Ntheta, Nalpha = phi.shape
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dth  = 2.0 * jnp.pi / Ntheta
    dal  = 2.0 * jnp.pi / Nalpha

    kpsi = jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi
    kth  = jnp.fft.fftfreq(Ntheta, d=dth ) * 2 * jnp.pi
    kal  = jnp.fft.fftfreq(Nalpha, d=dal ) * 2 * jnp.pi
    KPSI, KTH, KAL = jnp.meshgrid(kpsi, kth, kal, indexing='ij')

    phi_hat = jnp.fft.fftn(phi.astype(jnp.complex64))

    E_psi_hat  = -1j * KPSI * phi_hat
    E_th_hat   = -1j * KTH  * phi_hat
    E_al_hat   = -1j * KAL  * phi_hat

    E_psi  = jnp.fft.ifftn(E_psi_hat).real
    E_th   = jnp.fft.ifftn(E_th_hat ).real
    E_al   = jnp.fft.ifftn(E_al_hat ).real

    return E_psi, E_th, E_al


def gyroaverage_phi(
    phi: jnp.ndarray,
    geom: FieldAlignedGeometry,
    rho_i: float,
    n_ring_pts: int = 4,
) -> jnp.ndarray:
    """
    Apply gyroaveraging operator J₀ to φ using Γ₀(b) in Fourier space.

    ⟨φ⟩ = Γ₀^{1/2}(b) · φ̂  →  ifftn

    This is the field-level gyroaverage; for particles use the exact ring average.

    Parameters
    ----------
    phi : (Npsi, Ntheta, Nalpha)
    rho_i : ion Larmor radius (scalar)

    Returns
    -------
    phi_gyro : (Npsi, Ntheta, Nalpha) gyroaveraged potential
    """
    Npsi, Ntheta, Nalpha = phi.shape
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dal  = 2.0 * jnp.pi / Nalpha

    kpsi = jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi
    kal  = jnp.fft.fftfreq(Nalpha, d=dal ) * 2 * jnp.pi
    KPSI, _, KAL = jnp.meshgrid(kpsi,
                                  jnp.zeros(Ntheta),
                                  kal, indexing='ij')

    g_aa = geom.galphaalpha[:, :, None]
    kperp_sq = KPSI**2 + KAL**2 * g_aa
    b = kperp_sq * rho_i**2 / 2.0
    J0_op = jnp.sqrt(jnp.maximum(i0e(b), 1e-10))   # Γ₀^{1/2}(b) ≥ 0

    phi_hat = jnp.fft.fftn(phi.astype(jnp.complex64))
    phi_gyro = jnp.fft.ifftn(J0_op * phi_hat).real
    return phi_gyro


@jax.jit
def solve_poisson_pade_fa(
    delta_n: jnp.ndarray,
    geom: FieldAlignedGeometry,
    n0_avg: float,
    Te: float,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    Solve GK Poisson in field-aligned coords using Padé FLR: Γ₀(b) ≈ 1/(1+b).

    Provided for comparison with exact Γ₀ solver.
    """
    Npsi, Ntheta, Nalpha = delta_n.shape
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dal  = 2.0 * jnp.pi / Nalpha

    Omega_i  = e * geom.B0 / mi
    rho_i_sq = Ti / (mi * Omega_i**2)

    kpsi = jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi
    kth  = jnp.fft.fftfreq(Ntheta, d=2.0*jnp.pi/Ntheta) * 2 * jnp.pi
    kal  = jnp.fft.fftfreq(Nalpha, d=dal ) * 2 * jnp.pi
    KPSI, KTH, KAL = jnp.meshgrid(kpsi, kth, kal, indexing='ij')

    g_aa = geom.galphaalpha[:, :, None]
    kperp_sq = KPSI**2 + KAL**2 * g_aa
    b = kperp_sq * rho_i_sq / 2.0

    # Padé: Γ₀(b) ≈ 1/(1+b)  →  1 - Γ₀ ≈ b/(1+b)
    G0_pade = 1.0 / (1.0 + b)
    op = (Te / Ti) * (1.0 - G0_pade) + rho_i_sq * kperp_sq
    op = op.at[0, 0, 0].set(1.0)

    delta_n_hat = jnp.fft.fftn(delta_n.astype(jnp.complex64))
    phi_hat = (Te / (n0_avg * e)) * delta_n_hat / op
    phi_hat = phi_hat.at[0, 0, 0].set(0.0 + 0j)
    return jnp.fft.ifftn(phi_hat).real
