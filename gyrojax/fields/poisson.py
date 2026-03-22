# FILE: gyrojax/fields/poisson.py
"""
Gyrokinetic Poisson (quasineutrality) solver.

The gyrokinetic Poisson equation (electrostatic, adiabatic electrons):

    -∇⊥·(n0·mi/(e·B²)·∇⊥φ) + n0·e/Te·φ = δn_i^(gyroavg)

This can be written as:
    L[φ] = δn_i

where L is a linear operator. In Fourier space (θ, ζ periodic):

    For each mode (kθ, kζ):
        -d/dr(n0·mi/(e·B²)·dφ/dr) + (n0·mi·kperp²/(e·B²) + n0·e/Te)·φ = δn̂

This is solved as a tridiagonal system in r for each (kθ,kζ) mode,
then inverse-FFT'd back to real space.

For Phase 1 (quick start), we use the simplified long-wavelength limit:
    (1 - ρi²∇⊥²)φ ≈ (Te/e)·δn_i/n0

which is solved spectrally with FFT in all 3 directions.

References:
    Lee (1983) Phys. Fluids 26, 556  (original GK Poisson)
    Hahm et al. (1988) Phys. Fluids 31, 1940
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gyrojax.geometry.salpha import SAlphaGeometry


@jax.jit
def solve_poisson_gk(
    delta_n: jnp.ndarray,
    geom: SAlphaGeometry,
    n0_avg: float,
    Te: float,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    Solve the gyrokinetic Poisson equation spectrally.

    Uses the long-wavelength (Padé) approximation:
        (1 + (Ti/Te)*(1 - Γ0(b)))φ ≈ (Te/e) * δn_i / n0

    In the long-wavelength limit (b = k²ρi² << 1):
        Γ0(b) ≈ 1 - b,  so  (1 + (Ti/Te)*b) * φ = (Te/e)*δn/n0

    This is exact in Fourier space (spectral solve).

    Parameters
    ----------
    delta_n : (Nr, Ntheta, Nzeta)  gyroaveraged perturbed density
    geom    : SAlphaGeometry
    n0_avg  : average background density [m^-3]
    Te, Ti  : electron/ion temperature [J]
    mi      : ion mass [kg]
    e       : elementary charge [C]

    Returns
    -------
    phi : (Nr, Ntheta, Nzeta)  electrostatic potential [V]
    """
    Nr, Ntheta, Nzeta = delta_n.shape

    dr  = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    dth = 2.0 * jnp.pi / Ntheta
    dze = 2.0 * jnp.pi / Nzeta

    # Average B and ion gyroradius for FLR operator
    B_avg  = geom.B0
    Omega_i = e * B_avg / mi
    rho_i_sq = Ti / (mi * Omega_i**2)   # ρi² = vti²/Ωi² = Ti/(mi*Ωi²)

    # FFT in all 3 directions
    delta_n_hat = jnp.fft.fftn(delta_n)

    # Wavenumbers
    kr  = jnp.fft.fftfreq(Nr,  d=dr)  * 2 * jnp.pi
    kth = jnp.fft.fftfreq(Ntheta, d=dth) * 2 * jnp.pi
    kze = jnp.fft.fftfreq(Nzeta, d=dze)  * 2 * jnp.pi
    KR, KTH, KZE = jnp.meshgrid(kr, kth, kze, indexing='ij')

    # Perpendicular wavenumber squared (ζ is parallel → don't include kζ in kperp)
    kperp_sq = KR**2 + KTH**2   # (Nr, Ntheta, Nzeta)

    # GK Poisson operator in Fourier space:
    # (1 + (Ti/Te) * kperp²*ρi²) * φ̂ = (Te/e) * δn̂/n0
    # Handle zero mode separately (fix gauge)
    operator = 1.0 + (Ti / Te) * kperp_sq * rho_i_sq
    operator = operator.at[0, 0, :].set(1.0)  # remove zero-mode in (r,θ) to fix gauge

    rhs = (Te / e) * delta_n_hat / n0_avg

    phi_hat = rhs / operator
    phi_hat = phi_hat.at[0, 0, :].set(0.0)  # zero mean potential

    phi = jnp.fft.ifftn(phi_hat).real
    return phi


@jax.jit
def compute_efield(
    phi: jnp.ndarray,
    geom: SAlphaGeometry,
) -> tuple:
    """
    Compute electric field E = -∇φ using spectral differentiation.

    Returns
    -------
    E_r, E_theta, E_zeta : arrays of shape (Nr, Ntheta, Nzeta)
    """
    Nr, Ntheta, Nzeta = phi.shape
    dr  = (geom.r_grid[-1] - geom.r_grid[0]) / (Nr - 1)
    dth = 2.0 * jnp.pi / Ntheta
    dze = 2.0 * jnp.pi / Nzeta

    phi_hat = jnp.fft.fftn(phi)

    kr  = jnp.fft.fftfreq(Nr,     d=dr)  * 2 * jnp.pi
    kth = jnp.fft.fftfreq(Ntheta, d=dth) * 2 * jnp.pi
    kze = jnp.fft.fftfreq(Nzeta,  d=dze) * 2 * jnp.pi
    KR, KTH, KZE = jnp.meshgrid(kr, kth, kze, indexing='ij')

    E_r_hat     = -1j * KR  * phi_hat
    E_theta_hat = -1j * KTH * phi_hat
    E_zeta_hat  = -1j * KZE * phi_hat

    E_r     = jnp.fft.ifftn(E_r_hat).real
    E_theta = jnp.fft.ifftn(E_theta_hat).real
    E_zeta  = jnp.fft.ifftn(E_zeta_hat).real

    return E_r, E_theta, E_zeta


def gyroaverage_phi(
    phi: jnp.ndarray,
    rho_i: float,
) -> jnp.ndarray:
    """
    Apply gyroaveraging operator Γ0 to potential in Fourier space.

    Γ0(b) = I0(b)*exp(-b)  where b = k²ρi²
    Long-wavelength: Γ0 ≈ 1 - b

    Parameters
    ----------
    phi   : (Nr, Ntheta, Nzeta)
    rho_i : ion gyroradius [m]

    Returns
    -------
    gyroaveraged phi, same shape
    """
    Nr, Ntheta, Nzeta = phi.shape
    phi_hat = jnp.fft.fftn(phi)

    kr  = jnp.fft.fftfreq(Nr,     d=1.0/Nr)  * 2*jnp.pi / Nr
    kth = jnp.fft.fftfreq(Ntheta, d=1.0/Ntheta) * 2*jnp.pi / Ntheta
    kze = jnp.fft.fftfreq(Nzeta,  d=1.0/Nzeta)  * 2*jnp.pi / Nzeta
    KR, KTH, _ = jnp.meshgrid(kr, kth, kze, indexing='ij')

    b = (KR**2 + KTH**2) * rho_i**2
    # Use Bessel approximation: Γ0 = exp(-b) for Pade approximant
    gamma0 = jnp.exp(-b)

    return jnp.fft.ifftn(gamma0 * phi_hat).real
