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
import functools
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


@functools.partial(jax.jit, static_argnames=('use_radial_gaa',))
def solve_poisson_fa(
    delta_n: jnp.ndarray,
    geom: FieldAlignedGeometry,
    n0_avg: float,
    Te: float,
    Ti: float,
    mi: float,
    e: float,
    use_radial_gaa: bool = False,
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

    # Wavenumbers (cast to float32 — fftfreq returns float64 with x64 enabled)
    kpsi = (jnp.fft.fftfreq(Npsi,  d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kth  = (jnp.fft.fftfreq(Ntheta, d=dth) * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal) * 2 * jnp.pi).astype(jnp.float32)

    KPSI, KTH, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi, kth, kal, indexing='ij')]

    # g^{αα} at mid-radius, theta=0 (flux-tube approximation)
    # Using the full g_aa(psi,theta) in a 3D FFT operator causes the inner-psi
    # cells (small r, large g_aa ~ 1/r²) to dominate and suppress the response
    # at mid-radius where the particles live.  In a flux-tube code, the Poisson
    # operator is evaluated at the reference surface (r0, θ=0).
    Npsi_half = Npsi // 2
    Ntheta_half = Ntheta // 2
    g_aa_ref = geom.galphaalpha[Npsi_half, Ntheta_half]  # scalar: mid-psi, theta=0

    # Perpendicular wavenumber squared in field-aligned coords (flux-tube approx):
    #   k²⊥ = kψ²·g^{ψψ} + kα²·g^{αα}(r0)
    # (θ is the parallel direction so we exclude k_θ from k⊥)
    kperp_sq = KPSI**2 * 1.0 + KAL**2 * g_aa_ref   # (Npsi, Ntheta, Nalpha)

    # b = k⊥²·ρᵢ²/2
    b = kperp_sq * rho_i_sq / 2.0

    # Exact FLR operator
    G0 = _gamma0(b)   # Γ₀(b) = I₀(b)·e^{-b}

    # Radially-resolved Γ₀: use g^αα(ψ) profile for FLR correction per radial shell.
    # The inversion denominator still uses uniform g_aa_ref for spectral stability.
    if use_radial_gaa:
        g_aa_1d = geom.galphaalpha[:, Ntheta // 2]          # (Npsi,)
        kperp_sq_rad = KPSI**2 + KAL**2 * g_aa_1d[:, None, None]  # (Npsi, Ntheta, Nalpha)
        b_rad = kperp_sq_rad * rho_i_sq / 2.0
        G0 = _gamma0(b_rad)   # (Npsi, Ntheta, Nalpha) — overrides uniform G0

    # GK Poisson operator in Fourier space:
    #   op = (Te/Ti)·(1 - Γ₀) + ρᵢ²·k²⊥
    # (We divide out n₀·e/Te which cancels in (Te/e)·δn/n₀ form)
    op = (Te / Ti) * (1.0 - G0) + rho_i_sq * kperp_sq

    # Zero-out all purely parallel modes (kperp=0, kth≠0) — these have op≈0
    # and cannot be constrained by GK Poisson. Set phi=0 for these modes.
    # Also zero the (0,0,0) mean mode to fix gauge.
    kperp_zero = (kperp_sq < 1e-10)   # (Npsi, Ntheta, Nalpha)
    op = jnp.where(kperp_zero, 1.0, op)   # avoid /0; will zero phi_hat below

    # FFT of RHS: (Te/e) * δn/n₀
    # delta_n is dimensionless (δn/n₀ from scatter normalization)
    delta_n_hat = jnp.fft.fftn(delta_n.astype(jnp.complex64))
    rhs_hat = (Te / e) * delta_n_hat

    # Solve
    phi_hat = rhs_hat / op
    # Zero out all kperp=0 modes in phi (parallel modes undetermined by GK Poisson)
    phi_hat = jnp.where(kperp_zero, 0.0 + 0j, phi_hat)

    phi = jnp.fft.ifftn(phi_hat).real
    return phi, phi_hat


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def compute_efield_fa_from_hat(
    phi_hat: jnp.ndarray,
    geom: FieldAlignedGeometry,
    Npsi: int,
    Ntheta: int,
    Nalpha: int,
) -> tuple:
    """
    Compute E = -nabla phi from cached phi_hat (avoids redundant fftn).
    static_argnums ensures Npsi/Ntheta/Nalpha are compile-time constants.

    Returns E_psi, E_theta, E_alpha each (Npsi, Ntheta, Nalpha).
    """
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dth  = 2.0 * jnp.pi / Ntheta
    dal  = 2.0 * jnp.pi / Nalpha

    kpsi = (jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kth  = (jnp.fft.fftfreq(Ntheta, d=dth)  * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal)  * 2 * jnp.pi).astype(jnp.float32)
    KPSI, KTH, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi, kth, kal, indexing='ij')]

    E_psi_hat = 1j * (-KPSI) * phi_hat
    E_th_hat  = 1j * (-KTH)  * phi_hat
    E_al_hat  = 1j * (-KAL)  * phi_hat

    E_psi = jnp.fft.ifftn(E_psi_hat).real
    E_th  = jnp.fft.ifftn(E_th_hat).real
    E_al  = jnp.fft.ifftn(E_al_hat).real

    return E_psi, E_th, E_al


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

    kpsi = (jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kth  = (jnp.fft.fftfreq(Ntheta, d=dth)  * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal)  * 2 * jnp.pi).astype(jnp.float32)
    KPSI, KTH, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi, kth, kal, indexing='ij')]

    phi_hat = jnp.fft.fftn(phi.astype(jnp.complex64))

    # Stack k-vectors and do one batched ifftn: shape (3, Npsi, Ntheta, Nalpha)
    K_stack = jnp.stack([-KPSI, -KTH, -KAL], axis=0)   # (3, Npsi, Ntheta, Nalpha)
    E_hat_stack = 1j * K_stack * phi_hat[None]           # broadcast phi_hat
    E_stack = jnp.fft.ifftn(E_hat_stack, axes=(-3, -2, -1)).real  # ifftn over last 3 axes

    return E_stack[0], E_stack[1], E_stack[2]


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

    kpsi = (jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal)  * 2 * jnp.pi).astype(jnp.float32)
    KPSI, _, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi,
                                  jnp.zeros(Ntheta),
                                  kal, indexing='ij')]

    g_aa = geom.galphaalpha[:, :, None]
    kperp_sq = KPSI**2 + KAL**2 * g_aa
    b = kperp_sq * rho_i**2 / 2.0
    J0_op = jnp.sqrt(jnp.maximum(i0e(b), 1e-10))   # Γ₀^{1/2}(b) ≥ 0

    phi_hat = jnp.fft.fftn(phi.astype(jnp.complex64))
    phi_gyro = jnp.fft.ifftn(J0_op * phi_hat).real
    return phi_gyro


def gyroaverage_delta_n(
    delta_n: jnp.ndarray,
    geom: FieldAlignedGeometry,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    Apply sqrt(Gamma0(b)) gyroaveraging operator to scattered delta_n in Fourier space.

    delta_n_tilde = Gamma0^{1/2}(b) * delta_n_hat  (inverse FFT back to real space)

    This is the correct GK scatter gyroaverage. Without this, the source term
    in GK Poisson is too large at high k_perp (FLR effects underestimated).

    b = k_perp^2 * rho_i^2 / 2  where k_perp^2 = kpsi^2 + kalpha^2 * g^{alphaalpha}(r0)
    Gamma0^{1/2}(b) = sqrt(I0(b) * exp(-b)) = i0e(b)^{1/2}
    """
    Npsi, Ntheta, Nalpha = delta_n.shape
    dpsi = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)
    dal  = 2.0 * jnp.pi / Nalpha

    Omega_i = e * geom.B0 / mi
    rho_i_sq = Ti / (mi * Omega_i**2)

    kpsi = (jnp.fft.fftfreq(Npsi,  d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal) * 2 * jnp.pi).astype(jnp.float32)
    KPSI, _, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(
        kpsi,
        jnp.zeros(Ntheta, dtype=jnp.float32),
        kal, indexing='ij'
    )]

    Npsi_half = Npsi // 2
    Ntheta_half = Ntheta // 2
    g_aa_ref = geom.galphaalpha[Npsi_half, Ntheta_half]
    kperp_sq = KPSI**2 + KAL**2 * g_aa_ref
    b = kperp_sq * rho_i_sq / 2.0

    # sqrt(Gamma0(b)) = sqrt(I0(b)*exp(-b))
    sqrt_G0 = jnp.sqrt(jnp.maximum(i0e(b), 0.0))  # safe sqrt

    dn_hat = jnp.fft.fftn(delta_n.astype(jnp.complex64))
    dn_hat_ga = dn_hat * sqrt_G0  # apply sqrt(Gamma0)
    return jnp.fft.ifftn(dn_hat_ga).real.astype(jnp.float32)


def filter_single_mode(phi: jnp.ndarray, k_mode: int) -> jnp.ndarray:
    """
    Project phi onto a single binormal Fourier mode ±k_mode.

    Zeros all alpha-Fourier modes except k_alpha = ±k_mode.
    Used for linear benchmark runs to isolate single-mode growth.

    Parameters
    ----------
    phi    : (Npsi, Ntheta, Nalpha)
    k_mode : toroidal/binormal mode index to keep

    Returns
    -------
    phi_filtered : (Npsi, Ntheta, Nalpha)
    """
    Nalpha = phi.shape[-1]
    phi_hat = jnp.fft.fft(phi, axis=-1)   # FFT over alpha axis

    # Build mask: keep only indices k_mode and Nalpha-k_mode
    mask = jnp.zeros(Nalpha, dtype=jnp.bool_)
    # k_mode index and its conjugate (Nalpha - k_mode)
    k_pos = k_mode % Nalpha
    k_neg = (Nalpha - k_mode) % Nalpha
    mask = mask.at[k_pos].set(True)
    mask = mask.at[k_neg].set(True)
    # Also keep DC (k=0) only if k_mode == 0
    if k_mode == 0:
        mask = mask.at[0].set(True)

    phi_hat_filtered = jnp.where(mask[None, None, :], phi_hat, 0.0 + 0j)
    return jnp.fft.ifft(phi_hat_filtered, axis=-1).real


@jax.jit
def solve_poisson_tridiag(
    delta_n: jnp.ndarray,       # (Npsi, Ntheta, Nalpha) — perturbed density
    geom: FieldAlignedGeometry,
    n0_profile: jnp.ndarray,    # (Npsi,) — background density profile
    Te: float,
    Ti: float,
    mi: float,
    e: float,
) -> jnp.ndarray:
    """
    GTC-style tridiagonal GK Poisson solver in field-aligned coordinates.

    For each Fourier mode (kθ, kα), solves the 1D radial ODE:
        -d/dr[n₀(r)·ρᵢ²(r)·dφ/dr] + n₀(r)·(1-Γ₀(b))·(Te/Ti)·φ = (Te/Ti)·δn̂(r,kθ,kα)

    Discretized as a tridiagonal system with Dirichlet BCs (φ=0 at r=r_inner, r=a).

    Parameters
    ----------
    delta_n    : (Npsi, Ntheta, Nalpha) perturbed ion density
    geom       : FieldAlignedGeometry
    n0_profile : (Npsi,) background density profile
    Te, Ti     : electron/ion temperatures
    mi         : ion mass
    e          : elementary charge

    Returns
    -------
    phi : (Npsi, Ntheta, Nalpha) electrostatic potential (real)
    """
    Npsi, Ntheta, Nalpha = delta_n.shape
    Nmodes = Ntheta * Nalpha

    # Radial grid spacing
    dr = (geom.psi_grid[-1] - geom.psi_grid[0]) / (Npsi - 1)

    # B at mid-theta, shape (Npsi,)
    B_mid = geom.B_field[:, Ntheta // 2, 0]   # (Npsi,)

    # Ion gyroradius squared: ρᵢ²(r) = Ti / (mi * (e*B/mi)²) = Ti*mi / (e*B)²
    rho_sq = Ti * mi / (e * B_mid) ** 2         # (Npsi,)

    # g^αα at mid-theta: shape (Npsi,)
    g_aa_mid = geom.galphaalpha[:, Ntheta // 2]  # (Npsi,)

    # Wavenumber arrays
    dal = 2.0 * jnp.pi / Nalpha
    dth = 2.0 * jnp.pi / Ntheta
    kth_arr = (jnp.fft.fftfreq(Ntheta, d=dth) * 2 * jnp.pi).astype(jnp.float32)
    kal_arr = (jnp.fft.fftfreq(Nalpha, d=dal) * 2 * jnp.pi).astype(jnp.float32)

    # Build kperp_sq for all modes: shape (Nmodes, Npsi)
    KTH, KAL = jnp.meshgrid(kth_arr, kal_arr, indexing='ij')  # (Ntheta, Nalpha)
    KTH_flat = KTH.reshape(Nmodes)   # (Nmodes,)
    KAL_flat = KAL.reshape(Nmodes)   # (Nmodes,)

    # k⊥² = kα²·g^{αα}(r)  (kθ is parallel; radial kψ handled by finite diff)
    # shape (Nmodes, Npsi)
    kperp_sq = (KAL_flat[:, None] ** 2) * g_aa_mid[None, :]   # (Nmodes, Npsi)

    # b = k⊥²·ρᵢ²/2, Gamma0
    b = kperp_sq * rho_sq[None, :] / 2.0                     # (Nmodes, Npsi)
    G0 = i0e(b)                                               # (Nmodes, Npsi)

    # --- Build tridiagonal coefficients for all modes ---
    # a_i = c_i = -n0[i]*rho_sq[i]/dr²
    # b_i = 2*n0[i]*rho_sq[i]/dr² + n0[i]*(1-G0[i])*(Te/Ti)
    n0 = n0_profile  # (Npsi,)
    coeff_diff = n0[None, :] * rho_sq[None, :] / dr ** 2     # (1, Npsi) — same for all modes
    coeff_diff = jnp.broadcast_to(coeff_diff, (Nmodes, Npsi))  # (Nmodes, Npsi)

    a_diag = -coeff_diff                                      # (Nmodes, Npsi) lower
    c_diag = -coeff_diff                                      # (Nmodes, Npsi) upper
    b_diag = 2.0 * coeff_diff + n0[None, :] * (1.0 - G0) * (Te / Ti)  # (Nmodes, Npsi)

    # Dirichlet BC: rows 0 and Npsi-1 → φ=0 (using masks to avoid JAX at[] broadcasting issues)
    bc_mask_0 = jnp.zeros(Npsi, dtype=bool).at[0].set(True)    # (Npsi,)
    bc_mask_n = jnp.zeros(Npsi, dtype=bool).at[-1].set(True)   # (Npsi,)
    bc_mask = bc_mask_0 | bc_mask_n                             # (Npsi,)

    # On BC rows: b=1, a=0, c=0
    b_diag = jnp.where(bc_mask[None, :], 1.0, b_diag)
    a_diag = jnp.where(bc_mask[None, :], 0.0, a_diag)
    c_diag = jnp.where(bc_mask[None, :], 0.0, c_diag)

    # --- RHS: (Te/Ti)*delta_n_hat, with BC zeros ---
    # FFT in theta and alpha only
    delta_n_hat = jnp.fft.fft2(delta_n.astype(jnp.complex64), axes=(1, 2))  # (Npsi, Ntheta, Nalpha)
    # Reshape to (Nmodes, Npsi)
    rhs = (Te / Ti) * delta_n_hat.transpose(1, 2, 0).reshape(Nmodes, Npsi)  # (Nmodes, Npsi)
    rhs = jnp.where(bc_mask[None, :], jnp.zeros((), dtype=jnp.complex64), rhs.astype(jnp.complex64))

    # Cast real coefficients to complex64 for unified Thomas solve
    a_diag = a_diag.astype(jnp.complex64)
    b_diag = b_diag.astype(jnp.complex64)
    c_diag = c_diag.astype(jnp.complex64)

    # --- Thomas algorithm via jax.lax.scan (vectorized over modes via vmap) ---
    def thomas_single(a, b, c, d):
        """Solve one tridiagonal system of size Npsi."""
        # Forward sweep
        def fwd_step(carry, x):
            b_prev, d_prev, c_prev = carry
            a_i, b_i, c_i, d_i = x
            w = a_i / b_prev
            b_new = b_i - w * c_prev
            d_new = d_i - w * d_prev
            return (b_new, d_new, c_i), (b_new, d_new)

        # Initial carry: row 0 (already has BC applied)
        init_carry = (b[0], d[0], c[0])
        # Scan over rows 1..N-1
        _, (b_mod, d_mod) = jax.lax.scan(fwd_step, init_carry,
                                          (a[1:], b[1:], c[1:], d[1:]))
        # Prepend row 0
        b_full = jnp.concatenate([b[0:1], b_mod])
        d_full = jnp.concatenate([d[0:1], d_mod])

        # Back substitution
        def bwd_step(x_next, idx):
            i = Npsi - 2 - idx
            xi = (d_full[i] - c[i] * x_next) / b_full[i]
            return xi, xi

        x_last = d_full[-1] / b_full[-1]
        _, x_inner = jax.lax.scan(bwd_step, x_last, jnp.arange(Npsi - 1))
        x_inner = x_inner[::-1]
        x = jnp.concatenate([x_inner, x_last[None]])
        return x

    # Vectorize Thomas over all modes
    thomas_vmap = jax.vmap(thomas_single)
    phi_modes = thomas_vmap(a_diag, b_diag, c_diag, rhs)  # (Nmodes, Npsi)

    # Reshape back to (Ntheta, Nalpha, Npsi) then transpose to (Npsi, Ntheta, Nalpha)
    phi_hat = phi_modes.reshape(Ntheta, Nalpha, Npsi).transpose(2, 0, 1)

    phi = jnp.fft.ifft2(phi_hat, axes=(1, 2)).real
    return phi


def compute_growth_rate(
    phi_history: jnp.ndarray,
    dt: float,
    n_fit: int = 50,
) -> dict:
    """
    Compute linear growth rate γ and real frequency ω from phi time history.
    Like GTC: extracts dominant mode, fits log|A(t)| for γ, phase for ω.

    Parameters
    ----------
    phi_history : (n_steps, Npsi, Ntheta, Nalpha) OR (n_steps,) if already rms
    dt          : timestep
    n_fit       : number of steps to use for linear fit

    Returns
    -------
    dict with keys: gamma (growth rate), omega (frequency),
                    mode_amplitude (array of dominant mode amplitude vs time)
    """
    if phi_history.ndim == 1:
        amp = phi_history
        omega = 0.0
        n_steps = amp.shape[0]
        t = jnp.arange(n_steps) * dt
        t_fit = t[-n_fit:]
        amp_fit = amp[-n_fit:]
        log_amp = jnp.log(jnp.maximum(amp_fit, 1e-30))
        A = jnp.stack([t_fit, jnp.ones_like(t_fit)], axis=1)
        coeffs, _, _, _ = jnp.linalg.lstsq(A, log_amp, rcond=None)
        gamma = float(coeffs[0])
        return {'gamma': gamma, 'omega': omega, 'mode_amplitude': amp}

    # FFT over spatial dims
    phi_hat = jnp.fft.fftn(phi_history, axes=(1, 2, 3))  # (n_steps, Npsi, Ntheta, Nalpha)

    # Find dominant mode at mid-psi, time-averaged
    mode_amp = jnp.abs(phi_hat[:, phi_history.shape[1] // 2, :, :])  # (n_steps, Ntheta, Nalpha)
    avg_amp = jnp.mean(mode_amp, axis=0)  # (Ntheta, Nalpha)
    flat_idx = jnp.argmax(avg_amp)
    jth = flat_idx // phi_history.shape[3]
    jal = flat_idx % phi_history.shape[3]

    amp = jnp.abs(phi_hat[:, phi_history.shape[1] // 2, jth, jal])

    n_steps = amp.shape[0]
    t = jnp.arange(n_steps) * dt
    t_fit = t[-n_fit:]
    amp_fit = amp[-n_fit:]
    log_amp = jnp.log(jnp.maximum(amp_fit, 1e-30))

    A = jnp.stack([t_fit, jnp.ones_like(t_fit)], axis=1)
    coeffs, _, _, _ = jnp.linalg.lstsq(A, log_amp, rcond=None)
    gamma = float(coeffs[0])

    # Frequency from phase of complex mode amplitude
    complex_amp = phi_hat[-n_fit:, phi_history.shape[1] // 2, jth, jal]
    phase = jnp.angle(complex_amp)
    phase_unwrap = jnp.unwrap(phase)
    coeffs_ph, _, _, _ = jnp.linalg.lstsq(A, phase_unwrap, rcond=None)
    omega = float(coeffs_ph[0])

    return {
        'gamma': gamma,
        'omega': omega,
        'mode_amplitude': amp,
    }


def project_modes(phi: jnp.ndarray, n_modes: int = 8) -> dict:
    """
    Project phi onto poloidal/toroidal Fourier modes.
    Returns dict: {(m,n): complex_amplitude} for top n_modes modes.
    Like GTC's per-mode phi_real + i*phi_imag output.

    Parameters
    ----------
    phi     : (Npsi, Ntheta, Nalpha)
    n_modes : number of dominant modes to return

    Returns
    -------
    dict mapping (m_idx, n_idx) → complex amplitude at mid-psi
    """
    Npsi, Ntheta, Nalpha = phi.shape
    mid = Npsi // 2
    phi_hat = jnp.fft.fft2(phi[mid], axes=(0, 1))  # (Ntheta, Nalpha)

    # Find top n_modes by amplitude
    amps = jnp.abs(phi_hat)
    flat_idx = jnp.argsort(amps.ravel())[-n_modes:][::-1]

    result = {}
    for idx in flat_idx.tolist():
        m = int(idx) // Nalpha
        n = int(idx) % Nalpha
        result[(m, n)] = complex(phi_hat[m, n])
    return result


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

    kpsi = (jnp.fft.fftfreq(Npsi,   d=dpsi) * 2 * jnp.pi).astype(jnp.float32)
    kth  = (jnp.fft.fftfreq(Ntheta, d=2.0*jnp.pi/Ntheta) * 2 * jnp.pi).astype(jnp.float32)
    kal  = (jnp.fft.fftfreq(Nalpha, d=dal)  * 2 * jnp.pi).astype(jnp.float32)
    KPSI, KTH, KAL = [x.astype(jnp.float32) for x in jnp.meshgrid(kpsi, kth, kal, indexing='ij')]

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
