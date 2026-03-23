"""
VMEC geometry module for GyroJAX — Phase 2b.

Reads a VMEC wout_*.nc file and builds a geometry object compatible with
the FieldAlignedGeometry interface used in Phase 2a.

Coordinate system: straight-field-line (Boozer-like) VMEC coordinates
  s      = normalized toroidal flux ∈ [0, 1]   (radial)
  theta  = VMEC poloidal angle  ∈ [0, 2π)
  zeta   = VMEC toroidal angle  ∈ [0, 2π/nfp)  (one field period)

The field-line label in VMEC: α = theta - ι·zeta (ι = rotational transform = 1/q)

Outputs (compatible with FieldAlignedGeometry fields):
  B_field       : |B| on (Ns, Ntheta, Nzeta) grid
  gradB_psi     : ∂|B|/∂s
  gradB_th      : ∂|B|/∂theta
  kappa_psi     : radial curvature (from b⃗·∇b⃗ projected on ê_s)
  kappa_th      : poloidal curvature
  q_profile     : safety factor q = 1/ι
  shat          : magnetic shear ŝ = (s/q)·dq/ds
  twist_shift   : Δα = -2π·ŝ  (for twist-and-shift BC)
  galphaalpha   : g^{αα} from metric

References:
  Hirshman & Whitson (1983) Phys. Fluids 26, 3553 (VMEC)
  Boozer (1980) Phys. Fluids 23, 904 (Boozer coordinates)
"""

from __future__ import annotations
import dataclasses
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

from gyrojax.geometry.field_aligned import FieldAlignedGeometry


def _eval_fourier_sym(
    xmn: np.ndarray,   # (mnmax,)  Fourier coefficients
    xm:  np.ndarray,   # (mnmax,)  poloidal mode numbers
    xn:  np.ndarray,   # (mnmax,)  toroidal mode numbers
    theta: np.ndarray, # (Ntheta,)
    zeta:  np.ndarray, # (Nzeta,)
    cos_series: bool = True,
) -> np.ndarray:
    """
    Evaluate a stellarator-symmetric Fourier series on a (theta, zeta) grid.

    f(theta, zeta) = Σ_mn  xmn * cos(m*theta - n*zeta)   [cos_series=True]
                   = Σ_mn  xmn * sin(m*theta - n*zeta)   [cos_series=False]

    Returns shape (Ntheta, Nzeta).
    """
    # angle[mn, th, z] = m*theta[th] - n*zeta[z]
    angle = (xm[:, None, None] * theta[None, :, None]
             - xn[:, None, None] * zeta[None, None, :])    # (mnmax, Ntheta, Nzeta)

    if cos_series:
        basis = np.cos(angle)   # (mnmax, Ntheta, Nzeta)
    else:
        basis = np.sin(angle)

    return np.einsum('m,mij->ij', xmn, basis)   # (Ntheta, Nzeta)


def _eval_fourier_sym_surface(
    xmn_all: np.ndarray,   # (Ns, mnmax) — all surfaces
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,     # (Ntheta,)
    zeta: np.ndarray,      # (Nzeta,)
    cos_series: bool = True,
) -> np.ndarray:
    """
    Evaluate Fourier series on all surfaces. Returns (Ns, Ntheta, Nzeta).
    """
    angle = (xm[:, None, None] * theta[None, :, None]
             - xn[:, None, None] * zeta[None, None, :])    # (mnmax, Ntheta, Nzeta)
    if cos_series:
        basis = np.cos(angle)
    else:
        basis = np.sin(angle)
    # xmn_all: (Ns, mnmax);  basis: (mnmax, Ntheta, Nzeta)
    return np.tensordot(xmn_all, basis, axes=([1], [0]))    # (Ns, Ntheta, Nzeta)


def load_vmec_geometry(
    wout_path: str,
    Ntheta: int = 64,
    Nzeta: int = 32,
    s_min: float = 0.1,
    s_max: float = 0.9,
    Ns_out: Optional[int] = None,
) -> FieldAlignedGeometry:
    """
    Load a VMEC wout file and build a FieldAlignedGeometry.

    The returned geometry uses VMEC (s, theta, zeta) coordinates where:
      - s = normalized toroidal flux (like ψ/ψ_edge)
      - theta = VMEC poloidal angle (field-aligned in VMEC sense)
      - zeta = toroidal angle (one field period: [0, 2π/nfp))

    The FieldAlignedGeometry fields (psi_grid, theta_grid, alpha_grid) are
    repurposed as (s_grid, theta_grid, zeta_grid).

    Parameters
    ----------
    wout_path : str
        Path to VMEC wout_*.nc file.
    Ntheta, Nzeta : int
        Output grid resolution.
    s_min, s_max : float
        Radial range (normalized flux) to use; avoids axis (s=0) singularity.
    Ns_out : int, optional
        Number of radial (s) points in output. Default: use VMEC ns.

    Returns
    -------
    FieldAlignedGeometry
        Geometry compatible with Phase 2a simulation loop.
        Note: alpha_grid holds zeta values (one field period).
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required for VMEC geometry: pip install netCDF4")

    ds = nc.Dataset(wout_path, 'r')

    # --- Read scalars ---
    nfp   = int(ds.variables['nfp'][:])
    ns    = int(ds.variables['ns'][:])
    B0_axis = float(ds.variables['b0'][:])   # on-axis B [T]
    R0    = float(ds.variables['Rmajor_p'][:])
    a     = float(ds.variables['Aminor_p'][:])

    # --- Mode numbers ---
    # bmnc/gmnc/bsup*mnc use Nyquist-resolution modes (xm_nyq, xn_nyq)
    # rmnc/zmns use regular modes (xm, xn)
    xm  = np.asarray(ds.variables['xm'][:],     dtype=np.float64)
    xn  = np.asarray(ds.variables['xn'][:],     dtype=np.float64)
    xm_nyq = np.asarray(ds.variables['xm_nyq'][:], dtype=np.float64)
    xn_nyq = np.asarray(ds.variables['xn_nyq'][:], dtype=np.float64)

    # --- Profiles (on full-mesh: ns points) ---
    iotaf = np.asarray(ds.variables['iotaf'][:], dtype=np.float64)   # (ns,) rotational transform
    phi   = np.asarray(ds.variables['phi'][:],   dtype=np.float64)   # (ns,) toroidal flux

    # --- Fourier coefficients: shape (ns, mnmax) ---
    bmnc  = np.asarray(ds.variables['bmnc'][:],  dtype=np.float64)   # |B| cos
    gmnc  = np.asarray(ds.variables['gmnc'][:],  dtype=np.float64)   # Jacobian cos

    # bsupumnc: B^theta (B contravariant poloidal)
    # bsupvmnc: B^zeta  (B contravariant toroidal)
    bsupumnc = np.asarray(ds.variables['bsupumnc'][:], dtype=np.float64)
    bsupvmnc = np.asarray(ds.variables['bsupvmnc'][:], dtype=np.float64)

    ds.close()

    # --- Radial grid ---
    s_vmec = np.linspace(0.0, 1.0, ns)   # normalized flux
    if Ns_out is None:
        Ns_out = ns

    # Select radial range
    s_sel  = np.linspace(s_min, s_max, Ns_out)
    # Interpolation indices into s_vmec
    def interp_profile_1d(arr, s_out):
        return np.interp(s_out, s_vmec, arr)

    iota_out = interp_profile_1d(iotaf, s_sel)   # (Ns_out,)

    # Interpolate Fourier coefficients onto s_sel
    bmnc_out = np.stack([
        np.interp(s_sel, s_vmec, bmnc[:, m]) for m in range(bmnc.shape[1])
    ], axis=1)  # (Ns_out, mnmax)

    gmnc_out = np.stack([
        np.interp(s_sel, s_vmec, gmnc[:, m]) for m in range(gmnc.shape[1])
    ], axis=1)

    bsupumnc_out = np.stack([
        np.interp(s_sel, s_vmec, bsupumnc[:, m]) for m in range(bsupumnc.shape[1])
    ], axis=1)

    bsupvmnc_out = np.stack([
        np.interp(s_sel, s_vmec, bsupvmnc[:, m]) for m in range(bsupvmnc.shape[1])
    ], axis=1)

    # --- Angular grids ---
    theta_1d = np.linspace(0.0, 2*np.pi, Ntheta, endpoint=False)
    zeta_1d  = np.linspace(0.0, 2*np.pi / nfp, Nzeta, endpoint=False)

    # --- Evaluate B on grid (using Nyquist modes) ---
    B_3d = _eval_fourier_sym_surface(
        bmnc_out, xm_nyq, xn_nyq, theta_1d, zeta_1d, cos_series=True
    ).astype(np.float32)   # (Ns_out, Ntheta, Nzeta)

    # --- Jacobian (for normalization) ---
    Jac_3d = _eval_fourier_sym_surface(
        gmnc_out, xm_nyq, xn_nyq, theta_1d, zeta_1d, cos_series=True
    ).astype(np.float32)   # (Ns_out, Ntheta, Nzeta)

    # --- Numerical gradients of B ---
    # ∂B/∂s via finite differences (radial)
    ds_spacing = s_sel[1] - s_sel[0] if Ns_out > 1 else 1.0
    gradB_s = np.gradient(B_3d, ds_spacing, axis=0).astype(np.float32)

    # ∂B/∂theta via finite differences
    dth = theta_1d[1] - theta_1d[0]
    gradB_th = np.gradient(B_3d, dth, axis=1).astype(np.float32)

    # --- Curvature (field-line curvature κ = b⃗·∇b⃗) ---
    # In Boozer coords: κ_theta = -∂ln B / ∂theta · (B/Ω) = -(1/B)·∂B/∂theta
    # For GC drifts the relevant quantity is κ_s and κ_θ (curvature components)
    # κ_s = -(1/B)·∂B/∂s  (radial curvature)
    # κ_θ = -(1/B)·∂B/∂θ  (poloidal curvature driving radial drift)
    safe_B = np.maximum(B_3d, 1e-10)
    kappa_s  = (-gradB_s  / safe_B).astype(np.float32)
    kappa_th = (-gradB_th / safe_B).astype(np.float32)

    # --- Safety factor and shear ---
    q_out   = 1.0 / np.maximum(np.abs(iota_out), 1e-10) * np.sign(iota_out + 1e-30)
    q_out   = q_out.astype(np.float32)
    dqds    = np.gradient(q_out, ds_spacing)
    shat    = (s_sel * dqds / np.maximum(np.abs(q_out), 1e-10)).astype(np.float32)
    twist   = (-2 * np.pi * shat).astype(np.float32)

    # --- Metric g^{αα} ---
    # In VMEC Boozer coords: g^{αα} ≈ (iota·B/Bref)² + ... 
    # Use same formula as s-α: g^{αα} = (q/r)²(1+ŝ²θ²) but now r=sqrt(s)*a
    r_eff = (np.sqrt(np.maximum(s_sel, 1e-4)) * a).astype(np.float32)  # (Ns_out,)
    th2d  = theta_1d[None, :]   # (1, Ntheta)
    s2d   = s_sel[:, None]
    q2d   = q_out[:, None]
    r2d   = r_eff[:, None]
    shat2d = shat[:, None]
    galphaalpha_2d = ((q2d / r2d)**2 * (1.0 + shat2d**2 * (th2d - np.pi)**2)).astype(np.float32)

    return FieldAlignedGeometry(
        psi_grid    = jnp.array(s_sel.astype(np.float32)),
        theta_grid  = jnp.array(theta_1d.astype(np.float32)),
        alpha_grid  = jnp.array(zeta_1d.astype(np.float32)),
        B_field     = jnp.array(B_3d),
        gradB_psi   = jnp.array(gradB_s),
        gradB_th    = jnp.array(gradB_th),
        kappa_psi   = jnp.array(kappa_s),
        kappa_th    = jnp.array(kappa_th),
        gpsipsi     = jnp.ones(Ns_out, dtype=jnp.float32),
        galphaalpha = jnp.array(galphaalpha_2d),
        gpsialpha   = jnp.zeros(Ns_out, dtype=jnp.float32),
        q_profile   = jnp.array(q_out),
        shat        = jnp.array(shat),
        twist_shift = jnp.array(twist),
        R0=float(R0), a=float(a), B0=float(B0_axis),
    )
