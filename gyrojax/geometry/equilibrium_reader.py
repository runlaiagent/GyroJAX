"""
GTC equilibrium.out reader for GyroJAX.

File format (ASCII, based on GTC adipy loader):
------------------------------------------------
Line 0: nrplot   (number of 1-D profile rows, usually 30)
Line 1: lsp      (number of radial grid points)
Lines 2..2+lsp*(nrplot+1)-1 : 1-D profile data, one value per line
   Reshaped to (nrplot+1, lsp), rows are:
     0  : psi    (poloidal flux function)
     1  : torpsi_norm
     2  : Rminor (minor radius)
     3  : Rmajor (major radius)
     ...
     19 : q(psi) (safety factor)
     20 : d(ln q)/dr
     23 : r_1d   (minor radius grid)

After 1-D block:
  nplot  (number of 2-D field rows, usually 7)
  npsi   (radial grid size for 2-D arrays)
  ntheta (poloidal grid size)
  npsi*ntheta*(nplot+2) values, one per line
   Reshaped to (nplot+2, ntheta, npsi):
     0  : R(psi,theta)
     1  : Z(psi,theta)
     2  : B(psi,theta)
     3  : Jacobian
     4  : I
     5  : zeta2phi
     6  : delta

Reference: ~/wlhx/GTC/adipy_ver1.0/gtcdata/loader/equilibrium.py  load_old()
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from gyrojax.geometry.field_aligned import FieldAlignedGeometry


def read_gtc_equilibrium(filename: str, Nalpha: int = 32) -> FieldAlignedGeometry:
    """
    Read GTC equilibrium.out file and return a FieldAlignedGeometry.

    Parameters
    ----------
    filename : path to GTC equilibrium.out (ASCII format)
    Nalpha   : number of field-line labels α in the returned grid

    Returns
    -------
    FieldAlignedGeometry built from GTC numerical equilibrium.
    Curvature and ∇B are computed via finite differences of R, Z, B.
    """
    with open(filename, 'r') as fh:
        raw = [ln.strip() for ln in fh if ln.strip()]

    idx = 0

    # ---- 1-D profile block ----
    nrplot = int(raw[idx]);    idx += 1
    lsp    = int(raw[idx]);    idx += 1

    n1d = lsp * (nrplot + 1)
    pdata_flat = np.array([float(raw[idx + i]) for i in range(n1d)], dtype=np.float64)
    idx += n1d
    pdata = pdata_flat.reshape((nrplot + 1, lsp))

    # Row indices (same as GTC adipy loader):
    q_1d  = pdata[19, :]    # safety factor q(psi)
    r_1d  = pdata[23, :]    # minor radius grid [m]
    # Major radius R0 is approximately max of Rmajor (pdata[3])
    Rmajor_1d = pdata[3, :]

    # ---- 2-D spatial block ----
    nplot  = int(raw[idx]);  idx += 1
    npsi   = int(raw[idx]);  idx += 1
    ntheta = int(raw[idx]);  idx += 1

    n2d = npsi * ntheta * (nplot + 2)
    spdata_flat = np.array([float(raw[idx + i]) for i in range(n2d)], dtype=np.float64)
    spdata = spdata_flat.reshape((nplot + 2, ntheta, npsi))

    # Extract 2-D arrays: shape (ntheta, npsi) → transpose to (npsi, ntheta)
    R2d = spdata[0, :, :].T   # (npsi, ntheta)
    Z2d = spdata[1, :, :].T
    B2d = spdata[2, :, :].T

    # --- Derived 1-D quantities ---
    R0 = float(np.max(Rmajor_1d))    # magnetic axis major radius
    a  = float(r_1d[-1])             # last closed flux surface minor radius
    B0 = float(np.mean(B2d[0, :]))   # on-axis mean B

    # Regularise r grid (monotonically increasing, avoid zero)
    r_np = np.asarray(r_1d, dtype=np.float32)
    if r_np[0] == 0.0:
        r_np[0] = r_np[1] * 0.1

    q_np = np.asarray(q_1d, dtype=np.float32)

    # Magnetic shear ŝ = (r/q)*dq/dr via finite differences
    dqdr = np.gradient(q_np, r_np)
    s_np = (r_np / q_np * dqdr).astype(np.float32)
    twist = (-2 * np.pi * s_np).astype(np.float32)

    # ---- Poloidal angle grid ----
    # GTC 2-D arrays are on uniform theta ∈ [0, 2π)
    th_2d = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    # Field-aligned convention uses θ ∈ [-π, π)
    th_fa = np.linspace(-np.pi, np.pi, ntheta, endpoint=False).astype(np.float32)

    # Remap B2d from [0,2π) to [-π,π) ordering
    half = ntheta // 2
    B_reordered  = np.roll(B2d,  -half, axis=1).astype(np.float32)
    R_reordered  = np.roll(R2d,  -half, axis=1).astype(np.float32)
    Z_reordered  = np.roll(Z2d,  -half, axis=1).astype(np.float32)

    # ---- α grid ----
    al_np = np.linspace(0.0, 2.0 * np.pi, Nalpha, endpoint=False).astype(np.float32)

    # ---- Broadcast to (Npsi, Ntheta, Nalpha) ----
    Npsi = npsi

    # B is axisymmetric — broadcast over α
    B3 = np.broadcast_to(B_reordered[:, :, None],
                         (Npsi, ntheta, Nalpha)).copy().astype(np.float32)

    # ---- ∇B — finite differences in r and θ ----
    dr_np = np.gradient(r_np)             # (Npsi,)
    dth   = 2.0 * np.pi / ntheta

    # ∂B/∂r along flux surfaces
    dBdr = np.gradient(B_reordered, r_np, axis=0).astype(np.float32)   # (Npsi,Ntheta)
    # ∂B/∂θ along θ
    dBdth = np.gradient(B_reordered, th_fa, axis=1).astype(np.float32) # (Npsi,Ntheta)

    # Physical gradient along field line: dB/dl_∥ = (1/q*R0) * dB/dθ
    gradB_psi3 = np.broadcast_to(dBdr[:, :, None],
                                  (Npsi, ntheta, Nalpha)).copy().astype(np.float32)
    q3d = q_np[:, None, None] * np.ones((Npsi, ntheta, Nalpha), dtype=np.float32)
    dBdth3 = np.broadcast_to(dBdth[:, :, None],
                               (Npsi, ntheta, Nalpha)).copy().astype(np.float32)
    gradB_th3 = (dBdth3 / (q3d * R0)).astype(np.float32)

    # ---- Curvature from R, Z geometry ----
    # κ = -(∂ ln B / ∂r) ≈ -dBdr / B
    # Normal (ψ) curvature:
    kappa_psi3 = np.broadcast_to(
        (-dBdr / B_reordered)[:, :, None], (Npsi, ntheta, Nalpha)
    ).copy().astype(np.float32)

    # Geodesic (θ) curvature from Z geometry: κ_θ ≈ d²Z/dθ² / R
    d2Zdth2 = np.gradient(np.gradient(Z_reordered, th_fa, axis=1),
                           th_fa, axis=1).astype(np.float32)
    kappa_th3 = np.broadcast_to(
        (d2Zdth2 / R_reordered)[:, :, None], (Npsi, ntheta, Nalpha)
    ).copy().astype(np.float32)

    # ---- Contravariant metric ----
    gpsipsi_1d = np.ones(Npsi, dtype=np.float32)

    # g^{αα}(ψ,θ) = (q/r)²·(1 + ŝ²·θ²)
    q2d   = q_np[:, None]
    r2d   = r_np[:, None]
    s2d   = s_np[:, None]
    th2d  = th_fa[None, :]
    galphaalpha_2d = ((q2d / r2d)**2 * (1.0 + s2d**2 * th2d**2)).astype(np.float32)

    gpsialpha_1d = np.zeros(Npsi, dtype=np.float32)

    return FieldAlignedGeometry(
        psi_grid     = jnp.array(r_np),
        theta_grid   = jnp.array(th_fa),
        alpha_grid   = jnp.array(al_np),
        B_field      = jnp.array(B3),
        gradB_psi    = jnp.array(gradB_psi3),
        gradB_th     = jnp.array(gradB_th3),
        kappa_psi    = jnp.array(kappa_psi3),
        kappa_th     = jnp.array(kappa_th3),
        gpsipsi      = jnp.array(gpsipsi_1d),
        galphaalpha  = jnp.array(galphaalpha_2d),
        gpsialpha    = jnp.array(gpsialpha_1d),
        q_profile    = jnp.array(q_np),
        shat         = jnp.array(s_np),
        twist_shift  = jnp.array(twist),
        R0=float(R0), a=float(a), B0=float(B0),
    )
