"""
HDF5 checkpoint I/O for GyroJAX.

Usage (simple — save after run):
    diags, state, phi, geom = run_simulation_fa(cfg)
    from gyrojax.io import save_run
    save_run("output.h5", diags, state, phi, geom, cfg)

Usage (chunked — for long runs / low GPU memory):
    from gyrojax.io import CheckpointWriter
    writer = CheckpointWriter("output.h5", cfg)
    for chunk_start in range(0, cfg.n_steps, cfg.checkpoint_interval):
        chunk_cfg = cfg._replace(n_steps=min(cfg.checkpoint_interval, cfg.n_steps - chunk_start))
        diags, state, phi, geom = run_simulation_fa(chunk_cfg, ...)
        writer.append(diags, state, phi, geom, step_offset=chunk_start)
    writer.close()
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

if TYPE_CHECKING:
    pass


def _require_h5py():
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 I/O: pip install h5py")


def save_run(path: str, diags, state, phi, geom, cfg, step_offset: int = 0):
    """Save a completed simulation run to HDF5.

    File structure:
        /config/          <- SimConfigFA fields as attributes
        /diags/phi_rms    <- float32 array [n_steps]
        /diags/phi_max    <- float32 array [n_steps]
        /diags/weight_rms <- float32 array [n_steps]
        /phi              <- float32 array [Npsi, Ntheta, Nalpha] (final step)
        /particles/r      <- float32 array [N_particles] (final state)
        /particles/theta  <- float32 array [N_particles]
        /particles/zeta   <- float32 array [N_particles]
        /particles/vpar   <- float32 array [N_particles]
        /particles/mu     <- float32 array [N_particles]
        /particles/weight <- float32 array [N_particles]
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Config
        grp = f.create_group("config")
        cfg_dict = cfg._asdict() if hasattr(cfg, '_asdict') else vars(cfg)
        for field, val in cfg_dict.items():
            try:
                if val is None:
                    grp.attrs[field] = "None"
                elif isinstance(val, bool):
                    grp.attrs[field] = int(val)
                elif isinstance(val, (int, float, str)):
                    grp.attrs[field] = val
            except Exception:
                pass

        # Diagnostics timeseries
        dgrp = f.create_group("diags")
        phi_rms = np.array([float(d.phi_rms) for d in diags], dtype=np.float32)
        phi_max = np.array([float(d.phi_max) for d in diags], dtype=np.float32)
        w_rms   = np.array([float(d.weight_rms) for d in diags], dtype=np.float32)
        steps   = np.arange(step_offset, step_offset + len(diags), dtype=np.int32)
        dgrp.create_dataset("phi_rms",    data=phi_rms, compression="gzip")
        dgrp.create_dataset("phi_max",    data=phi_max, compression="gzip")
        dgrp.create_dataset("weight_rms", data=w_rms,   compression="gzip")
        dgrp.create_dataset("step",       data=steps)

        # Final phi field
        f.create_dataset("phi", data=np.array(phi, dtype=np.float32), compression="gzip")

        # Final particle state
        pgrp = f.create_group("particles")
        for name in ("r", "theta", "zeta", "vpar", "mu", "weight"):
            arr = np.array(getattr(state, name), dtype=np.float32)
            pgrp.create_dataset(name, data=arr, compression="gzip")


def append_run(path: str, diags, state, phi, step_offset: int):
    """Append a chunk of results to an existing HDF5 file."""
    _require_h5py()
    with h5py.File(path, "a") as f:
        dgrp = f["diags"]
        new_phi_rms = np.array([float(d.phi_rms)     for d in diags], dtype=np.float32)
        new_phi_max = np.array([float(d.phi_max)     for d in diags], dtype=np.float32)
        new_w_rms   = np.array([float(d.weight_rms)  for d in diags], dtype=np.float32)
        new_steps   = np.arange(step_offset, step_offset + len(diags), dtype=np.int32)
        for key, new_data in [("phi_rms", new_phi_rms), ("phi_max", new_phi_max),
                               ("weight_rms", new_w_rms), ("step", new_steps)]:
            old = dgrp[key][:]
            del dgrp[key]
            dgrp.create_dataset(key, data=np.concatenate([old, new_data]), compression="gzip")

        # Overwrite final phi + particles
        if "phi" in f:
            del f["phi"]
        f.create_dataset("phi", data=np.array(phi, dtype=np.float32), compression="gzip")
        pgrp = f["particles"]
        for name in ("r", "theta", "zeta", "vpar", "mu", "weight"):
            arr = np.array(getattr(state, name), dtype=np.float32)
            if name in pgrp:
                del pgrp[name]
            pgrp.create_dataset(name, data=arr, compression="gzip")


def load_run(path: str) -> dict:
    """Load a saved run. Returns dict with keys: diags, phi, particles, config."""
    _require_h5py()
    with h5py.File(path, "r") as f:
        result = {
            "phi":       f["phi"][:],
            "diags":     {k: f["diags"][k][:]     for k in f["diags"]},
            "particles": {k: f["particles"][k][:] for k in f["particles"]},
            "config":    dict(f["config"].attrs),
        }
    return result


class CheckpointWriter:
    """Context manager for chunked long runs."""

    def __init__(self, path: str, cfg, mode: str = "w"):
        _require_h5py()
        self.path = path
        self.cfg = cfg
        self.mode = mode
        self._initialized = False

    def append(self, diags, state, phi, geom, step_offset: int = 0):
        if not self._initialized:
            save_run(self.path, diags, state, phi, geom, self.cfg, step_offset=step_offset)
            self._initialized = True
        else:
            append_run(self.path, diags, state, phi, step_offset=step_offset)

    def close(self):
        pass  # file closed after each write

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
