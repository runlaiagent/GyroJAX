"""
Tests for the TOML input file system (gyrojax/input.py and gyrojax/runner.py).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_runner(*args, check=True):
    """Run `python -m gyrojax.runner <args>` and return CompletedProcess.

    Note: JAX sometimes triggers a SIGSEGV/SIGABRT on interpreter shutdown
    during dry-runs (a known upstream issue).  For dry-run tests we accept
    both exit-0 and signal-exit, and validate stdout instead.
    """
    cmd = [PYTHON, "-m", "gyrojax.runner"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=REPO_ROOT)
    if check and result.returncode not in (0, -11, -6):  # allow SIGSEGV/SIGABRT
        raise subprocess.CalledProcessError(result.returncode, cmd,
                                            result.stdout, result.stderr)
    return result


# ---------------------------------------------------------------------------
# Unit tests — load_config
# ---------------------------------------------------------------------------

def test_load_deltaf_config():
    """Load cbc.toml and verify it produces a valid SimConfigFA."""
    from gyrojax.input import load_config
    from gyrojax.simulation_fa import SimConfigFA

    cfg, raw = load_config(INPUTS_DIR / "cbc.toml")
    assert isinstance(cfg, SimConfigFA)
    # Check a few key values from the TOML
    assert cfg.Npsi == 16
    assert cfg.Ntheta == 32
    assert cfg.N_particles == 300_000
    assert cfg.single_mode is True
    assert cfg.k_mode == 18
    assert cfg.electron_model == "adiabatic"
    # rho_star → e
    assert abs(cfg.rho_star - 0.005556) < 1e-6
    assert abs(cfg.e - 1.0 / 0.005556) < 1.0   # ~ 180


def test_load_fullf_config():
    """Load cbc_fullf.toml and verify it produces a valid SimConfigFullF."""
    from gyrojax.input import load_config
    from gyrojax.simulation_fullf import SimConfigFullF

    cfg, raw = load_config(INPUTS_DIR / "cbc_fullf.toml")
    assert isinstance(cfg, SimConfigFullF)
    assert cfg.Npsi == 16
    assert cfg.N_particles == 200_000
    assert cfg.k_mode == 18


def test_load_rosenbluth_hinton():
    """Load rosenbluth_hinton.toml — zonal_init must be True, no drive."""
    from gyrojax.input import load_config
    from gyrojax.simulation_fa import SimConfigFA

    cfg, raw = load_config(INPUTS_DIR / "rosenbluth_hinton.toml")
    assert isinstance(cfg, SimConfigFA)
    assert cfg.zonal_init is True
    assert cfg.R0_over_LT == 0.0


def test_load_dimits():
    """Load dimits.toml (nonlinear template)."""
    from gyrojax.input import load_config
    from gyrojax.simulation_fa import SimConfigFA

    cfg, raw = load_config(INPUTS_DIR / "dimits.toml")
    assert isinstance(cfg, SimConfigFA)
    assert cfg.use_weight_spread is True
    assert cfg.k_alpha_min == 1


def test_unknown_method_raises():
    """load_config should raise ValueError for unknown method."""
    import tempfile, os
    from gyrojax.input import load_config

    toml_content = b'[run]\nmethod = "bogus"\n'
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="bogus"):
            load_config(tmp)
    finally:
        os.unlink(tmp)


def test_defaults_applied():
    """An empty TOML should produce a valid SimConfigFA with defaults."""
    import tempfile, os
    from gyrojax.input import load_config
    from gyrojax.simulation_fa import SimConfigFA

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"")   # completely empty
        tmp = f.name
    try:
        cfg, _ = load_config(tmp)
        assert isinstance(cfg, SimConfigFA)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# CLI dry-run tests
# ---------------------------------------------------------------------------

def test_dry_run_cbc():
    """--dry-run on cbc.toml should print config."""
    result = _run_runner(str(INPUTS_DIR / "cbc.toml"), "--dry-run")
    assert "SimConfig" in result.stdout or "Npsi" in result.stdout


def test_dry_run_fullf():
    """--dry-run on cbc_fullf.toml should print config."""
    result = _run_runner(str(INPUTS_DIR / "cbc_fullf.toml"), "--dry-run")
    assert "SimConfig" in result.stdout or "Npsi" in result.stdout


def test_dry_run_rh():
    """--dry-run on rosenbluth_hinton.toml."""
    result = _run_runner(str(INPUTS_DIR / "rosenbluth_hinton.toml"), "--dry-run")
    assert "SimConfig" in result.stdout or "Npsi" in result.stdout


# ---------------------------------------------------------------------------
# Short end-to-end run
# ---------------------------------------------------------------------------

def test_run_short_deltaf(tmp_path):
    """Run cbc.toml for 5 steps and verify JSON output is created."""
    import tempfile, os

    # Create a minimal TOML pointing output to tmp_path
    toml = f"""
[run]
method = "deltaf"

[grid]
Npsi   = 8
Ntheta = 16
Nalpha = 16

[particles]
N_particles = 5000

[time]
dt      = 0.05
n_steps = 5

[output]
output_dir = "{tmp_path}/"
"""
    inp = tmp_path / "short_deltaf.toml"
    inp.write_text(toml)

    result = subprocess.run(
        [PYTHON, "-m", "gyrojax.runner", str(inp)],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr

    out_file = tmp_path / "short_deltaf_results.json"
    assert out_file.exists(), "Expected results JSON not found"
    data = json.loads(out_file.read_text())
    assert data["n_steps"] == 5
    assert len(data["phi_max"]) == 5


def test_run_short_fullf(tmp_path):
    """Run fullf for 3 steps and verify JSON output."""
    toml = f"""
[run]
method = "fullf"

[grid]
Npsi   = 8
Ntheta = 16
Nalpha = 16

[particles]
N_particles = 2000

[time]
dt      = 0.05
n_steps = 3

[output]
output_dir = "{tmp_path}/"
"""
    inp = tmp_path / "short_fullf.toml"
    inp.write_text(toml)

    result = subprocess.run(
        [PYTHON, "-m", "gyrojax.runner", str(inp)],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr

    out_file = tmp_path / "short_fullf_results.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["n_steps"] == 3
