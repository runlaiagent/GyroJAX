"""Tests for TOML input file loading."""
import pytest
from pathlib import Path


def test_load_minimal_toml(tmp_path):
    """Minimal TOML with only [simulation] section loads correctly."""
    toml = tmp_path / "test.toml"
    toml.write_text("""
[simulation]
N_particles = 1000
n_steps = 5
""")
    from gyrojax.io.input_file import load_config
    cfg = load_config(toml)
    assert cfg.N_particles == 1000
    assert cfg.n_steps == 5


def test_load_dtype_section(tmp_path):
    """[dtype] section sets DtypeConfig correctly."""
    toml = tmp_path / "test.toml"
    toml.write_text("""
[dtype]
velocity = "bfloat16"
phi = "bfloat16"
""")
    from gyrojax.io.input_file import load_config
    cfg = load_config(toml)
    assert cfg.dtype_config.velocity == "bfloat16"
    assert cfg.dtype_config.phi == "bfloat16"
    assert cfg.dtype_config.weight == "float32"  # default unchanged


def test_load_all_sections(tmp_path):
    """Full TOML with all sections loads correctly."""
    toml = tmp_path / "test.toml"
    toml.write_text("""
[simulation]
Npsi = 8
Ntheta = 16
Nalpha = 16
N_particles = 500
n_steps = 3

[physics]
R0_over_LT = 7.5
beta = 0.01

[dtype]
velocity = "bfloat16"

[flags]
single_mode = true
absorbing_wall = true
""")
    from gyrojax.io.input_file import load_config
    cfg = load_config(toml)
    assert cfg.Npsi == 8
    assert cfg.R0_over_LT == 7.5
    assert cfg.beta == 0.01
    assert cfg.dtype_config.velocity == "bfloat16"
    assert cfg.single_mode == True
    assert cfg.absorbing_wall == True


def test_save_and_reload_template(tmp_path):
    """save_config_template writes a valid TOML that can be reloaded."""
    from gyrojax.io.input_file import load_config, save_config_template
    from gyrojax.simulation_fa import SimConfigFA

    path = tmp_path / "template.toml"
    save_config_template(path)
    cfg = load_config(path)
    assert isinstance(cfg, SimConfigFA)


def test_empty_toml_uses_defaults(tmp_path):
    """Empty TOML file produces default SimConfigFA."""
    toml = tmp_path / "empty.toml"
    toml.write_text("")
    from gyrojax.io.input_file import load_config
    from gyrojax.simulation_fa import SimConfigFA
    cfg = load_config(toml)
    default = SimConfigFA()
    assert cfg.N_particles == default.N_particles
    assert cfg.dt == default.dt
