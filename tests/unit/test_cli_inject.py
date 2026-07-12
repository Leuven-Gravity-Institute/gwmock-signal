"""CLI smoke tests for the `gwmock-signal inject cbc` command."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from gwmock_signal.cli.main import app

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


runner = CliRunner()

# ---------------------------------------------------------------------------
# Minimal params — light masses so the waveform is short even at low f_min
# ---------------------------------------------------------------------------

_PARAMS: dict = {
    "detector_frame_mass_1": 10.0,
    "detector_frame_mass_2": 10.0,
    "spin_1z": 0.0,
    "spin_2z": 0.0,
    "coa_time": 1126259462.4,
    "distance": 410.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 2.659,
    "inclination": 2.5,
    "coa_phase": 0.0,
}

# Short duration + low sample rate keeps the test fast
_DURATION = "0.5"
_SAMPLE_RATE = "256"
_F_MIN = "50.0"


@pytest.fixture
def params_file(tmp_path: Path) -> Path:
    """Write minimal CBC params to a temporary JSON file."""
    p = tmp_path / "params.json"
    p.write_text(json.dumps(_PARAMS))
    return p


def test_inject_cbc_stdout_summary(params_file: Path) -> None:
    """Inject cbc without --output prints a per-detector summary."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "H1" in result.output
    assert "L1" in result.output
    assert "rms=" in result.output


def test_inject_cbc_hdf5_output(params_file: Path, tmp_path: Path) -> None:
    """Inject cbc with --output writes a readable HDF5 file."""
    from gwpy.timeseries import TimeSeries

    out_path = str(tmp_path / "output.hdf5")

    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--output",
            out_path,
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
        ],
    )
    assert result.exit_code == 0, result.output

    ts = TimeSeries.read(out_path, format="hdf5", path="H1")
    assert len(ts) > 0


def test_inject_cbc_bad_params_file(tmp_path: Path) -> None:
    """Inject cbc raises BadParameter for a missing params file."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(tmp_path / "nonexistent.json"),
            "--network",
            "H1L1",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_invalid_json(tmp_path: Path) -> None:
    """Inject cbc raises BadParameter for invalid JSON."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(bad),
            "--network",
            "H1L1",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_unknown_network(params_file: Path) -> None:
    """Inject cbc raises BadParameter for an unknown network name."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "UNKNOWN",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_comma_separated_codes(params_file: Path) -> None:
    """--network accepts comma-separated PyCBC detector codes (no named preset needed)."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1,L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "H1" in result.output
    assert "L1" in result.output


def test_inject_cbc_file_based_network_rejected(params_file: Path, tmp_path: Path) -> None:
    """Passing an invalid JSON file as --network exits with non-zero status."""
    net_file = tmp_path / "network.json"
    net_file.write_text("{}")
    result = runner.invoke(
        app,
        ["inject", "cbc", "--params", str(params_file), "--network", str(net_file)],
    )
    assert result.exit_code != 0


def test_inject_cbc_yaml_network_file_accepted(params_file: Path, tmp_path: Path) -> None:
    """Passing a valid YAML network file as --network succeeds and prints a summary."""
    import yaml

    net_file = tmp_path / "h1l1.yaml"
    net_file.write_text(yaml.dump({"name": "H1L1", "detectors": [{"name": "H1"}, {"name": "L1"}]}))
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            str(net_file),
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "H1" in result.output
    assert "rms=" in result.output


def test_inject_cbc_seed_accepted(params_file: Path) -> None:
    """--seed is accepted and the command completes successfully."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inject_cbc_explicit_lal_backend_accepted(params_file: Path) -> None:
    """--backend lal is accepted explicitly and the command succeeds."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
            "--backend",
            "lal",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inject_cbc_gwsignal_backend_accepted(params_file: Path) -> None:
    """--backend gwsignal is accepted and the command succeeds."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
            "--backend",
            "gwsignal",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inject_cbc_pycbc_backend_surfaces_install_hint(params_file: Path) -> None:
    """--backend pycbc surfaces the optional-extra install hint on ImportError."""
    with patch(
        "gwmock_signal.cli.inject.PyCBCBackend",
        side_effect=ImportError("pycbc is not installed. Run: pip install 'gwmock-signal[pycbc]'"),
    ):
        result = runner.invoke(
            app,
            [
                "inject",
                "cbc",
                "--params",
                str(params_file),
                "--network",
                "H1L1",
                "--backend",
                "pycbc",
            ],
        )
    assert result.exit_code == 2
    # ImportError from PyCBCBackend is wrapped as typer.BadParameter (SystemExit(2)).
    assert "gwmock-signal[pycbc]" in result.output


def test_inject_cbc_ripple_backend_accepted(params_file: Path) -> None:
    """--backend ripple runs the injection pipeline end-to-end."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            _DURATION,
            "--sample-rate",
            _SAMPLE_RATE,
            "--f-min",
            _F_MIN,
            "--backend",
            "ripple",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inject_cbc_ripple_backend_surfaces_install_hint(params_file: Path) -> None:
    """--backend ripple surfaces the optional-extra install hint on ImportError."""
    with patch(
        "gwmock_signal.cli.inject.RippleBackend",
        side_effect=ImportError("ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'"),
    ):
        result = runner.invoke(
            app,
            [
                "inject",
                "cbc",
                "--params",
                str(params_file),
                "--network",
                "H1L1",
                "--backend",
                "ripple",
            ],
        )
    assert result.exit_code == 2
    # ImportError from RippleBackend is wrapped as typer.BadParameter (SystemExit(2)).
    assert "gwmock-signal[jax]" in result.output


def test_inject_cbc_invalid_backend(params_file: Path) -> None:
    """--backend rejects values other than lal, pycbc, and ripple."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--backend",
            "not-a-backend",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_invalid_sample_rate(params_file: Path) -> None:
    """--sample-rate <= 0 must exit non-zero."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--sample-rate",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_invalid_duration(params_file: Path) -> None:
    """--duration <= 0 must exit non-zero."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--duration",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_invalid_f_min(params_file: Path) -> None:
    """--f-min <= 0 must exit non-zero."""
    result = runner.invoke(
        app,
        [
            "inject",
            "cbc",
            "--params",
            str(params_file),
            "--network",
            "H1L1",
            "--f-min",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_inject_cbc_missing_tc_key(tmp_path: Path) -> None:
    """Params file without 'coa_time' key must exit non-zero."""
    params = {k: v for k, v in _PARAMS.items() if k != "coa_time"}
    p = tmp_path / "params_no_tc.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(
        app,
        ["inject", "cbc", "--params", str(p), "--network", "H1L1"],
    )
    assert result.exit_code != 0


def test_inject_cbc_invalid_tc_value(tmp_path: Path) -> None:
    """Params file with non-numeric 'coa_time' must exit non-zero."""
    params = {**_PARAMS, "coa_time": None}
    p = tmp_path / "params_bad_tc.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(
        app,
        ["inject", "cbc", "--params", str(p), "--network", "H1L1"],
    )
    assert result.exit_code != 0


def test_inject_cbc_help() -> None:
    """`inject cbc --help` displays all expected options."""
    result = runner.invoke(app, ["inject", "cbc", "--help"])
    assert result.exit_code == 0
    clean = _ANSI_ESCAPE.sub("", result.output)
    for opt in [
        "--params",
        "--network",
        "--output",
        "--sample-rate",
        "--f-min",
        "--duration",
        "--approximant",
        "--seed",
    ]:
        assert opt in clean, f"Missing option {opt!r} in --help output"
