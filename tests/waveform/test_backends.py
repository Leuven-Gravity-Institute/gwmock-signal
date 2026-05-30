"""Tests for waveform backends."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import LALSimulationBackend, PyCBCBackend

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lalsimulation_backend_available_approximants_includes_imrphenomd() -> None:
    """The default LAL backend exposes standard TD approximants."""
    models = LALSimulationBackend().available_approximants()
    assert "IMRPhenomD" in models


def test_lalsimulation_backend_generates_timeseries_dict() -> None:
    """A minimal LAL waveform call returns GWpy time series."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=36.0,
        detector_frame_mass_2=29.0,
        luminosity_distance=410.0,
        inclination=0.0,
        coa_phase=0.0,
    )
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)
    assert isinstance(result["cross"], TimeSeries)


def test_pycbc_backend_raises_helpful_import_error_when_pycbc_missing() -> None:
    """PyCBCBackend fails at instantiation time with installation guidance."""
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "pycbc.waveform":
            raise ImportError("pycbc unavailable")
        return real_import_module(name, package)

    with (
        patch("gwmock_signal.waveform.backends.pycbc.importlib.import_module", side_effect=_import_module),
        pytest.raises(ImportError, match=r"gwmock-signal\[pycbc\]"),
    ):
        PyCBCBackend()


def test_pycbc_backend_available_approximants_when_installed() -> None:
    """PyCBCBackend works normally when PyCBC is present."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    assert "IMRPhenomD" in PyCBCBackend().available_approximants()


def test_lal_backend_accepts_tidal_params_for_nrtidal_approximant() -> None:
    """lambda_1/lambda_2 do not raise ValueError for NRTidal waveforms."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomPv2_NRTidalv2",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=1.4,
        detector_frame_mass_2=1.4,
        luminosity_distance=40.0,
        lambda_1=500.0,
        lambda_2=300.0,
    )
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)


def test_lal_backend_accepts_tidal_aliases() -> None:
    """tidal_1/tidal_2 are accepted as aliases for lambda_1/lambda_2."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomPv2_NRTidalv2",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=1.4,
        detector_frame_mass_2=1.4,
        luminosity_distance=40.0,
        tidal_1=500.0,
        tidal_2=300.0,
    )
    assert set(result) == {"plus", "cross"}


def test_lal_backend_nontidal_unaffected_by_default_lambdas() -> None:
    """Non-tidal waveforms work without tidal params (lambda=0 is safe)."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=36.0,
        detector_frame_mass_2=29.0,
        luminosity_distance=410.0,
    )
    assert set(result) == {"plus", "cross"}


def test_lal_backend_tidal_alias_conflict_raises() -> None:
    """Passing lambda_1 and tidal_1 simultaneously raises ValueError."""
    with pytest.raises(ValueError, match="Do not mix aliases"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=1_126_259_462.4,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=36.0,
            detector_frame_mass_2=29.0,
            luminosity_distance=410.0,
            lambda_1=500.0,
            tidal_1=500.0,
        )


def test_top_level_backend_import_succeeds_without_pycbc() -> None:
    """Top-level backend exports do not require importing PyCBC."""
    code = """
import builtins

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("pycbc"):
        raise ModuleNotFoundError("No module named 'pycbc'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
from gwmock_signal import LALSimulationBackend, WaveformBackend
assert WaveformBackend.__name__ == "WaveformBackend"
assert LALSimulationBackend.__name__ == "LALSimulationBackend"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
