"""The finite-arm option reaches the projection through every public path that offers it.

A dropped keyword anywhere between a caller and the projection would leave the option silently off
while the projection's own tests stay green, so each forward is exercised here: end to end where the
result can discriminate, and by recording the call where generating the signal would be the whole cost.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from gwmock_signal.cli.main import app
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.geometry import arm_lengths
from gwmock_signal.simulator import CBCSimulator
from gwmock_signal.stochastic.simulator import StochasticBackgroundSimulator
from gwmock_signal.timeseries import TimeSeries

_PARAMS = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
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
_FS = 2048.0


def _zeros(names, start=1126259458.0, duration=8.0):
    return {name: TimeSeries(np.zeros(int(duration * _FS)), t0=start, sample_rate=_FS) for name in names}


@pytest.mark.requires_lalsuite
def test_inject_cbc_signal_applies_the_finite_arm_response():
    """Through ``inject_cbc_signal`` and ``CBCSimulator.simulate`` the option changes the ET strain by ~1e-2."""
    detectors = list(Network.from_preset("ET-Triangle-EMR").detector_names)
    names = [d.name for d in detectors]
    common = {"sampling_frequency": _FS, "minimum_frequency": 20.0}
    arms = dataclasses.replace(ProjectionEffects.long_wavelength(earth_rotation=True), finite_arm_length=True)
    finite = inject_cbc_signal("IMRPhenomD", _PARAMS, detectors, _zeros(names), effects=arms, **common)
    vertex = inject_cbc_signal(
        "IMRPhenomD",
        _PARAMS,
        detectors,
        _zeros(names),
        effects=ProjectionEffects.long_wavelength(earth_rotation=True),
        **common,
    )
    peak = np.max(np.abs(vertex.data))
    assert np.max(np.abs(finite.data - vertex.data)) / peak > 1e-3
    null = np.max(np.abs(finite.data.sum(axis=0))) / peak
    assert null < np.max(np.abs(vertex.data.sum(axis=0))) / peak / 100


@pytest.mark.requires_lalsuite
def test_write_forwards_the_effects(monkeypatch, tmp_path):
    """``CBCSimulator.write`` passes the effects on to ``simulate``."""
    seen = {}

    def recording(self, *args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("recorded")

    monkeypatch.setattr("gwmock_signal.simulator.CBCSimulator.simulate", recording)
    with pytest.raises(RuntimeError, match="recorded"):
        CBCSimulator(waveform_model="IMRPhenomD").write(
            tmp_path / "out.hdf5",
            _PARAMS,
            ["H1"],
            {},
            sampling_frequency=_FS,
            minimum_frequency=20.0,
            effects=ProjectionEffects(shapiro_delay=False),
        )
    assert seen["effects"] == ProjectionEffects(shapiro_delay=False)


@pytest.mark.requires_lalsuite
def test_the_cli_flag_reaches_the_pipeline(monkeypatch, tmp_path):
    """Every effect is on by default, and each ``--no-...`` flag turns off exactly its own."""
    seen = []

    def recording(**kwargs):
        seen.append(kwargs["effects"])
        raise RuntimeError("recorded")

    monkeypatch.setattr("gwmock_signal.cli.inject.inject_cbc_signal", recording)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(_PARAMS))
    base = ["inject", "cbc", "--params", str(params), "--network", "H1L1", "--duration", "0.5"]
    flags = [
        "earth-rotation",
        "finite-arm-length",
        "source-precession",
        "roemer-delay",
        "einstein-delay",
        "shapiro-delay",
    ]
    for extra in [[]] + [[f"--no-{flag}"] for flag in flags]:
        result = CliRunner().invoke(app, base + extra)
        assert isinstance(result.exception, RuntimeError), result.output
    assert seen[0] == ProjectionEffects()
    for flag, effects in zip(flags, seen[1:], strict=True):
        assert effects == dataclasses.replace(ProjectionEffects(), **{flag.replace("-", "_"): False})


def test_a_network_files_arm_length_reaches_the_detector(tmp_path):
    """A custom entry's ``arm_length_m`` becomes the detector's arm length for projection."""
    path = tmp_path / "net.yaml"
    path.write_text(
        "name: n\ndetectors:\n  - name: X1\n    latitude_deg: 40\n    longitude_deg: 9\n    elevation_m: 0\n"
        "    xarm_azimuth_deg: 10\n    yarm_azimuth_deg: 100\n    arm_length_m: 12345\n"
    )
    (detector,) = Network.from_file(path).detector_names
    assert arm_lengths(detector) == (12345.0, 12345.0)


@pytest.mark.requires_lalsuite
def test_the_arm_length_survives_registration_with_lal():
    """``to_lal`` records the arm length as LAL does, so the registered code reads back the same length."""
    detector = CustomDetector("X1", 0.7, 0.16, 100.0, 1.23, 2.80, arm_length_m=750.0)
    prefix = detector.to_lal().frDetector.prefix
    assert arm_lengths(prefix) == arm_lengths(detector) == (750.0, 750.0)


def test_the_stochastic_simulator_accepts_and_ignores_projection_effects():
    """Nothing is projected for a background, whose response is the overlap reduction function it is given."""
    pytest.importorskip("gwmock_noise", reason="the sgwb extra is not installed")
    frequencies = np.fft.rfftfreq(256, d=1.0 / 128.0)
    overlap = {("H1", "L1"): np.full(np.count_nonzero(frequencies >= 8.0), 0.25)}
    common = {"sampling_frequency": 128.0, "minimum_frequency": 8.0}
    stacks = [
        StochasticBackgroundSimulator(duration=2.0, seed=10, overlap_reduction=overlap).simulate(
            {"omega_ref": 1.0e30}, ["H1", "L1"], effects=effects, **common
        )
        for effects in (None, ProjectionEffects.long_wavelength())
    ]
    np.testing.assert_array_equal(stacks[0].data, stacks[1].data)
