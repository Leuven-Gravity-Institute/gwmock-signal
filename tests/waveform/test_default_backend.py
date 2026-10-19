"""The waveform backend used when none is given: LAL where lalsuite is installed, ripple where it is not."""

from __future__ import annotations

import logging

import pytest

from gwmock_signal.waveform import factory
from gwmock_signal.waveform.factory import WaveformFactory, default_waveform_backend

pytestmark = pytest.mark.unit


class _StubRipple:
    """Stands in for RippleBackend, so the choice is tested without JAX."""

    def available_approximants(self):
        return ["IMRPhenomD"]


def _unavailable():
    raise ImportError("ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'")


class _StubLAL(_StubRipple):
    """Stands in for LALSimulationBackend, so the choice is tested without lalsuite."""


def test_lal_is_the_default_where_lalsuite_is_installed(monkeypatch, caplog):
    """Where lalsuite is installed nothing changes: LAL, and no warning."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: True)
    monkeypatch.setattr(factory, "LALSimulationBackend", _StubLAL)
    monkeypatch.setattr(factory, "RippleBackend", _StubRipple)

    with caplog.at_level(logging.WARNING, logger="gwmock_signal.waveform"):
        backend = default_waveform_backend()

    assert type(backend) is _StubLAL
    assert caplog.records == []


def test_ripple_is_used_where_lalsuite_is_not_and_says_so(monkeypatch, caplog):
    """Without lalsuite but with ripple, ripple is used, with one warning saying the waveforms are ripple's."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", _StubRipple)

    with caplog.at_level(logging.WARNING, logger="gwmock_signal.waveform"):
        backend = default_waveform_backend()

    assert isinstance(backend, _StubRipple)
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "ripple (JAX) backend rather than the default LAL one" in caplog.text


def test_without_either_the_error_names_both_remedies(monkeypatch):
    """With neither installed, the error says how to install lalsuite or the jax extra."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", _unavailable)

    with pytest.raises(ImportError, match="requires lalsuite") as raised:
        default_waveform_backend()

    assert "pip install 'gwmock-signal[jax]'" in str(raised.value)


def test_the_factory_uses_it_when_given_no_backend(monkeypatch):
    """``WaveformFactory()`` takes the same default, so every caller that passes no backend gets it."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", _StubRipple)

    assert WaveformFactory().list_models() == ["IMRPhenomD"]


def test_a_given_backend_is_never_replaced(monkeypatch):
    """An explicitly chosen backend is used as given, whatever is installed."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", _unavailable)
    given = _StubRipple()

    assert WaveformFactory(backend=given)._backend is given


def test_a_real_ripple_waveform_where_lalsuite_is_hidden(monkeypatch):
    """End to end with the real ripple backend, as on a Python lalsuite does not install on."""
    pytest.importorskip("ripplegw")
    from gwmock_signal import CBCSimulator

    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    stack = CBCSimulator("IMRPhenomD").simulate(
        {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "distance": 400.0,
            "coa_time": 1_000_000_000.0,
            "right_ascension": 1.0,
            "declination": 0.3,
            "polarization_angle": 0.2,
            "inclination": 0.5,
            "coa_phase": 0.0,
        },
        ["H1", "L1"],
        sampling_frequency=2048.0,
        minimum_frequency=20.0,
    )

    assert stack.detector_names == ("H1", "L1")
    assert stack.data.shape[0] == 2
    assert 1e-23 < float(abs(stack.data).max()) < 1e-20
