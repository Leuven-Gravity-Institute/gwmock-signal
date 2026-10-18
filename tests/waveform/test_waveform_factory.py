"""Tests for `WaveformFactory`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

from gwmock_signal.waveform import LALSimulationBackend, WaveformBackend, WaveformFactory, pycbc_waveform_wrapper


@pytest.mark.requires_lalsuite
@patch.object(LALSimulationBackend, "available_approximants", return_value=["IMRPhenomD", "SEOBNRv4"])
def test_list_models_returns_list(mock_available_approximants) -> None:
    """list_models() reflects the default backend's available approximants."""
    factory = WaveformFactory()
    models = factory.list_models()
    assert isinstance(models, list)
    assert "IMRPhenomD" in models
    mock_available_approximants.assert_called_once()


@pytest.mark.requires_lalsuite
def test_register_model_non_callable_raises_type_error() -> None:
    """Registering a non-callable raises TypeError."""
    factory = WaveformFactory()
    with pytest.raises(TypeError, match="not callable"):
        factory.register_model("bad", 42)  # type: ignore[arg-type]


@pytest.mark.requires_lalsuite
def test_register_model_string_no_colon_no_dot_raises_value_error() -> None:
    """A bare string without ':' or '.' raises ValueError."""
    factory = WaveformFactory()
    with pytest.raises(ValueError, match=r"module.path:callable"):
        factory.register_model("bad", "nodots_and_no_colon")


@pytest.mark.requires_lalsuite
def test_generate_with_waveform_model_in_params_raises_value_error() -> None:
    """Passing 'waveform_model' inside parameters raises ValueError."""
    factory = WaveformFactory()
    factory.register_model("toy", lambda **kw: {"plus": None, "cross": None})
    with pytest.raises(ValueError, match="waveform_model"):
        factory.generate("toy", {"waveform_model": "toy"})


@pytest.mark.requires_lalsuite
def test_get_model_unknown_raises():
    """Unknown model names raise `ValueError`."""
    factory = WaveformFactory()
    with pytest.raises(ValueError, match="not found"):
        factory.get_model("definitely_not_a_real_approximant_name_xyz")


@pytest.mark.requires_gwpy
def test_register_and_generate_toy():
    """Custom model returns consistent plus/cross GWpy series."""

    def toy(
        *,
        waveform_model: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **kwargs,
    ):
        width_s = float(kwargs.get("width", 0.1))
        f_hz = float(kwargs.get("f0", 150.0))
        dt = 1.0 / sampling_frequency
        n_half = max(8, int(width_s * sampling_frequency))
        t = (np.arange(-n_half, n_half + 1) * dt) + tc
        env = np.exp(-0.5 * ((t - tc) / (width_s / 3)) ** 2)
        phase = 2 * np.pi * f_hz * (t - tc)
        hp = env * np.cos(phase)
        hc = env * np.sin(phase)
        return {
            "plus": TimeSeries(hp, t0=float(t[0]), dt=dt, unit=u.dimensionless_unscaled),
            "cross": TimeSeries(hc, t0=float(t[0]), dt=dt, unit=u.dimensionless_unscaled),
        }

    factory = WaveformFactory()
    factory.register_model("toy_burst", toy)
    out = factory.generate(
        "toy_burst",
        {"tc": 1_400_000_000.0, "f0": 120.0, "width": 0.05},
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
    )
    assert set(out) == {"plus", "cross"}
    assert len(out["plus"]) == len(out["cross"])


@pytest.mark.requires_lalsuite
def test_register_model_string_import():
    """String `register_model` path with ':' resolves to the same callable object."""
    factory = WaveformFactory()
    factory.register_model("pycbc_alias", "gwmock_signal.waveform.pycbc_wrapper:pycbc_waveform_wrapper")
    assert factory.get_model("pycbc_alias") is pycbc_waveform_wrapper


@pytest.mark.requires_lalsuite
def test_register_model_string_dotted_import() -> None:
    """Dotted string without ':' resolves the callable via rsplit on the last '.'."""
    factory = WaveformFactory()
    factory.register_model("pycbc_dotted", "gwmock_signal.waveform.pycbc_wrapper.pycbc_waveform_wrapper")
    assert factory.get_model("pycbc_dotted") is pycbc_waveform_wrapper


@pytest.mark.requires_gwpy
def test_factory_generate_delegates_to_backend() -> None:
    """Built-in model dispatch uses the configured backend implementation."""
    backend = MagicMock(spec=WaveformBackend)
    backend.available_approximants.return_value = ["IMRPhenomD"]
    hp = TimeSeries(np.ones(16), t0=100.0, dt=1.0 / 4096)
    hc = TimeSeries(np.zeros(16), t0=100.0, dt=1.0 / 4096)
    backend.generate_td_waveform.return_value = {"plus": hp, "cross": hc}

    factory = WaveformFactory(backend=backend)
    factory.generate(
        "IMRPhenomD",
        {"tc": 100.0, "mass1": 10.0, "mass2": 10.0},
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
    )
    backend.generate_td_waveform.assert_called_once_with(
        approximant="IMRPhenomD",
        tc=100.0,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        mass1=10.0,
        mass2=10.0,
    )
