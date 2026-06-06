"""Tests for `project_polarizations_to_network`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.projection.network import project_polarizations_to_network


def _uniform_series(n: int = 128, fs: float = 4096.0, t0: float = 100.0) -> TimeSeries:
    t = np.arange(n) / fs + t0
    return TimeSeries(np.sin(2 * np.pi * 10.0 * t), t0=t0, sample_rate=fs)


def _project_with_pycbc_reference(  # noqa: PLR0913
    hp: TimeSeries,
    hc: TimeSeries,
    detector_names: list[str],
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
) -> dict[str, np.ndarray]:
    pytest.importorskip("pycbc", reason="pycbc not installed")
    from pycbc.detector import Detector as PyCBCDetector

    time_array = np.asarray(hp.times.value, dtype=float)
    reference_time = float(0.5 * (time_array[0] + time_array[-1]))
    dt = float(hp.dt.value)
    freqs = np.fft.rfftfreq(len(hp.value), d=dt)
    hp_rfft = np.fft.rfft(hp.value)
    hc_rfft = np.fft.rfft(hc.value)

    strains: dict[str, np.ndarray] = {}
    for name in detector_names:
        detector = PyCBCDetector(name)
        time_delay = detector.time_delay_from_earth_center(
            right_ascension=right_ascension,
            declination=declination,
            t_gps=reference_time,
        )
        fp, fc = detector.antenna_pattern(
            right_ascension=right_ascension,
            declination=declination,
            polarization=polarization_angle,
            t_gps=reference_time,
            polarization_type="tensor",
        )
        phase = np.exp(-2j * np.pi * freqs * time_delay)
        hp_shifted = np.fft.irfft(hp_rfft * phase, n=len(hp.value))
        hc_shifted = np.fft.irfft(hc_rfft * phase, n=len(hc.value))
        strains[name] = np.asarray(fp * hp_shifted + fc * hc_shifted, dtype=float)
    return strains


def test_polarizations_not_a_mapping_raises_type_error() -> None:
    """Passing a non-mapping raises TypeError."""
    with pytest.raises(TypeError, match="mapping"):
        project_polarizations_to_network(
            [1, 2, 3],  # type: ignore[arg-type]
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_polarizations_wrong_series_type_raises_type_error() -> None:
    """Plus/cross values that are not GWpy TimeSeries raise TypeError."""
    with pytest.raises(TypeError, match=r"gwpy.timeseries.TimeSeries"):
        project_polarizations_to_network(
            {"plus": np.ones(8), "cross": np.zeros(8)},  # type: ignore[arg-type]
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_mismatched_sample_rates_raises_value_error() -> None:
    """Plus and cross with different sample rates raise ValueError."""
    hp = _uniform_series(fs=4096.0)
    hc = _uniform_series(fs=2048.0)
    with pytest.raises(ValueError, match="same sample rate"):
        project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_mismatched_time_grids_raises_value_error() -> None:
    """Plus and cross on different time grids raise ValueError."""
    hp = _uniform_series(t0=100.0)
    hc = _uniform_series(t0=200.0)
    with pytest.raises(ValueError, match="same time samples"):
        project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_duplicate_detector_names_raises_value_error() -> None:
    """Duplicate entries in detector_names raise ValueError."""
    hp = hc = _uniform_series()
    with pytest.raises(ValueError, match="duplicates"):
        project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1", "H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_requires_plus_cross_keys():
    """Polarizations mapping must include plus and cross."""
    hp = _uniform_series()
    with pytest.raises(ValueError, match=r"plus.*cross"):
        project_polarizations_to_network(
            {"plus": hp},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_requires_matching_length():
    """Plus and cross must have the same number of samples."""
    hp = _uniform_series(n=64)
    hc = _uniform_series(n=32)
    with pytest.raises(ValueError, match="same number of samples"):
        project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


def test_unknown_detector_name():
    """Invalid IFO codes raise ValueError with a helpful message."""
    hp = hc = _uniform_series()
    with pytest.raises(ValueError, match="Unknown or unsupported"):
        project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["NOT_A_REAL_DETECTOR_XYZ"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
        )


@patch("gwmock_signal.projection.network._antenna_pattern_lal", return_value=(1.0, 0.0))
@patch("gwmock_signal.projection.network._time_delay_from_earth_center_lal", return_value=0.0)
def test_delegates_to_lal(mock_time_delay, mock_antenna_pattern):
    """Each built-in detector name resolves through the LAL detector path."""
    t0 = 0.0
    fs = 8.0
    hp = TimeSeries(np.ones(8), t0=t0, sample_rate=fs)
    hc = TimeSeries(np.zeros(8), t0=t0, sample_rate=fs)

    names = ["H1", "L1"]
    with patch.dict(
        "gwmock_signal.projection.network.lal.cached_detector_by_prefix",
        {"H1": object(), "L1": object()},
        clear=False,
    ):
        out = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            names,
            right_ascension=0.1,
            declination=0.2,
            polarization_angle=0.3,
            earth_rotation=False,
        )
    assert set(out) == set(names)
    assert mock_time_delay.call_count == len(names)
    assert mock_antenna_pattern.call_count == len(names)


@patch("gwmock_signal.projection.network._antenna_pattern_lal", return_value=(1.0, 0.0))
@patch("gwmock_signal.projection.network._time_delay_from_earth_center_lal")
def test_half_sample_shift_preserves_amplitude(mock_td, mock_ap) -> None:
    """FD phase shift retains 500 Hz RMS amplitude to <0.05% after a half-sample delay."""
    fs = 2048.0
    n = 2048
    mock_td.return_value = 0.5 / fs
    t = np.arange(n) / fs
    amplitude = 3.0
    hp = TimeSeries(amplitude * np.sin(2 * np.pi * 500.0 * t), t0=0.0, sample_rate=fs)
    hc = TimeSeries(np.zeros(n), t0=0.0, sample_rate=fs)
    out = project_polarizations_to_network(
        {"plus": hp, "cross": hc},
        ["H1"],
        right_ascension=0.0,
        declination=0.0,
        polarization_angle=0.0,
        earth_rotation=False,
    )
    rms_out = np.sqrt(np.mean(out["H1"].value ** 2))
    rms_in = amplitude / np.sqrt(2)  # RMS of a pure sinusoid
    assert abs(rms_out - rms_in) / rms_in < 5e-4


def test_matches_pycbc_reference_on_gw150914_like_case() -> None:
    """The direct-LAL projection path matches the previous PyCBC detector result."""
    n = 1024
    fs = 4096.0
    t0 = 1126259462.4 - 0.125
    times = np.arange(n) / fs
    taper = np.hanning(n)
    hp = TimeSeries(np.sin(2 * np.pi * 35.0 * times) * taper, t0=t0, sample_rate=fs)
    hc = TimeSeries(np.cos(2 * np.pi * 35.0 * times) * taper, t0=t0, sample_rate=fs)

    detector_names = ["H1", "L1", "V1"]
    kwargs = {
        "right_ascension": 1.375,
        "declination": -1.211,
        "polarization_angle": 0.0,
    }
    projected = project_polarizations_to_network(
        {"plus": hp, "cross": hc},
        detector_names,
        earth_rotation=False,
        **kwargs,
    )
    expected = _project_with_pycbc_reference(hp, hc, detector_names, **kwargs)

    for name in detector_names:
        np.testing.assert_allclose(projected[name].value, expected[name], rtol=1e-10, atol=1e-12)
