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

    hp_vals = np.asarray(hp.value, dtype=float)
    hc_vals = np.asarray(hc.value, dtype=float)
    n = len(hp_vals)
    dt = float(hp.dt.value)
    freqs = np.fft.rfftfreq(n, d=dt)
    rfft_hp = np.fft.rfft(hp_vals)
    rfft_hc = np.fft.rfft(hc_vals)

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
        hp_shifted = np.fft.irfft(rfft_hp * phase, n=n)
        hc_shifted = np.fft.irfft(rfft_hc * phase, n=n)
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


def test_fd_shift_earth_rotation_false_rms_amplitude_retention() -> None:
    """Tapered 500 Hz sinusoid projected with a known delay retains RMS amplitude to <0.01%."""
    fs, n = 2048.0, 2048
    t = np.arange(n) / fs
    taper = np.hanning(n)
    signal = np.sin(2 * np.pi * 500.0 * t) * taper
    hp = TimeSeries(signal, t0=0.0, sample_rate=fs)
    hc = TimeSeries(np.zeros(n), t0=0.0, sample_rate=fs)
    with (
        patch("gwmock_signal.projection.network._time_delay_from_earth_center_lal", return_value=3e-3),
        patch("gwmock_signal.projection.network._antenna_pattern_lal", return_value=(1.0, 0.0)),
    ):
        out = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
            earth_rotation=False,
        )
    rms_out = float(np.sqrt(np.mean(out["H1"].value ** 2)))
    rms_in = float(np.sqrt(np.mean(signal**2)))
    assert abs(rms_out / rms_in - 1.0) < 1e-4


def test_fd_shift_earth_rotation_false_matches_direct_fd_reference() -> None:
    """Broadband projected signal matches a directly computed FD-shift reference to <1e-10."""
    rng = np.random.default_rng(42)
    fs, n = 2048.0, 8192
    taper = np.hanning(n)
    signal = rng.standard_normal(n) * taper
    hp = TimeSeries(signal, t0=0.0, sample_rate=fs)
    hc = TimeSeries(np.zeros(n), t0=0.0, sample_rate=fs)
    delay = 3e-3
    with (
        patch("gwmock_signal.projection.network._time_delay_from_earth_center_lal", return_value=delay),
        patch("gwmock_signal.projection.network._antenna_pattern_lal", return_value=(1.0, 0.0)),
    ):
        out = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
            earth_rotation=False,
        )
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    phase = np.exp(-2j * np.pi * freqs * delay)
    ref = np.fft.irfft(np.fft.rfft(signal) * phase, n=n)
    np.testing.assert_allclose(out["H1"].value, ref, rtol=1e-10, atol=1e-12)


def test_fd_shift_earth_rotation_false_no_circular_wrap_leakage() -> None:
    """Tapered input does not leak energy across segment boundary under the FD shift."""
    fs, n = 2048.0, 4096
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 100.0 * t) * np.hanning(n)
    hp = TimeSeries(sig, t0=0.0, sample_rate=fs)
    hc = TimeSeries(np.zeros(n), t0=0.0, sample_rate=fs)
    with (
        patch("gwmock_signal.projection.network._time_delay_from_earth_center_lal", return_value=3e-3),
        patch("gwmock_signal.projection.network._antenna_pattern_lal", return_value=(1.0, 0.0)),
    ):
        out = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            ["H1"],
            right_ascension=0.0,
            declination=0.0,
            polarization_angle=0.0,
            earth_rotation=False,
        )
    vals = out["H1"].value
    edge_rms = float(np.sqrt(np.mean(np.concatenate([vals[:5], vals[-5:]]) ** 2)))
    peak_rms = float(np.sqrt(np.mean(vals**2)))
    assert edge_rms < 1e-4 * peak_rms
