"""Tests for `project_polarizations_to_network`."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.projection.network import (
    _CONSTANT_PATTERN_ERROR_PER_SECOND,
    _CONSTANT_PATTERN_SATURATION,
    _CONSTANT_PATTERN_WARN_SECONDS,
    project_polarizations_to_network,
)


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
        "gwmock_signal.projection.geometry.lal.cached_detector_by_prefix",
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


class TestConstantPatternDurationWarning:
    """`earth_rotation=False` is a good trade for a short signal and wrong for a long one.

    The branch evaluates the antenna pattern once, at the midpoint of the span it is handed, so its
    error grows with how far Earth turns across that span -- and nothing in the output reveals it.
    Measured against the rotating branch across 48 detector/sky/polarization/frequency combinations,
    the worst deviation grows at up to 2.9e-4 of peak per second. The warning exists because the
    parameter is a bare boolean with no stated domain of validity.
    """

    @staticmethod
    def _polarizations(seconds: float, sampling_frequency: float = 16.0):
        n = int(seconds * sampling_frequency)
        t = np.arange(n) / sampling_frequency
        return {
            "plus": TimeSeries(
                np.cos(2 * np.pi * 4.0 * t), t0=1577491218.0, sample_rate=sampling_frequency, unit="strain"
            ),
            "cross": TimeSeries(
                np.sin(2 * np.pi * 4.0 * t), t0=1577491218.0, sample_rate=sampling_frequency, unit="strain"
            ),
        }

    def _project(self, seconds: float, *, earth_rotation: bool):
        return project_polarizations_to_network(
            self._polarizations(seconds),
            ["H1"],
            right_ascension=1.1,
            declination=0.3,
            polarization_angle=0.2,
            earth_rotation=earth_rotation,
        )

    def test_a_long_span_warns(self, caplog):
        """The warning fires at all, and names the span responsible."""
        span = 4 * _CONSTANT_PATTERN_WARN_SECONDS

        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            self._project(span, earth_rotation=False)

        assert "earth_rotation=False" in caplog.text
        assert f"{span:.1f} s" in caplog.text or f"{span - 1 / 16.0:.1f} s" in caplog.text, (
            "the message must name the span that triggered it"
        )

    def test_the_estimate_is_derived_from_the_measured_rate(self, caplog):
        """A bare 'this may be inaccurate' does not tell the reader whether to care.

        Computed from the constants rather than hard-coded, so revising the measurement updates the
        test with the code instead of leaving a stale literal behind.
        """
        span = 4 * _CONSTANT_PATTERN_WARN_SECONDS
        expected = 100.0 * span * _CONSTANT_PATTERN_ERROR_PER_SECOND

        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            self._project(span, earth_rotation=False)

        assert f"up to about {expected:.0f}%" in caplog.text

    def test_a_saturating_span_names_saturation_rather_than_a_percentage(self, caplog):
        """The linear rate is local. Extrapolated to a day it would claim over 1000% of peak.

        Past the saturation point the honest statement is that the deviation is of order the signal
        itself, not a number the model cannot support.
        """
        span = 10.0 * _CONSTANT_PATTERN_SATURATION / _CONSTANT_PATTERN_ERROR_PER_SECOND

        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            self._project(span, earth_rotation=False)

        assert "of order the signal amplitude itself" in caplog.text
        assert "%" not in caplog.text.split("turns across it")[1].split(".")[0], (
            "a percentage was quoted past the point where the linear model holds"
        )

    def test_a_span_exactly_at_the_threshold_warns(self, caplog):
        """The boundary is inclusive, and pinned here because it is otherwise a coin toss.

        A span of exactly the threshold sits at the one-percent mark. Warning there is the safer
        way round for a guard against an error nothing else reveals, and an off-by-one in the
        comparison would otherwise go unnoticed.
        """
        rate = 16.0
        # span = (n - 1) / rate, so this lands on the threshold exactly rather than near it.
        samples = int(_CONSTANT_PATTERN_WARN_SECONDS * rate) + 1
        t = np.arange(samples) / rate
        polarizations = {
            "plus": TimeSeries(np.cos(2 * np.pi * 4.0 * t), t0=1577491218.0, sample_rate=rate, unit="strain"),
            "cross": TimeSeries(np.sin(2 * np.pi * 4.0 * t), t0=1577491218.0, sample_rate=rate, unit="strain"),
        }
        span = float(np.asarray(polarizations["plus"].times.to_value())[-1] - 1577491218.0)
        assert span == _CONSTANT_PATTERN_WARN_SECONDS, f"test setup gave a span of {span}, not the threshold"

        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            project_polarizations_to_network(
                polarizations,
                ["H1"],
                right_ascension=1.1,
                declination=0.3,
                polarization_angle=0.2,
                earth_rotation=False,
            )

        assert "earth_rotation=False" in caplog.text

    def test_a_short_span_is_silent(self, caplog):
        """Otherwise every compact-binary projection would warn and the message would be ignored."""
        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            self._project(_CONSTANT_PATTERN_WARN_SECONDS / 3.0, earth_rotation=False)

        assert "earth_rotation" not in caplog.text

    def test_the_rotating_branch_never_warns(self, caplog):
        """The warning is about the approximation, not about long signals."""
        with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
            self._project(4 * _CONSTANT_PATTERN_WARN_SECONDS, earth_rotation=True)

        assert "earth_rotation" not in caplog.text


def _flat_polarizations(duration: float, sampling_frequency: float = 512.0, t0: float = 1.4e9):
    """A monochromatic pair of polarizations spanning *duration*."""
    n = int(duration * sampling_frequency)
    t = np.arange(n) / sampling_frequency
    return {
        "plus": TimeSeries(1e-24 * np.cos(2 * np.pi * 20.0 * t), t0=t0, sample_rate=sampling_frequency),
        "cross": TimeSeries(1e-24 * np.sin(2 * np.pi * 20.0 * t), t0=t0, sample_rate=sampling_frequency),
    }


_SKY = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}


class TestTheBackendArgumentValidation:
    """Rejections that are pure argument checking, so they must hold without JAX installed.

    Kept out of :class:`TestTheDeviceBackend`, whose autouse fixture skips the whole class when
    JAX is absent. These two need no JAX and are exactly the checks a base installation should
    still be running.
    """

    def test_an_unknown_backend_is_refused(self):
        """A typo must not fall through to the host path while the caller believes otherwise."""
        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            project_polarizations_to_network(_flat_polarizations(1.0), ["H1"], backend="cuda", **_SKY)

    def test_the_constant_pattern_branch_has_no_device_path(self):
        """Serving it from the host would report a backend that did not run."""
        with pytest.raises(ValueError, match="backend='jax' is only available with earth_rotation=True"):
            project_polarizations_to_network(
                _flat_polarizations(1.0), ["H1"], earth_rotation=False, backend="jax", **_SKY
            )


class TestTheDeviceBackend:
    """``backend="jax"`` must be the same projection, and must refuse what it cannot serve."""

    @pytest.fixture(autouse=True)
    def _sixty_four_bit(self):
        """Enable x64 for these tests and restore whatever the flag was.

        Restored rather than left set, because tests run in random order and a leaked flag would
        change the dtype of unrelated JAX work. The device path requires x64; most of this module
        tests the host path and should not need JAX at all, which is why this is a class fixture
        rather than a module-level import-time call.
        """
        jax = pytest.importorskip("jax")
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", previous)

    @staticmethod
    def _polarizations(duration: float, sampling_frequency: float = 512.0):
        return _flat_polarizations(duration, sampling_frequency)

    def test_it_reproduces_the_host_path(self):
        """Same algorithm, so the two may differ only by floating-point reassociation.

        Checked through this function rather than the primitive underneath, which
        ``test_rotating_projection_matches_numpy_path`` already covers -- what is new here is the
        dispatch: the geometry lookup, the sidereal anchor, and the series that comes back.
        """
        polarizations = self._polarizations(64.0)
        detectors = ["H1", "L1"]

        host = project_polarizations_to_network(polarizations, detectors, earth_rotation=True, **_SKY)
        device = project_polarizations_to_network(polarizations, detectors, earth_rotation=True, backend="jax", **_SKY)

        for name in detectors:
            peak = float(np.max(np.abs(host[name].value)))
            assert peak > 0.0, "a null response would make the comparison below vacuous"
            worst = float(np.max(np.abs(device[name].value - host[name].value))) / peak
            assert worst < 1e-10, f"{name} differs from the host path by {worst:.3e} of peak"

    @pytest.mark.parametrize(("duration", "sampling_frequency"), [(600.0, 256.0), (64.0, 2048.0)])
    def test_it_reproduces_the_host_path_at_other_shapes(self, duration, sampling_frequency):
        """One shape is not evidence for the rest.

        The single agreement test above is 64 s at 512 Hz while the continuous-wave simulator runs
        segments of hundreds of seconds. A divergence that grew with span, or one that depended on
        sample rate through the resampler's Nyquist margin, would not show up there.
        """
        polarizations = self._polarizations(duration, sampling_frequency)

        host = project_polarizations_to_network(polarizations, ["H1"], earth_rotation=True, **_SKY)
        device = project_polarizations_to_network(polarizations, ["H1"], earth_rotation=True, backend="jax", **_SKY)

        peak = float(np.max(np.abs(host["H1"].value)))
        assert peak > 0.0
        worst = float(np.max(np.abs(device["H1"].value - host["H1"].value))) / peak
        assert worst < 1e-10, f"{duration} s at {sampling_frequency} Hz differs by {worst:.3e} of peak"

    def test_a_non_default_kernel_still_agrees(self):
        """The tap count and window reach both implementations, or only one honoured them."""
        polarizations = self._polarizations(16.0)
        kernel = {"sinc_taps": 63, "kaiser_beta": 8.0}

        host = project_polarizations_to_network(polarizations, ["H1"], earth_rotation=True, **kernel, **_SKY)
        device = project_polarizations_to_network(
            polarizations, ["H1"], earth_rotation=True, backend="jax", **kernel, **_SKY
        )

        peak = float(np.max(np.abs(host["H1"].value)))
        worst = float(np.max(np.abs(device["H1"].value - host["H1"].value))) / peak
        assert worst < 1e-10, f"a 63-tap kernel differs between paths by {worst:.3e} of peak"

    def test_the_epoch_and_rate_survive_the_round_trip(self):
        """A device path returning bare arrays could lose the time coordinate silently."""
        polarizations = self._polarizations(8.0)

        device = project_polarizations_to_network(polarizations, ["H1"], earth_rotation=True, backend="jax", **_SKY)[
            "H1"
        ]

        assert float(device.t0.value) == 1.4e9
        assert float(device.sample_rate.value) == 512.0
        assert len(device.value) == len(polarizations["plus"].value)

    def test_the_device_kernel_is_what_runs(self):
        """Asserted directly, because no comparison can establish it.

        A ``backend="jax"`` that quietly fell through to the host path would satisfy every
        equivalence check in this class -- the two sides would be the same code.
        """
        from gwmock_signal.projection import network

        network._compiled_rotating_projection.cache_clear()
        with patch.object(network, "_project_rotating_on_device") as device:
            device.return_value = {}
            network.project_polarizations_to_network(
                self._polarizations(4.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
            )

        device.assert_called_once()

    def test_the_host_only_preparation_is_skipped(self):
        """The device path must dispatch before the host branch's setup, not after it.

        That setup -- two rffts, a frequency grid, and per-sample Astropy GMST with its sines and
        cosines -- serves only the NumPy branches. Computed before the dispatch it was pure waste:
        4.3 s for the Astropy call alone at 4096 s and 512 Hz, and ~120 MiB of arrays, on a path
        whose whole point is not to scale with the host. Nothing about the output changes when it
        is skipped, so only this test stands between the fix and a silent regression.
        """
        from gwmock_signal.projection import network

        with patch.object(network, "_gmst_accurate_array", side_effect=AssertionError("host GMST ran")) as gmst:
            network.project_polarizations_to_network(
                self._polarizations(4.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
            )

        gmst.assert_not_called()

    def test_one_kernel_is_compiled_per_shape_not_per_detector(self):
        """Pins the caching contract at the layer that actually caches.

        An earlier version of this test counted calls to the factory and concluded it proved a
        single JAX compilation. It did not: the factory is called once per projection regardless,
        so the test passed whether or not the geometry was traced. What is checkable here is the
        cache itself -- a three-detector network must not add three entries, and a second call at
        the same shape must hit rather than compile again.
        """
        from gwmock_signal.projection import network

        network._compiled_rotating_projection.cache_clear()
        network.project_polarizations_to_network(
            self._polarizations(4.0), ["H1", "L1", "V1"], earth_rotation=True, backend="jax", **_SKY
        )
        after_first = network._compiled_rotating_projection.cache_info()
        assert after_first.currsize == 1, (
            f"three detectors produced {after_first.currsize} cached kernels; the geometry should "
            f"be traced so one kernel serves the network"
        )

        network.project_polarizations_to_network(
            self._polarizations(4.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
        )
        assert network._compiled_rotating_projection.cache_info().hits > after_first.hits

        network.project_polarizations_to_network(
            self._polarizations(8.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
        )
        assert network._compiled_rotating_projection.cache_info().currsize == 2, (
            "a different segment length must compile its own kernel"
        )

    def test_the_kernel_cache_is_bounded(self):
        """An unbounded cache of XLA executables is a process-lifetime memory leak.

        `jax_batch` caps its equivalent caches at the same bound; this one is reached by a
        long-lived worker projecting varied segment lengths, which is an ordinary way to use the
        library rather than an exotic one.
        """
        from gwmock_signal.projection import network

        assert network._compiled_rotating_projection.cache_info().maxsize == network._KERNEL_CACHE_SIZE
        assert network._KERNEL_CACHE_SIZE is not None

    def test_the_host_backend_does_not_reach_the_device(self):
        """The default must stay on the host, or installing JAX would change existing output."""
        from gwmock_signal.projection import network

        # Cleared because a kernel cached by an earlier test closes over the real primitive and
        # could run without touching the patched symbol, hiding a mutated default.
        network._compiled_rotating_projection.cache_clear()
        with patch.object(network, "_project_rotating_on_device") as device:
            network.project_polarizations_to_network(self._polarizations(4.0), ["H1"], earth_rotation=True, **_SKY)

        device.assert_not_called()

    def test_a_span_beyond_the_validated_sidereal_range_is_refused(self):
        """The device path extrapolates sidereal time linearly from a single anchor.

        Accepted to 86400 s, a ceiling set by an error budget: at a day the model costs 6.2e-11 s
        of geocenter delay, six orders below the 8.6e-05 s precession offset the projection
        already carries. Beyond that nothing has been measured, and a single segment that long
        would get a quietly degraded answer rather than an error. Consecutive segments are
        unaffected at any run length, because each re-anchors against Astropy.
        """
        with pytest.raises(ValueError, match="accepts spans up to"):
            project_polarizations_to_network(
                self._polarizations(90000.0, 1.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
            )

    def test_a_day_long_segment_is_accepted(self):
        """The ceiling must not refuse spans the error budget says are fine.

        An earlier version set it at 8192 s, where the *validation table* stopped rather than
        where the error mattered, which made this simulator reject day-long segments that had
        worked before. Pins the boundary from the accepting side so tightening it silently is a
        test failure.
        """
        result = project_polarizations_to_network(
            self._polarizations(80000.0, 1.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
        )

        assert np.all(np.isfinite(result["H1"].value))

    def test_thirty_two_bit_jax_is_refused(self):
        """In 32-bit mode the device path returns plausible strain that is materially wrong.

        The real flag is flipped here rather than patched. An earlier version of this test
        asserted that JAX rejects changing x64 once arrays exist; that is false for the installed
        version, and patching the property therefore tested only that the guard reads an
        attribute.
        """
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", False)

        with pytest.raises(RuntimeError, match="requires JAX in 64-bit mode"):
            project_polarizations_to_network(
                self._polarizations(1.0), ["H1"], earth_rotation=True, backend="jax", **_SKY
            )

    def test_the_hazard_the_x64_guard_protects_against_is_real(self):
        """The guard is only worth having if 32-bit output is wrong, so measure it.

        Calls the primitive directly, bypassing the guard, and compares against the 64-bit host
        result. Without this the guard's justification -- "wrong by of order a percent of peak
        while still looking like strain" -- lives only in a docstring.
        """
        jax = pytest.importorskip("jax")
        from gwmock_signal.projection.geometry import reconstructed_geometry
        from gwmock_signal.projection.jax_projection import project_polarizations_td_rotating
        from gwmock_signal.projection.sidereal import gmst_anchor_and_rate

        polarizations = self._polarizations(256.0)
        reference = project_polarizations_to_network(polarizations, ["H1"], earth_rotation=True, **_SKY)["H1"]
        plus = np.asarray(polarizations["plus"].value)
        cross = np.asarray(polarizations["cross"].value)
        response, location = reconstructed_geometry("H1")
        anchors, rate = gmst_anchor_and_rate(1.4e9)

        jax.config.update("jax_enable_x64", False)
        degraded = np.asarray(
            project_polarizations_td_rotating(
                plus,
                cross,
                response=response,
                location=location,
                sampling_frequency=512.0,
                n_samples=len(plus),
                gmst_start=float(np.atleast_1d(anchors)[0]),
                gmst_rate=float(rate),
                **_SKY,
            )
        )

        peak = float(np.max(np.abs(reference.value)))
        worst = float(np.max(np.abs(degraded - reference.value))) / peak
        assert worst > 1e-4, (
            f"32-bit output differed from the 64-bit host path by only {worst:.3e} of peak, so the "
            f"guard is refusing something harmless and its message overstates the hazard"
        )
