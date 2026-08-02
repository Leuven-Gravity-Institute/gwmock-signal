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


class TestTheDeviceBackend:
    """``backend="jax"`` must be the same projection, and must refuse what it cannot serve."""

    @pytest.fixture(autouse=True)
    def _sixty_four_bit(self):
        """Enable x64, which the device path requires and this module does not otherwise need.

        Set here rather than at module import, the convention in the JAX-only test modules,
        because most of this file tests the host path and should not depend on JAX being
        installed at all. The flag is process-global and cannot be restored -- JAX rejects
        flipping it once arrays exist -- but every other module that touches JAX enables it too,
        so the suite is consistent either way.
        """
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)

    @staticmethod
    def _polarizations(duration: float, sampling_frequency: float = 512.0):
        n = int(duration * sampling_frequency)
        t = np.arange(n) / sampling_frequency
        return {
            "plus": TimeSeries(1e-24 * np.cos(2 * np.pi * 20.0 * t), t0=1.4e9, sample_rate=sampling_frequency),
            "cross": TimeSeries(1e-24 * np.sin(2 * np.pi * 20.0 * t), t0=1.4e9, sample_rate=sampling_frequency),
        }

    def test_it_reproduces_the_host_path(self):
        """Same algorithm, so the two may differ only by floating-point reassociation.

        Checked through this function rather than the primitive underneath, which
        ``test_rotating_projection_matches_numpy_path`` already covers -- what is new here is the
        dispatch: the geometry lookup, the sidereal anchor, and the series that comes back.

        The span is 64 s, long enough that the linear sidereal model and the per-sample Astropy
        one have somewhere to disagree, and short enough to keep the host reference affordable.
        """
        pytest.importorskip("jax")
        polarizations = self._polarizations(64.0)
        sky = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}
        detectors = ["H1", "L1"]

        host = project_polarizations_to_network(polarizations, detectors, earth_rotation=True, **sky)
        device = project_polarizations_to_network(polarizations, detectors, earth_rotation=True, backend="jax", **sky)

        for name in detectors:
            peak = float(np.max(np.abs(host[name].value)))
            assert peak > 0.0, "a null response would make the comparison below vacuous"
            worst = float(np.max(np.abs(device[name].value - host[name].value))) / peak
            assert worst < 1e-10, f"{name} differs from the host path by {worst:.3e} of peak"

    def test_the_epoch_and_rate_survive_the_round_trip(self):
        """A device path returning bare arrays could lose the time coordinate silently."""
        pytest.importorskip("jax")
        polarizations = self._polarizations(8.0)

        device = project_polarizations_to_network(
            polarizations,
            ["H1"],
            earth_rotation=True,
            backend="jax",
            right_ascension=1.3,
            declination=-0.4,
            polarization_angle=0.7,
        )["H1"]

        assert float(device.t0.value) == 1.4e9
        assert float(device.sample_rate.value) == 512.0
        assert len(device.value) == len(polarizations["plus"].value)

    def test_the_device_primitive_is_the_thing_that_runs(self):
        """Asserted directly, because no comparison can establish it.

        A ``backend="jax"`` that quietly fell through to the host path would satisfy every
        equivalence check in this class -- the two sides would be the same code. Mutating the
        dispatch away leaves only this test failing, which is why it exists separately from the
        agreement test rather than being folded into it.
        """
        pytest.importorskip("jax")
        from gwmock_signal.projection import jax_projection

        calls = []
        original = jax_projection.project_polarizations_td_rotating

        def _spy(*args, **kwargs):
            calls.append(kwargs.get("n_samples"))
            return original(*args, **kwargs)

        with patch.object(jax_projection, "project_polarizations_td_rotating", _spy):
            project_polarizations_to_network(
                self._polarizations(4.0),
                ["H1", "L1"],
                earth_rotation=True,
                backend="jax",
                right_ascension=1.3,
                declination=-0.4,
                polarization_angle=0.7,
            )

        assert len(calls) == 2, f"expected one device call per detector, got {len(calls)}"
        assert set(calls) == {int(4.0 * 512.0)}

    def test_the_host_backend_does_not_reach_the_device(self):
        """The default must stay on the host, or installing JAX would change existing output."""
        pytest.importorskip("jax")
        from gwmock_signal.projection import jax_projection

        with patch.object(jax_projection, "project_polarizations_td_rotating") as device:
            project_polarizations_to_network(
                self._polarizations(4.0),
                ["H1"],
                earth_rotation=True,
                right_ascension=1.3,
                declination=-0.4,
                polarization_angle=0.7,
            )

        device.assert_not_called()

    def test_an_unknown_backend_is_refused(self):
        """A typo must not fall through to the host path while the caller believes otherwise."""
        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            project_polarizations_to_network(
                self._polarizations(1.0),
                ["H1"],
                right_ascension=1.3,
                declination=-0.4,
                polarization_angle=0.7,
                backend="cuda",
            )

    def test_the_constant_pattern_branch_has_no_device_path(self):
        """Serving it from the host would report a backend that did not run."""
        with pytest.raises(ValueError, match="backend='jax' is only available with earth_rotation=True"):
            project_polarizations_to_network(
                self._polarizations(1.0),
                ["H1"],
                right_ascension=1.3,
                declination=-0.4,
                polarization_angle=0.7,
                earth_rotation=False,
                backend="jax",
            )

    def test_thirty_two_bit_jax_is_refused_rather_than_degraded(self, monkeypatch):
        """In 32-bit mode the device path returns plausible strain that is wrong by ~1% of peak.

        Measured against the host path while writing this: 2.7e-03 over 256 s and 1.8e-02 over
        1024 s, growing with the span as the GPS times lose precision. Nothing downstream can
        distinguish such a series from a correct one, so it has to fail here. It has not bitten
        anyone yet only because importing ``ripplegw`` happens to enable x64 -- an accident of an
        unrelated import, which is exactly the kind of thing that stops being true.
        """
        jax = pytest.importorskip("jax")
        # `jax.config.jax_enable_x64` is a read-only property, and flipping the real flag after
        # arrays exist is rejected by JAX. Patching the property on the class is what is left;
        # the guard reads exactly this attribute, which is the thing under test.
        monkeypatch.setattr(type(jax.config), "jax_enable_x64", property(lambda _self: False))

        with pytest.raises(RuntimeError, match="requires JAX in 64-bit mode"):
            project_polarizations_to_network(
                self._polarizations(1.0),
                ["H1"],
                right_ascension=1.3,
                declination=-0.4,
                polarization_angle=0.7,
                backend="jax",
            )
