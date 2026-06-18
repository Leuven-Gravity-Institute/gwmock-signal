"""Unit tests for CustomDetector user-defined geometry."""

from __future__ import annotations

import math
from unittest.mock import patch

import lal
import numpy as np
import pytest

from gwmock_signal.detector import CustomDetector

# Exact H1 geodetic parameters from LAL CachedDetectors
# (lal.CachedDetectors, prefix == "H1").
_H1_LATITUDE_RAD = 0.81079526383
_H1_LONGITUDE_RAD = -2.08405676917
_H1_ELEVATION_M = 142.5540008544922
_H1_XARM_AZIMUTH_RAD = 5.654877185821533
_H1_YARM_AZIMUTH_RAD = 4.084080696105957
_H1_XARM_TILT_RAD = -0.0006195000023581088
_H1_YARM_TILT_RAD = 1.249999968422344e-05


def _make_h1_custom(name: str = "H1_custom", prefix: str = "") -> CustomDetector:
    """Make a CustomDetector with H1 coordinates."""
    return CustomDetector(
        name=name,
        latitude_rad=_H1_LATITUDE_RAD,
        longitude_rad=_H1_LONGITUDE_RAD,
        elevation_m=_H1_ELEVATION_M,
        xarm_azimuth_rad=_H1_XARM_AZIMUTH_RAD,
        yarm_azimuth_rad=_H1_YARM_AZIMUTH_RAD,
        xarm_tilt_rad=_H1_XARM_TILT_RAD,
        yarm_tilt_rad=_H1_YARM_TILT_RAD,
        prefix=prefix,
    )


class TestCustomDetectorH1Validation:
    """Verify CustomDetector with H1 coordinates reproduces LAL H1 patterns."""

    def test_antenna_pattern_matches_h1_at_10_sky_positions(self) -> None:
        """H1 custom geometry must agree with the cached LAL H1 detector."""
        custom = _make_h1_custom()
        ref = lal.cached_detector_by_prefix["H1"]

        rng = np.random.default_rng(42)
        n = 10
        ras = rng.uniform(0, 2 * math.pi, n)
        decs = rng.uniform(-math.pi / 2, math.pi / 2, n)
        psis = rng.uniform(0, math.pi, n)
        t_ref = 1126259462.0
        ts = rng.uniform(t_ref, t_ref + 1e6, n)

        custom_lal = custom.to_lal()
        for i in range(n):
            gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(ts[i])))
            fp_c, fc_c = lal.ComputeDetAMResponse(custom_lal.response, ras[i], decs[i], psis[i], gmst)
            fp_r, fc_r = lal.ComputeDetAMResponse(ref.response, ras[i], decs[i], psis[i], gmst)
            err_p = abs(fp_c - fp_r) / (abs(fp_r) + 1e-30)
            err_c = abs(fc_c - fc_r) / (abs(fc_r) + 1e-30)
            assert err_p < 1e-5, f"F+ relative error {err_p:.2e} at sample {i} exceeds 1e-5"
            assert err_c < 1e-5, f"Fx relative error {err_c:.2e} at sample {i} exceeds 1e-5"

    def test_to_lal_returns_lal_detector(self) -> None:
        """to_lal() must return a LAL Detector instance."""
        custom = _make_h1_custom(name="H1_lal_type_check", prefix="X9")
        det = custom.to_lal()
        assert isinstance(det, lal.Detector)

    def test_to_lal_is_cached(self) -> None:
        """Repeated calls to to_lal() must return the same object."""
        custom = _make_h1_custom(name="H1_cache_check", prefix="Y9")
        det1 = custom.to_lal()
        det2 = custom.to_lal()
        assert det1 is det2


class TestCustomDetectorValidation:
    """Test __post_init__ geometry validation."""

    def test_invalid_latitude_too_high_raises_value_error(self) -> None:
        """Latitude > pi/2 must raise ValueError."""
        with pytest.raises(ValueError, match="latitude_rad"):
            CustomDetector(
                name="bad",
                latitude_rad=math.pi,
                longitude_rad=0.0,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_invalid_latitude_too_low_raises_value_error(self) -> None:
        """Latitude < -pi/2 must raise ValueError."""
        with pytest.raises(ValueError, match="latitude_rad"):
            CustomDetector(
                name="bad",
                latitude_rad=-math.pi,
                longitude_rad=0.0,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_invalid_longitude_too_high_raises_value_error(self) -> None:
        """Longitude > 2*pi must raise ValueError."""
        with pytest.raises(ValueError, match="longitude_rad"):
            CustomDetector(
                name="bad",
                latitude_rad=0.0,
                longitude_rad=2 * math.pi,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_invalid_longitude_too_low_raises_value_error(self) -> None:
        """Longitude < -2*pi must raise ValueError."""
        with pytest.raises(ValueError, match="longitude_rad"):
            CustomDetector(
                name="bad",
                latitude_rad=0.0,
                longitude_rad=-2 * math.pi,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_invalid_elevation_too_low_raises_value_error(self) -> None:
        """Elevation < -1e7 m (and < -1e4 m) must raise ValueError."""
        with pytest.raises(ValueError, match="elevation_m"):
            CustomDetector(
                name="bad",
                latitude_rad=0.0,
                longitude_rad=0.0,
                elevation_m=-1e7,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_invalid_elevation_too_high_raises_value_error(self) -> None:
        """Elevation > 2e5 m must raise ValueError."""
        with pytest.raises(ValueError, match="elevation_m"):
            CustomDetector(
                name="bad",
                latitude_rad=0.0,
                longitude_rad=0.0,
                elevation_m=2e5,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_valid_geometry_does_not_raise(self) -> None:
        """Boundary-safe values must not raise."""
        det = CustomDetector(
            name="ok",
            latitude_rad=0.0,
            longitude_rad=0.0,
            elevation_m=0.0,
            xarm_azimuth_rad=0.0,
            yarm_azimuth_rad=math.pi / 2,
        )
        assert det.name == "ok"

    def test_empty_name_raises_value_error(self) -> None:
        """Blank or whitespace-only name must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CustomDetector(
                name="   ",
                latitude_rad=0.0,
                longitude_rad=0.0,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_non_finite_xarm_azimuth_raises_value_error(self) -> None:
        """Non-finite xarm_azimuth_rad must raise ValueError."""
        with pytest.raises(ValueError, match="xarm_azimuth_rad"):
            CustomDetector(
                name="inf_arm",
                latitude_rad=0.0,
                longitude_rad=0.0,
                elevation_m=0.0,
                xarm_azimuth_rad=math.inf,
                yarm_azimuth_rad=math.pi / 2,
            )

    def test_non_finite_yarm_tilt_raises_value_error(self) -> None:
        """Non-finite yarm_tilt_rad must raise ValueError."""
        with pytest.raises(ValueError, match="yarm_tilt_rad"):
            CustomDetector(
                name="nan_tilt",
                latitude_rad=0.0,
                longitude_rad=0.0,
                elevation_m=0.0,
                xarm_azimuth_rad=0.0,
                yarm_azimuth_rad=math.pi / 2,
                yarm_tilt_rad=math.nan,
            )

    def test_to_lal_raises_runtime_error_when_detector_is_none(self) -> None:
        """RuntimeError is raised if LAL detector construction returns None."""
        det = CustomDetector(
            name="mock_det",
            latitude_rad=0.0,
            longitude_rad=0.0,
            elevation_m=0.0,
            xarm_azimuth_rad=0.0,
            yarm_azimuth_rad=math.pi / 2,
            prefix="Z9",
        )
        with (
            patch("gwmock_signal.detector.lal.CreateDetector", return_value=None),
            pytest.raises(RuntimeError, match="Failed to register"),
        ):
            det.to_lal()


class TestMixedNetworkProjection:
    """Test project_polarizations_to_network with a mixed str/CustomDetector list."""

    def test_mixed_list_produces_output_for_each_detector(self) -> None:
        """project_polarizations_to_network must accept str + CustomDetector."""
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries

        from gwmock_signal.projection.network import project_polarizations_to_network

        n = 128
        sample_rate = 4096.0
        t0 = 1126259462.0
        dt = 1.0 / sample_rate
        times = np.arange(n) * dt

        hp = GWpyTimeSeries(np.sin(2 * math.pi * 30 * times), t0=t0, sample_rate=sample_rate)
        hc = GWpyTimeSeries(np.cos(2 * math.pi * 30 * times), t0=t0, sample_rate=sample_rate)
        polarizations = {"plus": hp, "cross": hc}

        custom_l1 = CustomDetector(
            name="L1_custom",
            latitude_rad=0.53342313506,
            longitude_rad=-1.58430937078,
            elevation_m=-6.574,
            xarm_azimuth_rad=4.40317772346,
            yarm_azimuth_rad=2.83238139666,
            xarm_tilt_rad=-3.121e-4,
            yarm_tilt_rad=-6.107e-4,
        )

        result = project_polarizations_to_network(
            polarizations,
            ["H1", custom_l1],
            right_ascension=1.2,
            declination=0.3,
            polarization_angle=0.5,
        )

        assert set(result.keys()) == {"H1", "L1_custom"}
        for key, ts in result.items():
            assert isinstance(ts, GWpyTimeSeries), f"{key}: expected GWpyTimeSeries"
            assert len(ts) == n


def test_to_lal_reallocates_auto_generated_prefix_on_collision() -> None:
    """Two auto-prefixed detectors that initially draw the same prefix must not clash.

    Reproduces the construct-then-register window: prefixes are allocated at
    construction, but two siblings built before either registers can draw the same
    auto prefix. The second ``to_lal`` must re-allocate instead of raising.
    """
    prefixes = iter(["ZZ", "ZZ", "ZY"])  # both built as 'ZZ'; the re-allocation yields 'ZY'
    for stale in ("ZZ", "ZY"):
        lal.cached_detector_by_prefix.pop(stale, None)
    try:
        with patch("gwmock_signal.detector._generate_detector_prefix", lambda: next(prefixes)):
            first = _make_h1_custom(name="auto_first")  # prefix="" -> 'ZZ'
            second = _make_h1_custom(name="auto_second")  # -> 'ZZ'
            first_lal = first.to_lal()  # registers 'ZZ'
            second_lal = second.to_lal()  # 'ZZ' taken + auto-generated -> re-allocate -> 'ZY'
        assert first_lal.frDetector.prefix == "ZZ"
        assert second_lal.frDetector.prefix == "ZY"
        assert second.to_lal() is second_lal  # cached, idempotent
    finally:
        for prefix in ("ZZ", "ZY"):
            lal.cached_detector_by_prefix.pop(prefix, None)


def test_to_lal_explicit_prefix_collision_still_raises() -> None:
    """An explicitly chosen prefix that clashes remains a hard error."""
    detector = _make_h1_custom(name="explicit", prefix="H1")  # H1 is a built-in LAL prefix
    with pytest.raises(ValueError, match="already registered in LAL"):
        detector.to_lal()
