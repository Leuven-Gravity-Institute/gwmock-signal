"""A CustomDetector projects from its own fields, in double precision unless asked for LAL's single.

Projection no longer registers a CustomDetector with LAL: it builds the geometry from the detector's
fields. ``precision="double"`` (the default) uses them as given. ``precision="single"`` rounds the
elevation and the arm angles to single precision first, as LAL stores them, so it reproduces what a
LAL-registered copy of the detector gives.
"""

from __future__ import annotations

import pytest

from gwmock_signal._detector_table import builtin_detectors, lal_storage_precision
from gwmock_signal.detector import CustomDetector
from gwmock_signal.projection.geometry import _defining_fields, resolve_detectors

#: Arm angles and elevation chosen *not* exact in single precision, so rounding them changes them.
FIELDS = {
    "latitude_rad": 0.7615,
    "longitude_rad": 0.1833,
    "elevation_m": 51.884,
    "xarm_azimuth_rad": 0.3387,
    "yarm_azimuth_rad": 1.3861,
    "xarm_tilt_rad": 1.1e-4,
    "yarm_tilt_rad": -2.3e-4,
}


def _detector(name: str = "CUSTOM", **overrides: object) -> CustomDetector:
    return CustomDetector(name=name, **{**FIELDS, **overrides})


def test_double_precision_uses_the_fields_as_given():
    """The default keeps every field exactly."""
    detector = _detector()

    assert detector.precision == "double"
    assert detector.geometry_fields() == (0.1833, 0.7615, 51.884, 0.3387, 1.3861, 1.1e-4, -2.3e-4)


def test_single_precision_rounds_what_lal_stores_in_single_precision():
    """Elevation and arm angles are rounded; the vertex longitude and latitude stay double."""
    double = _detector().geometry_fields()
    single = _detector(precision="single").geometry_fields()

    assert single == lal_storage_precision(double)
    assert single[:2] == double[:2]
    assert all(rounded != exact for rounded, exact in zip(single[2:], double[2:], strict=True))


def test_single_precision_equals_the_fields_lal_registers():
    """The single-precision fields are exactly what LAL holds once the detector is registered."""
    pytest.importorskip("lal")
    detector = _detector(precision="single")

    assert detector.geometry_fields() == _defining_fields(detector.to_lal().frDetector)


def test_an_unknown_precision_is_rejected():
    """Only the two precisions are defined."""
    with pytest.raises(ValueError, match="precision must be 'double' or 'single'"):
        _detector(precision="half")


def test_single_precision_rejects_an_angle_beyond_its_range():
    """A finite arm angle too large for single precision fails at construction, as a ValueError."""
    with pytest.raises(ValueError, match="within the single-precision range"):
        _detector(precision="single", xarm_azimuth_rad=1e40)

    assert _detector(xarm_azimuth_rad=1e40).precision == "double"


@pytest.mark.parametrize("value", [3.4028234663852886e38, -3.4028234663852886e38])
def test_the_largest_single_precision_value_is_kept(value: float):
    """The overflow check rejects only what single precision cannot hold, not its own maximum."""
    assert lal_storage_precision((0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0))[3] == value


def test_the_lookup_key_is_the_geometry():
    """A CustomDetector resolves to its own fields, without registering anything with LAL."""
    detector = _detector()

    assert resolve_detectors([detector]) == [("CUSTOM", detector.geometry_fields())]
    assert detector._lal_prefix == ""


class TestDuplicates:
    """The same detector twice is rejected; distinct detectors may share a geometry, as before."""

    def test_the_same_object_twice_is_rejected(self):
        """One detector passed twice would appear under one name twice."""
        detector = _detector()

        with pytest.raises(ValueError, match="resolve to the same detector"):
            resolve_detectors([detector, detector])

    def test_a_detector_with_its_own_registered_prefix_is_rejected(self):
        """A CustomDetector and the prefix it registered with LAL are the same detector."""
        pytest.importorskip("lal")
        detector = _detector(name="Q4_NAMED")
        prefix = detector.to_lal().frDetector.prefix

        with pytest.raises(ValueError, match="resolve to the same detector"):
            resolve_detectors([detector, prefix])

    def test_two_distinct_detectors_may_share_a_geometry(self):
        """Two CustomDetectors with identical fields but different names stay allowed."""
        names = [name for name, _ in resolve_detectors([_detector("LEFT"), _detector("RIGHT")])]

        assert names == ["LEFT", "RIGHT"]

    def test_a_custom_detector_on_a_built_in_site_is_distinct_from_the_code(self):
        """A CustomDetector placed on H1 stays distinct from the code ``H1``, as before."""
        longitude, latitude, elevation, x_azimuth, y_azimuth, x_altitude, y_altitude = builtin_detectors()["H1"]
        on_h1 = CustomDetector(
            name="ON_H1",
            latitude_rad=latitude,
            longitude_rad=longitude,
            elevation_m=elevation,
            xarm_azimuth_rad=x_azimuth,
            yarm_azimuth_rad=y_azimuth,
            xarm_tilt_rad=x_altitude,
            yarm_tilt_rad=y_altitude,
        )

        assert [name for name, _ in resolve_detectors([on_h1, "H1"])] == ["ON_H1", "H1"]
