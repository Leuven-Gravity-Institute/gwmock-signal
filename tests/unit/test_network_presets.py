"""Tests for bundled detector geometry presets."""

from __future__ import annotations

from importlib.resources import files

import lal
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.projection.network import project_polarizations_to_network

_PRESET_NAMES = {
    "ET-Triangle-Sardinia": ("ET1_SARD", "ET2_SARD", "ET3_SARD"),
    "ET-Triangle-EMR": ("ET1_EMR", "ET2_EMR", "ET3_EMR"),
    "ET-2L-Aligned": ("ET1_2L_ALIGNED_SARD", "ET2_2L_ALIGNED_EMR"),
    "ET-2L-Misaligned": ("ET1_2L_MISALIGNED_SARD", "ET2_2L_MISALIGNED_EMR"),
}

_REFERENCE_TRIANGLE_SARDINIA = (
    {
        "name": "ET1_SARD",
        "latitude_deg": 40.5166666668747,
        "longitude_deg": 9.416666666888249,
        "elevation_m": 51.884,
        "xarm_azimuth_deg": 19.432600234737947,
        "yarm_azimuth_deg": 319.4326002345667,
        "xarm_tilt_rad": 2.64278457158e-10,
        "yarm_tilt_rad": 2.64278469515e-10,
    },
    {
        "name": "ET2_SARD",
        "latitude_deg": 40.60158246647564,
        "longitude_deg": 9.455973799867088,
        "elevation_m": 59.739,
        "xarm_azimuth_deg": 259.4581895871824,
        "yarm_azimuth_deg": 199.45815909933023,
        "xarm_tilt_rad": -0.000783862464265,
        "yarm_tilt_rad": -0.0015710370848,
    },
    {
        "name": "ET3_SARD",
        "latitude_deg": 40.585048823916765,
        "longitude_deg": 9.339849816516432,
        "elevation_m": 59.73,
        "xarm_azimuth_deg": 139.38265970390688,
        "yarm_azimuth_deg": 79.38262936387784,
        "xarm_tilt_rad": -0.00156913521227,
        "yarm_tilt_rad": -0.000781960584882,
    },
)


def _custom_detector(entry: dict[str, float | str]) -> CustomDetector:
    return CustomDetector(
        name=str(entry["name"]),
        latitude_rad=np.deg2rad(float(entry["latitude_deg"])),
        longitude_rad=np.deg2rad(float(entry["longitude_deg"])),
        elevation_m=float(entry["elevation_m"]),
        xarm_azimuth_rad=np.deg2rad(float(entry["xarm_azimuth_deg"])),
        yarm_azimuth_rad=np.deg2rad(float(entry["yarm_azimuth_deg"])),
        xarm_tilt_rad=float(entry["xarm_tilt_rad"]),
        yarm_tilt_rad=float(entry["yarm_tilt_rad"]),
    )


@pytest.mark.parametrize(("preset", "expected_names"), sorted(_PRESET_NAMES.items()))
def test_from_preset_returns_custom_detectors(preset: str, expected_names: tuple[str, ...]) -> None:
    """Bundled ET presets resolve to ordered CustomDetector objects."""
    net = Network.from_preset(preset)
    assert net.name == preset
    assert tuple(det.name for det in net.detector_names) == expected_names
    assert all(isinstance(det, CustomDetector) for det in net.detector_names)


def test_from_name_supports_file_backed_preset() -> None:
    """from_name resolves the bundled triangle Sardinia preset."""
    net = Network.from_name("ET-Triangle-Sardinia")
    assert net.name == "ET-Triangle-Sardinia"
    assert tuple(det.name for det in net.detector_names) == ("ET1_SARD", "ET2_SARD", "ET3_SARD")


def test_from_name_supports_compatibility_alias() -> None:
    """Roadmap compatibility aliases resolve to the bundled preset geometry."""
    canonical = Network.from_preset("ET-Triangle-Sardinia")
    alias = Network.from_name("ET-Sardinia")
    assert alias.name == canonical.name
    assert alias.detector_names == canonical.detector_names


def test_list_names_includes_bundled_presets() -> None:
    """Bundled preset names and compatibility aliases appear in list_names()."""
    names = Network.list_names()
    assert {
        "ET-Triangle-Sardinia",
        "ET-Triangle-EMR",
        "ET-2L-Aligned",
        "ET-2L-Misaligned",
        "ET-Sardinia",
        "ET-EMR",
    } <= set(names)


def test_from_preset_unknown_name_raises_value_error() -> None:
    """Unknown bundled preset names raise a helpful ValueError."""
    with pytest.raises(ValueError, match="Known presets"):
        Network.from_preset("not-a-preset")


def test_preset_yaml_is_packaged_as_resource() -> None:
    """Preset YAML files are accessible through importlib.resources."""
    resource = files("gwmock_signal.data.detectors").joinpath("et-triangle-sardinia.yaml")
    assert resource.is_file()


def test_triangle_sardinia_projection_matches_reference_geometry() -> None:
    """The packaged Sardinia triangle preset matches the reference geometry."""
    n = 256
    fs = 2048.0
    t0 = 1126259462.4 - 0.0625
    times = np.arange(n) / fs
    taper = np.hanning(n)
    hp = TimeSeries(np.sin(2 * np.pi * 32.0 * times) * taper, t0=t0, sample_rate=fs)
    hc = TimeSeries(np.cos(2 * np.pi * 32.0 * times) * taper, t0=t0, sample_rate=fs)

    preset = Network.from_preset("ET-Triangle-Sardinia")
    reference = tuple(_custom_detector(entry) for entry in _REFERENCE_TRIANGLE_SARDINIA)
    kwargs = {
        "right_ascension": 1.375,
        "declination": -1.211,
        "polarization_angle": 0.0,
        "earth_rotation": False,
    }

    projected = project_polarizations_to_network({"plus": hp, "cross": hc}, preset.detector_names, **kwargs)
    expected = project_polarizations_to_network({"plus": hp, "cross": hc}, reference, **kwargs)

    assert tuple(projected) == ("ET1_SARD", "ET2_SARD", "ET3_SARD")
    for name in projected:
        np.testing.assert_allclose(projected[name].value, expected[name].value, rtol=0.0, atol=1e-12)


def _arm_direction(latitude: float, longitude: float, azimuth: float, tilt: float) -> np.ndarray:
    """Earth-fixed unit vector of an arm with LAL's azimuth convention (clockwise from North)."""
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array([-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)])
    up = np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
    return np.cos(tilt) * (np.sin(azimuth) * east + np.cos(azimuth) * north) + np.sin(tilt) * up


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


@pytest.mark.parametrize("preset", ["ET-Triangle-Sardinia", "ET-Triangle-EMR"])
def test_triangle_preset_arms_lie_along_tunnel_chords(preset: str) -> None:
    """Every triangle arm points along the tunnel to a neighbouring vertex.

    The x-arm of vertex ``i`` runs to vertex ``i + 1`` and the y-arm to vertex ``i - 1``. The
    chord comes from the vertex locations LAL computes, the arm from the stored angles read in
    LAL's convention, and the response tensor LAL builds from those angles is checked against the
    chords as well, so the test fails whichever convention the stored azimuths are written in
    unless it is the one LAL applies.
    """
    detectors = Network.from_preset(preset).detector_names
    lal_detectors = [det.to_lal() for det in detectors]
    locations = [np.asarray(det.location, dtype=float) for det in lal_detectors]

    for i, (det, lal_det) in enumerate(zip(detectors, lal_detectors, strict=True)):
        chords = []
        for neighbour, azimuth, tilt in (
            ((i + 1) % 3, det.xarm_azimuth_rad, det.xarm_tilt_rad),
            ((i - 1) % 3, det.yarm_azimuth_rad, det.yarm_tilt_rad),
        ):
            chord = locations[neighbour] - locations[i]
            chord /= np.linalg.norm(chord)
            chords.append(chord)
            arm = _arm_direction(det.latitude_rad, det.longitude_rad, azimuth, tilt)
            assert _angle(arm, chord) < 1e-6, f"{det.name}: arm misses the chord to vertex {neighbour}"
        x_chord, y_chord = chords
        expected_response = (np.outer(x_chord, x_chord) - np.outer(y_chord, y_chord)) / 2.0
        np.testing.assert_allclose(np.asarray(lal_det.response), expected_response, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(("preset", "relative_angle"), [("ET-2L-Aligned", 0.0), ("ET-2L-Misaligned", np.pi / 4)])
def test_two_l_preset_arms_follow_virgo_reference(preset: str, relative_angle: float) -> None:
    """Each 2L x-arm has the azimuth of Virgo's arm 1, the second site turned by the relative angle.

    Both detectors are oriented against their local North: the Sardinia x-arm has the azimuth of
    LAL's V1 x-arm, the Meuse-Rhine x-arm that azimuth plus the relative angle (0 aligned, 45 degrees
    misaligned), and each y-arm is ``x-arm x up``, i.e. 90 degrees clockwise of the x-arm. The
    response tensor LAL builds from the stored angles is checked against those arms too.
    """
    virgo_azimuth = lal.cached_detector_by_prefix["V1"].frDetector.xArmAzimuthRadians
    detectors = Network.from_preset(preset).detector_names

    for det, azimuth in zip(detectors, (virgo_azimuth, virgo_azimuth + relative_angle), strict=True):
        up = _arm_direction(det.latitude_rad, det.longitude_rad, 0.0, np.pi / 2)
        x_expected = _arm_direction(det.latitude_rad, det.longitude_rad, azimuth, 0.0)
        y_expected = np.cross(x_expected, up)
        for arm_azimuth, arm_tilt, expected in (
            (det.xarm_azimuth_rad, det.xarm_tilt_rad, x_expected),
            (det.yarm_azimuth_rad, det.yarm_tilt_rad, y_expected),
        ):
            arm = _arm_direction(det.latitude_rad, det.longitude_rad, arm_azimuth, arm_tilt)
            assert _angle(arm, expected) < 1e-6, f"{det.name}: arm misses its reference direction"
        expected_response = (np.outer(x_expected, x_expected) - np.outer(y_expected, y_expected)) / 2.0
        np.testing.assert_allclose(np.asarray(det.to_lal().response), expected_response, rtol=0.0, atol=1e-6)
