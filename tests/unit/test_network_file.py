"""Unit tests for Network.from_file and the YAML/JSON network config format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET_TRIANGLE_YAML = Path(__file__).parents[2] / "examples" / "networks" / "et_triangle.yaml"

_MIXED_CONFIG = {
    "name": "mixed-net",
    "detectors": [
        {"name": "H1"},
        {
            "name": "MY-DET",
            "latitude_deg": 46.455,
            "longitude_deg": -119.408,
            "elevation_m": 142.554,
            "xarm_azimuth_deg": 125.9994,
            "yarm_azimuth_deg": 215.9994,
            "xarm_tilt_rad": 0.0,
            "yarm_tilt_rad": 0.0,
        },
    ],
}


@pytest.fixture
def mixed_yaml(tmp_path: Path) -> Path:
    """Write a mixed (alias + custom) network config to a temporary YAML file."""
    p = tmp_path / "mixed.yaml"
    p.write_text(yaml.dump(_MIXED_CONFIG))
    return p


@pytest.fixture
def mixed_json(tmp_path: Path) -> Path:
    """Write a mixed (alias + custom) network config to a temporary JSON file."""
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(_MIXED_CONFIG))
    return p


# ---------------------------------------------------------------------------
# et_triangle.yaml
# ---------------------------------------------------------------------------


def test_load_et_triangle_returns_three_detectors() -> None:
    """Test that the ET-triangle network has three detectors."""
    net = Network.from_file(_ET_TRIANGLE_YAML)
    assert net.name == "ET-triangle"
    assert len(net.detector_names) == 3
    assert net.detector_names == ("E1", "E2", "E3")


@pytest.mark.requires_gwpy
def test_load_et_triangle_passes_through_project_polarizations() -> None:
    """Network.from_file result can be passed to project_polarizations_to_network."""
    import numpy as np
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network

    net = Network.from_file(_ET_TRIANGLE_YAML)
    n = 512
    hp = TimeSeries(np.zeros(n), t0=0.0, sample_rate=512)
    hc = TimeSeries(np.zeros(n), t0=0.0, sample_rate=512)
    projected = project_polarizations_to_network(
        {"plus": hp, "cross": hc},
        net.detector_names,
        right_ascension=0.0,
        declination=0.0,
        polarization_angle=0.0,
    )
    assert set(projected.keys()) == {"E1", "E2", "E3"}


# ---------------------------------------------------------------------------
# Mixed config (YAML and JSON)
# ---------------------------------------------------------------------------


def test_load_mixed_yaml_alias_and_custom(mixed_yaml: Path) -> None:
    """Test that a mixed (alias + custom) network config can be loaded from a YAML file."""
    net = Network.from_file(mixed_yaml)
    assert net.name == "mixed-net"
    assert len(net.detector_names) == 2
    assert net.detector_names[0] == "H1"
    assert isinstance(net.detector_names[1], CustomDetector)
    assert net.detector_names[1].name == "MY-DET"


def test_load_mixed_json_alias_and_custom(mixed_json: Path) -> None:
    """Test that a mixed (alias + custom) network config can be loaded from a JSON file."""
    net = Network.from_file(mixed_json)
    assert len(net.detector_names) == 2
    assert net.detector_names[0] == "H1"
    assert isinstance(net.detector_names[1], CustomDetector)


# ---------------------------------------------------------------------------
# Missing required geometry field raises ValueError
# ---------------------------------------------------------------------------


def test_missing_latitude_raises_value_error(tmp_path: Path) -> None:
    """Test that a missing latitude field raises a ValueError."""
    config = {
        "name": "bad-net",
        "detectors": [
            {
                "name": "MY-DET",
                # latitude_deg intentionally omitted
                "longitude_deg": -119.408,
                "elevation_m": 142.554,
                "xarm_azimuth_deg": 125.9994,
                "yarm_azimuth_deg": 215.9994,
            }
        ],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="latitude_deg"):
        Network.from_file(p)


# ---------------------------------------------------------------------------
# *_rad variants for required angles
# ---------------------------------------------------------------------------


def test_required_angles_as_radians(tmp_path: Path) -> None:
    """All required angles can be supplied as *_rad instead of *_deg."""
    import math

    lat_rad = math.radians(46.455)
    lon_rad = math.radians(-119.408)
    xaz_rad = math.radians(125.9994)
    yaz_rad = math.radians(215.9994)
    config = {
        "name": "rad-net",
        "detectors": [
            {
                "name": "MY-DET",
                "latitude_rad": lat_rad,
                "longitude_rad": lon_rad,
                "elevation_m": 142.554,
                "xarm_azimuth_rad": xaz_rad,
                "yarm_azimuth_rad": yaz_rad,
            }
        ],
    }
    p = tmp_path / "rad.yaml"
    p.write_text(yaml.dump(config))
    net = Network.from_file(p)
    det = net.detector_names[0]
    assert isinstance(det, CustomDetector)
    assert abs(det.latitude_rad - lat_rad) < 1e-12
    assert abs(det.longitude_rad - lon_rad) < 1e-12
    assert abs(det.xarm_azimuth_rad - xaz_rad) < 1e-12
    assert abs(det.yarm_azimuth_rad - yaz_rad) < 1e-12


# ---------------------------------------------------------------------------
# Conflict: both *_deg and *_rad for the same angle raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conflict_extra",
    [
        {"latitude_deg": 46.455, "latitude_rad": 0.81},
        {"longitude_deg": -119.408, "longitude_rad": -2.08},
        {"xarm_azimuth_deg": 125.9994, "xarm_azimuth_rad": 2.2},
        {"yarm_azimuth_deg": 215.9994, "yarm_azimuth_rad": 3.77},
        {"xarm_tilt_deg": 0.1, "xarm_tilt_rad": 0.001},
        {"yarm_tilt_deg": 0.1, "yarm_tilt_rad": 0.001},
    ],
)
def test_conflicting_deg_and_rad_raises_value_error(tmp_path: Path, conflict_extra: dict) -> None:
    """Providing both *_deg and *_rad for the same angle must raise ValueError."""
    base_entry = {
        "name": "MY-DET",
        "latitude_deg": 46.455,
        "longitude_deg": -119.408,
        "elevation_m": 142.554,
        "xarm_azimuth_deg": 125.9994,
        "yarm_azimuth_deg": 215.9994,
    }
    entry = {**base_entry, **conflict_extra}
    config = {"name": "conflict-net", "detectors": [entry]}
    p = tmp_path / "conflict.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Conflicting angle"):
        Network.from_file(p)


# ---------------------------------------------------------------------------
# Unsupported extension raises ValueError
# ---------------------------------------------------------------------------


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    """Test that an unsupported file extension raises a ValueError."""
    p = tmp_path / "network.toml"
    p.write_text("[network]\nname = 'test'\n")
    with pytest.raises(ValueError, match=r"Unsupported file extension"):
        Network.from_file(p)


# ---------------------------------------------------------------------------
# Out-of-range geometry value raises ValueError
# ---------------------------------------------------------------------------


def test_tilt_deg_keys_accepted_and_converted(tmp_path: Path) -> None:
    """xarm_tilt_deg / yarm_tilt_deg are converted to radians correctly."""
    import math

    tilt_deg = 1.0
    config = {
        "name": "tilt-deg-net",
        "detectors": [
            {
                "name": "MY-DET",
                "latitude_deg": 46.455,
                "longitude_deg": -119.408,
                "elevation_m": 142.554,
                "xarm_azimuth_deg": 125.9994,
                "yarm_azimuth_deg": 215.9994,
                "xarm_tilt_deg": tilt_deg,
                "yarm_tilt_deg": tilt_deg,
            }
        ],
    }
    p = tmp_path / "tilt_deg.yaml"
    p.write_text(yaml.dump(config))
    net = Network.from_file(p)
    det = net.detector_names[0]
    assert isinstance(det, CustomDetector)
    assert abs(det.xarm_tilt_rad - math.radians(tilt_deg)) < 1e-12
    assert abs(det.yarm_tilt_rad - math.radians(tilt_deg)) < 1e-12


def test_tilt_both_deg_and_rad_raises_value_error(tmp_path: Path) -> None:
    """Providing both xarm_tilt_deg and xarm_tilt_rad simultaneously must raise ValueError."""
    config = {
        "name": "tilt-conflict-net",
        "detectors": [
            {
                "name": "MY-DET",
                "latitude_deg": 46.455,
                "longitude_deg": -119.408,
                "elevation_m": 142.554,
                "xarm_azimuth_deg": 125.9994,
                "yarm_azimuth_deg": 215.9994,
                "xarm_tilt_deg": 2.0,
                "xarm_tilt_rad": 99.0,  # conflict — must be rejected
                "yarm_tilt_rad": 0.0,
            }
        ],
    }
    p = tmp_path / "tilt_conflict.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Conflicting angle"):
        Network.from_file(p)


def test_out_of_range_latitude_raises_value_error(tmp_path: Path) -> None:
    """Test that an out-of-range latitude raises a ValueError."""
    config = {
        "name": "bad-range-net",
        "detectors": [
            {
                "name": "MY-DET",
                "latitude_deg": 200.0,  # > 90 deg → latitude_rad > pi/2
                "longitude_deg": -119.408,
                "elevation_m": 142.554,
                "xarm_azimuth_deg": 125.9994,
                "yarm_azimuth_deg": 215.9994,
            }
        ],
    }
    p = tmp_path / "out_of_range.yaml"
    p.write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="latitude_rad"):
        Network.from_file(p)


class TestPrecision:
    """A custom detector's ``precision`` is read from the file, and refused where it means nothing."""

    @staticmethod
    def _load(tmp_path: Path, **extra: object) -> Network:
        entry = {**_MIXED_CONFIG["detectors"][1], **extra}
        p = tmp_path / "precision.yaml"
        p.write_text(yaml.dump({"name": "precision-net", "detectors": [entry]}))
        return Network.from_file(p)

    def test_the_default_is_double(self, tmp_path: Path) -> None:
        """Without the key, the detector uses the fields as given."""
        (detector,) = self._load(tmp_path).detector_names
        assert detector.precision == "double"

    def test_single_is_passed_through(self, tmp_path: Path) -> None:
        """``precision: single`` reaches the CustomDetector instead of being dropped."""
        (detector,) = self._load(tmp_path, precision="single").detector_names
        assert detector.precision == "single"

    def test_an_unknown_precision_is_rejected(self, tmp_path: Path) -> None:
        """A misspelt value fails with the detector named, not silently as double."""
        with pytest.raises(ValueError, match=r"'MY-DET'.*precision must be 'double' or 'single'"):
            self._load(tmp_path, precision="float32")

    def test_precision_on_a_detector_code_is_rejected(self, tmp_path: Path) -> None:
        """A detector code has no fields to round, so the key there is a mistake."""
        p = tmp_path / "code.yaml"
        p.write_text(yaml.dump({"name": "code-net", "detectors": [{"name": "H1", "precision": "single"}]}))
        with pytest.raises(ValueError, match=r"'H1'.*applies only to a custom detector"):
            Network.from_file(p)
