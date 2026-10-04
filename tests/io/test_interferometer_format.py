"""Tests for the Bilby ``.interferometer`` compatibility shim."""

from __future__ import annotations

import math
import warnings
from ast import literal_eval
from pathlib import Path

import lal
import pytest

from gwmock_signal.detector import CustomDetector
from gwmock_signal.io.interferometer_format import (
    interferometer_config_to_custom_detector,
    read_interferometer_config,
)
from gwmock_signal.network import Network

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "interferometers"
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.interferometer"))


def _read_fixture_assignments(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, value_text = raw_line.split("=", maxsplit=1)
        key = key.strip()
        if key == "power_spectral_density":
            continue
        data[key] = literal_eval(value_text.strip())
    return data


@pytest.mark.parametrize("fixture_path", _FIXTURES, ids=lambda path: path.name)
def test_interferometer_config_to_custom_detector_matches_golden_fixture(fixture_path: Path) -> None:
    """Golden fixtures match the converted CustomDetector geometry exactly."""
    expected = _read_fixture_assignments(fixture_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        detector = interferometer_config_to_custom_detector(fixture_path)

    assert isinstance(detector, CustomDetector)
    assert detector.name == expected["name"]
    assert detector.latitude_rad == pytest.approx(math.radians(float(expected["latitude"])))
    assert detector.longitude_rad == pytest.approx(math.radians(float(expected["longitude"])))
    assert detector.elevation_m == pytest.approx(float(expected["elevation"]))
    # Bilby azimuths are North of East; CustomDetector follows LAL (clockwise from North).
    assert detector.xarm_azimuth_rad == pytest.approx(math.radians((90.0 - float(expected["xarm_azimuth"])) % 360.0))
    assert detector.yarm_azimuth_rad == pytest.approx(math.radians((90.0 - float(expected["yarm_azimuth"])) % 360.0))
    assert detector.xarm_tilt_rad == pytest.approx(float(expected["xarm_tilt"]))
    assert detector.yarm_tilt_rad == pytest.approx(float(expected["yarm_tilt"]))


def test_interferometer_x_arm_matches_virgo_reference() -> None:
    """The 2L Sardinia x-arm has the azimuth of LAL's V1 x-arm, as its fixture states."""
    fixture_path = _FIXTURE_DIR / "E1_2L_Aligned_Sardinia.interferometer"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        detector = interferometer_config_to_custom_detector(fixture_path)

    virgo_azimuth = lal.cached_detector_by_prefix["V1"].frDetector.xArmAzimuthRadians
    assert detector.xarm_azimuth_rad == pytest.approx(virgo_azimuth, abs=1e-9)
    assert detector.yarm_azimuth_rad == pytest.approx(virgo_azimuth + math.pi / 2, abs=1e-9)


def test_read_interferometer_config_ignores_psd_line() -> None:
    """The raw reader keeps scalar parameters and skips the PSD constructor line."""
    fixture_path = next(p for p in _FIXTURES if _read_fixture_assignments(p).get("name") == "E1_2L_aligned_sardinia")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = read_interferometer_config(fixture_path)

    assert "power_spectral_density" not in config
    assert config["name"] == "E1_2L_aligned_sardinia"
    assert config["length"] == 15


def test_network_from_file_loads_one_custom_detector(tmp_path: Path) -> None:
    """Network.from_file accepts ``.interferometer`` input as a single-detector shim."""
    fixture_path = _FIXTURE_DIR / "E1_Triangle_Sardinia.interferometer"
    config_path = tmp_path / fixture_path.name
    config_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="YAML detector preset/network format"):
        network = Network.from_file(config_path)

    assert network.name == "E1_triangle_sardinia"
    assert len(network.detector_names) == 1
    assert isinstance(network.detector_names[0], CustomDetector)
    assert network.detector_names[0].name == "E1_triangle_sardinia"


def test_network_from_file_warns_once_per_file(tmp_path: Path) -> None:
    """Repeated reads of the same file only emit one deprecation warning."""
    fixture_path = _FIXTURE_DIR / "E1_Triangle_EMR.interferometer"
    config_path = tmp_path / fixture_path.name
    config_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        Network.from_file(config_path)
        Network.from_file(config_path)

    deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
    assert len(deprecations) == 1
