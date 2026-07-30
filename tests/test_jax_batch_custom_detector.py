"""The device path must accept a CustomDetector, not only built-in LAL codes.

`jax_batch` used one string as both the LAL registry key and the output channel name. That
silently restricts it to built-in interferometer codes: a
:class:`~gwmock_signal.detector.CustomDetector` registers itself under a generated
two-character prefix, so looking it up by its ``name`` fails. gwmock's ET presets are custom
detectors, so before this the device path could not simulate the configuration it exists for.

The load-bearing test here is agreement with the NumPy path, which already supported custom
detectors: it checks the device path resolves the *same geometry*, not merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("ripplegw", reason="ripple not installed")

from gwmock_signal.detector import CustomDetector  # noqa: E402
from gwmock_signal.jax_batch import simulate_cbc_batch  # noqa: E402
from gwmock_signal.projection.geometry import reconstructed_geometry, resolve_detectors  # noqa: E402

_FS = 2048.0
_F_MIN = 30.0
_T0 = 1.4e9

_PARAMETERS = {
    "detector_frame_mass_1": np.array([30.0]),
    "detector_frame_mass_2": np.array([28.0]),
    "luminosity_distance": np.array([400.0]),
    "inclination": np.array([0.3]),
    "coa_phase": np.array([0.0]),
    "right_ascension": np.array([1.3]),
    "declination": np.array([-0.4]),
    "polarization_angle": np.array([0.7]),
    "coa_time": np.array([_T0 + 20.0]),
}


def _et_like(name: str = "ET1", longitude: float = 0.1833) -> CustomDetector:
    """A Virgo-site triangular-arm detector, standing in for a gwmock ET preset."""
    return CustomDetector(
        name=name,
        latitude_rad=0.7615,
        longitude_rad=longitude,
        elevation_m=51.884,
        xarm_azimuth_rad=0.3387,
        yarm_azimuth_rad=1.3861,
    )


def _batch(detectors: list, earth_rotation: bool = True) -> object:
    return simulate_cbc_batch(
        "IMRPhenomD",
        detectors,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=dict(_PARAMETERS),
        earth_rotation=earth_rotation,
    )


@pytest.mark.parametrize("earth_rotation", [True, False])
def test_custom_detector_is_accepted_by_both_branches(earth_rotation: bool) -> None:
    """Both projection branches must resolve a custom detector and key it by its own name."""
    detector = _et_like()
    batch = _batch([detector], earth_rotation=earth_rotation)
    assert batch.detector_names == ("ET1",)
    strain = np.asarray(batch.strain)
    assert strain.shape[:2] == (1, 1)
    assert np.all(np.isfinite(strain))
    assert np.any(strain != 0.0)


@pytest.mark.parametrize("earth_rotation", [True, False])
def test_custom_detector_matches_the_same_geometry_given_as_a_prefix(earth_rotation: bool) -> None:
    """The strain must be identical whether the detector is named or given by its LAL prefix.

    This is what pins that the *geometry* is resolved correctly rather than merely that some
    detector was found: passing the registered prefix goes through the built-in code path, so
    agreement means both routes reached the same response tensor and location.
    """
    detector = _et_like()
    prefix = detector.to_lal().frDetector.prefix

    by_object = np.asarray(_batch([detector], earth_rotation=earth_rotation).strain)
    by_prefix = np.asarray(_batch([prefix], earth_rotation=earth_rotation).strain)
    assert np.array_equal(by_object, by_prefix)


def test_custom_and_builtin_detectors_can_be_mixed() -> None:
    """A network of built-in and custom detectors must keep channel order and naming."""
    detector = _et_like()
    batch = _batch(["H1", detector, "L1"])
    assert batch.detector_names == ("H1", "ET1", "L1")
    strain = np.asarray(batch.strain)
    assert strain.shape[:2] == (1, 3)
    # Distinct geometries must give distinct strain; identical rows would mean one lookup key
    # was used for several channels. atol=0 is required: strain is ~1e-24, so allclose's default
    # atol=1e-8 calls any two strain arrays equal and the assertion could never fail.
    assert not np.allclose(strain[0, 0], strain[0, 1], rtol=1e-6, atol=0.0)
    assert not np.allclose(strain[0, 1], strain[0, 2], rtol=1e-6, atol=0.0)


def test_two_custom_detectors_get_distinct_geometry() -> None:
    """Separately constructed custom detectors must not collide in the geometry cache.

    ``reconstructed_geometry`` is cached by prefix, so two custom detectors sharing a prefix
    would silently return one geometry for both. They are auto-assigned distinct prefixes; this
    pins that, since the failure is silent and looks like a physically plausible network.
    """
    first = _et_like("ET1", longitude=0.1833)
    second = _et_like("ET2", longitude=-1.2)
    keys = [key for _, key in resolve_detectors([first, second])]
    assert keys[0] != keys[1]
    _, location_first = reconstructed_geometry(keys[0])
    _, location_second = reconstructed_geometry(keys[1])
    assert not np.allclose(location_first, location_second, rtol=1e-9, atol=0.0)

    batch = _batch([first, second])
    assert batch.detector_names == ("ET1", "ET2")
    strain = np.asarray(batch.strain)
    # atol=0 for the same reason as above.
    assert not np.allclose(strain[0, 0], strain[0, 1], rtol=1e-6, atol=0.0)


def test_duplicate_detector_names_are_rejected() -> None:
    """Two channels with one name would make the strain's detector axis unattributable."""
    with pytest.raises(ValueError, match="must be unique"):
        _batch([_et_like("ET1", longitude=0.1833), _et_like("ET1", longitude=-1.2)])


def test_an_empty_detector_list_is_rejected() -> None:
    """Zero detectors would yield an empty detector axis rather than an error."""
    with pytest.raises(ValueError, match="At least one detector"):
        _batch([])


def test_an_unsupported_specification_type_is_rejected() -> None:
    """Anything other than a code or a CustomDetector is a caller error, named as such."""
    with pytest.raises(TypeError, match="Unsupported detector specification type"):
        _batch([object()])


def test_an_unknown_detector_code_fails_before_generation() -> None:
    """An unusable detector must be reported without first generating a catalogue."""
    with pytest.raises(ValueError, match="Unknown or unsupported detector"):
        _batch(["ZZ"])


def test_bundled_et_preset_reaches_the_device_path() -> None:
    """The motivating case: gwmock's own ET preset must simulate on device.

    ``ET-Sardinia`` resolves to three ``CustomDetector`` instances, so before detector
    specifications were resolved separately from output names this raised
    ``Unknown or unsupported detector CustomDetector(...)`` -- the target configuration for the
    whole device path was the one configuration it could not run.
    """
    from gwmock_signal.network import Network

    network = Network.from_preset("ET-Sardinia")
    batch = _batch(list(network.detector_names))
    assert batch.detector_names == ("ET1_SARD", "ET2_SARD", "ET3_SARD")
    strain = np.asarray(batch.strain)
    assert strain.shape[:2] == (1, 3)
    assert np.all(np.isfinite(strain))
    for detector in range(3):
        assert np.any(strain[0, detector] != 0.0)


def test_et_preset_geometry_matches_the_published_design() -> None:
    """Anchor the resolved geometry against ET's design, not against our own other code path.

    Two properties that do not depend on anything in this repository:

    * The three vertices sit 10 km apart -- ET's design arm length.
    * A closed triangle has a null stream: the three response tensors sum to zero. The residual
      here is not zero but ~5e-3 of a single tensor, which is the size Earth's curvature over a
      10 km triangle predicts (10 km / 6371 km ~ 1.6e-3), so it is checked as a bound rather
      than asserted to vanish.
    """
    import itertools

    from gwmock_signal.network import Network

    keys = [key for _, key in resolve_detectors(list(Network.from_preset("ET-Sardinia").detector_names))]
    responses = [reconstructed_geometry(key)[0] for key in keys]
    locations = [reconstructed_geometry(key)[1] for key in keys]

    for first, second in itertools.combinations(locations, 2):
        separation_km = float(np.linalg.norm(first - second)) / 1e3
        assert separation_km == pytest.approx(10.0, abs=0.05)

    scale = max(float(np.max(np.abs(response))) for response in responses)
    residual = float(np.max(np.abs(sum(responses)))) / scale
    assert residual < 2e-2, f"null-stream residual {residual:.2e} is too large for a closed triangle"
    assert residual > 1e-4, (
        f"null-stream residual {residual:.2e} is suspiciously small; the vertices are 10 km apart "
        f"on a curved Earth, so an exactly vanishing sum would suggest co-located geometry"
    )
