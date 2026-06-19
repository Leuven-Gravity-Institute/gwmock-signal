"""Tests for batched on-device CBC simulation (simulate_cbc_batch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ripplegw", reason="ripplegw not installed")
import jax

jax.config.update("jax_enable_x64", True)

from gwmock_signal.jax_batch import BatchedDetectorStrain, simulate_cbc_batch  # noqa: E402
from gwmock_signal.waveform.backends import RippleBackend  # noqa: E402

_FS = 2048.0
_F_MIN = 20.0
_DETECTORS = ["H1", "L1"]

# Three-event catalogue (struct-of-arrays of canonical parameter names).
_CATALOGUE = {
    "detector_frame_mass_1": np.array([40.0, 36.0, 55.0]),
    "detector_frame_mass_2": np.array([31.0, 29.0, 48.0]),
    "luminosity_distance": np.array([400.0, 410.0, 800.0]),
    "spin_1z": np.array([0.5, 0.0, 0.2]),
    "spin_2z": np.array([-0.2, 0.0, 0.1]),
    "inclination": np.array([0.9, 0.4, 1.2]),
    "coa_phase": np.array([0.3, 0.0, 1.0]),
    "right_ascension": np.array([1.375, 0.5, 5.0]),
    "declination": np.array([-1.211, 0.3, 1.0]),
    "polarization_angle": np.array([2.659, 0.0, 1.5]),
    "coa_time": np.array([1.1262594624e9, 1.1262594640e9, 1.1262594610e9]),
}
_N_EVENTS = 3


def test_simulate_cbc_batch_shapes_and_metadata() -> None:
    """The batched simulation returns the expected shapes and timing metadata."""
    backend = RippleBackend(segment_duration=8.0)
    result = simulate_cbc_batch(
        "IMRPhenomD",
        _DETECTORS,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=_CATALOGUE,
        backend=backend,
    )
    assert isinstance(result, BatchedDetectorStrain)
    n_samples = np.asarray(result.strain).shape[-1]
    assert np.asarray(result.strain).shape == (_N_EVENTS, len(_DETECTORS), n_samples)
    assert result.detector_names == tuple(_DETECTORS)
    np.testing.assert_array_equal(result.coa_time, _CATALOGUE["coa_time"])
    _, expected_epoch = backend.coalescence_placement(n_samples, _FS)
    assert result.epoch == pytest.approx(expected_epoch)
    assert np.all(np.isfinite(np.asarray(result.strain)))


def test_simulate_cbc_batch_missing_sky_parameter_raises() -> None:
    """Omitting a required sky parameter raises ValueError."""
    params = {k: v for k, v in _CATALOGUE.items() if k != "coa_time"}
    with pytest.raises(ValueError, match="coa_time"):
        simulate_cbc_batch(
            "IMRPhenomD", _DETECTORS, sampling_frequency=_FS, minimum_frequency=_F_MIN, parameters=params
        )


@pytest.mark.integration
def test_simulate_cbc_batch_matches_host_pipeline() -> None:
    """Each batched event/detector matches the host NumPy pipeline (overlap > 0.999)."""
    from gwmock_signal.projection.network import project_polarizations_to_network

    backend = RippleBackend(segment_duration=8.0)  # fixed grid shared with the host path
    result = simulate_cbc_batch(
        "IMRPhenomD",
        _DETECTORS,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=_CATALOGUE,
        backend=backend,
    )
    device = np.asarray(result.strain)

    intrinsic_keys = (
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "luminosity_distance",
        "spin_1z",
        "spin_2z",
        "inclination",
        "coa_phase",
    )
    for i in range(_N_EVENTS):
        td = backend.generate_td_waveform(
            "IMRPhenomD",
            tc=float(_CATALOGUE["coa_time"][i]),
            sampling_frequency=_FS,
            minimum_frequency=_F_MIN,
            **{key: float(_CATALOGUE[key][i]) for key in intrinsic_keys},
        )
        for j, detector in enumerate(_DETECTORS):
            host = project_polarizations_to_network(
                {"plus": td["plus"], "cross": td["cross"]},
                [detector],
                right_ascension=float(_CATALOGUE["right_ascension"][i]),
                declination=float(_CATALOGUE["declination"][i]),
                polarization_angle=float(_CATALOGUE["polarization_angle"][i]),
                earth_rotation=False,
            )[detector]
            a, b = host.value, device[i, j]
            overlap = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
            assert overlap > 0.999, f"event {i} {detector} overlap {overlap:.5f}"


def test_simulate_cbc_catalogue_tiles_span_and_places_signals() -> None:
    """The catalogue wrapper tiles the span and places each signal in its segment(s)."""
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    catalogue = {
        "detector_frame_mass_1": np.array([40.0, 38.0]),
        "detector_frame_mass_2": np.array([31.0, 33.0]),
        "luminosity_distance": np.array([400.0, 450.0]),
        "spin_1z": np.array([0.3, -0.1]),
        "spin_2z": np.array([-0.2, 0.1]),
        "inclination": np.array([0.9, 0.6]),
        "coa_phase": np.array([0.3, 1.0]),
        "right_ascension": np.array([1.375, 0.5]),
        "declination": np.array([-1.211, 0.3]),
        "polarization_angle": np.array([2.659, 0.0]),
        "coa_time": np.array([100.0, 130.0]),
    }
    segment_duration, start_time, end_time = 16.0, 64.0, 160.0
    segments = simulate_cbc_catalogue(
        "IMRPhenomD",
        _DETECTORS,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=catalogue,
        segment_duration=segment_duration,
        start_time=start_time,
        end_time=end_time,
        backend=RippleBackend(segment_duration=8.0),  # 8 s generation buffer
    )

    assert len(segments) == 6  # ceil((160 - 64) / 16)
    for k, stack in enumerate(segments):
        assert stack["H1"].t0.value == pytest.approx(start_time + k * segment_duration)

    # Segments before/after every signal are zero; segments overlapping a coalescence are not.
    assert np.all(np.asarray(segments[0]["H1"].value) == 0.0)  # [64, 80): no signal
    assert np.all(np.asarray(segments[5]["H1"].value) == 0.0)  # [144, 160): no signal
    assert np.any(np.asarray(segments[2]["H1"].value) != 0.0)  # [96, 112): event 0 coalescence (100)
    assert np.any(np.asarray(segments[3]["H1"].value) != 0.0)  # [112, 128): event 1 inspiral toward 130
