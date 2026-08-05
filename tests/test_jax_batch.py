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


_WIDE_MASS_CATALOGUE = {
    "detector_frame_mass_1": np.array([45.0, 30.0, 8.0, 1.6]),
    "detector_frame_mass_2": np.array([40.0, 25.0, 6.0, 1.4]),
    "luminosity_distance": np.array([400.0, 500.0, 300.0, 100.0]),
    "spin_1z": np.array([0.3, -0.1, 0.2, 0.0]),
    "spin_2z": np.array([-0.2, 0.1, 0.0, 0.0]),
    "inclination": np.array([0.9, 0.6, 1.0, 0.7]),
    "coa_phase": np.array([0.3, 1.0, 0.5, 0.0]),
    "right_ascension": np.array([1.375, 0.5, 2.0, 3.0]),
    "declination": np.array([-1.211, 0.3, 0.5, -0.2]),
    "polarization_angle": np.array([2.659, 0.0, 1.0, 0.5]),
    "coa_time": np.array([200.0, 260.0, 320.0, 380.0]),
}


@pytest.mark.integration
def test_simulate_cbc_catalogue_binning_agrees_with_single_grid() -> None:
    """Chirp-mass binning agrees with a single grid up to per-event discretization."""
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    common = {
        "sampling_frequency": 2048.0,
        "minimum_frequency": 30.0,
        "parameters": _WIDE_MASS_CATALOGUE,
        "segment_duration": 64.0,
        "start_time": 0.0,
        "end_time": 512.0,
    }
    unbinned = simulate_cbc_catalogue("IMRPhenomD", _DETECTORS, n_chirp_mass_bins=1, **common)
    binned = simulate_cbc_catalogue("IMRPhenomD", _DETECTORS, n_chirp_mass_bins=3, **common)

    assert len(binned) == len(unbinned)
    for single, split in zip(unbinned, binned, strict=True):
        assert single["H1"].t0.value == split["H1"].t0.value
    for detector in _DETECTORS:
        a = np.concatenate([s[detector].value for s in unbinned])
        b = np.concatenate([s[detector].value for s in binned])
        overlap = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        assert overlap > 0.99, f"{detector} binned/unbinned overlap {overlap:.4f}"


def test_simulate_cbc_catalogue_more_bins_than_events() -> None:
    """Asking for more bins than events drops the empty bins and still runs."""
    catalogue = {key: value[:2] for key, value in _WIDE_MASS_CATALOGUE.items()}
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    segments = simulate_cbc_catalogue(
        "IMRPhenomD",
        _DETECTORS,
        sampling_frequency=2048.0,
        minimum_frequency=30.0,
        parameters=catalogue,
        segment_duration=64.0,
        start_time=0.0,
        end_time=512.0,
        n_chirp_mass_bins=5,  # > 2 events
        backend=RippleBackend(segment_duration=8.0),
    )
    assert len(segments) == 8


@pytest.mark.integration
def test_simulate_cbc_catalogue_chunking_is_output_identical() -> None:
    """Count-chunking yields exactly the same segments as one batch (same grid)."""
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    common = {
        "sampling_frequency": 2048.0,
        "minimum_frequency": 30.0,
        "parameters": _WIDE_MASS_CATALOGUE,
        "segment_duration": 64.0,
        "start_time": 0.0,
        "end_time": 512.0,
    }
    whole = simulate_cbc_catalogue("IMRPhenomD", _DETECTORS, chunk_size=None, **common)
    chunked = simulate_cbc_catalogue("IMRPhenomD", _DETECTORS, chunk_size=2, **common)

    assert len(chunked) == len(whole)
    for single, split in zip(whole, chunked, strict=True):
        for detector in _DETECTORS:
            a, b = single[detector].value, split[detector].value
            peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
            if peak > 0.0:
                assert np.max(np.abs(a - b)) < 1e-9 * peak


#: Catalogue for the two batched-versus-NumPy equivalence tests below.
#:
#: Module-level rather than copied into each: the two tests must compare against the *same*
#: waveforms, and two literal copies would let the catalogues drift apart while both tests stayed
#: green against different signals.
_EQUIVALENCE_DETECTORS = ["E1", "E2"]
_EQUIVALENCE_FS, _EQUIVALENCE_F_MIN = 4096.0, 25.0
_EQUIVALENCE_CATALOGUE = {
    "detector_frame_mass_1": np.array([30.0, 25.0]),
    "detector_frame_mass_2": np.array([28.0, 22.0]),
    "luminosity_distance": np.array([900.0, 1200.0]),
    "inclination": np.array([0.3, 1.1]),
    "coa_phase": np.array([0.0, 2.0]),
    "right_ascension": np.array([1.3, 4.0]),
    "declination": np.array([-0.4, 0.6]),
    "polarization_angle": np.array([0.7, 2.1]),
    "coa_time": np.array([1.4e9, 1.4e9 + 300.0]),
}


def _numpy_reference_strain(*, earth_rotation: bool):
    """Yield ``(event, detector_index, name, expected)`` from the per-event NumPy projection.

    Rebuilds the same time-domain polarizations the device path projects -- one shared frequency
    grid, inverse-FFT, roll to place coalescence -- then runs the host projection on them. Shared by
    both branches so neither can be compared against a different waveform than the other.
    """
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.waveform.backends.ripple import RippleBackend

    backend = RippleBackend()
    fd = backend.generate_fd_polarizations_batch(
        "IMRPhenomD",
        sampling_frequency=_EQUIVALENCE_FS,
        minimum_frequency=_EQUIVALENCE_F_MIN,
        parameters=_EQUIVALENCE_CATALOGUE,
    )
    n_samples = fd.n_samples
    merger_index, epoch = backend.coalescence_placement(n_samples, _EQUIVALENCE_FS)

    for event in range(len(_EQUIVALENCE_CATALOGUE["coa_time"])):
        start = epoch + _EQUIVALENCE_CATALOGUE["coa_time"][event]
        hp = np.roll(np.fft.irfft(np.asarray(fd.plus[event]), n=n_samples) * _EQUIVALENCE_FS, merger_index)
        hc = np.roll(np.fft.irfft(np.asarray(fd.cross[event]), n=n_samples) * _EQUIVALENCE_FS, merger_index)
        reference = project_polarizations_to_network(
            {
                "plus": GWpyTimeSeries(hp, t0=start, sample_rate=_EQUIVALENCE_FS),
                "cross": GWpyTimeSeries(hc, t0=start, sample_rate=_EQUIVALENCE_FS),
            },
            _EQUIVALENCE_DETECTORS,
            right_ascension=float(_EQUIVALENCE_CATALOGUE["right_ascension"][event]),
            declination=float(_EQUIVALENCE_CATALOGUE["declination"][event]),
            polarization_angle=float(_EQUIVALENCE_CATALOGUE["polarization_angle"][event]),
            earth_rotation=earth_rotation,
        )
        for index, name in enumerate(_EQUIVALENCE_DETECTORS):
            yield event, index, name, reference[name].value


def _batched_strain(*, earth_rotation: bool):
    """Return the batched device strain for the shared catalogue."""
    from gwmock_signal.jax_batch import simulate_cbc_batch

    batch = simulate_cbc_batch(
        "IMRPhenomD",
        _EQUIVALENCE_DETECTORS,
        sampling_frequency=_EQUIVALENCE_FS,
        minimum_frequency=_EQUIVALENCE_F_MIN,
        parameters=_EQUIVALENCE_CATALOGUE,
        earth_rotation=earth_rotation,
    )
    return np.asarray(batch.strain)


def test_simulate_cbc_batch_earth_rotation_matches_numpy_path():
    """The batched rotating path agrees with the per-event NumPy projection.

    Anchors the device path against ``project_polarizations_to_network`` on the same
    polarizations, so the two implementations of Earth rotation cannot drift apart.
    """
    pytest.importorskip("jax", reason="jax not installed")
    pytest.importorskip("ripplegw", reason="ripple not installed")

    device = _batched_strain(earth_rotation=True)

    for event, index, name, expected in _numpy_reference_strain(earth_rotation=True):
        scale = np.max(np.abs(expected))
        # Relative tolerances are meaningless at an antenna null; the sky positions
        # are fixed, but assert the premise rather than assume it.
        assert scale > 0.0, f"event {event} {name} has a null response"
        difference = np.abs(device[event, index] - expected)
        # Round-off: one resampling kernel and one sidereal implementation across
        # both paths. See test_jax_projection.py for the history of this tolerance.
        assert np.sqrt(np.mean(difference**2)) < 1e-11 * scale
        assert np.max(difference) < 1e-10 * scale


def test_simulate_cbc_batch_static_pattern_matches_numpy_path():
    """The batched ``earth_rotation=False`` branch agrees with the per-event NumPy projection.

    The same anchoring as the rotating test above, for the branch that evaluates the response at a
    single instant. It exists separately because that branch has its own reference time -- the
    segment midpoint rather than the first sample -- which the rotating test cannot reach.

    ``test_simulate_cbc_batch_matches_host_pipeline`` already compares these two paths, but it gates
    on ``overlap > 0.999``, and an overlap is nearly blind to a common-mode error: giving one side
    the wrong sky-frame convention moves the strain by 4.4e-03 of peak and still leaves that test
    passing at 0.9999997, both measured. So the residual is asserted directly instead, which is what
    makes this a check on the *convention* reaching both paths and not only on the arithmetic.
    """
    pytest.importorskip("jax", reason="jax not installed")
    pytest.importorskip("ripplegw", reason="ripple not installed")

    device = _batched_strain(earth_rotation=False)

    for event, index, name, expected in _numpy_reference_strain(earth_rotation=False):
        scale = np.max(np.abs(expected))
        assert scale > 0.0, f"event {event} {name} has a null response"
        difference = np.abs(device[event, index] - expected)
        # Measured at 7.7e-12 to 1.3e-11 of peak across the four event/detector pairs, so the
        # bound sits just above that rather than at a round number. It is round-off from the two
        # paths associating the same arithmetic differently, the same size the rotating branch
        # shows, and eight orders below the 4.4e-03 a wrong sky frame costs.
        assert np.sqrt(np.mean(difference**2)) < 3e-11 * scale
