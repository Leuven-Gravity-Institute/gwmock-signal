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
    from gwmock_signal.projection.resampling import edge_padding

    padding = edge_padding(_FS)
    n_samples = round(8.0 * _FS)  # the pinned waveform buffer
    # The waveform buffer padded at each end against the geocentre delay, as the host projection is.
    assert np.asarray(result.strain).shape == (_N_EVENTS, len(_DETECTORS), n_samples + 2 * padding)
    assert result.detector_names == tuple(_DETECTORS)
    np.testing.assert_array_equal(result.coa_time, _CATALOGUE["coa_time"])
    # The epoch describes the buffer returned, so it starts the padding earlier than the waveform's.
    _, waveform_epoch = backend.coalescence_placement(n_samples, _FS)
    assert result.epoch == pytest.approx(waveform_epoch - padding / _FS)
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
            # Both are padded at each end against the geocentre delay, by the same amount.
            assert host.t0 == pytest.approx(float(_CATALOGUE["coa_time"][i]) + result.epoch, rel=0, abs=1e-6)
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
        assert stack["H1"].t0 == pytest.approx(start_time + k * segment_duration)

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
        assert single["H1"].t0 == split["H1"].t0
    for detector in _DETECTORS:
        a = np.concatenate([s[detector].value for s in unbinned])
        b = np.concatenate([s[detector].value for s in binned])
        overlap = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
        residual = float(np.max(np.abs(a - b)) / peak)

        # The overlap is kept, but it is not the check, and measurement says so plainly: scaling one
        # side by +0.5%, +2%, +5% and even +20% leaves this overlap at 0.9997787 to seven digits, so
        # it passes the old 0.99 gate *and* a tightened 0.999 one. That matches the lesson
        # `test_simulate_cbc_batch_matches_host_pipeline_static` already records in this file -- an
        # overlap is nearly blind to a common-mode error. Tightening it would have looked stricter
        # and discriminated no better.
        #
        # The residual is the check, and it is a *band* rather than a ceiling. Binning is a
        # grid-discretization trade: the difference is supposed to be there, about 1.3e-02 of peak
        # (H1 1.324e-02, L1 1.181e-02 measured). A ceiling alone would pass a change that made the
        # difference vanish -- binning silently reduced to a single grid -- which is the failure the
        # old assertion could not see either. The floor catches that; the ceiling catches drift.
        assert overlap > 0.999, f"{detector} binned/unbinned overlap {overlap:.6f}"
        assert 0.8e-2 < residual < 1.8e-2, (
            f"{detector} binned/unbinned residual {residual:.3e} of peak is outside the measured "
            "discretization band; below it, binning is not actually splitting the grid"
        )


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
def test_simulate_cbc_catalogue_chunking_does_not_change_the_model() -> None:
    """Count-chunking leaves the simulated signals alone -- to 1e-9 of peak, not bit-for-bit.

    The name and this docstring used to say "exactly the same segments", which the assertion below
    has never checked and which is false. Measured on both assembly paths: a chunked run differs
    from a whole one at a few times 1e-13 of peak, while whole-vs-whole is bit-identical run to run.
    The cause is not superposition arithmetic -- it is per-event frequency-domain generation, where
    XLA picks different reduction orderings for different batch shapes, so the grid-defining
    worst-case event comes out bit-identical and the others do not.

    The 1e-9 bound is therefore right and deliberately loose: the worst residual on this catalogue
    measures 7.0e-11 of peak, leaving about a factor of fourteen for platform arithmetic. What was
    wrong was the promise, not the number.
    """
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


def _host_reference(*, earth_rotation: bool, backend=None, catalogue=None):
    """Yield ``(event, detector_index, name, expected)`` from the per-event host projection.

    Rebuilds the same time-domain polarizations the device path projects -- one shared frequency
    grid, inverse-FFT, roll to place coalescence -- then runs
    :func:`~gwmock_signal.projection.network.project_polarizations_to_network` on them, which pads
    each detector series against the geocentre delay. ``expected`` is that padded
    :class:`~gwmock_signal.timeseries.TimeSeries`, start time included, so a comparison checks where
    the batch says its buffer starts as well as what it holds. Shared by both branches so neither
    can be compared against a different waveform than the other.
    """
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.waveform.backends.ripple import RippleBackend

    backend = backend or RippleBackend()
    catalogue = _EQUIVALENCE_CATALOGUE if catalogue is None else catalogue
    fd = backend.generate_fd_polarizations_batch(
        "IMRPhenomD",
        sampling_frequency=_EQUIVALENCE_FS,
        minimum_frequency=_EQUIVALENCE_F_MIN,
        parameters=catalogue,
    )
    n_samples = fd.n_samples
    merger_index, epoch = backend.coalescence_placement(n_samples, _EQUIVALENCE_FS)

    for event in range(len(catalogue["coa_time"])):
        start = epoch + catalogue["coa_time"][event]
        hp = np.roll(np.fft.irfft(np.asarray(fd.plus[event]), n=n_samples) * _EQUIVALENCE_FS, merger_index)
        hc = np.roll(np.fft.irfft(np.asarray(fd.cross[event]), n=n_samples) * _EQUIVALENCE_FS, merger_index)
        reference = project_polarizations_to_network(
            {
                "plus": GWpyTimeSeries(hp, t0=start, sample_rate=_EQUIVALENCE_FS),
                "cross": GWpyTimeSeries(hc, t0=start, sample_rate=_EQUIVALENCE_FS),
            },
            _EQUIVALENCE_DETECTORS,
            right_ascension=float(catalogue["right_ascension"][event]),
            declination=float(catalogue["declination"][event]),
            polarization_angle=float(catalogue["polarization_angle"][event]),
            earth_rotation=earth_rotation,
        )
        for index, name in enumerate(_EQUIVALENCE_DETECTORS):
            yield event, index, name, reference[name]


def _batched(*, earth_rotation: bool, backend=None, catalogue=None):
    """Return the batched device result for the shared catalogue."""
    from gwmock_signal.jax_batch import simulate_cbc_batch

    return simulate_cbc_batch(
        "IMRPhenomD",
        _EQUIVALENCE_DETECTORS,
        sampling_frequency=_EQUIVALENCE_FS,
        minimum_frequency=_EQUIVALENCE_F_MIN,
        parameters=_EQUIVALENCE_CATALOGUE if catalogue is None else catalogue,
        earth_rotation=earth_rotation,
        backend=backend,
    )


def _placed_on(expected, batch, event: int, detector: int) -> np.ndarray:
    """Return the batch's strain for one event and detector, placed on ``expected``'s samples.

    Placed by the start time the batch reports for its own buffer (``epoch + coa_time``), not by
    assuming the two arrays line up, and zero wherever the batch holds no sample. A batch that is
    shorter than the host projection, or reports a different start, therefore differs from it by
    the signal it is missing rather than failing on a shape before any value is compared.
    """
    strain = np.asarray(batch.strain)[event, detector]
    offset = round((batch.epoch + float(batch.coa_time[event]) - float(expected.t0)) * _EQUIVALENCE_FS)
    placed = np.zeros(expected.n_samples)
    first, last = max(offset, 0), min(offset + strain.size, expected.n_samples)
    placed[first:last] = strain[first - offset : last - offset]
    return placed


@pytest.mark.requires_gwpy
def test_simulate_cbc_batch_earth_rotation_matches_numpy_path():
    """The batched rotating path agrees with the per-event NumPy projection.

    Anchors the device path against ``project_polarizations_to_network`` on the same
    polarizations, so the two implementations of Earth rotation cannot drift apart -- including
    the padding that keeps the geocentre delay from moving signal out of the buffer.
    """
    pytest.importorskip("jax", reason="jax not installed")
    pytest.importorskip("ripplegw", reason="ripple not installed")

    batch = _batched(earth_rotation=True)

    for event, index, name, expected in _host_reference(earth_rotation=True):
        scale = np.max(np.abs(expected.value))
        # Relative tolerances are meaningless at an antenna null; the sky positions
        # are fixed, but assert the premise rather than assume it.
        assert scale > 0.0, f"event {event} {name} has a null response"
        difference = np.abs(_placed_on(expected, batch, event, index) - expected.value)
        # Round-off: one resampling kernel and one sidereal implementation across
        # both paths. See test_jax_projection.py for the history of this tolerance.
        assert np.sqrt(np.mean(difference**2)) < 1e-11 * scale
        assert np.max(difference) < 1e-10 * scale
        assert np.asarray(batch.strain).shape[-1] == expected.n_samples


@pytest.mark.requires_gwpy
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

    batch = _batched(earth_rotation=False)

    for event, index, name, expected in _host_reference(earth_rotation=False):
        scale = np.max(np.abs(expected.value))
        assert scale > 0.0, f"event {event} {name} has a null response"
        difference = np.abs(_placed_on(expected, batch, event, index) - expected.value)
        # Measured at 9e-17 to 1e-16 of peak (RMS) across the four event/detector pairs: both paths
        # pad the same buffer and phase-shift it through a real FFT of the same length, so they
        # differ only in the last bits. The bound leaves three orders of magnitude for other FFT
        # implementations, and is far below both the 4.4e-03 a wrong sky frame costs and the
        # 2.9e-05 by which an unpadded phase shift -- wrapping the delayed edges round the buffer --
        # differed from the host here.
        assert np.sqrt(np.mean(difference**2)) < 1e-13 * scale
        assert np.max(difference) < 1e-12 * scale
        assert np.asarray(batch.strain).shape[-1] == expected.n_samples


#: Two sky positions on opposite sides of the Earth from the Einstein Telescope site, so the
#: geocentre delay moves the signal later at one and earlier at the other.
_ANTIPODAL_CATALOGUE = {
    key: np.array([value[0], value[0]]) for key, value in _EQUIVALENCE_CATALOGUE.items() if key not in {"coa_time"}
} | {
    "right_ascension": np.array([1.3, 1.3 + np.pi]),
    "declination": np.array([0.7, -0.7]),
    "coa_time": np.array([1.4e9, 1.4e9 + 300.0]),
}


@pytest.mark.requires_gwpy
@pytest.mark.parametrize("earth_rotation", [True, False])
@pytest.mark.parametrize("merger_at", ["end", "start"])
def test_simulate_cbc_batch_keeps_signal_the_delay_moves_past_an_end(earth_rotation: bool, merger_at: str):
    """The batch keeps what the geocentre delay moves past either end of the waveform buffer.

    The merger is placed two samples from one end of the buffer, so on whichever of the two sky
    positions the delay points outwards the peak itself leaves the buffer. Before the batch was
    padded, the rotating branch dropped it and the phase-shift branch wrapped it round to the other
    end, while the host projection, padded since, keeps it beyond the end. The host is the
    reference; the batch is placed on the host's samples by its own reported start time.
    """
    pytest.importorskip("jax", reason="jax not installed")
    pytest.importorskip("ripplegw", reason="ripple not installed")
    from gwmock_signal.projection.resampling import edge_padding
    from gwmock_signal.waveform.backends.ripple import RippleBackend

    # A fixed buffer, because a ringdown fraction this close to 0 or 1 would otherwise size it for
    # an inspiral occupying a sliver of it. The merger is then two samples from the chosen end.
    ringdown_fraction = 0.0001 if merger_at == "end" else 0.9999
    backend = RippleBackend(segment_duration=4.0, ringdown_fraction=ringdown_fraction)
    batch = _batched(earth_rotation=earth_rotation, backend=backend, catalogue=_ANTIPODAL_CATALOGUE)
    reference = list(_host_reference(earth_rotation=earth_rotation, backend=backend, catalogue=_ANTIPODAL_CATALOGUE))

    padding = edge_padding(_EQUIVALENCE_FS)
    beyond = slice(-padding, None) if merger_at == "end" else slice(0, padding)
    # The premise: on at least one event the host keeps a large part of the signal beyond that end.
    # Without it the comparison below would pass whatever the batch did there.
    assert (
        max(np.max(np.abs(expected.value[beyond])) / np.max(np.abs(expected.value)) for *_, expected in reference) > 0.5
    )

    for event, index, name, expected in reference:
        scale = np.max(np.abs(expected.value))
        difference = np.abs(_placed_on(expected, batch, event, index) - expected.value)
        assert np.max(difference) < 1e-10 * scale, (
            f"event {event} {name}: {np.max(difference) / scale:.3e} of peak from the host projection"
        )
