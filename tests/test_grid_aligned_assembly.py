"""Assembly must be exact when the batch is generated against an output lattice.

The device projection is accurate to ~1e-12, but superposing its output onto a segment used
to resample with a cubic spline whenever a signal started between samples — which, since
coalescence times are continuous, is essentially always. Measured on a real IMRPhenomD signal
the two paths differ by 18%, so the assembly step was setting the accuracy of the pipeline
and discarding what the projection achieved.

Generating against a :class:`~gwmock_signal.sampling_grid.SamplingGrid` folds the sub-sample
offset into the shift the projection already applies, so superposition becomes an integer add.
These tests pin that the aligned path introduces no error of its own and that off-lattice
inputs are refused rather than quietly rounded.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("ripplegw", reason="ripple not installed")

from gwmock_signal.jax_batch import assemble_segments, simulate_cbc_batch  # noqa: E402
from gwmock_signal.sampling_grid import SamplingGrid  # noqa: E402

_FS = 2048.0
_F_MIN = 30.0
_SEGMENT = 64.0
_T0 = 1.4e9
_STARTS = [_T0 + k * _SEGMENT for k in range(2)]

_BASE = {
    "detector_frame_mass_1": np.array([30.0, 25.0]),
    "detector_frame_mass_2": np.array([28.0, 22.0]),
    "luminosity_distance": np.array([400.0, 500.0]),
    "inclination": np.array([0.3, 1.1]),
    "coa_phase": np.array([0.0, 2.0]),
    "right_ascension": np.array([1.3, 4.0]),
    "declination": np.array([-0.4, 0.6]),
    "polarization_angle": np.array([0.7, 2.1]),
}
_ON_LATTICE = np.array([_T0 + 20.0, _T0 + 41.5])
_OFF_LATTICE = np.array([_T0 + 20.3179, _T0 + 41.77123])


def _grid() -> SamplingGrid:
    return SamplingGrid.from_segment_starts(np.array(_STARTS), _FS)


def _assembled(coa_time: np.ndarray, *, grid: SamplingGrid | None, earth_rotation: bool = True) -> np.ndarray:
    parameters = dict(_BASE)
    parameters["coa_time"] = coa_time
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        earth_rotation=earth_rotation,
        output_grid=grid,
    )
    segments = assemble_segments(batch, segment_duration=_SEGMENT, segment_start_times=_STARTS)
    return np.concatenate([s.to_dict()["E1"].value for s in segments])


@pytest.mark.parametrize("earth_rotation", [False, True])
def test_aligned_and_unaligned_agree_when_nothing_needs_resampling(earth_rotation: bool) -> None:
    """With on-lattice coalescence times both paths do integer adds, so they must match exactly.

    This is what establishes that the aligned path is *correct* rather than merely different:
    it introduces no error where no resampling is required.
    """
    unaligned = _assembled(_ON_LATTICE, grid=None, earth_rotation=earth_rotation)
    aligned = _assembled(_ON_LATTICE, grid=_grid(), earth_rotation=earth_rotation)
    scale = np.max(np.abs(unaligned))
    assert scale > 0.0
    assert np.max(np.abs(aligned - unaligned)) < 1e-12 * scale


def test_alignment_changes_the_result_when_coalescence_is_off_lattice() -> None:
    """The fix must actually change the off-lattice case, or it is doing nothing.

    The difference is the cubic resampling the aligned path removes; on a real IMRPhenomD
    signal it is of order 10%, not a rounding detail.
    """
    unaligned = _assembled(_OFF_LATTICE, grid=None)
    aligned = _assembled(_OFF_LATTICE, grid=_grid())
    scale = max(np.max(np.abs(unaligned)), np.max(np.abs(aligned)))
    assert np.max(np.abs(aligned - unaligned)) > 0.01 * scale


def test_aligned_batch_reports_lattice_indices() -> None:
    """The batch must carry where each event starts, so assembly needs no float arithmetic."""
    grid = _grid()
    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        output_grid=grid,
    )
    assert batch.grid == grid
    assert batch.start_index is not None
    assert batch.start_index.dtype.kind == "i"
    # The recorded index must be the lattice sample at or before the requested start.
    expected, _ = grid.split_index(batch.coa_time + batch.epoch)
    assert np.array_equal(batch.start_index, expected)


def test_unaligned_batch_carries_no_grid() -> None:
    """Omitting the grid must leave the previous behaviour and metadata untouched."""
    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
    )
    assert batch.grid is None
    assert batch.start_index is None


@pytest.mark.parametrize("earth_rotation", [False, True])
def test_both_branches_report_alignment(earth_rotation: bool) -> None:
    """Both branches must carry the metadata, or assembly silently resamples anyway.

    The static branch originally omitted it, and the omission was invisible because the
    aligned-versus-unaligned static comparison then assembled *both* sides unaligned.
    """
    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        earth_rotation=earth_rotation,
        output_grid=_grid(),
    )
    assert batch.grid is not None, "alignment metadata missing, so assembly would resample"
    assert batch.start_index is not None


@pytest.mark.parametrize("earth_rotation", [False, True])
def test_alignment_changes_the_result_on_both_branches(earth_rotation: bool) -> None:
    """Alignment must actually take effect on the static branch too, not only the rotating one."""
    unaligned = _assembled(_OFF_LATTICE, grid=None, earth_rotation=earth_rotation)
    aligned = _assembled(_OFF_LATTICE, grid=_grid(), earth_rotation=earth_rotation)
    scale = max(np.max(np.abs(unaligned)), np.max(np.abs(aligned)))
    assert np.max(np.abs(aligned - unaligned)) > 0.01 * scale


def test_overlap_uses_the_aligned_buffer_start() -> None:
    """An event a fraction of a sample inside a segment must be classified by where it lands.

    The overlap test originally used the *requested* start, which differs from the aligned one
    by the fractional remainder, so an event within a fraction of a sample of a boundary could
    be attributed to the wrong segment.
    """
    grid = _grid()
    # Place coalescence so the buffer start sits just before a segment boundary, with a
    # fractional remainder that the device absorbs.
    boundary = _STARTS[1]
    parameters = dict(_BASE)
    parameters["coa_time"] = np.array([boundary + 0.4 / _FS, boundary + 10.0])
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        output_grid=grid,
    )
    # The recorded start must be the lattice sample the data really begins on.
    assert np.allclose(grid.time_of(batch.start_index), grid.time_of(grid.split_index(batch.coa_time + batch.epoch)[0]))
    segments = assemble_segments(batch, segment_duration=_SEGMENT, segment_start_times=_STARTS)
    assert len(segments) == len(_STARTS)


def test_catalogue_aligns_by_default() -> None:
    """The production entry point must use the grid, not just expose the option.

    It previously built its own segment starts and then called the primitive without a grid, so
    the advertised path kept the cubic error.
    """
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    common = {
        "sampling_frequency": _FS,
        "minimum_frequency": _F_MIN,
        "parameters": parameters,
        "segment_duration": _SEGMENT,
        "start_time": _T0,
        "end_time": _T0 + 2 * _SEGMENT,
    }
    aligned = simulate_cbc_catalogue("IMRPhenomD", ["E1"], **common)
    legacy = simulate_cbc_catalogue("IMRPhenomD", ["E1"], align_to_output_grid=False, **common)
    a = np.concatenate([s.to_dict()["E1"].value for s in aligned])
    b = np.concatenate([s.to_dict()["E1"].value for s in legacy])
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)))
    assert scale > 0.0
    assert np.max(np.abs(a - b)) > 0.01 * scale, "default catalogue output is not aligned"


def test_catalogue_rejects_a_fractional_sample_segment() -> None:
    """A segment boundary landing mid-sample cannot share a lattice with the others."""
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    parameters = dict(_BASE)
    parameters["coa_time"] = _ON_LATTICE
    with pytest.raises(ValueError, match="whole number of samples"):
        simulate_cbc_catalogue(
            "IMRPhenomD",
            ["E1"],
            sampling_frequency=_FS,
            minimum_frequency=_F_MIN,
            parameters=parameters,
            segment_duration=_SEGMENT + 0.3 / _FS,
            start_time=_T0,
            end_time=_T0 + 2 * _SEGMENT,
        )


def test_off_lattice_segment_starts_are_rejected() -> None:
    """An aligned batch cannot be scattered onto segments that miss its lattice."""
    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        output_grid=_grid(),
    )
    with pytest.raises(ValueError, match="must lie on the sampling grid"):
        assemble_segments(
            batch,
            segment_duration=_SEGMENT,
            segment_start_times=[_T0 + 0.3, _T0 + _SEGMENT + 0.3],
        )


def test_grid_sample_rate_must_match_the_batch() -> None:
    """A grid at a different rate describes a different lattice and cannot be honoured."""
    parameters = dict(_BASE)
    parameters["coa_time"] = _ON_LATTICE
    with pytest.raises(ValueError, match="must equal"):
        simulate_cbc_batch(
            "IMRPhenomD",
            ["E1"],
            sampling_frequency=_FS,
            minimum_frequency=_F_MIN,
            parameters=parameters,
            output_grid=SamplingGrid(epoch=_T0, sampling_frequency=_FS / 2),
        )


def _aligned_batch(coa_time: np.ndarray, detectors: list[str] | None = None) -> object:
    parameters = dict(_BASE)
    parameters["coa_time"] = coa_time
    return simulate_cbc_batch(
        "IMRPhenomD",
        detectors or ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        output_grid=_grid(),
    )


def test_segment_ownership_is_exact_at_a_boundary() -> None:
    """A signal ending at a boundary belongs to the earlier segment only, and vice versa.

    The half-open convention ``[start, start + duration)`` is only meaningful if ownership at the
    seam is exact. An earlier version of this test asserted merely that both segments contained
    *some* nonzero data, which stays true when a boundary event is misattributed in either
    direction -- and both its events shared one strain array, so their contributions were not even
    distinguishable. It could not fail.

    This builds a synthetic batch whose two events carry distinct constant values, so every output
    sample can be attributed to exactly one of them.
    """
    from gwmock_signal.jax_batch import BatchedDetectorStrain

    n_signal = 64
    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    segment_duration = n_segment / _FS
    starts = [grid.time_of(0), grid.time_of(n_segment)]

    # Event 0 ends on the sample before the second segment starts; event 1 starts exactly on it.
    start_index = np.array([n_segment - n_signal, n_segment], dtype=np.int64)
    strain = np.stack([np.full((1, n_signal), 1.0), np.full((1, n_signal), 2.0)])
    batch = BatchedDetectorStrain(
        strain=strain,
        detector_names=("E1",),
        coa_time=np.asarray(grid.time_of(start_index), dtype=float),
        epoch=0.0,
        sampling_frequency=_FS,
        start_index=start_index,
        grid=grid,
    )

    first, second = (
        s.to_dict()["E1"].value
        for s in assemble_segments(batch, segment_duration=segment_duration, segment_start_times=starts)
    )

    # Event 0 occupies exactly the tail of the first segment, and nothing of the second.
    assert np.array_equal(first[: n_segment - n_signal], np.zeros(n_segment - n_signal))
    assert np.array_equal(first[n_segment - n_signal :], np.ones(n_signal))
    assert 2.0 not in set(np.unique(first)), "event 1 leaked into the segment before it starts"

    # Event 1 occupies exactly the head of the second segment, and nothing of the first.
    assert np.array_equal(second[:n_signal], np.full(n_signal, 2.0))
    assert np.array_equal(second[n_signal:], np.zeros(n_segment - n_signal))
    assert 1.0 not in set(np.unique(second)), "event 0 leaked into the segment after it ends"


def test_signal_straddling_a_boundary_is_split_exactly() -> None:
    """A signal crossing a segment boundary must be cropped at exactly the right sample.

    The ownership test above places each event wholly inside one segment, so it never exercises
    the cropping arithmetic -- an off-by-one in the slice bounds survives it untouched, because
    the signal length caps the bound. This one straddles the seam, so both slice ends are load
    bearing: 24 samples must land at the end of the first segment and the remaining 40 at the
    start of the second, with nothing spilling either way.
    """
    from gwmock_signal.jax_batch import BatchedDetectorStrain

    n_signal = 64
    n_segment = 128
    before_boundary = 24
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    starts = [grid.time_of(0), grid.time_of(n_segment)]
    start_index = np.array([n_segment - before_boundary], dtype=np.int64)

    batch = BatchedDetectorStrain(
        strain=np.full((1, 1, n_signal), 5.0),
        detector_names=("E1",),
        coa_time=np.asarray(grid.time_of(start_index), dtype=float),
        epoch=0.0,
        sampling_frequency=_FS,
        start_index=start_index,
        grid=grid,
    )
    first, second = (
        s.to_dict()["E1"].value
        for s in assemble_segments(batch, segment_duration=n_segment / _FS, segment_start_times=starts)
    )

    assert np.array_equal(first[:-before_boundary], np.zeros(n_segment - before_boundary))
    assert np.array_equal(first[-before_boundary:], np.full(before_boundary, 5.0))

    after_boundary = n_signal - before_boundary
    assert np.array_equal(second[:after_boundary], np.full(after_boundary, 5.0))
    assert np.array_equal(second[after_boundary:], np.zeros(n_segment - after_boundary))

    # Every sample of the signal is placed exactly once, in total.
    assert np.count_nonzero(first) + np.count_nonzero(second) == n_signal


def test_signal_spanning_many_segments() -> None:
    """A signal longer than several segments must contribute its overlapping part to each."""
    # Coalescence placed inside the span the segments below actually cover: coalescence sits
    # near the end of its buffer, so the buffer occupies roughly [coa - 3.6 s, coa + 0.4 s].
    batch = _aligned_batch(np.array([_T0 + 8.0, _T0 + 9.0]))
    n_signal = np.asarray(batch.strain).shape[2]
    # Segments short enough that one buffer genuinely spans several of them, chosen in whole
    # *samples*. Dividing the duration in seconds happened to be whole-sample only while buffer
    # lengths were powers of two; the backend now sizes to the next 5-smooth length, and an aligned
    # batch rightly refuses a fractional-sample segment.
    short_samples = n_signal // 4
    short_segment = short_samples / _FS
    starts = [_T0 + k * short_segment for k in range(16)]
    segments = assemble_segments(batch, segment_duration=short_segment, segment_start_times=starts)
    assert len(segments) == len(starts)
    occupied = [int(np.count_nonzero(s.to_dict()["E1"].value)) for s in segments]
    assert sum(1 for n in occupied if n > 0) >= 4, (n_signal, short_samples, occupied)


def test_negative_start_index_is_handled() -> None:
    """An event beginning before the grid epoch must still contribute its overlapping tail."""
    grid = _grid()
    batch = _aligned_batch(np.array([_T0 + 5.0, _T0 + 6.0]))
    n_signal = np.asarray(batch.strain).shape[2]
    shifted = object.__new__(type(batch))
    object.__setattr__(shifted, "strain", batch.strain)
    for field in ("detector_names", "coa_time", "epoch", "sampling_frequency", "grid"):
        object.__setattr__(shifted, field, getattr(batch, field))
    # Start well before the epoch, so only the tail of each buffer lands in segment zero.
    object.__setattr__(shifted, "start_index", np.array([-n_signal // 2, -n_signal // 3], dtype=np.int64))

    segments = assemble_segments(shifted, segment_duration=_SEGMENT, segment_start_times=_STARTS)
    values = segments[0].to_dict()["E1"].value
    assert np.count_nonzero(values) > 0, "a partially pre-epoch signal contributed nothing"
    assert np.all(np.isfinite(values))
    del grid


def test_multiple_detectors_share_one_alignment() -> None:
    """Detector delays differ, but the lattice index must not: it is not detector-specific."""
    batch = _aligned_batch(_OFF_LATTICE, detectors=["E1", "E2", "E3"])
    assert batch.start_index is not None
    assert batch.start_index.shape == _OFF_LATTICE.shape
    strain = np.asarray(batch.strain)
    assert strain.shape[1] == 3
    # Different geometry must give different strain, while sharing the same start indices.
    # Compared relative to the signal amplitude: strain is ~1e-24, so numpy's default
    # atol of 1e-8 would call any two of these arrays equal.
    scale = np.max(np.abs(strain))
    assert np.max(np.abs(strain[:, 0] - strain[:, 1])) > 1e-6 * scale
    segments = assemble_segments(batch, segment_duration=_SEGMENT, segment_start_times=_STARTS)
    assert set(segments[0].to_dict()) == {"E1", "E2", "E3"}


def test_aligned_assembly_rejects_a_fractional_sample_segment() -> None:
    """A direct caller must not get integer arithmetic on a rounded segment length."""
    batch = _aligned_batch(_ON_LATTICE)
    with pytest.raises(ValueError, match="whole number of samples"):
        assemble_segments(batch, segment_duration=_SEGMENT + 0.3 / _FS, segment_start_times=_STARTS)


def test_sidereal_anchor_uses_the_aligned_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rotating path must anchor sidereal time at the aligned start, not the requested one.

    Tested at the seam rather than by its numerical effect. Alignment moves a buffer back by a
    fractional sample; if the anchor does not move with it, ``F(t)`` and ``tau(t)`` lag the
    samples they multiply by up to a full sample. At Earth's real rotation rate that is ~3e-8 in
    ``F``, far too small for an end-to-end comparison to resolve, so this captures the times the
    projection actually anchors on and checks them against the aligned starts.
    """
    from gwmock_signal import jax_batch

    captured: list[np.ndarray] = []
    original = jax_batch.gmst_anchor_and_rate

    def _recording(start_times, **kwargs):
        captured.append(np.atleast_1d(np.asarray(start_times, dtype=float)).copy())
        return original(start_times, **kwargs)

    monkeypatch.setattr(jax_batch, "gmst_anchor_and_rate", _recording)

    grid = _grid()
    parameters = dict(_BASE)
    parameters["coa_time"] = _OFF_LATTICE
    batch = simulate_cbc_batch(
        "IMRPhenomD",
        ["E1"],
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=parameters,
        output_grid=grid,
    )

    assert captured, "the rotating path did not anchor sidereal time at all"
    # split_index is applied to coa_time + epoch, so time_of(start_index) *is* the aligned
    # buffer start; adding epoch again would count it twice.
    requested = _OFF_LATTICE + batch.epoch
    aligned = np.asarray(grid.time_of(batch.start_index), dtype=float)
    # The two differ by the fractional remainder, which is exactly what must have been applied.
    assert not np.allclose(requested, aligned, rtol=0.0, atol=1e-9)
    assert np.allclose(captured[0], aligned, rtol=0.0, atol=1e-9), (
        f"anchored at {captured[0]} but the aligned buffers start at {aligned}"
    )


def _synthetic_aligned_batch(n_signal: int, n_segment: int, grid: SamplingGrid) -> object:
    """One event of constant strain, aligned to *grid*, for background-validation tests."""
    from gwmock_signal.jax_batch import BatchedDetectorStrain

    return BatchedDetectorStrain(
        strain=np.ones((1, 1, n_signal)),
        detector_names=("E1",),
        coa_time=np.array([0.0]),
        epoch=_T0,
        sampling_frequency=_FS,
        start_index=np.array([0], dtype=np.int64),
        grid=grid,
    )


def _background_case(n_segment: int, grid: SamplingGrid, **overrides: float) -> tuple[object, dict, list[float]]:
    """Return ``(batch, backgrounds, starts)`` with one background, optionally inconsistent with its segment."""
    from gwpy.timeseries import TimeSeries

    batch = _synthetic_aligned_batch(64, n_segment, grid)
    length = int(overrides.get("length", n_segment))
    background = TimeSeries(
        np.zeros(length),
        t0=overrides.get("t0", grid.time_of(0)),
        sample_rate=overrides.get("sample_rate", _FS),
    )
    return batch, [{"E1": background}], [float(grid.time_of(0))]


def test_a_correct_background_is_accepted_and_added() -> None:
    """The matching case must still work, and the background must actually be included."""
    from gwpy.timeseries import TimeSeries

    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch = _synthetic_aligned_batch(64, n_segment, grid)
    background = TimeSeries(np.full(n_segment, 7.0), t0=grid.time_of(0), sample_rate=_FS)
    segments = assemble_segments(
        batch,
        segment_duration=n_segment / _FS,
        segment_start_times=[float(grid.time_of(0))],
        backgrounds=[{"E1": background}],
    )
    data = np.asarray(segments[0]["E1"].value)
    assert len(data) == n_segment
    # Background everywhere, plus the signal over its first 64 samples.
    assert np.allclose(data[:64], 8.0)
    assert np.allclose(data[64:], 7.0)


def test_background_of_the_wrong_length_is_rejected() -> None:
    """Measured: a half-length background was accepted and yielded a half-length segment."""
    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch, backgrounds, starts = _background_case(n_segment, grid, length=n_segment // 2)
    with pytest.raises(ValueError, match="samples, but the segment is"):
        assemble_segments(batch, segment_duration=n_segment / _FS, segment_start_times=starts, backgrounds=backgrounds)


def test_background_at_the_wrong_sample_rate_is_rejected() -> None:
    """Measured: the batch's rate silently replaced the background's."""
    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch, backgrounds, starts = _background_case(n_segment, grid, sample_rate=_FS * 2)
    with pytest.raises(ValueError, match="is sampled at"):
        assemble_segments(batch, segment_duration=n_segment / _FS, segment_start_times=starts, backgrounds=backgrounds)


def test_background_starting_elsewhere_is_rejected() -> None:
    """Measured: a background 100 s from the segment was accepted and silently relabelled.

    This is the case Copilot raised on the PR. Its suggested fix -- take the background's own
    ``t0`` for the result -- would have swapped one silent inconsistency for another: the data is
    placed by integer offsets from the *segment* start, so labelling the output with a different
    epoch describes an interval the samples do not occupy.
    """
    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch, backgrounds, starts = _background_case(n_segment, grid, t0=float(grid.time_of(0)) + 100.0)
    with pytest.raises(ValueError, match="but the segment it is paired with starts at"):
        assemble_segments(batch, segment_duration=n_segment / _FS, segment_start_times=starts, backgrounds=backgrounds)


def test_background_off_the_lattice_is_rejected() -> None:
    """A background starting between samples cannot be reconciled with integer placement."""
    n_segment = 128
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch, backgrounds, starts = _background_case(n_segment, grid, t0=float(grid.time_of(0)) + 0.4 / _FS)
    with pytest.raises(ValueError, match="must lie on the sampling grid"):
        assemble_segments(batch, segment_duration=n_segment / _FS, segment_start_times=starts, backgrounds=backgrounds)
