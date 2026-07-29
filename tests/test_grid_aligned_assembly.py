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
