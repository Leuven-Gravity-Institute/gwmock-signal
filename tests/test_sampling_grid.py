"""Tests for the canonical sample lattice and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.sampling_grid import SamplingGrid

_EPOCH = 1.4e9
_FS = 2048.0


def _grid() -> SamplingGrid:
    return SamplingGrid(epoch=_EPOCH, sampling_frequency=_FS)


@pytest.mark.parametrize(("epoch", "fs"), [(_EPOCH, 0.0), (_EPOCH, -1.0), (_EPOCH, float("nan")), (float("inf"), _FS)])
def test_invalid_lattice_rejected(epoch: float, fs: float) -> None:
    """A lattice needs a finite epoch and a positive finite rate."""
    with pytest.raises(ValueError, match=r"finite|positive"):
        SamplingGrid(epoch=epoch, sampling_frequency=fs)


def test_index_and_time_round_trip() -> None:
    """``time_of`` must invert ``index_of`` at sample resolution."""
    grid = _grid()
    indices = np.array([0, 1, 12345, 10**7])
    assert np.allclose(grid.index_of(grid.time_of(indices)), indices, rtol=0.0, atol=1e-6)


def test_index_keeps_sub_sample_resolution_at_gps_magnitudes() -> None:
    """The sub-sample part must survive GPS-scale times.

    Guards the ordering inside ``index_of``: GPS times are ~1.4e9 and the spacing ~5e-4 s, so
    scaling before subtracting would lose the fraction that the whole contract is about.
    """
    grid = _grid()
    offset = 0.25 / _FS
    assert grid.index_of(_EPOCH + 1000.0 + offset) % 1.0 == pytest.approx(0.25, abs=1e-9)


def test_split_index_reconstructs_the_time() -> None:
    """``split_index`` must decompose a time without losing it."""
    grid = _grid()
    times = _EPOCH + np.array([0.0, 20.3179, 41.77123, 1000.5])
    index, fraction = grid.split_index(times)
    assert np.all((fraction >= 0.0) & (fraction < 1.0))
    assert np.allclose(grid.time_of(index) + fraction / _FS, times, rtol=0.0, atol=1e-9)


def test_on_lattice_detection() -> None:
    """Exact lattice times are on it; a half-sample offset is not."""
    grid = _grid()
    assert bool(grid.is_on_lattice(grid.time_of(17)))
    assert not bool(grid.is_on_lattice(grid.time_of(17) + 0.5 / _FS))


def test_off_lattice_times_are_rejected_not_rounded() -> None:
    """Rounding would silently displace data by up to half a sample."""
    grid = _grid()
    with pytest.raises(ValueError, match="must lie on the sampling grid"):
        grid.require_on_lattice(grid.time_of(3) + 0.4 / _FS, name="segment_start_times")


def test_require_on_lattice_reports_the_offender() -> None:
    """The message must identify which entry is wrong and by how much.

    The reported offset is ~0.2998 rather than exactly 0.3: adding 1.5e-4 s to a GPS time of
    1.4e9 is at the edge of float64 resolution, so the requested offset is not exactly
    representable. The message reports what the value actually is, which is the useful thing.
    """
    grid = _grid()
    times = np.array([grid.time_of(0), grid.time_of(1), grid.time_of(2) + 0.3 / _FS])
    with pytest.raises(ValueError, match=r"entry 2 .*\+0\.29"):
        grid.require_on_lattice(times, name="segment_start_times")


def test_from_segment_starts_anchors_on_the_first() -> None:
    """A contiguous tiling defines a valid grid anchored on its first segment."""
    starts = _EPOCH + np.arange(4) * 64.0
    grid = SamplingGrid.from_segment_starts(starts, _FS)
    assert grid.epoch == _EPOCH
    assert np.array_equal(grid.require_on_lattice(starts, name="starts"), np.arange(4) * int(64.0 * _FS))


def test_from_segment_starts_rejects_inconsistent_starts() -> None:
    """Starts that do not share one lattice cannot have a single canonical grid."""
    starts = np.array([_EPOCH, _EPOCH + 64.0 + 0.3 / _FS])
    with pytest.raises(ValueError, match="must lie on the sampling grid"):
        SamplingGrid.from_segment_starts(starts, _FS)


def test_from_segment_starts_rejects_empty() -> None:
    """There is no grid to build from nothing."""
    with pytest.raises(ValueError, match="must not be empty"):
        SamplingGrid.from_segment_starts(np.array([]), _FS)
