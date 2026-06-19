"""Tests for scattering batched signals into fixed-duration data segments."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.jax_batch import BatchedDetectorStrain, assemble_segments, simulate_cbc_catalogue

# 8 Hz sampling; 4 s segments => 32 samples per segment.
_FS = 8.0
_SEGMENT_DURATION = 4.0
_SEG_SAMPLES = 32


def _batch(strain: np.ndarray, detectors: tuple[str, ...], coa_time: np.ndarray, epoch: float) -> BatchedDetectorStrain:
    return BatchedDetectorStrain(
        strain=strain,
        detector_names=detectors,
        coa_time=coa_time,
        epoch=epoch,
        sampling_frequency=_FS,
    )


def test_assemble_segments_spanning_signal_split_across_segments() -> None:
    """A signal longer than a segment contributes its overlapping part to each segment."""
    # 16-sample (2 s) signal of distinct values, starting at t=3.0 -> spans [3, 5):
    # [3, 4) lands in segment 0 (indices 24..31); [4, 5) lands in segment 1 (indices 0..7).
    signal = (np.arange(16) + 1.0).reshape(1, 1, 16)
    batch = _batch(signal, ("H1",), coa_time=np.array([3.0]), epoch=0.0)

    segments = assemble_segments(batch, segment_duration=_SEGMENT_DURATION, segment_start_times=[0.0, 4.0])
    assert len(segments) == 2

    expected0 = np.zeros(_SEG_SAMPLES)
    expected0[24:32] = np.arange(1, 9)
    expected1 = np.zeros(_SEG_SAMPLES)
    expected1[0:8] = np.arange(9, 17)
    np.testing.assert_allclose(segments[0]["H1"].value, expected0)
    np.testing.assert_allclose(segments[1]["H1"].value, expected1)
    # Each segment carries the right start time and sample rate.
    assert segments[0]["H1"].t0.value == 0.0
    assert segments[1]["H1"].t0.value == 4.0


def test_assemble_segments_zero_noise_when_no_overlap() -> None:
    """A segment with no overlapping signal is all zeros (zero-noise default)."""
    signal = np.ones((1, 1, 16))
    batch = _batch(signal, ("H1",), coa_time=np.array([3.0]), epoch=0.0)  # spans [3, 5)
    segments = assemble_segments(batch, segment_duration=_SEGMENT_DURATION, segment_start_times=[0.0, 4.0, 8.0])
    np.testing.assert_array_equal(segments[2]["H1"].value, np.zeros(_SEG_SAMPLES))  # [8, 12): no signal


def test_assemble_segments_adds_onto_provided_background() -> None:
    """A provided background is preserved and the signal is added on top."""
    signal = np.full((1, 1, 16), 2.0)
    batch = _batch(signal, ("H1",), coa_time=np.array([0.0]), epoch=0.0)  # spans [0, 2) within segment 0
    background = [{"H1": TimeSeries(np.full(_SEG_SAMPLES, 5.0), t0=0.0, sample_rate=_FS)}]
    segments = assemble_segments(
        batch, segment_duration=_SEGMENT_DURATION, segment_start_times=[0.0], backgrounds=background
    )
    result = segments[0]["H1"].value
    np.testing.assert_allclose(result[:16], 7.0)  # 5 background + 2 signal
    np.testing.assert_allclose(result[16:], 5.0)  # background only


def test_assemble_segments_multiple_detectors_preserve_order() -> None:
    """Channels are returned per detector in the batch order."""
    strain = np.zeros((1, 2, 16))
    strain[0, 0] = 1.0  # H1 signal
    strain[0, 1] = 3.0  # L1 signal
    batch = _batch(strain, ("H1", "L1"), coa_time=np.array([0.0]), epoch=0.0)
    segments = assemble_segments(batch, segment_duration=_SEGMENT_DURATION, segment_start_times=[0.0])
    assert segments[0].detector_names == ("H1", "L1")
    assert segments[0]["H1"].value[0] == 1.0
    assert segments[0]["L1"].value[0] == 3.0


def test_assemble_segments_rejects_misaligned_backgrounds() -> None:
    """Backgrounds must be aligned one-to-one with segment_start_times."""
    batch = _batch(np.ones((1, 1, 16)), ("H1",), coa_time=np.array([0.0]), epoch=0.0)
    with pytest.raises(ValueError, match="aligned one-to-one"):
        assemble_segments(
            batch,
            segment_duration=_SEGMENT_DURATION,
            segment_start_times=[0.0, 4.0],
            backgrounds=[{"H1": TimeSeries(np.zeros(_SEG_SAMPLES), t0=0.0, sample_rate=_FS)}],
        )


def test_simulate_cbc_catalogue_rejects_nonpositive_segment_duration() -> None:
    """segment_duration must be > 0 (validated before any generation)."""
    with pytest.raises(ValueError, match="segment_duration must be > 0"):
        simulate_cbc_catalogue(
            "IMRPhenomD",
            ["H1"],
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            parameters={},
            segment_duration=0.0,
            start_time=0.0,
            end_time=16.0,
        )


def test_simulate_cbc_catalogue_rejects_empty_span() -> None:
    """end_time must be after start_time (validated before any generation)."""
    with pytest.raises(ValueError, match="end_time must be greater"):
        simulate_cbc_catalogue(
            "IMRPhenomD",
            ["H1"],
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            parameters={},
            segment_duration=4.0,
            start_time=10.0,
            end_time=10.0,
        )
