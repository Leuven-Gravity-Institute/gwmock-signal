"""The stack, segment assembly and SNR take the package's own time series as well as GWpy's.

Native and GWpy inputs give identical results, and what comes back is still a GWpy series: the one
given, when it was GWpy, or a single conversion that is returned every time it is asked for.
"""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock_signal.jax_batch import BatchedDetectorStrain, assemble_segments
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.sampling_grid import SamplingGrid
from gwmock_signal.snr import network_optimal_snr
from gwmock_signal.timeseries import TimeSeries

_FS = 64.0
_T0 = 1_400_000_000.0


def _native(value: float, *, n: int = 32, t0: float = _T0, **kwargs: object) -> TimeSeries:
    return TimeSeries(np.full(n, value), t0=t0, sample_rate=_FS, **kwargs)


class TestStack:
    """Channels of either kind; read back as GWpy series."""

    def test_native_channels_are_read_back_as_one_gwpy_series_each(self):
        """The same GWpy object every time, with the channel's samples and timing."""
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": _native(2.0)})

        first = stack["H1"]
        assert isinstance(first, GWpyTimeSeries)
        assert stack["H1"] is first
        assert stack[0] is first
        assert stack.to_dict()["H1"] is first
        assert np.array_equal(first.value, np.full(32, 1.0))
        assert first.t0.value == _T0
        assert stack.t0.value == _T0
        assert stack.sample_rate.value == _FS
        assert np.array_equal(stack.data, np.stack([np.full(32, 1.0), np.full(32, 2.0)]))

    def test_a_gwpy_channel_comes_back_as_the_object_given(self):
        """Mixed inputs: the GWpy one is returned as is, the native one converted."""
        given = _native(2.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": given})

        assert stack["L1"] is given
        assert isinstance(stack["H1"], GWpyTimeSeries)

    @pytest.mark.parametrize(
        ("other", "match"),
        [
            (_native(1.0, n=16), "length 16 does not match"),
            (_native(1.0, t0=_T0 + 1.0), "time grid does not match"),
            (_native(1.0, unit="strain"), "not compatible"),
            (TimeSeries(np.zeros((2, 32)), t0=_T0, sample_rate=_FS), "must be one-dimensional"),
        ],
    )
    def test_misaligned_native_channels_are_rejected(self, other, match):
        """Length, start, unit and shape are checked on native channels as on GWpy ones."""
        with pytest.raises(ValueError, match=match):
            DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": other})

    def test_anything_else_is_rejected_naming_the_detector(self):
        """A bare array has no timing."""
        with pytest.raises(TypeError, match=r"strains\['L1'\] must be a gwmock_signal.TimeSeries"):
            DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": np.zeros(32)})

    @pytest.mark.parametrize("fmt", ["hdf5", "npy"])
    def test_a_native_stack_survives_a_file_round_trip(self, tmp_path, fmt):
        """Samples, start time and rate come back exactly."""
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": _native(2.0)})
        path = tmp_path / f"stack.{fmt}"

        stack.write(path, format=fmt)
        back = DetectorStrainStack.read(path, format=fmt)

        assert back.detector_names == ("H1", "L1")
        assert np.array_equal(back.data, stack.data)
        assert back.t0.value == _T0
        assert back.sample_rate.value == _FS


@pytest.mark.parametrize("aligned", [True, False])
def test_assembly_takes_native_backgrounds(aligned):
    """Native and GWpy backgrounds give identical segments, on and off the output lattice."""
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch = BatchedDetectorStrain(
        strain=np.arange(16.0).reshape(1, 1, 16),
        detector_names=("H1",),
        coa_time=np.array([0.0 if aligned else 0.37 / _FS]),
        epoch=_T0 + 8 / _FS,
        sampling_frequency=_FS,
        start_index=np.array([8], dtype=np.int64) if aligned else None,
        grid=grid if aligned else None,
    )
    native = [{"H1": _native(5.0)}]
    gwpy = [{"H1": _native(5.0).to_gwpy()}]
    kwargs = {"segment_duration": 32 / _FS, "segment_start_times": [_T0]}

    from_native = assemble_segments(batch, backgrounds=native, **kwargs)[0]["H1"]
    from_gwpy = assemble_segments(batch, backgrounds=gwpy, **kwargs)[0]["H1"]

    assert isinstance(from_native, GWpyTimeSeries)
    assert np.array_equal(from_native.value, from_gwpy.value)
    assert np.any(from_native.value != 5.0)
    assert not np.any(native[0]["H1"].data != 5.0)


def test_network_snr_is_the_same_for_native_and_gwpy_channels():
    """It reads the channels as held, so their kind does not matter."""
    rng = np.random.default_rng(1)
    h1, l1 = (TimeSeries(rng.standard_normal(256), t0=_T0, sample_rate=_FS) for _ in range(2))
    psds = {"H1": np.ones(129), "L1": np.full(129, 2.0)}

    native = network_optimal_snr(DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": h1, "L1": l1}), psds)
    gwpy = network_optimal_snr(
        DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": h1.to_gwpy(), "L1": l1.to_gwpy()}), psds
    )

    assert native == gwpy
    assert native > 0.0


@pytest.mark.parametrize("aligned", [True, False])
def test_assembly_keeps_the_background_unit_and_never_shares_its_samples(aligned):
    """A segment with no signal in it is a copy of its background, unit included, not the background itself."""
    grid = SamplingGrid(epoch=_T0, sampling_frequency=_FS)
    batch = BatchedDetectorStrain(
        strain=np.ones((1, 1, 4)),
        detector_names=("H1",),
        coa_time=np.array([100.0]),
        epoch=_T0,
        sampling_frequency=_FS,
        start_index=np.array([100 * 64], dtype=np.int64) if aligned else None,
        grid=grid if aligned else None,
    )
    background = _native(5.0, unit="strain")

    segment = assemble_segments(
        batch, segment_duration=32 / _FS, segment_start_times=[_T0], backgrounds=[{"H1": background}]
    )[0]["H1"]
    segment.value[0] = -1.0

    assert segment.unit.to_string() == "strain"
    assert background.data[0] == 5.0


class TestOneSetOfSamples:
    """A stack's channel, its ``data`` and the caller's series stay one set of samples."""

    def test_a_native_channel_and_its_gwpy_reading_share_samples(self):
        """A write through either is seen by the other, and by ``data``."""
        given = _native(1.0)
        stack = DetectorStrainStack.from_mapping(["H1"], {"H1": given})

        stack["H1"].value[0] = 7.0
        given.data[1] = 8.0

        assert stack.data[0, :2].tolist() == [7.0, 8.0]
        assert stack["H1"].value[:2].tolist() == [7.0, 8.0]

    def test_written_timing_is_the_timing_read_back(self, tmp_path):
        """A file records the start time the stack reports, even after the GWpy series given is moved."""
        given = _native(1.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1"], {"H1": given})
        given.t0 = _T0 + 5.0

        stack.write(tmp_path / "stack.npy", format="npy")

        assert stack.t0.value == _T0 + 5.0
        assert DetectorStrainStack.read(tmp_path / "stack.npy", format="npy").t0.value == _T0 + 5.0

    def test_frames_record_the_timing_read_back(self, monkeypatch, tmp_path):
        """GWF output carries the start time the stack reports, as the other formats do."""
        from gwpy.timeseries import TimeSeriesDict

        written = {}
        monkeypatch.setattr(TimeSeriesDict, "write", lambda self, *args, **kwargs: written.update(self))
        given = _native(1.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": given, "L1": _native(2.0)})
        given.t0 = _T0 + 5.0

        stack.write(tmp_path / "stack.gwf", format="gwf")

        assert [written[name].t0.value for name in ("H1", "L1")] == [stack["H1"].t0.value, stack["L1"].t0.value]
        assert written["H1"].t0.value == _T0 + 5.0
        assert [written[name].name for name in ("H1", "L1")] == ["H1", "L1"]
        assert np.array_equal(written["L1"].value, np.full(32, 2.0))
