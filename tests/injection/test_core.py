"""Tests for strain injection."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.injection import inject_strain, inject_strains_sequential


def _segment(n: int = 32, fs: float = 128.0, t0: float = 100.0) -> TimeSeries:
    return TimeSeries(np.zeros(n), t0=t0, sample_rate=fs)


def test_inject_returns_new_object():
    """Result is never the same instance as target (contract)."""
    target = _segment()
    inj = TimeSeries(np.ones(8), t0=target.t0.value, sample_rate=target.sample_rate)
    out = inject_strain(target, inj)
    assert out is not target


def test_inject_aligned_adds_in_window():
    """Aligned injection adds samples where grids overlap."""
    target = _segment(n=16, fs=4.0, t0=0.0)
    inj = TimeSeries(np.full(4, 2.0), t0=1.0, sample_rate=4.0)
    out = inject_strain(target, inj)
    assert np.allclose(out.value[4:8], 2.0)
    assert np.allclose(out.value[:4], 0.0)
    assert np.allclose(out.value[8:], 0.0)


def test_no_overlap_returns_copy():
    """Injection entirely after target returns unchanged values on a new object."""
    target = _segment(n=8, fs=4.0, t0=0.0)
    inj = TimeSeries(np.ones(4), t0=10.0, sample_rate=4.0)
    out = inject_strain(target, inj)
    assert out is not target
    assert np.allclose(out.value, target.value)


def test_sequential_empty_is_copy():
    """Empty injection list yields a copy of target."""
    target = _segment()
    out = inject_strains_sequential(target, [])
    assert out is not target
    assert np.array_equal(out.value, target.value)


def test_sequential_two():
    """Two aligned injections accumulate on the segment."""
    target = _segment(n=16, fs=4.0, t0=0.0)
    a = TimeSeries(np.ones(2), t0=0.0, sample_rate=4.0)
    b = TimeSeries(np.full(2, 2.0), t0=0.5, sample_rate=4.0)
    out = inject_strains_sequential(target, [a, b], interpolate_if_offset=True)
    assert out.value[0] == pytest.approx(1.0)
    assert out.value[2] == pytest.approx(2.0)


def test_incompatible_raises():
    """Mismatched sample rates should fail GWpy compatibility."""
    target = _segment(fs=128.0)
    inj = TimeSeries(np.ones(4), t0=target.t0.value, sample_rate=256.0)
    with pytest.raises(ValueError, match="sample sizes"):
        inject_strain(target, inj)


def _gaussian_tone(times: np.ndarray, centre: float, sigma: float, frequency: float) -> np.ndarray:
    """Band-limited reference signal: a Gaussian-windowed cosine, evaluated analytically."""
    return np.exp(-0.5 * ((times - centre) / sigma) ** 2) * np.cos(2.0 * np.pi * frequency * (times - centre))


@pytest.mark.parametrize("start_offset_seconds", [2.0, 100.0])
@pytest.mark.parametrize("nyquist_fraction", [0.2, 0.5])
def test_off_lattice_injection_matches_analytic_signal(start_offset_seconds: float, nyquist_fraction: float):
    """A half-sample offset is resampled accurately, however far into the segment it lands.

    The alignment decision must not depend on how large the start offset is: a relative
    tolerance accepts any fractional part once the offset is large enough and snaps the
    injection a whole half sample. The fractional shift must also be band-limited, since a
    cubic spline is already wrong at the percent level by half Nyquist.
    """
    fs = 1024.0
    dt = 1.0 / fs
    target = TimeSeries(np.zeros(int(128 * fs)), t0=0.0, sample_rate=fs)
    n_inj = 512
    inj_t0 = start_offset_seconds + 0.5 * dt
    inj_times = inj_t0 + np.arange(n_inj) * dt
    centre = inj_t0 + 0.5 * n_inj * dt
    sigma = 40.0 * dt
    frequency = nyquist_fraction * 0.5 * fs
    injection = TimeSeries(_gaussian_tone(inj_times, centre, sigma, frequency), t0=inj_t0, sample_rate=fs)

    out = inject_strain(target, injection)

    expected = _gaussian_tone(np.arange(len(target)) * dt, centre, sigma, frequency)
    expected[np.abs(np.arange(len(target)) * dt - centre) > 0.5 * n_inj * dt] = 0.0
    assert np.max(np.abs(out.value - expected)) <= 1e-6


def test_off_lattice_far_into_segment_is_skipped_without_interpolation():
    """With interpolation disabled, a fractional offset is skipped rather than snapped."""
    fs = 1024.0
    target = TimeSeries(np.zeros(int(128 * fs)), t0=0.0, sample_rate=fs)
    injection = TimeSeries(np.ones(16), t0=100.0 + 0.5 / fs, sample_rate=fs)

    out = inject_strain(target, injection, interpolate_if_offset=False)

    assert np.array_equal(out.value, target.value)


def test_on_lattice_at_gps_epoch_is_exact_add():
    """An injection on the lattice at a realistic GPS epoch is added sample-for-sample."""
    fs = 4096.0
    t0 = 1_400_000_000.0
    target = TimeSeries(np.zeros(int(200 * fs)), t0=t0, sample_rate=fs)
    start_index = int(150 * fs) + 3
    values = np.arange(1.0, 33.0)
    injection = TimeSeries(values, t0=t0 + start_index / fs, sample_rate=fs)

    out = inject_strain(target, injection)

    assert np.array_equal(out.value[start_index : start_index + values.size], values)
    assert np.count_nonzero(out.value) == values.size
