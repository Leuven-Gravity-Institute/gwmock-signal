"""Injection takes the package's own time series as well as GWpy's, and uses the whole injection.

Inputs of either kind give the same samples. The result is still a GWpy series. An injection off the
target's lattice is resampled from all of its samples, so splitting a target in two and injecting
into each half gives what injecting into the whole gives, up to rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.injection import inject_strain, inject_strains_sequential
from gwmock_signal.timeseries import TimeSeries
from gwmock_signal.timeseries import TimeSeries as NativeTimeSeries

_FS = 256.0
_T0 = 1_400_000_000.0


def _tone(t0: float, n: int) -> TimeSeries:
    times = t0 + np.arange(n) / _FS
    centre = t0 + 0.5 * n / _FS
    return TimeSeries(
        np.exp(-0.5 * ((times - centre) / 0.05) ** 2) * np.cos(2 * np.pi * 40.0 * (times - centre)),
        t0=t0,
        sample_rate=_FS,
    )


@pytest.mark.parametrize("target_kind", ["native", "gwpy"])
@pytest.mark.parametrize("injection_kind", ["native", "gwpy"])
@pytest.mark.parametrize("offset_samples", [5.0, 5.37])
def test_either_kind_of_input_gives_the_same_samples(target_kind, injection_kind, offset_samples):
    """Native and GWpy inputs, on and off the lattice, mixed in any combination."""
    target = TimeSeries(np.zeros(512), t0=_T0, sample_rate=_FS)
    injection = _tone(_T0 + offset_samples / _FS, 128)
    convert = {"native": lambda series: series, "gwpy": TimeSeries.to_gwpy}

    result = inject_strain(convert[target_kind](target), convert[injection_kind](injection))
    reference = inject_strain(target, injection)

    assert isinstance(result, NativeTimeSeries)
    assert np.array_equal(result.value, reference.value)
    assert result.t0 == _T0
    assert result.sample_rate == _FS
    assert np.any(result.value != 0.0)


def test_sequential_accepts_native_series():
    """Each injection is added in turn, whatever its kind, and the target is not modified."""
    target = TimeSeries(np.zeros(16), t0=0.0, sample_rate=4.0)
    first = TimeSeries(np.ones(2), t0=0.0, sample_rate=4.0)
    second = TimeSeries(np.full(2, 2.0), t0=0.5, sample_rate=4.0).to_gwpy()

    result = inject_strains_sequential(target, [first, second])

    assert list(result.value[:4]) == [1.0, 1.0, 2.0, 2.0]
    assert not np.any(target.data)


def test_the_target_keeps_its_name_and_unit():
    """The result describes the target's channel."""
    target = TimeSeries(np.zeros(8), t0=0.0, sample_rate=4.0, name="H1", unit="strain")

    result = inject_strain(target, TimeSeries(np.ones(2), t0=0.0, sample_rate=4.0, unit="strain"))

    assert (result.name, result.unit) == ("H1", "strain")


def test_a_different_unit_is_not_compatible():
    """Strain cannot be added to a dimensionless series."""
    target = TimeSeries(np.zeros(8), t0=0.0, sample_rate=4.0, unit="strain")

    with pytest.raises(ValueError, match="not compatible"):
        inject_strain(target, TimeSeries(np.ones(2), t0=0.0, sample_rate=4.0))


def test_an_off_lattice_injection_split_across_two_targets_matches_one_target():
    """Each half resamples from the whole injection, so the halves join where the whole is.

    The injection straddles the boundary between the halves. A target sample near its end has
    kernel taps that reach past that end, and those read the injection's samples there rather
    than nothing.
    """
    whole = TimeSeries(np.zeros(1024), t0=_T0, sample_rate=_FS)
    first, second = whole.crop(end=_T0 + 512 / _FS), whole.crop(start=_T0 + 512 / _FS)
    injection = _tone(_T0 + 448.37 / _FS, 128)

    joined = np.concatenate([inject_strain(first, injection).value, inject_strain(second, injection).value])

    assert np.max(np.abs(joined - inject_strain(whole, injection).value)) <= 1e-15


@pytest.mark.parametrize("offset_samples", [10.25, 10.75, -3.4, 500.6])
def test_an_off_lattice_injection_fills_exactly_the_samples_it_covers(offset_samples):
    """Every target sample whose time falls within the injection is filled, and no other.

    The covered samples are those from the first at or after the injection's start to the last at
    or before its end, clipped to the target. Samples outside are left exactly as they were.
    """
    n_target, n_injection = 512, 64
    target = TimeSeries(np.zeros(n_target), t0=_T0, sample_rate=_FS)
    injection = TimeSeries(np.ones(n_injection), t0=_T0 + offset_samples / _FS, sample_rate=_FS)

    result = inject_strain(target, injection).value

    first = max(int(np.ceil(offset_samples)), 0)
    last = min(int(np.floor(offset_samples + n_injection - 1)), n_target - 1)
    assert np.flatnonzero(result).tolist() == list(range(first, last + 1))


@pytest.mark.parametrize("role", ["target", "injection"])
def test_a_multi_row_series_is_rejected(role):
    """Injection adds one channel into one channel; several rows are refused with a clear message."""
    series = {
        "target": TimeSeries(np.zeros(8), t0=0.0, sample_rate=4.0),
        "injection": TimeSeries(np.ones(4), t0=0.0, sample_rate=4.0),
    }
    series[role] = TimeSeries(np.zeros((2, 8)), t0=0.0, sample_rate=4.0)

    with pytest.raises(ValueError, match=f"the {role} must be one-dimensional"):
        inject_strain(series["target"], series["injection"])
