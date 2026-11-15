"""Tests for the sidereal-time model shared by both projection paths.

The device path cannot call Astropy inside a compiled kernel, so the host gives it GMST at every
sample: an anchor and rate, plus Astropy's departure from that line sampled at regular nodes and
interpolated between them on the host. These tests pin how closely that reproduces Astropy, within
a UTC day and across a UTC midnight, and that it does not depend on the length of the series.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from gwmock_signal.projection.sidereal import (
    gmst_anchor_and_rate,
    gmst_rad_astropy,
    gmst_rate_rad_per_second,
)

_T0 = 1.4e9

#: A UTC midnight at which Astropy's GMST changes rate by 1.25e-13 rad/s: the IERS UT1 - UTC
#: values it interpolates between are daily, and 2011 is covered by their final values.
_MIDNIGHT = float(Time("2011-10-01T00:00:00", scale="utc").gps)


def _astropy_gmst(start: float, offsets: np.ndarray) -> np.ndarray:
    """Astropy GMST at ``start + offsets``, the two passed apart so no instant is rounded."""
    times = Time(np.full(offsets.size, start), offsets, format="gps", scale="utc", location=(0, 0))
    return np.unwrap(np.asarray(times.sidereal_time("mean").rad, dtype=float))


def _wrapped(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@pytest.mark.parametrize("span", [256.0, 2048.0, 8192.0])
def test_gmst_is_linear_over_a_segment(span: float) -> None:
    """The anchor-and-rate model must reproduce Astropy across realistic segments.

    The bound is expressed as a geocenter delay error, because that is how a sidereal
    error actually enters the projection: an Earth radius over the speed of light times
    the angular error.
    """
    times = np.linspace(_T0, _T0 + span, 2001)
    exact = np.unwrap(gmst_rad_astropy(times))
    anchors, rate = gmst_anchor_and_rate(_T0)
    linear = anchors[0] + rate * (times - _T0)

    angular_error = float(np.max(np.abs(exact - linear)))
    delay_error = angular_error * 6.4e6 / 2.998e8
    assert delay_error < 1e-12, (angular_error, delay_error)


def test_rate_matches_the_sidereal_day() -> None:
    """The finite-difference rate must equal the known sidereal rotation rate.

    An external check rather than a self-consistency one: 2*pi over a sidereal day of
    86164.0905 s is 7.2921159e-5 rad/s.
    """
    expected = 2.0 * np.pi / 86164.0905
    assert gmst_rate_rad_per_second(_T0) == pytest.approx(expected, rel=1e-8)


def test_rate_is_correct_at_every_epoch_including_the_wrap() -> None:
    """The rate must not depend on where in the sidereal day the epoch falls.

    Regression test for a real bug: ``gmst_rad_astropy`` wraps to ``[0, 2*pi)``, and a
    600 s baseline that straddles the wrap gave a rate of about -1.04e-2 rad/s instead of
    +7.29e-5 -- wrong sign and 143x too large -- corrupting the projection for roughly
    0.7% of possible start times. It survived review because every test used a single
    benign epoch, so this one sweeps a whole sidereal day.
    """
    expected = 2.0 * np.pi / 86164.0905
    epochs = _T0 + np.linspace(0.0, 86400.0, 577)
    rates = np.array([gmst_rate_rad_per_second(float(t)) for t in epochs])
    assert np.allclose(rates, expected, rtol=1e-6), (
        f"worst rate {rates[np.argmax(np.abs(rates - expected))]:.6e} vs {expected:.6e}"
    )


def test_rate_is_correct_when_the_baseline_straddles_the_wrap() -> None:
    """Target the wrap directly rather than relying on a sweep happening to hit it."""
    expected = 2.0 * np.pi / 86164.0905
    times = _T0 + np.arange(0.0, 90000.0, 50.0)
    gmst = gmst_rad_astropy(times)
    just_before_wrap = float(times[int(np.argmax(np.diff(gmst) < 0))])
    for offset in (-500.0, -300.0, -100.0, -10.0):
        rate = gmst_rate_rad_per_second(just_before_wrap + offset)
        assert rate == pytest.approx(expected, rel=1e-6), (offset, rate)


def test_anchors_are_returned_per_start_time() -> None:
    """One anchor per segment, each matching a direct Astropy evaluation."""
    starts = _T0 + np.array([0.0, 300.0, 4096.0])
    anchors, _ = gmst_anchor_and_rate(starts)
    assert anchors.shape == starts.shape
    assert np.allclose(anchors, gmst_rad_astropy(starts), rtol=0.0, atol=1e-15)


def test_scalar_start_time_accepted() -> None:
    """A scalar start time must work and yield a length-one anchor array."""
    anchors, rate = gmst_anchor_and_rate(_T0)
    assert anchors.shape == (1,)
    assert rate > 0.0


def _piecewise(start: float, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The device model at ``offsets`` from ``start``, evaluated as the kernel evaluates it, and the line alone.

    The new helper is imported here rather than at module level, so this module still collects
    against a version without it and each test that needs it fails on its own.
    """
    from gwmock_signal.projection.sidereal import GMST_KINKS, gmst_piecewise

    anchors, rates, kinks, changes, _ = gmst_piecewise(start)
    model = anchors[0] + rates[0] * offsets
    for k in range(GMST_KINKS):
        model = model + changes[0, k] * np.maximum(offsets - kinks[0, k], 0.0)
    line_anchor, line_rate = gmst_anchor_and_rate(start)
    return model, line_anchor[0] + line_rate * offsets


def _model_error(start: float, span: float, step: float) -> tuple[float, float]:
    """The device model's and the old line's largest error against Astropy over a segment."""
    offsets = np.arange(0.0, span, step)
    exact = _astropy_gmst(start, offsets)
    model, line = _piecewise(start, offsets)
    return float(np.max(np.abs(_wrapped(model - exact)))), float(np.max(np.abs(_wrapped(line - exact))))


def test_gmst_model_follows_astropy_across_a_utc_midnight() -> None:
    """Chords bending at each UTC midnight must reproduce Astropy across one.

    Regression test for the device paths' sidereal time. A line from the segment's first sample is
    exact within a UTC day, but Astropy's GMST changes rate at each UTC midnight, so after one the
    line drifts: here by 3.9e-10 rad over the 3096 s that follow, a geocentre delay of 8e-12 s. The
    model is measured at 3.4e-14 rad here; the largest in a survey of every midnight from 2005 to 2026
    is 9.95e-14 rad, so 1e-13 holds at this pinned midnight, whose IERS values are final, with margin.

    The start is off the whole second, as an event's buffer is, and the comparison is every 0.25 s.
    """
    model, line = _model_error(_MIDNIGHT - 1000.0 + 0.3225957, 4096.0, 0.25)
    # The premise: across this midnight the line alone is far off, so the test can fail.
    assert line > 1e-10
    assert model < 1e-13


def test_gmst_model_follows_astropy_across_a_leap_second() -> None:
    """The same across the UTC midnight that ends 2016-12-31, which had a leap second.

    UTC repeats a second there while GPS time and UT1 run on, so Astropy's UT1 - UTC steps by one
    second; GMST, from UT1, does not. Measured at 5.1e-14 rad.
    """
    midnight = float(Time("2017-01-01T00:00:00", scale="utc").gps)
    model, _ = _model_error(midnight - 2000.0 + 0.3225957, 4096.0, 0.05)
    assert model < 1e-13


def test_gmst_model_follows_astropy_on_a_day_ending_with_a_leap_second() -> None:
    """A segment inside a UTC day that ends with a leap second, crossing no midnight.

    Regression test. The rate was a chord to Astropy's value at the next midnight, and an instant at a
    midnight after a leap second can round into the leap second, where Astropy's UT1 - UTC has not yet
    stepped: its GMST is a second of rotation off there, and the whole chord was 1.2e-5 rad wrong for
    this start. Rates are now measured inside the day. Measured at 3.1e-14 rad.
    """
    model, _ = _model_error(1119719212.607171, 4096.0, 0.25)
    assert model < 1e-13


def test_gmst_model_follows_astropy_over_the_longest_accepted_span() -> None:
    """Over a day, the longest span the device path accepts, the model stays within 1e-13 rad.

    The line alone drifts by 5.8e-9 rad over this day; the chords bend where Astropy does, so the error
    does not grow with the span. Measured at 4.3e-14 rad.
    """
    model, line = _model_error(_MIDNIGHT - 40000.0 + 0.3225957, 86400.0, 0.37)
    assert line > 1e-9
    assert model < 1e-13


def test_gmst_piecewise_bends_at_the_next_utc_midnights() -> None:
    """Two bends per start, at the next two UTC midnights; rates chords between Astropy values there."""
    from gwmock_signal.projection.sidereal import GMST_KINKS, gmst_piecewise

    starts = np.array([_MIDNIGHT - 1000.25, _T0, float(Time("2016-12-31T23:59:00", scale="utc").gps)])
    anchors, rates, kinks, changes, valid_until = gmst_piecewise(starts)
    assert anchors.shape == rates.shape == valid_until.shape == (3,)
    assert kinks.shape == changes.shape == (3, GMST_KINKS)
    # The anchor is the first piece's line at the first sample, which is Astropy there to round-off.
    np.testing.assert_allclose(_wrapped(anchors - gmst_rad_astropy(starts)), 0.0, rtol=0.0, atol=1e-13)
    for row, start in enumerate(starts):
        first = Time(start, format="gps", scale="utc")
        midnights = [Time(np.floor(first.utc.mjd) + day, format="mjd", scale="utc") for day in (1, 2, 3)]
        expected = np.array([(midnight - first).sec for midnight in midnights])
        np.testing.assert_allclose(kinks[row], expected[:GMST_KINKS], rtol=0.0, atol=1e-6)
        assert valid_until[row] == pytest.approx(expected[-1], abs=1e-6)
        assert valid_until[row] > 86400.0
        # The first piece's line reaches the first midnight on Astropy's value there.
        at_kink = _astropy_gmst(start, kinks[row, :1])
        assert abs(_wrapped(anchors[row] + rates[row] * kinks[row, 0] - at_kink[0])) < 1e-13
    # The bends are the size Astropy's rate changes by at a midnight, not whole turns.
    assert np.all(np.abs(changes) < 1e-12)
    # Only the start matters: the same start gives the same numbers whatever else is in the batch.
    alone = gmst_piecewise(starts[1])
    for together, single in zip((anchors, rates, kinks, changes, valid_until), alone, strict=True):
        np.testing.assert_array_equal(together[1], single[0])


def test_gmst_model_crosses_the_wrap() -> None:
    """Astropy wraps GMST to [0, 2*pi); a segment that crosses the wrap must not gain a whole turn."""
    times = _T0 + np.arange(0.0, 90000.0, 50.0)
    gmst = gmst_rad_astropy(times)
    start = float(times[int(np.argmax(np.diff(gmst) < 0))]) - 5.0
    model, _ = _model_error(start, 200.0, 0.5)
    assert model < 1e-13
