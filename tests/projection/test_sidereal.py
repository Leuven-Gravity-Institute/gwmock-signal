"""Tests for the sidereal-time model shared by both projection paths.

The device path cannot call Astropy inside a compiled kernel, so it consumes a
host-computed anchor and rate, plus Astropy's departure from that line at nodes, instead.
These tests pin how closely that reproduces Astropy, within a UTC day and across a UTC midnight.
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


def _model_error(start: float, span: float, step: float) -> tuple[float, float, np.ndarray]:
    """The device model's and the line's largest error against Astropy over a segment, and the nodes.

    The model is the line plus the correction nodes interpolated linearly, as the device evaluates it.
    The new helpers are imported here rather than at module level, so this module still collects
    against a version without them and each test that needs them fails on its own.
    """
    from gwmock_signal.projection.sidereal import (
        GMST_NODE_SPACING_SECONDS,
        gmst_correction_nodes,
        gmst_node_count,
    )

    anchors, rate = gmst_anchor_and_rate(start)
    correction = gmst_correction_nodes(start, anchors, rate, gmst_node_count(span))[0]
    offsets = np.arange(0.0, span, step)
    exact = _astropy_gmst(start, offsets)
    line = anchors[0] + rate * offsets
    model = line + np.interp(offsets, np.arange(correction.size) * GMST_NODE_SPACING_SECONDS, correction)
    return (
        float(np.max(np.abs(_wrapped(model - exact)))),
        float(np.max(np.abs(_wrapped(line - exact)))),
        correction,
    )


def test_gmst_model_follows_astropy_across_a_utc_midnight() -> None:
    """The line plus interpolated correction nodes must reproduce Astropy across a UTC midnight.

    Regression test for the device paths' sidereal time. A line from the segment's first sample is
    exact within a UTC day, but Astropy's GMST changes rate at each UTC midnight, so after one the
    line drifts: here by 3.9e-10 rad over the 3096 s that follow, a geocentre delay of 8e-12 s. The
    model is measured at 2.8e-14 rad here. Its bound for the bend alone is 3.1e-14 (the 1.25e-13 rad/s
    change in rate times a quarter of the 1 s spacing); the largest total measured at any midnight
    from 2005 to 2026 is 1.07e-13 rad, so 1e-13 holds with margin at this pinned midnight, whose IERS
    values are final, but is not a bound for every midnight.

    The start is off the whole second, as an event's buffer is, and the comparison is every 0.25 s,
    so it covers points between nodes as well as the interval that contains the midnight.
    """
    model, line, correction = _model_error(_MIDNIGHT - 1000.0 + 0.3225957, 4096.0, 0.25)
    # The premise: across this midnight the line alone is far off, so the test can fail.
    assert line > 1e-10
    assert model < 1e-13
    # Node zero is the anchor itself, so the correction there is zero.
    assert correction[0] == 0.0


def test_gmst_model_follows_astropy_across_a_leap_second() -> None:
    """The same across the UTC midnight that ends 2016-12-31, which had a leap second.

    UTC repeats a second there while GPS time and UT1 run on, so Astropy's UT1 - UTC steps by one
    second; GMST, from UT1, does not. Measured at 5.7e-14 rad.
    """
    midnight = float(Time("2017-01-01T00:00:00", scale="utc").gps)
    model, _, _ = _model_error(midnight - 2000.0 + 0.3225957, 4096.0, 0.05)
    assert model < 1e-13


def test_gmst_model_follows_astropy_over_the_longest_accepted_span() -> None:
    """Over a day, the longest span the device path accepts, the model stays within 1e-13 rad.

    The line alone drifts by 1e-8 rad over this day; the nodes keep the error local to the one
    interval that holds the midnight, so it does not grow with the span. Measured at 3.7e-14 rad.
    """
    model, line, _ = _model_error(_MIDNIGHT - 40000.0 + 0.3225957, 86400.0, 0.37)
    assert line > 1e-9
    assert model < 1e-13


def test_gmst_correction_nodes_cover_each_segment() -> None:
    """One row of nodes per start, each Astropy less the line at that node."""
    from gwmock_signal.projection.sidereal import (
        GMST_NODE_SPACING_SECONDS,
        gmst_correction_nodes,
    )

    starts = np.array([_MIDNIGHT - 10.5, _T0])
    anchors, rate = gmst_anchor_and_rate(starts)
    correction = gmst_correction_nodes(starts, anchors, rate, 22)
    assert correction.shape == (2, 22)
    offsets = np.arange(22) * GMST_NODE_SPACING_SECONDS
    for row, start in enumerate(starts):
        expected = _wrapped(_astropy_gmst(start, offsets) - (anchors[row] + rate * offsets))
        np.testing.assert_allclose(correction[row], expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("duration", "count"),
    [(4096.0, 4097), (4096.0 + 479.0 / 8192.0, 4098), (20.25, 22), (1.0, 2), (0.5, 2), (0.0, 2)],
)
def test_gmst_node_count_covers_the_last_sample(duration: float, count: int) -> None:
    """``ceil(duration / spacing) + 1`` nodes, the last at or after the last sample; never fewer than two."""
    from gwmock_signal.projection.sidereal import gmst_node_count

    assert gmst_node_count(duration) == count


@pytest.mark.parametrize(
    ("duration", "spacing"), [(-1.0, 1.0), (np.inf, 1.0), (np.nan, 1.0), (1.0, 0.0), (1.0, -1.0), (1.0, np.inf)]
)
def test_gmst_node_count_refuses_a_bad_duration_or_spacing(duration: float, spacing: float) -> None:
    """A negative or non-finite duration, or a spacing that is not finite and positive, is refused."""
    from gwmock_signal.projection.sidereal import gmst_node_count

    with pytest.raises(ValueError, match="must be finite"):
        gmst_node_count(duration, spacing)


def test_gmst_correction_unwraps_a_start_just_before_the_wrap() -> None:
    """Astropy wraps GMST to [0, 2*pi); a segment that crosses the wrap must not gain a whole turn."""
    from gwmock_signal.projection.sidereal import gmst_correction_nodes, gmst_node_count

    times = _T0 + np.arange(0.0, 90000.0, 50.0)
    gmst = gmst_rad_astropy(times)
    start = float(times[int(np.argmax(np.diff(gmst) < 0))]) - 5.0
    anchors, rate = gmst_anchor_and_rate(start)
    correction = gmst_correction_nodes(start, anchors, rate, gmst_node_count(200.0))[0]
    assert np.max(np.abs(correction)) < 1e-12
