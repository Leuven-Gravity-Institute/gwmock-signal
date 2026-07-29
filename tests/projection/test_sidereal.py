"""Tests for the sidereal-time model shared by both projection paths.

The device path cannot call Astropy inside a compiled kernel, so it consumes a
host-computed anchor and rate instead. That substitution is only legitimate if GMST really
is linear over a segment, which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.projection.sidereal import (
    gmst_anchor_and_rate,
    gmst_rad_astropy,
    gmst_rate_rad_per_second,
)

_T0 = 1.4e9


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
