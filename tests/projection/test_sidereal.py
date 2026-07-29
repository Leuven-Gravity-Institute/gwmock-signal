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
