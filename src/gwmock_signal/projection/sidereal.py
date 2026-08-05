#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""Sidereal time for the projection paths, with Astropy as the single authority.

Both projection paths need Greenwich Mean Sidereal Time, and for a while they computed it
two different ways: Astropy on the host, and the IAU 2006 series
(:func:`~gwmock_signal.projection.jax_projection.gmst_rad`) on device, because Astropy
cannot be called from inside a JIT-compiled kernel. Those two differ by about 1.5e-6 rad,
which enters the geocenter delay as a ~3e-8 s timing error and was, once the resampling
kernel was made accurate, the largest disagreement between the paths.

The fix is not a better series but a different decomposition. GMST is very nearly linear in
time, so a segment needs only an anchor and a rate — two host-computed scalars that a
device kernel can consume. Astropy therefore remains the only implementation of the
sidereal model, including its IERS UT1 handling, while the kernel does one multiply-add.

!!! note "How linear is it?"

    Fitting the endpoints and measuring the worst deviation in between, against Astropy:

    | segment | max deviation | as a geocenter delay error |
    |---|---|---|
    | 256 s | 6.4e-14 rad | 1.4e-15 s |
    | 2048 s | 6.4e-14 rad | 1.4e-15 s |
    | 8192 s | 5.6e-14 rad | 1.2e-15 s |
    | 86400 s | 8.6e-10 rad | 1.8e-11 s |

    So over any segment length these simulations use, the linear model is exact to well
    below every other error in the projection. ``tests/projection/test_sidereal.py``
    pins this.

Using Astropy also removes the ``dut1`` approximation rather than merely exposing it:
Astropy applies real IERS UT1-UTC, whereas the on-device series defaulted to ``dut1 = 0``
and so carried up to 0.9 s of sidereal error, worth 6.7e-4 in the antenna pattern.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

#: Baseline for the finite-difference rate estimate, in seconds. Long enough that Astropy's
#: own round-off does not dominate the slope, short enough that GMST advances only
#: ~0.044 rad across it -- which is what makes the wrap correction below unambiguous.
_RATE_BASELINE_SECONDS = 600.0


def gmst_rad_astropy(t_gps: np.ndarray | float) -> np.ndarray:
    """Return Greenwich Mean Sidereal Time in radians, from Astropy.

    Args:
        t_gps: GPS time(s) in seconds.

    Returns:
        GMST in radians, wrapped to ``[0, 2*pi)``, with the shape of ``t_gps``.
    """
    return np.asarray(
        Time(t_gps, format="gps", scale="utc", location=(0, 0)).sidereal_time("mean").rad,
        dtype=float,
    )


def gmst_rate_rad_per_second(t_gps: float) -> float:
    """Return dGMST/dt in radians per second near *t_gps*.

    Estimated by finite difference over :data:`_RATE_BASELINE_SECONDS` rather than taken
    as the nominal sidereal rate, so that the precession terms Astropy models are carried
    through instead of being silently dropped.

    Args:
        t_gps: GPS time in seconds at which to evaluate the rate.

    Returns:
        The rate in rad/s.
    """
    start = float(t_gps)
    pair = np.unwrap(gmst_rad_astropy(np.array([start, start + _RATE_BASELINE_SECONDS])))
    # Unwrapped because gmst_rad_astropy wraps to [0, 2*pi): a short baseline does not
    # prevent it from *straddling* the wrap, only from spanning more than one. Without
    # this, an epoch in the last 600 s before the wrap yields a rate of about
    # -1.04e-2 rad/s instead of +7.29e-5 -- wrong sign, 143x too large -- which would
    # corrupt every rotating projection anchored there. That is roughly 0.7% of epochs.
    return float((pair[1] - pair[0]) / _RATE_BASELINE_SECONDS)


def gmst_anchor_and_rate(
    start_times: np.ndarray | float, *, rate_epoch: float | None = None
) -> tuple[np.ndarray, float]:
    """Return per-segment GMST anchors and one shared rate.

    The anchors are *not* wrapped into ``[0, 2*pi)`` by the caller's linear extrapolation,
    and they need not be: the antenna pattern and delay depend on GMST only through sine
    and cosine, so an unwrapped phase is equivalent and keeps the model continuous across
    a segment that would otherwise straddle the wrap.

    Args:
        start_times: GPS time(s) at which each segment begins.
        rate_epoch: Time at which to evaluate the rate. Defaults to the first start time;
            the rate varies far too slowly for the choice to matter over a catalogue.

    Returns:
        ``(anchors, rate)`` where ``anchors`` has the shape of ``start_times`` and is GMST
        in radians at those times, and ``rate`` is dGMST/dt in rad/s.
    """
    start_times = np.atleast_1d(np.asarray(start_times, dtype=float))
    anchors = gmst_rad_astropy(start_times)
    epoch = float(start_times.flat[0]) if rate_epoch is None else float(rate_epoch)
    return anchors, gmst_rate_rad_per_second(epoch)


#: Seconds per Julian century, for the precession polynomials.
_SECONDS_PER_JULIAN_CENTURY = 36525.0 * 86400.0

#: GPS seconds at the J2000.0 epoch (2000-01-01T12:00:00 TT), the origin of those polynomials.
_GPS_AT_J2000 = 630763213.0

#: Arcseconds to radians, written as LAL writes it (pi / 6.48e5) so the two can be compared literally.
_ARCSEC_TO_RAD = np.pi / 6.48e5


def lunisolar_precession_angles(t_gps: float) -> tuple[float, float, float]:
    """Return the three lunisolar precession angles ``(zeta_A, z_A, theta_A)`` in radians.

    Equations 3.212 of the *Explanatory Supplement to the Astronomical Almanac*, which is what
    LAL's ``XLALBarycenterEarth`` evaluates as ``tzeA``, ``zA`` and ``thetaA``. The polynomials are
    reproduced rather than taken from Astropy on purpose: the reference this pipeline is validated
    against is LAL, and Astropy's IAU 2006 model differs from these truncated series at the
    milliarcsecond level. Agreeing with the reference beats agreeing with a better model that the
    reference does not use.

    Args:
        t_gps: GPS time in seconds.

    Returns:
        ``(zeta_A, z_A, theta_A)`` in radians.
    """
    centuries = (float(t_gps) - _GPS_AT_J2000) / _SECONDS_PER_JULIAN_CENTURY
    zeta_a = centuries * (2306.2181 + (0.30188 + 0.017998 * centuries) * centuries) * _ARCSEC_TO_RAD
    z_a = centuries * (2306.2181 + (1.09468 + 0.018203 * centuries) * centuries) * _ARCSEC_TO_RAD
    theta_a = centuries * (2004.3109 - (0.42665 + 0.041833 * centuries) * centuries) * _ARCSEC_TO_RAD
    return zeta_a, z_a, theta_a


def precess_to_epoch(right_ascension: float, declination: float, t_gps: float) -> tuple[float, float]:
    """Rotate a J2000 sky position into the mean equator and equinox of *t_gps*.

    **Why this is needed at all.** Greenwich Mean Sidereal Time measures the Earth's rotation from
    the mean equinox *of date*. Combining it with a right ascension referred to J2000 mixes two
    frames, and the mismatch grows as the equinox precesses -- 0.43 degrees in right ascension by
    2030. That is not a convention; it is an inconsistency, and it showed up as a 1.8e-04 s
    disagreement with ``lalpulsar.Barycenter`` in the geocentre-to-detector delay, worst case over
    detectors, sky positions and epochs.

    Applying this rotation first brings both quantities into the frame of date. The same rotation
    LAL applies inside ``XLALBarycenter``, expressed as a sky position rather than inlined into a
    dot product, so that one call serves the delay, the antenna pattern and both backends.

    **Evaluated once per segment, not per sample.** The angles move by 2306 arcseconds per century,
    so over even a 4096 s segment they change by 3e-6 arcseconds -- fourteen orders of magnitude
    below the effect being corrected. The caller passes a reference epoch.

    **What it does not include** is nutation, the short-period part of the same motion, which LAL
    adds separately. Leaving it out is what remains of the disagreement: 8.7e-07 s worst case,
    against 1.8e-04 s before this rotation, so the correction removes a factor of 204 and the
    residue is nutation-scale (its amplitude is ~17 arcseconds).

    Args:
        right_ascension: J2000 right ascension in radians.
        declination: J2000 declination in radians.
        t_gps: GPS time defining the target frame.

    Returns:
        ``(right_ascension, declination)`` in the mean frame of *t_gps*, in radians.
    """
    zeta_a, z_a, theta_a = lunisolar_precession_angles(t_gps)
    cos_dec = np.cos(declination)
    sin_dec = np.sin(declination)
    cos_ra_zeta = np.cos(right_ascension + zeta_a)
    sin_ra_zeta = np.sin(right_ascension + zeta_a)

    # The three components of the rotated unit vector, in LAL's own grouping so the port can be
    # checked line against line: see the cosDelta*/sinDelta expressions in LALBarycenter.c.
    east = cos_dec * sin_ra_zeta
    north = cos_ra_zeta * np.cos(theta_a) * cos_dec - np.sin(theta_a) * sin_dec
    pole = cos_ra_zeta * np.sin(theta_a) * cos_dec + np.cos(theta_a) * sin_dec

    return float(np.arctan2(east, north) + z_a), float(np.arcsin(pole))


def precession_anchors_and_rates(t_gps: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the precession angles at *t_gps* and their time derivatives, in radians and rad/s.

    The same decomposition :func:`gmst_anchor_and_rate` uses, and for the same reason: a device
    kernel cannot evaluate a polynomial in Julian centuries per sample any more than it can call
    Astropy, but it can do a multiply-add.

    **Why a rate at all, when the angles barely move.** They move 2306 arcseconds per century, so
    across one segment the change is negligible -- which is exactly the reasoning that produced a bug.
    Evaluating them once per segment makes the sky position piecewise constant in time, and the step
    at each boundary broke continuous-wave phase coherence between segments at 1.6e-08 of peak
    against a 1e-09 tolerance. Drift *within* a segment being negligible says nothing about the
    discontinuity *between* two of them. LAL evaluates precession at each sample time; a rate
    reproduces that continuously at the cost of one multiply-add.

    Linear is not an approximation worth worrying about here: the quadratic term is 0.30 arcseconds
    per century squared, which over a day contributes 2e-11 arcseconds.

    Args:
        t_gps: GPS time at which to anchor.

    Returns:
        ``((zeta_A, z_A, theta_A), (dzeta_A, dz_A, dtheta_A))``, radians and radians per second.
    """
    centuries = (float(t_gps) - _GPS_AT_J2000) / _SECONDS_PER_JULIAN_CENTURY
    anchors = lunisolar_precession_angles(t_gps)
    # d/dt of each polynomial, converted from per-century to per-second.
    per_century = (
        (2306.2181 + (2.0 * 0.30188 + 3.0 * 0.017998 * centuries) * centuries) * _ARCSEC_TO_RAD,
        (2306.2181 + (2.0 * 1.09468 + 3.0 * 0.018203 * centuries) * centuries) * _ARCSEC_TO_RAD,
        (2004.3109 - (2.0 * 0.42665 + 3.0 * 0.041833 * centuries) * centuries) * _ARCSEC_TO_RAD,
    )
    rates = tuple(value / _SECONDS_PER_JULIAN_CENTURY for value in per_century)
    return anchors, (rates[0], rates[1], rates[2])


#: Half-interval for differencing the precessed sky position, in seconds.
#:
#: One day. Large enough that the difference is not float64 noise on two nearly equal angles, small
#: enough that the quadratic term contributes nothing: the angles' curvature is 0.30 arcseconds per
#: century squared, so over a day the second-order term is 2e-11 arcseconds.
_PRECESSION_DIFFERENCE_SECONDS = 86400.0


def precessed_sky_anchor_and_rate(
    right_ascension: float, declination: float, t_gps: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the precessed sky position at *t_gps* and how fast it moves, in radians and rad/s.

    Precession is slow -- the position drifts about 3.5e-12 rad/s -- but it is *not* constant, and
    treating it as constant per segment is what broke continuous-wave phase coherence between
    segments: each segment used a slightly different sky position, so the stitched signal stepped at
    every boundary by 1.6e-08 of peak against a 1e-09 tolerance.

    A linear model removes that. The position is continuous in absolute time, so two segments that
    abut agree exactly where they meet, and the residual curvature is fourteen orders below the
    effect being modelled.

    The rate is measured by differencing :func:`precess_to_epoch` rather than differentiating the
    rotation analytically. The derivative of an ``arctan2`` composed with three polynomials is easy
    to get subtly wrong, and there is nothing to gain: this is evaluated once per segment on the
    host.

    Args:
        right_ascension: J2000 right ascension in radians.
        declination: J2000 declination in radians.
        t_gps: GPS time to anchor at, normally the segment midpoint.

    Returns:
        ``((right_ascension, declination), (d_right_ascension, d_declination))`` -- the position in
        the mean frame of *t_gps*, and its rate in radians per second.
    """
    half = _PRECESSION_DIFFERENCE_SECONDS
    anchor = precess_to_epoch(right_ascension, declination, t_gps)
    before = precess_to_epoch(right_ascension, declination, t_gps - half)
    after = precess_to_epoch(right_ascension, declination, t_gps + half)
    # Unwrapped, so a position straddling the 2*pi seam does not produce a spurious rate.
    d_ra = float(np.unwrap([before[0], after[0]])[1] - np.unwrap([before[0], after[0]])[0]) / (2.0 * half)
    d_dec = (after[1] - before[1]) / (2.0 * half)
    return anchor, (d_ra, d_dec)
