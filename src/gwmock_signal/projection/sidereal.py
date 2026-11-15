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

The fix is not a better series but a different decomposition. Astropy takes UT1 - UTC from the
IERS tables, which give it once per day at 0h UTC, and interpolates linearly in between, so its
GMST is a straight line within each UTC day and bends at every UTC midnight, by however much the
length of day changed. :func:`gmst_piecewise` describes that exactly in a few host-computed
numbers -- the value at the segment's first sample, the rate to the next UTC midnight, and the
change of rate at that midnight and the one after -- each rate a chord between Astropy values,
and the device kernels evaluate it per sample with a multiply-add and two ramps. Astropy therefore
remains the only implementation of the sidereal model, including its IERS UT1 handling.

!!! note "Why not a line, and why not per-sample values"

    A line from the first sample, which this used to be, cannot follow the bend. Within one UTC
    day it was exact to 6e-14 rad, but after a midnight it drifted by the change in rate times the
    time since the midnight: 9.3e-11 rad (a 2e-12 s geocenter delay) at the end of one 4096 s
    segment that crosses midnight 744 s before its end. From 2005 to 2026 the change in rate was
    9.3e-14 rad/s at the median and at most 3.5e-13 rad/s; over a month of segments from GPS 1e9,
    one starting every 30 minutes, the line's worst error was 7.2e-10 rad for 4096 s segments and
    1.6e-9 rad for 8192 s, and about one segment in 21 of 4096 s crosses a midnight.

    Sidereal time evaluated on the host at every sample and passed to the kernel as an array is
    as accurate, but on one platform (macOS on Apple silicon) the compiled projection of a sample
    can depend on the length the kernel was compiled for: a burst padded at its end projected
    differently from the unpadded one in about a tenth of its samples, by up to 2.2e-13 of its
    peak, although the sidereal times it read were identical. The same happened when the bends
    were added to every kernel, though they add exactly zero before a midnight. The compiled
    arithmetic is not guaranteed to be independent of the length; the line alone, as it was, is
    the form that has been checked to be. So the device kernels add the bends only when the
    caller's samples cross a UTC midnight, and are otherwise unchanged; the piecewise form also
    needs no per-sample memory.

    Measured against Astropy (sampled every 0.05 s or 0.5 s), the piecewise form's largest error
    was 8.0e-14 rad in segments crossing every seventh UTC midnight from 2005 to 2026, 8.7e-14 rad
    in 4096 s segments crossing every thirtieth, 7.5e-14 rad in 4096 s segments crossing none, at
    most 5.0e-14 rad across the five midnights after a leap second, and 5.9e-14 rad over a whole
    day: a geocenter delay below 2e-15 s. ``tests/projection/test_sidereal.py`` pins a midnight,
    a leap-second midnight and a full day.

Using Astropy also removes the ``dut1`` approximation rather than merely exposing it:
Astropy applies real IERS UT1-UTC, whereas the on-device series defaulted to ``dut1 = 0``
and so carried up to 0.9 s of sidereal error, worth 6.7e-4 in the antenna pattern.

!!! note "UT1, deliberately, and not the pipelines' UTC"

    The Earth's orientation is set by the Earth Rotation Angle, which IAU 2000 Resolution B1.8
    defines as a linear function of UT1, and sidereal time is that angle plus the accumulated
    precession (Kaplan 2005, USNO Circular 179, eq. 2.12; arXiv:astro-ph/0602086). UT1 is the
    measured rotation; UTC is kept within 0.9 s of it by leap seconds. Sidereal time here therefore
    uses the IERS UT1-UTC, so the detectors rotate with the Earth's measured rotation. It is the mean
    sidereal time: nutation is not modelled (see :func:`precess_to_epoch`).

    LAL's ``XLALGreenwichMeanSiderealTime``, which PyCBC and Bilby analyses use, converts GPS time to
    UTC and evaluates its own series with UTC in place of UT1. It agrees with the IAU 2006 series at
    UT1 = UTC to 1e-7 rad, so the difference from this module is that substitution: LAL's exceeds
    this module's by minus UT1-UTC times the Earth's rotation rate (+2.2e-5 rad at GPS 1e9, where
    UT1-UTC is -0.305 s), plus the 1e-7 rad between the two series. Sidereal time enters
    the projection only through the hour angle, so at any one time that difference is a shift of
    right ascension -- of date, when the source direction is precessed: such a pipeline recovers the
    right ascension offset by it. The shift follows UT1-UTC, which drifts by about 0.7 ms a day near
    GPS 1e9 (4e-9 rad across a 1.8-hour signal) and changes by 1 s at a leap second.
"""

from __future__ import annotations

# Imported on first use from Python 3.15 (PEP 810), and as usual before it: astropy.time is most
# of the cost of importing the projection, and only the GMST functions below need it.
__lazy_modules__ = ["astropy.time"]

import numpy as np
from astropy.time import Time

#: Baseline for the finite-difference rate estimate, in seconds. Long enough that Astropy's
#: own round-off does not dominate the slope, short enough that GMST advances only
#: ~0.044 rad across it -- which is what makes the wrap correction below unambiguous.
_RATE_BASELINE_SECONDS = 600.0

#: UTC midnights after a segment's first sample at which :func:`gmst_piecewise` lets sidereal time bend.
#: Two cover any segment shorter than a day from wherever it starts, and the device path accepts no
#: span longer than a day.
GMST_KINKS = 2

#: Earth's sidereal rotation rate, used only to count whole turns between two Astropy values.
_NOMINAL_RATE_RAD_PER_SECOND = 2.0 * np.pi / 86164.0905


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


def gmst_piecewise(
    start_times: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each segment's sidereal time as a line that bends at the next :data:`GMST_KINKS` UTC midnights.

    Sidereal time ``t`` seconds after a segment's first sample is
    ``anchor + rate * t + sum_k change[k] * max(t - kink[k], 0)``: a chord between Astropy's values at
    the first sample and the next UTC midnight, then chords from midnight to midnight. Astropy's GMST
    is a straight line between UTC midnights (module docstring), so the chords follow it to its own
    round-off. Every number depends on the first sample's time alone, not on how long the segment is,
    so a series padded at its end gets exactly the sidereal time of the unpadded one.

    The model covers ``valid_until`` seconds from each start, the :data:`GMST_KINKS` + 1-th midnight:
    more than a day.

    Args:
        start_times: GPS time of each segment's first sample, scalar or shape ``(n_segments,)``.

    Returns:
        ``(anchors, rates, kinks, changes, valid_until)``: anchors and rates of shape ``(n_segments,)``
        (GMST in radians at the first sample; rad/s), kink offsets in seconds and rate changes in rad/s
        of shape ``(n_segments, GMST_KINKS)``, and the covered span in seconds, ``(n_segments,)``.
    """
    starts = np.atleast_1d(np.asarray(start_times, dtype=float))
    first = Time(starts, format="gps", scale="utc", location=(0, 0))
    midnight = np.floor(first.utc.mjd)
    # The next GMST_KINKS + 1 midnights, as offsets from each first sample: Time differences, not GPS
    # floats, which near 1e9 s would round to 1.2e-7 s.
    days = midnight[:, None] + 1.0 + np.arange(GMST_KINKS + 1)[None, :]
    offsets = (Time(days, format="mjd", scale="utc") - first[:, None]).sec
    bounds = np.concatenate([np.zeros((starts.size, 1)), offsets], axis=1)
    shape = bounds.shape
    instants = Time(np.broadcast_to(starts[:, None], shape), bounds, format="gps", scale="utc", location=(0, 0))
    raw = np.asarray(instants.sidereal_time("mean").rad, dtype=float)
    # Whole turns between consecutive values, counted from the nominal rotation rate: the values are
    # up to a day apart, too far for an unwrap to tell a turn from a step.
    steps = np.diff(raw, axis=1)
    steps = steps + 2.0 * np.pi * np.round(
        (_NOMINAL_RATE_RAD_PER_SECOND * np.diff(bounds, axis=1) - steps) / (2.0 * np.pi)
    )
    rates = steps / np.diff(bounds, axis=1)
    return raw[:, 0], rates[:, 0], offsets[:, :GMST_KINKS], np.diff(rates, axis=1), offsets[:, GMST_KINKS]


#: Seconds per Julian century, for the precession polynomials.
_SECONDS_PER_JULIAN_CENTURY = 36525.0 * 86400.0

#: GPS seconds at the J2000.0 epoch (2000-01-01T12:00:00 TT), the origin of those polynomials.
_GPS_AT_J2000 = 630763213.0

#: Arcseconds to radians, written as LAL writes it (pi / 6.48e5) so the two can be compared literally.
_ARCSEC_TO_RAD = np.pi / 6.48e5


def lunisolar_precession_angles(
    t_gps: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return the three lunisolar precession angles ``(zeta_A, z_A, theta_A)`` in radians.

    Equations 3.212 of the *Explanatory Supplement to the Astronomical Almanac*, which is what
    LAL's ``XLALBarycenterEarth`` evaluates as ``tzeA``, ``zA`` and ``thetaA``. The polynomials are
    reproduced rather than taken from Astropy on purpose: the reference this pipeline is validated
    against is LAL, and Astropy's IAU 2006 model differs from these truncated series at the
    milliarcsecond level. Agreeing with the reference beats agreeing with a better model that the
    reference does not use.

    Args:
        t_gps: GPS time in seconds, scalar or array. Arrays are needed by the batched path, which
            precesses a whole catalogue of events at their own segment epochs.

    Returns:
        ``(zeta_A, z_A, theta_A)`` in radians, shaped like *t_gps*.
    """
    centuries = (np.asarray(t_gps, dtype=float) - _GPS_AT_J2000) / _SECONDS_PER_JULIAN_CENTURY
    zeta_a = centuries * (2306.2181 + (0.30188 + 0.017998 * centuries) * centuries) * _ARCSEC_TO_RAD
    z_a = centuries * (2306.2181 + (1.09468 + 0.018203 * centuries) * centuries) * _ARCSEC_TO_RAD
    theta_a = centuries * (2004.3109 - (0.42665 + 0.041833 * centuries) * centuries) * _ARCSEC_TO_RAD
    return zeta_a, z_a, theta_a


def precess_to_epoch(
    right_ascension: np.ndarray | float, declination: np.ndarray | float, t_gps: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
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
        right_ascension: J2000 right ascension in radians, scalar or array.
        declination: J2000 declination in radians, scalar or array.
        t_gps: GPS time defining the target frame, scalar or array. All three broadcast together,
            which is what lets the batched path precess a catalogue of events at their own epochs
            through the same code the single-event path uses.

    Returns:
        ``(right_ascension, declination)`` in the mean frame of *t_gps*, in radians, as arrays
        (0-d for scalar input).
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

    return np.arctan2(east, north) + z_a, np.arcsin(pole)


#: Half-interval for differencing the precessed sky position, in seconds.
#:
#: One day. Large enough that the difference is not float64 noise on two nearly equal angles, small
#: enough that the quadratic term contributes nothing: the angles' curvature is 0.30 arcseconds per
#: century squared, so over a day the second-order term is 2e-11 arcseconds.
_PRECESSION_DIFFERENCE_SECONDS = 86400.0


def precessed_sky_anchor_and_rate(
    right_ascension: np.ndarray | float, declination: np.ndarray | float, t_gps: np.ndarray | float
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
        right_ascension: J2000 right ascension in radians, scalar or array.
        declination: J2000 declination in radians, scalar or array.
        t_gps: GPS time to anchor at, scalar or array. One entry per event on the batched path,
            where each event has its own segment.

    Returns:
        ``((right_ascension, declination), (d_right_ascension, d_declination))`` -- the position in
        the mean frame of *t_gps*, and its rate in radians per second, as arrays (0-d for scalar
        input).
    """
    half = _PRECESSION_DIFFERENCE_SECONDS
    t_gps = np.asarray(t_gps, dtype=float)
    anchor = precess_to_epoch(right_ascension, declination, t_gps)
    before_ra, before_dec = precess_to_epoch(right_ascension, declination, t_gps - half)
    after_ra, after_dec = precess_to_epoch(right_ascension, declination, t_gps + half)
    # Wrapped into (-pi, pi] before dividing, so a position straddling the 2*pi seam -- where
    # `arctan2` returns values at opposite ends of its range for two epochs a day apart -- gives the
    # small true rate rather than a spurious 2*pi/(2 days). Elementwise, so it holds for every entry
    # of a batch independently.
    d_ra = (np.mod(after_ra - before_ra + np.pi, 2.0 * np.pi) - np.pi) / (2.0 * half)
    d_dec = (after_dec - before_dec) / (2.0 * half)
    return anchor, (d_ra, d_dec)
