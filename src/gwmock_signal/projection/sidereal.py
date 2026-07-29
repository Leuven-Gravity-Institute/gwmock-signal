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
