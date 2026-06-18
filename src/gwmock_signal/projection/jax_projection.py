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
"""JAX building blocks for on-device detector projection.

These are the traceable counterparts of the NumPy/Astropy helpers in
:mod:`gwmock_signal.projection.network`, used by the planned on-device (GPU)
simulation path. This module imports JAX at import time, so it must only be
imported by the device path or its tests, never by always-loaded modules
(the package must still import without the optional ``[jax]`` extra).

!!! important "Double precision required"

    GPS times are ~1e9 and Julian dates ~2.45e6, so these helpers require JAX's
    64-bit mode (``jax.config.update("jax_enable_x64", True)``). The ripple
    backend enables this on construction; callers of this module must ensure it
    is enabled or results are silently wrong in float32.

!!! note "Evaluate once when Earth rotation is off"

    For ``earth_rotation=False`` the sidereal time is the same for the whole
    segment, so callers should evaluate :func:`gmst_rad` **once** at the
    reference time (a scalar) rather than per sample. Only the time-dependent
    (``earth_rotation=True``) path passes an array of times.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike

# Julian Date of the GPS epoch (1980-01-06 00:00:00 UTC).
_JD_GPS_EPOCH = 2444244.5
# Julian Date of J2000.0.
_JD_J2000 = 2451545.0
_SECONDS_PER_DAY = 86400.0
_JULIAN_CENTURY_DAYS = 36525.0
# GPS = TAI - 19 s (fixed); TT = TAI + 32.184 s (fixed).
_GPS_MINUS_TAI = -19.0
_TT_MINUS_TAI = 32.184
# Default TAI - UTC (leap seconds), valid from 2017-01-01. Step IERS data: pass an
# explicit value for other epochs. UTC = GPS - (tai_minus_utc + _GPS_MINUS_TAI).
_DEFAULT_TAI_MINUS_UTC = 37.0
_ARCSEC_TO_RAD = math.pi / 648000.0
# Speed of light (exact, SI); matches astropy.constants.c.value.
_SPEED_OF_LIGHT_M_S = 299792458.0


def gmst_rad(
    t_gps: ArrayLike,
    *,
    tai_minus_utc: float = _DEFAULT_TAI_MINUS_UTC,
    dut1: float = 0.0,
) -> Array:
    """Greenwich Mean Sidereal Time in radians (IAU 2006), as a JAX array.

    Mirrors Astropy's ``Time(..., format="gps").sidereal_time("mean")`` using the
    IAU 2006 model: the Earth Rotation Angle plus the sidereal precession
    polynomial. Fed the leap-second count and ``dut1`` that Astropy uses, it
    reproduces Astropy's GMST to better than 1e-6 rad.

    Args:
        t_gps: GPS time(s) in seconds. Scalar or any-shaped array.
        tai_minus_utc: Leap-second offset TAI - UTC in seconds. Step IERS data;
            the default (37 s) is valid from 2017-01-01. Pass the value for the
            relevant epoch for sub-millisecond accuracy.
        dut1: UT1 - UTC in seconds (IERS, |dut1| < 0.9 s). Defaults to 0; with
            the default the result is accurate to ~1e-4 rad (the neglected dut1).
            Pass Astropy's ``delta_ut1_utc`` to anchor to ~1e-6 rad.

    Returns:
        GMST in radians wrapped to ``[0, 2*pi)``, same shape as ``t_gps``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    t_gps = jnp.asarray(t_gps, dtype=jnp.float64)
    utc_seconds = t_gps - (tai_minus_utc + _GPS_MINUS_TAI)
    jd_utc = _JD_GPS_EPOCH + utc_seconds / _SECONDS_PER_DAY
    jd_ut1 = jd_utc + dut1 / _SECONDS_PER_DAY
    jd_tt = jd_utc + (tai_minus_utc + _TT_MINUS_TAI) / _SECONDS_PER_DAY

    # Earth Rotation Angle (the dominant, linear-in-UT1 term).
    tu = jd_ut1 - _JD_J2000
    era = 2.0 * jnp.pi * jnp.mod(0.7790572732640 + 1.00273781191135448 * tu, 1.0)

    # Sidereal precession polynomial (arcseconds), evaluated in TT centuries.
    t_cent = (jd_tt - _JD_J2000) / _JULIAN_CENTURY_DAYS
    poly_arcsec = (
        0.014506
        + 4612.156534 * t_cent
        + 1.3915817 * t_cent**2
        - 0.00000044 * t_cent**3
        - 0.000029956 * t_cent**4
        - 0.0000000368 * t_cent**5
    )

    return jnp.mod(era + poly_arcsec * _ARCSEC_TO_RAD, 2.0 * jnp.pi)


def antenna_pattern(
    response: ArrayLike,
    gmst: ArrayLike,
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
) -> tuple[Array, Array]:
    """Tensor antenna-pattern factors (F+, Fx) for one detector, as JAX arrays.

    Traceable counterpart of ``_antenna_pattern_lal`` in
    :mod:`gwmock_signal.projection.network`, using the same formula but taking the
    detector response tensor and sidereal time as inputs (single source of truth:
    ``response`` from :func:`gwmock_signal.projection.geometry.reconstructed_geometry`,
    ``gmst`` from :func:`gmst_rad`).

    For ``earth_rotation=False`` callers pass a scalar ``gmst`` (evaluated once at
    the reference time); the time-dependent path passes an array (e.g. via ``vmap``).

    Args:
        response: 3x3 detector response tensor.
        gmst: Greenwich Mean Sidereal Time in radians (scalar or array).
        right_ascension: Source right ascension in radians.
        declination: Source declination in radians.
        polarization_angle: Polarization angle psi in radians.

    Returns:
        ``(f_plus, f_cross)``, each the shape of ``gmst``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    response = jnp.asarray(response, dtype=jnp.float64)
    gha = jnp.asarray(gmst, dtype=jnp.float64) - right_ascension
    ones = jnp.ones_like(gha)  # broadcast the gha-independent z-component to gmst's shape
    cosgha, singha = jnp.cos(gha), jnp.sin(gha)
    cosdec, sindec = jnp.cos(declination), jnp.sin(declination)
    cospsi, sinpsi = jnp.cos(polarization_angle), jnp.sin(polarization_angle)

    x = jnp.stack(
        [
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec * ones,
        ]
    )
    y = jnp.stack(
        [
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec * ones,
        ]
    )
    dx = response @ x
    dy = response @ y
    f_plus = jnp.sum(x * dx - y * dy, axis=0)
    f_cross = jnp.sum(x * dy + y * dx, axis=0)
    return f_plus, f_cross


def time_delay_from_geocenter(
    location: ArrayLike,
    gmst: ArrayLike,
    *,
    right_ascension: float,
    declination: float,
) -> Array:
    """Geocenter-to-detector time delay (seconds) for one detector, as a JAX array.

    Traceable counterpart of ``_time_delay_from_earth_center_lal`` in
    :mod:`gwmock_signal.projection.network`, using the same formula but taking the
    detector location and sidereal time as inputs. As with :func:`antenna_pattern`,
    pass a scalar ``gmst`` for ``earth_rotation=False``.

    Args:
        location: Earth-fixed detector position in metres (3-vector).
        gmst: Greenwich Mean Sidereal Time in radians (scalar or array).
        right_ascension: Source right ascension in radians.
        declination: Source declination in radians.

    Returns:
        Time delay in seconds, the shape of ``gmst``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    location = jnp.asarray(location, dtype=jnp.float64)
    gha = jnp.asarray(gmst, dtype=jnp.float64) - right_ascension
    cosdec, sindec = jnp.cos(declination), jnp.sin(declination)
    propagation_direction = jnp.stack(
        [
            cosdec * jnp.cos(gha),
            -cosdec * jnp.sin(gha),
            sindec * jnp.ones_like(gha),
        ]
    )
    return -jnp.tensordot(location, propagation_direction, axes=1) / _SPEED_OF_LIGHT_M_S
