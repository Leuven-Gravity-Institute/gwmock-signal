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


def project_polarizations_fd(  # noqa: PLR0913
    frequencies: ArrayLike,
    plus: ArrayLike,
    cross: ArrayLike,
    *,
    f_plus: ArrayLike,
    f_cross: ArrayLike,
    time_delay: ArrayLike,
    n_samples: int,
    sampling_frequency: float,
) -> Array:
    """Project frequency-domain polarizations onto one detector, returning strain in time.

    Forms the detector response ``F+ h+ + Fx hx`` in the frequency domain, applies the
    geocenter-to-detector delay as the exact phase shift ``exp(-2j pi f tau)``, and
    inverse real-FFTs to the time-domain strain. This is the on-device counterpart of
    the ``earth_rotation=False`` branch of
    :func:`gwmock_signal.projection.network.project_polarizations_to_network`.

    The coalescence stays where the input ``plus``/``cross`` place it (``t = 0`` for
    :class:`~gwmock_signal.waveform.backends.ripple.FrequencyDomainPolarizations`); the
    caller positions it within the analysis segment.

    Args:
        frequencies: One-sided frequency grid in Hz, shape ``(n_samples // 2 + 1,)``.
        plus: Frequency-domain plus polarization on ``frequencies``.
        cross: Frequency-domain cross polarization on ``frequencies``.
        f_plus: Plus antenna-pattern factor (scalar for ``earth_rotation=False``).
        f_cross: Cross antenna-pattern factor.
        time_delay: Geocenter-to-detector delay in seconds.
        n_samples: Length of the real time series the inverse FFT produces.
        sampling_frequency: Sample rate in Hz (the ``irfft`` is scaled by it, i.e. ``/dt``).

    Returns:
        The time-domain detector strain, shape ``(n_samples,)``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    frequencies = jnp.asarray(frequencies)
    strain_f = f_plus * jnp.asarray(plus) + f_cross * jnp.asarray(cross)
    strain_f = strain_f * jnp.exp(-2j * jnp.pi * frequencies * time_delay)
    return jnp.fft.irfft(strain_f, n=n_samples) * sampling_frequency


def _interpolate_uniform_cubic(samples: ArrayLike, index: ArrayLike, n_samples: int) -> Array:
    """Catmull-Rom cubic interpolation of a uniformly sampled series.

    ``index`` is a fractional sample index into ``samples``. Positions outside
    ``[0, n_samples - 1]`` return zero, matching the ``bounds_error=False,
    fill_value=0.0`` behaviour of the SciPy interpolation used by the NumPy path.

    Catmull-Rom is used rather than SciPy's natural cubic spline because the latter is
    a global tridiagonal solve, which is sequential and therefore poorly suited to a
    device kernel. Both are C1 cubics with O(h^4) error on smooth data; for the
    band-limited, heavily oversampled strain here the difference is far below the
    interpolation error itself, which ``tests/projection/test_jax_projection.py``
    pins against the NumPy path.

    Args:
        samples: Uniformly sampled series, shape ``(n_samples,)``.
        index: Fractional sample positions to evaluate at, any shape.
        n_samples: Length of ``samples``.

    Returns:
        Interpolated values, the shape of ``index``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    samples = jnp.asarray(samples, dtype=jnp.float64)
    index = jnp.asarray(index, dtype=jnp.float64)

    base = jnp.floor(index)
    frac = index - base
    base_int = base.astype(jnp.int32)

    def _at(offset: int) -> Array:
        # Clamp the gather so out-of-range reads stay in bounds; the values they
        # produce are discarded by the mask below.
        return samples[jnp.clip(base_int + offset, 0, n_samples - 1)]

    p_prev, p0, p1, p_next = _at(-1), _at(0), _at(1), _at(2)
    interpolated = 0.5 * (
        2.0 * p0
        + (p1 - p_prev) * frac
        + (2.0 * p_prev - 5.0 * p0 + 4.0 * p1 - p_next) * frac**2
        + (p_next - 3.0 * p1 + 3.0 * p0 - p_prev) * frac**3
    )
    in_range = (index >= 0.0) & (index <= n_samples - 1)
    return jnp.where(in_range, interpolated, 0.0)


def project_polarizations_td_rotating(  # noqa: PLR0913
    plus: ArrayLike,
    cross: ArrayLike,
    *,
    response: ArrayLike,
    location: ArrayLike,
    start_time: float,
    sampling_frequency: float,
    n_samples: int,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    tai_minus_utc: float = _DEFAULT_TAI_MINUS_UTC,
    dut1: float = 0.0,
) -> Array:
    """Project time-domain polarizations with a time-dependent antenna pattern.

    The on-device counterpart of the ``earth_rotation=True`` branch of
    :func:`gwmock_signal.projection.network.project_polarizations_to_network`, and the
    algorithm is deliberately identical to it, step for step: the geocenter delay and
    the antenna-pattern factors are evaluated **per sample**, the polarizations are
    resampled at the delayed times, and the two are combined as
    ``F+(t) h+(t - tau(t)) + Fx(t) hx(t - tau(t))``.

    This matters for long signals. Earth turns 15 degrees per hour, so over the
    2048 s (10 Hz) to 16384 s (5 Hz) segments a binary-neutron-star inspiral occupies
    in the Einstein Telescope band, the detector sweeps tens of degrees. Evaluating the
    response once at the segment midpoint — all the frequency-domain path can do,
    since a time-varying response is not a frequency-domain multiply — is not a small
    approximation there, it is the wrong answer.

    !!! warning "Oversample the strain"

        Resampling at the delayed times is a cubic interpolation, so its error grows
        steeply as the signal approaches Nyquist and is worst at the merger, where the
        waveform peaks. Against the NumPy path the largest sample-wise difference falls
        roughly 8x per doubling of the sample rate. This is a property of the algorithm,
        not of this implementation — ``project_polarizations_to_network`` carries the
        same error — but it means the sample rate should be chosen well above the
        signal's highest frequency, not merely above it.

    !!! note "Delay convention follows the NumPy path"

        The delay applied to the strain uses the sidereal time at ``t``, while the
        antenna pattern is evaluated at ``t + tau(t)``. That asymmetry is inherited
        from ``project_polarizations_to_network`` so the two paths agree; it is a
        sub-sample effect (``|tau| <= 21 ms``) on a quantity that varies on an hourly
        timescale.

    Args:
        plus: Time-domain plus polarization, shape ``(n_samples,)``.
        cross: Time-domain cross polarization, shape ``(n_samples,)``.
        response: 3x3 detector response tensor.
        location: Earth-fixed detector position in metres (3-vector).
        start_time: GPS time of the first sample.
        sampling_frequency: Sample rate in Hz.
        n_samples: Number of samples.
        right_ascension: Source right ascension in radians.
        declination: Source declination in radians.
        polarization_angle: Polarization angle psi in radians.
        tai_minus_utc: Leap-second offset passed through to :func:`gmst_rad`.
        dut1: UT1 - UTC passed through to :func:`gmst_rad`.

    Returns:
        The time-domain detector strain, shape ``(n_samples,)``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    dt = 1.0 / sampling_frequency
    times = start_time + jnp.arange(n_samples, dtype=jnp.float64) * dt

    gmst = gmst_rad(times, tai_minus_utc=tai_minus_utc, dut1=dut1)
    time_delays = time_delay_from_geocenter(location, gmst, right_ascension=right_ascension, declination=declination)

    gmst_antenna = gmst_rad(times + time_delays, tai_minus_utc=tai_minus_utc, dut1=dut1)
    f_plus, f_cross = antenna_pattern(
        response,
        gmst_antenna,
        right_ascension=right_ascension,
        declination=declination,
        polarization_angle=polarization_angle,
    )

    # Fractional sample index of t - tau(t) on the uniform input grid.
    index = jnp.arange(n_samples, dtype=jnp.float64) - time_delays * sampling_frequency
    plus_shifted = _interpolate_uniform_cubic(plus, index, n_samples)
    cross_shifted = _interpolate_uniform_cubic(cross, index, n_samples)

    return f_plus * plus_shifted + f_cross * cross_shifted
