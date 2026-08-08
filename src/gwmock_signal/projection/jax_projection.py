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

from gwmock_signal.projection.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    SPEED_OF_LIGHT_M_S,
    edge_padding,
    kaiser_window_chebyshev,
    validate_kernel,
)

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
        right_ascension: Source right ascension in radians, in the **mean equator and equinox of
            date** rather than J2000. `project_polarizations_to_network` applies that rotation once
            before dispatching here; a direct caller must do the same, or the sidereal angle and the
            sky position refer to different frames -- worth 3.7% of peak strain when it was missed.
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
        right_ascension: Source right ascension in radians, in the **mean equator and equinox of
            date** rather than J2000. `project_polarizations_to_network` applies that rotation once
            before dispatching here; a direct caller must do the same, or the sidereal angle and the
            sky position refer to different frames -- worth 3.7% of peak strain when it was missed.
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
    return -jnp.tensordot(location, propagation_direction, axes=1) / SPEED_OF_LIGHT_M_S


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


def _interpolate_uniform_sinc(
    samples: ArrayLike,
    index: ArrayLike,
    n_samples: int,
    *,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> Array:
    """Kaiser-windowed sinc resampling of a uniformly sampled series.

    The device counterpart of
    :func:`gwmock_signal.projection.resampling.resample_uniform_sinc`; both must use the
    same kernel or the two projection paths disagree by more than either one's own error,
    so the tap count and window parameter are imported from that module rather than
    restated here.

    For a band-limited uniformly sampled signal the sinc series is the exact interpolant,
    so accuracy is set by the tap count and can be refined until it stops mattering. A
    cubic cannot be refined at all.

    Positions whose centre index falls outside ``[0, n_samples - 1]`` return zero. Taps that
    reach outside while the centre is inside are *clamped* to the first or last sample, which
    repeats it -- so this helper alone does not zero-pad. Callers that need true zero padding,
    including :func:`project_polarizations_td_rotating`, pass an already-padded array and an
    index offset by that padding; see
    :func:`gwmock_signal.projection.resampling.edge_padding`.

    The taps are accumulated in a ``fori_loop`` rather than gathered into one
    ``(taps, ...)`` array: at Einstein Telescope BNS segment lengths a 127-tap gather
    would need over a hundred arrays of ``n_samples`` live at once, and device memory —
    not arithmetic — is the binding constraint on this path.

    Args:
        samples: Uniformly sampled series, shape ``(n_samples,)``.
        index: Fractional sample positions to evaluate at, any shape.
        n_samples: Length of ``samples``.
        taps: Number of kernel taps; odd.
        beta: Kaiser window shape parameter.

    Returns:
        Interpolated values, the shape of ``index``.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.scipy.special import i0  # noqa: PLC0415

    taps, beta = validate_kernel(taps, beta)
    samples = jnp.asarray(samples, dtype=jnp.float64)
    index = jnp.asarray(index, dtype=jnp.float64)
    half = (taps - 1) // 2
    denominator = half + 1.0
    normalisation = i0(jnp.asarray(beta, dtype=jnp.float64))

    # The window is `i0(beta * sqrt(1 - v)) / i0(beta)` with `v = (x / denominator) ** 2`, evaluated
    # once per tap. Because `i0` is even, that composition is analytic in `v`, so a polynomial in `v`
    # replaces both the `i0` and the `sqrt`: measured 2.18x on this kernel in float64, with the error
    # against an analytic sinusoid unchanged at 4.027e-12 of peak. `beta` is static here, so the fit
    # happens on the host once per distinct beta. `None` means no degree reached the accuracy target,
    # and then `i0` is kept rather than a worse window accepted.
    window_coefficients = kaiser_window_chebyshev(float(beta))
    coefficients = None if window_coefficients is None else jnp.asarray(window_coefficients, dtype=jnp.float64)

    def _window(x: Array) -> Array:
        if coefficients is None:
            return i0(beta * jnp.sqrt(jnp.maximum(0.0, 1.0 - (x / denominator) ** 2))) / normalisation
        # Clamped at 1 to mirror the exact form's `maximum(0.0, 1 - v)`, and **unreachable by
        # construction**: `x = frac - offset` with `frac` in [0, 1) and `offset` in [-half, half]
        # gives `|x| < half + 1 = denominator`, so `v < 1` strictly. Verified by mutation -- removing
        # the clamp changes no test result, because no input reaches it. Kept so the two
        # implementations read alike, not because it guards a live case; a reader who assumes it is
        # load-bearing would be wrong.
        #
        # Clenshaw in the Chebyshev basis, never monomials: converting a degree-24 fit and using
        # Horner was measured to cost three orders of accuracy (1.9e-14 against 1.8e-11).
        v = jnp.minimum(1.0, (x / denominator) ** 2)
        t = 2.0 * v - 1.0
        b_kp1 = jnp.zeros_like(t)
        b_kp2 = jnp.zeros_like(t)
        for coefficient in coefficients[:0:-1]:
            b_kp1, b_kp2 = 2.0 * t * b_kp1 - b_kp2 + coefficient, b_kp1
        return t * b_kp1 - b_kp2 + coefficients[0]

    base = jnp.floor(index)
    frac = index - base
    base_int = base.astype(jnp.int32)

    def _accumulate(step: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        total, weight_sum = carry
        offset = step - half
        x = frac - offset
        weight = jnp.sinc(x) * _window(x)
        gathered = samples[jnp.clip(base_int + offset, 0, n_samples - 1)]
        return total + weight * gathered, weight_sum + weight

    zeros = jnp.zeros_like(index)
    total, weight_sum = jax.lax.fori_loop(0, taps, _accumulate, (zeros, zeros))

    # Normalised to unit DC gain, matching the NumPy implementation.
    interpolated = jnp.where(weight_sum != 0.0, total / weight_sum, 0.0)
    in_range = (index >= 0.0) & (index <= n_samples - 1)
    return jnp.where(in_range, interpolated, 0.0)


def project_polarizations_td_rotating(  # noqa: PLR0913
    plus: ArrayLike,
    cross: ArrayLike,
    *,
    response: ArrayLike,
    location: ArrayLike,
    sampling_frequency: float,
    n_samples: int,
    # `ArrayLike`, not `float`, for everything `jax_batch` maps over: under its `jax.vmap` these
    # arrive as traced per-event arrays, so `float` describes only the single-event caller. Kept as
    # `float` where the batched path passes one unmapped value -- `gmst_rate` is shared across a batch.
    right_ascension: ArrayLike,
    declination: ArrayLike,
    right_ascension_rate: ArrayLike,
    declination_rate: ArrayLike,
    polarization_angle: ArrayLike,
    gmst_start: ArrayLike,
    gmst_rate: float,
    extra_shift_samples: ArrayLike = 0.0,
    sinc_taps: int = DEFAULT_SINC_TAPS,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
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

    !!! note "Edge support is zero-padded, not clipped"

        The delay and the sinc kernel both reach outside the input buffer near its ends. The
        polarizations are therefore gathered from a zero-padded copy, so out-of-range taps
        contribute zero. Without that they would clip to the first or last sample and *repeat*
        it, which distorts the edges by an amount that depends on how large the waveform is
        there -- fine for a tapered inspiral, wrong for a waveform with abrupt support, and this
        primitive is waveform-agnostic. The width comes from
        :func:`~gwmock_signal.projection.resampling.edge_padding`, which both projection paths
        call so neither can pad differently from the other. It covers the largest geocenter
        delay *any ground-based detector* can have -- Earth's equatorial radius over c, not this
        detector's own distance, because ``location`` is a traced argument here and making it
        static would cost one compiled kernel per detector -- plus the kernel half-width and the
        sub-sample alignment shift. ``require_terrestrial_location`` enforces that assumption on
        the host, where the geometry is still concrete.

    !!! warning "Oversample the strain"

        Resampling at the delayed times uses the Kaiser-windowed sinc kernel in
        :mod:`gwmock_signal.projection.resampling`, which is the exact interpolant for a
        band-limited series in the limit of many taps. Its error is nonetheless a steep
        function of how close the signal comes to Nyquist: the shipped kernel reaches ~1e-12
        up to about 0.8 x Nyquist and degrades sharply above that. The sample rate should
        therefore be chosen well above the signal's highest frequency, not merely above it.
        ``project_polarizations_to_network`` uses the same kernel and has the same limit.

    !!! note "Time coordinate"

        The output series is labelled in the same time coordinate as the input, and the
        detector strain at sample ``t`` is ``F(t) h_geo(t - tau(t))``: the antenna pattern
        is evaluated at ``t``, and the polarizations are resampled at ``t - tau(t)``.
        Taking ``t_geo = t - tau(t)`` is the first Newton step of the exact relation
        ``t = t_geo + tau(t_geo)``; the residual is of order ``tau * dtau/dt`` (about
        3e-8 s, i.e. 6e-5 samples at 2048 Hz), so no iteration is performed.

    Args:
        plus: Time-domain plus polarization, shape ``(n_samples,)``.
        cross: Time-domain cross polarization, shape ``(n_samples,)``.
        response: 3x3 detector response tensor.
        location: Earth-fixed detector position in metres (3-vector).
        sampling_frequency: Sample rate in Hz.
        n_samples: Number of samples.
        right_ascension: Source right ascension in radians, in the **mean equator and equinox of
            date** rather than J2000. `project_polarizations_to_network` applies that rotation once
            before dispatching here; a direct caller must do the same, or the sidereal angle and the
            sky position refer to different frames -- worth 3.7% of peak strain when it was missed.
        declination: Source declination in radians.
        polarization_angle: Polarization angle psi in radians.
        gmst_start: Greenwich Mean Sidereal Time in radians at the first sample, computed on
            the host by :func:`~gwmock_signal.projection.sidereal.gmst_anchor_and_rate`.
            Supplied rather than computed here so that Astropy remains the only
            implementation of the sidereal model; see that module for why a linear model
            is exact at these segment lengths.
        gmst_rate: dGMST/dt in radians per second.
        right_ascension_rate: How fast the precessed right ascension moves, in radians per second,
            from :func:`~gwmock_signal.projection.sidereal.precessed_sky_anchor_and_rate`. Required
            rather than defaulted, because a caller that forgets it gets a position frozen for the
            whole segment -- which steps at every segment boundary and broke continuous-wave phase
            coherence at 1.6e-08 of peak against a 1e-09 tolerance. Anchored at the first sample, the
            same origin as ``gmst_start``.
        declination_rate: The same for declination.
        extra_shift_samples: Additional shift, in samples, applied together with the
            geocenter delay. Used to land the output on a caller's sample lattice; because it
            joins the delay inside one resampling, the alignment costs no extra interpolation
            and inherits the kernel's accuracy rather than a downstream cubic's.
        sinc_taps: Taps in the resampling kernel. More taps cost arithmetic and buy
            accuracy; the default is set by measured convergence.
        kaiser_beta: Kaiser window shape parameter for the resampling kernel.

    Returns:
        The time-domain detector strain, shape ``(n_samples,)``.
    """
    import jax.numpy as jnp  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    dt = 1.0 / sampling_frequency
    sample_offsets = jnp.arange(n_samples, dtype=jnp.float64) * dt

    # GMST from the host-supplied anchor and rate. Deliberately left unwrapped: only its
    # sine and cosine are used, and wrapping would put a discontinuity mid-segment.
    gmst = gmst_start + gmst_rate * sample_offsets

    # The sky position moves too, because precession is a rotation into the frame *of date* and the
    # date advances across the segment. Linear in absolute time, so two abutting segments agree
    # exactly where they meet; frozen per segment they do not, and the step is what a continuous-wave
    # coherence test sees.
    right_ascension_of_date = right_ascension + right_ascension_rate * sample_offsets
    declination_of_date = declination + declination_rate * sample_offsets

    time_delays = time_delay_from_geocenter(
        location, gmst, right_ascension=right_ascension_of_date, declination=declination_of_date
    )

    # Antenna pattern at the detector-time sample, i.e. the time coordinate the output
    # series is labelled with. Evaluating it at t + tau would mix the detector and
    # geocenter time coordinates.
    f_plus, f_cross = antenna_pattern(
        response,
        gmst,
        right_ascension=right_ascension_of_date,
        declination=declination_of_date,
        polarization_angle=polarization_angle,
    )

    # Gather from a zero-padded copy so kernel taps reaching past either end read zero rather
    # than clamping to the first or last sample and repeating it. Sized for the largest
    # geocenter delay this detector can have (|location| / c, a sky-independent upper bound),
    # the kernel half-width, and the sub-sample alignment shift.
    pad = edge_padding(sampling_frequency, sinc_taps, kaiser_beta)
    padded_plus = jnp.pad(jnp.asarray(plus, dtype=jnp.float64), (pad, pad))
    padded_cross = jnp.pad(jnp.asarray(cross, dtype=jnp.float64), (pad, pad))
    padded_length = n_samples + 2 * pad

    # Fractional sample index of t - tau(t) on the padded input grid, plus any lattice
    # alignment the caller asked for -- one shift, one resampling.
    index = pad + jnp.arange(n_samples, dtype=jnp.float64) - time_delays * sampling_frequency - extra_shift_samples
    plus_shifted = _interpolate_uniform_sinc(padded_plus, index, padded_length, taps=sinc_taps, beta=kaiser_beta)
    cross_shifted = _interpolate_uniform_sinc(padded_cross, index, padded_length, taps=sinc_taps, beta=kaiser_beta)

    return f_plus * plus_shifted + f_cross * cross_shifted
