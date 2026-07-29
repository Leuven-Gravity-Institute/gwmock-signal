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
"""Shared specification for the band-limited resampling kernel.

Projecting onto a rotating detector requires evaluating the polarizations at
``t - tau(t)``, where the delay changes every sample, so every output sample needs a
different sub-sample shift. For a band-limited, uniformly sampled signal the *exact*
interpolant is the sinc series; truncating it to a finite window is the only
approximation, and its error falls steeply with the number of taps. That makes the
accuracy a tunable rather than a fixed property of the method — unlike a cubic, whose
error is fixed at O(h^4) no matter how much compute is available.

This module holds the kernel *specification* only — tap count, window shape and the
weight formula — because the NumPy path (:mod:`gwmock_signal.projection.network`) and
the device path (:mod:`gwmock_signal.projection.jax_projection`) must resample
identically or the two implementations disagree by more than either one's own error.
The array evaluation is written twice, once per backend; the definition lives here once.

!!! note "Choosing the tap count"

    The defaults below were set by measured convergence, not from a design formula:
    see ``tests/projection/test_resampling.py``, which refines both the tap count and
    the sample rate against an interpolation-free direct-Fourier reference. A Kaiser
    window is used because its stopband is controlled by a single parameter, so tap
    count and attenuation can be traded explicitly.

    No windowed-sinc kernel resamples accurately for signal content arbitrarily close
    to Nyquist. The strain must be oversampled with margin; the tests quantify how much.
"""

from __future__ import annotations

import math

import numpy as np

#: Fewest taps that still defines a symmetric kernel.
_MINIMUM_TAPS = 3

#: Default number of kernel taps. Odd, so the kernel is symmetric about the sample
#: preceding the requested position.
DEFAULT_SINC_TAPS = 127

#: Default Kaiser window shape parameter. This, not the tap count, sets the error floor:
#: at 127 taps, beta = 12 stalls near 1e-8 while beta = 32 reaches 1e-12. It cannot be
#: raised freely, because a larger beta needs more taps to hold its transition band --
#: see :data:`_TAPS_PER_BETA` and :func:`validate_kernel`.
DEFAULT_KAISER_BETA = 32.0

#: Taps required per unit of Kaiser beta. Measured, not derived: at beta = 32, 127 taps
#: reach 2.2e-12 at 0.8 x Nyquist while 63 taps give only 1.7e-4, because the transition
#: band no longer fits. The bound ``taps >= 4 * beta - 1`` reproduces where that
#: transition happens across the beta values tested.
_TAPS_PER_BETA = 4.0


#: WGS84 equatorial radius: the largest geocentre distance a ground-based detector can have,
#: used only to bound the geocenter delay when sizing edge padding.
EARTH_RADIUS_M = 6378137.0

#: Speed of light (exact, SI); matches ``astropy.constants.c.value``. Exported, and imported by
#: the projection paths rather than restated there, because the delay a path computes and the
#: padding sized to cover that delay must come from one value -- two copies are a latent
#: inconsistency even when both happen to be the same exact integer today.
SPEED_OF_LIGHT_M_S = 299792458.0


#: Largest sub-sample alignment shift the padding is sized for. ``SamplingGrid.split_index``
#: returns a remainder in ``[0, 1)`` by construction, so one sample of slack suffices -- but a
#: direct caller passing more than this would reach past the padding, hence
#: :func:`require_shift_within_padding`.
MAXIMUM_ALIGNMENT_SHIFT_SAMPLES = 1.0


def require_terrestrial_location(location: np.ndarray, *, name: str = "location") -> None:
    """Reject a detector further from the geocentre than :data:`EARTH_RADIUS_M`.

    The padding bound assumes a ground-based detector. A space-based or incorrectly specified location
    would need more padding than is allocated, and the shortfall would show up as quiet edge
    corruption rather than an error, so it is checked where the location is still concrete.

    Args:
        location: Earth-fixed position in metres (3-vector).
        name: What is being checked, for the error message.

    Raises:
        ValueError: If the position lies outside Earth's equatorial radius.
    """
    radius = float(np.linalg.norm(np.asarray(location, dtype=float)))
    if radius > EARTH_RADIUS_M:
        raise ValueError(
            f"{name} is {radius:.0f} m from the geocentre, beyond Earth's equatorial radius "
            f"({EARTH_RADIUS_M:.0f} m). The resampling edge padding is sized for ground-based "
            f"detectors; a more distant one needs a larger bound than edge_padding() allocates."
        )


def require_shift_within_padding(shift_samples: np.ndarray | float, *, name: str = "shift") -> None:
    """Reject an alignment shift larger than the padding is sized for.

    Args:
        shift_samples: Sub-sample shift(s), in samples.
        name: What is being checked, for the error message.

    Raises:
        ValueError: If any shift is negative, NaN, or at least
            :data:`MAXIMUM_ALIGNMENT_SHIFT_SAMPLES`.
    """
    shifts = np.atleast_1d(np.asarray(shift_samples, dtype=float))
    # NaN fails both comparisons, so it is rejected explicitly rather than passing as in-range and
    # then poisoning every interpolated sample downstream.
    offenders = ~((shifts >= 0.0) & (shifts < MAXIMUM_ALIGNMENT_SHIFT_SAMPLES))
    if np.any(offenders):
        # Reported from among the *offending* entries. Taking the largest magnitude over all
        # shifts would name a perfectly valid 0.9 while the actual offender was a -0.1, sending
        # the reader to inspect the wrong input.
        index = int(np.argmax(offenders))
        raise ValueError(
            f"{name} must lie in [0, {MAXIMUM_ALIGNMENT_SHIFT_SAMPLES}) samples, but entry {index} "
            f"is {shifts[index]!r}. The resampling edge padding is sized for that range, so a "
            f"larger shift would read past the padded region."
        )


def edge_padding(sampling_frequency: float, taps: int = DEFAULT_SINC_TAPS, beta: float = DEFAULT_KAISER_BETA) -> int:
    """Return the zero-padding, in samples, each end of a resampled series needs.

    Both projection paths must pad identically or they disagree at the buffer edges by the
    padding difference alone -- which is how the device path came to diverge from the NumPy
    reference after only one of them was changed. The rule therefore lives here, beside the
    kernel it belongs to, rather than in either path.

    Covers the largest geocenter delay any ground-based detector can have, the kernel's
    half-width, and one sample of sub-sample alignment shift, plus one for rounding. The delay
    bound is Earth's equatorial radius over the speed of light rather than an individual
    detector's distance: on the device that value is a traced argument and so unavailable when
    the padding must be sized, and making it static per detector would cost one compiled kernel
    per detector. The bound over-pads by at most a few samples on buffers of millions.

    Args:
        sampling_frequency: Sample rate in Hz.
        taps: Taps in the resampling kernel.
        beta: Kaiser window shape parameter the caller will actually use.

    Returns:
        Padding in samples for each end.

    Raises:
        ValueError: If the kernel configuration is invalid, or the sample rate is not positive
            and finite.
    """
    # Validated before sizing, so an invalid kernel cannot reserve a padded buffer and only fail
    # afterwards inside the interpolation. The caller's own beta is validated, not the default:
    # checking against the default both rejected valid small-taps/small-beta pairs and let an
    # invalid large-beta pair through until after the allocation.
    taps, _ = validate_kernel(taps, beta)
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError(f"sampling_frequency must be positive and finite; got {sampling_frequency}.")
    max_delay_seconds = EARTH_RADIUS_M / SPEED_OF_LIGHT_M_S
    return (
        math.ceil(max_delay_seconds * sampling_frequency)
        + (taps - 1) // 2
        + math.ceil(MAXIMUM_ALIGNMENT_SHIFT_SAMPLES)
        + 1
    )


def validate_kernel(taps: int, beta: float) -> tuple[int, float]:
    """Validate a kernel specification and return it normalised.

    Args:
        taps: Number of kernel taps; must be an odd integer >= 3.
        beta: Kaiser window shape parameter; must be finite, non-negative and small enough
            that its transition band fits in ``taps``.

    Returns:
        ``(taps, beta)`` as an ``int`` and a ``float``.

    Raises:
        ValueError: If ``taps`` is not an odd integer >= 3, if ``beta`` is not finite or is
            negative, or if ``beta`` is too large for ``taps``. The last case is rejected
            rather than warned about because it fails *quietly*: the kernel still returns
            plausible numbers, several orders of magnitude less accurate than the tap count
            implies.
    """
    # Rejected rather than truncated: int(127.9) would silently accept a value the
    # integer-only contract forbids, and the caller would never learn which kernel ran.
    if isinstance(taps, float) and not taps.is_integer():
        raise ValueError(f"taps must be an odd integer >= 3; got {taps}.")
    taps = int(taps)
    if taps < _MINIMUM_TAPS or taps % 2 == 0:
        raise ValueError(f"taps must be an odd integer >= 3; got {taps}.")
    beta = float(beta)
    # NaN fails every comparison, so it would slip past a bare `beta < 0` check and then
    # propagate silently through every interpolated sample.
    if not np.isfinite(beta):
        raise ValueError(f"Kaiser beta must be finite; got {beta}.")
    if beta < 0.0:
        raise ValueError(f"Kaiser beta must be non-negative; got {beta}.")
    minimum_taps = _TAPS_PER_BETA * beta - 1.0
    if taps < minimum_taps:
        raise ValueError(
            f"Kaiser beta={beta} needs at least {minimum_taps:.0f} taps to hold its transition "
            f"band, but taps={taps}. Either raise taps or lower beta; a beta too large for the "
            f"kernel silently loses several orders of magnitude of accuracy."
        )
    return taps, beta


def resample_uniform_sinc(
    samples: np.ndarray,
    index: np.ndarray,
    *,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """Resample a uniformly sampled series at fractional positions (NumPy).

    Evaluates the Kaiser-windowed sinc series. Positions outside
    ``[0, len(samples) - 1]`` return zero, matching the ``fill_value=0.0`` behaviour of
    the interpolation this replaces.

    Weights are normalised to sum to one, which enforces unit gain at DC and removes
    the amplitude bias the window would otherwise introduce near the ends of the series.

    Args:
        samples: Uniformly sampled series, shape ``(n,)``.
        index: Fractional sample positions to evaluate at, any shape.
        taps: Number of kernel taps.
        beta: Kaiser window shape parameter.

    Returns:
        Interpolated values, the shape of ``index``.
    """
    taps, beta = validate_kernel(taps, beta)
    samples = np.asarray(samples, dtype=float)
    index = np.asarray(index, dtype=float)
    n = samples.shape[0]
    half = (taps - 1) // 2

    base = np.floor(index)
    frac = index - base
    base_int = base.astype(np.int64)

    total = np.zeros(index.shape, dtype=float)
    weight_sum = np.zeros(index.shape, dtype=float)
    # Kaiser window evaluated on the kernel's own support, so the taper is a property
    # of the kernel rather than of where in the series it happens to land.
    denominator = half + 1.0
    for offset in range(-half, half + 1):
        x = frac - offset
        window = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (x / denominator) ** 2))) / np.i0(beta)
        weight = np.sinc(x) * window
        gathered = samples[np.clip(base_int + offset, 0, n - 1)]
        total += weight * gathered
        weight_sum += weight

    interpolated = np.where(weight_sum != 0.0, total / weight_sum, 0.0)
    in_range = (index >= 0.0) & (index <= n - 1)
    return np.where(in_range, interpolated, 0.0)
