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

The kernel *specification* — tap count, window shape and the weight formula — is
:mod:`gwmock_data.resampling`'s, shared by the gwmock packages and re-exported here,
because the NumPy path (:mod:`gwmock_signal.projection.network`) and the device path
(:mod:`gwmock_signal.projection.jax_projection`) must resample identically or the two
implementations disagree by more than either one's own error. This module adds what
projection needs on top: the edge padding sized for the largest geocenter delay, and
the checks that keep a detector and a shift within it.

!!! note "Choosing the tap count"

    The defaults below were set by measured convergence, not from a design formula:
    see ``tests/projection/test_resampling.py``, which refines both the tap count and
    the sample rate against an interpolation-free direct-Fourier reference. A Kaiser
    window is used because its stopband is controlled by a single parameter, so tap
    count and attenuation can be traded explicitly.

    No windowed-sinc kernel resamples accurately for signal content arbitrarily close
    to Nyquist: the default is accurate to 0.8 x Nyquist. A whole signal tapered to zero
    at both ends, such as a generated compact-binary waveform, is therefore oversampled
    exactly first (``oversample_fourier``, the ``tapered`` option of the projection and of
    injection), which puts its whole band at or below half the kernel's Nyquist frequency.
    A piece cut from a longer signal cannot be: the exact interpolant is global, and its cut
    ends would ring across the whole piece.
"""

from __future__ import annotations

import math

import numpy as np
from gwmock_data.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    OVERSAMPLING_FACTOR,
    kaiser_window_chebyshev,
    oversample_fourier,
    resample_uniform_sinc,
    validate_kernel,
)

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


__all__ = [
    "DEFAULT_KAISER_BETA",
    "DEFAULT_SINC_TAPS",
    "EARTH_RADIUS_M",
    "MAXIMUM_ALIGNMENT_SHIFT_SAMPLES",
    "OVERSAMPLING_FACTOR",
    "SPEED_OF_LIGHT_M_S",
    "edge_padding",
    "kaiser_window_chebyshev",
    "oversample_fourier",
    "require_shift_within_padding",
    "require_terrestrial_location",
    "resample_uniform_sinc",
    "validate_kernel",
]
