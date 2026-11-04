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
"""The finite-arm detector response on the host projection path.

The long-wavelength response treats a detector as a point at its vertex. Each arm in fact measures
the strain averaged along the light's round trip to its end mirror and back (R. Essick, S. Vitale and
M. Evans, Phys. Rev. D 96, 084004 (2017), Eq. (4)). For an arm of length ``L`` the difference is of
order ``f L / c``: 1.7e-4 at 5 Hz and 0.14 at 4096 Hz for 10 km. It matters most where detectors share
arms. In a triangle each arm line is used by two interferometers with opposite signs, so the signal
cancels in the null stream ``E1 + E2 + E3`` only if both read it as the same average; placed at their
vertices, they read it 10 km apart.

The response is split in two, see :func:`~gwmock_signal.projection.kernels.finite_arm_residual`:

* the strain at the arm's **midpoint** rather than at the vertex, a pure time shift. It changes as
  Earth turns, and is applied sample by sample with the same resampling kernel as the vertex delay;
* the **residual** ``R(f L / c, mu)``, the average over the two legs of the trip about that midpoint,
  where ``mu`` is the cosine between the arm and the direction to the source. It is a filter that
  depends on ``mu``, which changes as Earth turns. It is applied as filters at a few fixed values of
  ``mu`` (:func:`alignment_nodes`), whose outputs are interpolated in ``mu`` sample by sample.

Both are applied to the arm's strain after the antenna pattern, in detector time. Applying the residual
there rather than to the polarizations before the delay treats the slowly changing antenna pattern,
delay and ``mu`` as constant across the residual's length of two light crossings of the arm (67 us for
10 km). The delay changes by at most 1.5e-6 s per second as Earth turns, so the error is at most of that
order times the residual itself. Against the strain averaged by quadrature along each arm's light path,
with Earth turning during the trip, it is 1e-11 of peak at 100 Hz, 4e-10 at 700 Hz and 3e-9 at 1.8 kHz
for the ET triangle, within that bound (``tests/projection/test_finite_arm.py``).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator

import numpy as np
from gwmock_data.resampling import resample_uniform_sinc

from gwmock_signal.projection.kernels import finite_arm_residual
from gwmock_signal.projection.resampling import SPEED_OF_LIGHT_M_S

#: Largest error tolerated, at any ``mu`` and any frequency up to Nyquist, in interpolating the residual
#: between :func:`alignment_nodes`. The residual is of order one, so this is relative to the signal.
ALIGNMENT_INTERPOLATION_TOLERANCE = 1e-14

#: Most alignment nodes :func:`alignment_nodes` will use. 14 suffice for 15 km arms at 16384 Hz.
_MAX_ALIGNMENT_NODES = 64

#: Frequencies, as fractions of the largest, at which :func:`alignment_nodes` checks the interpolation.
_CHECKED_FRACTIONS = np.linspace(0.125, 1.0, 8)

#: Values of ``mu`` at which :func:`alignment_nodes` checks the interpolation.
_CHECKED_ALIGNMENTS = np.linspace(-1.0, 1.0, 2001)


def _chebyshev_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``count`` Chebyshev points of the second kind on ``[-1, 1]`` and their barycentric weights."""
    order = np.arange(count)
    weights = (-1.0) ** order
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return np.cos(np.pi * order / (count - 1)), weights


def _interpolation_weights(alignment: np.ndarray, nodes: np.ndarray, weights: np.ndarray) -> Iterator[np.ndarray]:
    """Yield, for each node in turn, its barycentric interpolation weight at every ``alignment``.

    The second barycentric form, which is exact at the nodes and stable for Chebyshev points. A sample
    whose ``alignment`` equals a node exactly takes that node's value. One node at a time, so a long
    series holds two arrays of its length here rather than one per node.
    """
    alignment = np.asarray(alignment, dtype=float)
    hit = np.full(alignment.shape, -1)
    total = np.zeros(alignment.shape)
    for index, node in enumerate(nodes):
        difference = alignment - node
        exact = difference == 0.0
        hit[exact] = index
        total += weights[index] / np.where(exact, 1.0, difference)
    for index, node in enumerate(nodes):
        difference = alignment - node
        term = weights[index] / np.where(difference == 0.0, 1.0, difference) / total
        yield np.where(hit >= 0, (hit == index).astype(float), term)


@functools.lru_cache(maxsize=64)
def alignment_nodes(max_arm_light_crossings: float) -> np.ndarray:
    """Return the values of ``mu`` at which the residual is filtered, for frequencies up to a bound.

    The fewest Chebyshev points of the second kind on ``[-1, 1]`` from which barycentric interpolation
    reproduces :func:`~gwmock_signal.projection.kernels.finite_arm_residual` within
    :data:`ALIGNMENT_INTERPOLATION_TOLERANCE` at every ``mu`` checked, for ``x = f L / c`` up to
    ``max_arm_light_crossings``: 11 for 10 km arms at 8192 Hz, 12 for 15 km.

    Raises:
        ValueError: If no number up to the cap reaches the tolerance, which would take arms of
            hundreds of kilometres at these sample rates.
    """
    crossings = float(max_arm_light_crossings) * _CHECKED_FRACTIONS
    exact = finite_arm_residual(crossings[:, None], _CHECKED_ALIGNMENTS[None, :], xp=np)
    for count in range(2, _MAX_ALIGNMENT_NODES + 1):
        nodes, weights = _chebyshev_points(count)
        values = finite_arm_residual(crossings[:, None], nodes[None, :], xp=np)
        basis = np.stack(list(_interpolation_weights(_CHECKED_ALIGNMENTS, nodes, weights)))
        if np.max(np.abs(values @ basis - exact)) < ALIGNMENT_INTERPOLATION_TOLERANCE:
            return nodes
    raise ValueError(
        f"The finite-arm residual at {max_arm_light_crossings:.3g} arm light crossings per period cannot be "
        f"interpolated in the arm's alignment within {ALIGNMENT_INTERPOLATION_TOLERANCE:g} with "
        f"{_MAX_ALIGNMENT_NODES} nodes. That is far beyond any ground-based arm at a usual sample rate."
    )


def _fast_length(minimum: int) -> int:
    """Return the smallest integer at least ``minimum`` with no prime factor above 5, a fast FFT length."""
    best = 1 << max(0, (minimum - 1).bit_length())
    power5 = 1
    while power5 < best:
        power35 = power5
        while power35 < best:
            candidate = power35
            while candidate < minimum:
                candidate *= 2
            best = min(best, candidate)
            power35 *= 3
        power5 *= 5
    return best


def _leg_nodes(alignment: float, crossing_samples: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the delays, in samples, and the weights averaging the strain over the two legs of the trip.

    Gauss-Legendre points on each leg: the outward leg over delays ``[-mu, 2 + mu] T / 2`` and the
    return leg over ``[-2 + mu, -mu] T / 2``, with ``T`` the one-way crossing time, each carrying half the
    weight. For a band-limited signal the error falls like ``(pi W)^(2n) / (2n)!^2`` for a leg ``W``
    samples long and ``n`` points; ``8 + ceil(4 W)`` points keep it below 1e-12 up to Nyquist.
    """
    half = 0.5 * crossing_samples
    legs = (
        (-alignment * half, crossing_samples + alignment * half),
        (-crossing_samples + alignment * half, -alignment * half),
    )
    points, weights = np.polynomial.legendre.leggauss(8 + math.ceil(4.0 * crossing_samples))
    delays = [0.5 * (start + end) + 0.5 * (end - start) * points for start, end in legs]
    return np.concatenate(delays), np.concatenate([0.25 * weights, 0.25 * weights])


@functools.lru_cache(maxsize=256)
def residual_kernel(alignment: float, crossing_samples: float, *, taps: int, beta: float) -> np.ndarray:
    """Return the residual as a finite filter, for a series that is a piece of a longer signal.

    The average over the two legs of the trip of the resampling kernel's own interpolant, minus the
    sample itself: what interpolating the series at each Gauss-Legendre point of :func:`_leg_nodes` and
    averaging gives, less the series. It is as accurate as that kernel -- to 0.8 of Nyquist -- and as
    local, so the ends of a piece cut from a longer signal disturb only the kernel's half-width at
    each end rather than ringing across it, which the exact frequency-domain residual would do.

    Cached, and read-only: every arm of a given length at a given rate shares it.

    Returns:
        Taps centred on lag zero, an odd number of them.
    """
    delays, weights = _leg_nodes(alignment, crossing_samples)
    half_width = (taps - 1) // 2
    reach = half_width + math.ceil(crossing_samples) + 1
    # The impulse sits a further half-width from each end, so every tap of every position read lies
    # inside the array whatever the kernel does at an array's ends.
    centre = reach + half_width
    impulse = np.zeros(2 * centre + 1)
    impulse[centre] = 1.0
    lags = np.arange(-reach, reach + 1, dtype=np.int64)
    # Every Gauss-Legendre point at every lag in one call: row k is the kernel read at delay k.
    interpolated = resample_uniform_sinc(
        impulse,
        np.broadcast_to(centre + lags, (len(delays), len(lags))).ravel(),
        fraction=np.repeat(-delays, len(lags)),
        taps=taps,
        beta=beta,
    ).reshape(len(delays), len(lags))
    kernel = weights @ interpolated - impulse[centre - reach : centre + reach + 1]
    kernel.setflags(write=False)
    return kernel


def apply_residual(  # noqa: PLR0913 -- the series, its geometry and the kernel settings
    arm_strain: np.ndarray,
    alignment: np.ndarray,
    *,
    arm_length_m: float,
    sample_rate: float,
    tapered: bool,
    taps: int,
    beta: float,
) -> np.ndarray:
    """Return one arm's strain with the residual of its finite length applied, sample by sample.

    Args:
        arm_strain: The arm's long-wavelength strain, taken at its midpoint, in detector time.
        alignment: ``mu`` at each sample, the cosine between the arm and the direction to the source.
        arm_length_m: The arm's length in metres.
        sample_rate: Samples per second.
        tapered: Whether ``arm_strain`` is a whole signal that falls to zero at both ends. If so the
            residual is applied exactly, in the frequency domain, up to Nyquist; otherwise as the
            finite filter of :func:`residual_kernel`, accurate to 0.8 of Nyquist.
        taps: Taps of the resampling kernel, which sizes the finite filter.
        beta: Kaiser window shape parameter of that kernel.

    Returns:
        The arm's strain, of the same length.
    """
    count = len(arm_strain)
    crossing_samples = arm_length_m / SPEED_OF_LIGHT_M_S * sample_rate
    nodes = alignment_nodes(0.5 * crossing_samples)
    interpolation = _interpolation_weights(alignment, nodes, _chebyshev_points(len(nodes))[1])
    kernels = (
        [] if tapered else [residual_kernel(float(node), crossing_samples, taps=taps, beta=beta) for node in nodes]
    )
    reach = 0 if tapered else (len(kernels[0]) - 1) // 2
    # Twice the series, so the product of spectra is a linear convolution: what the residual spreads past
    # one end does not wrap round to the other.
    length = _fast_length(2 * count + 2 * reach)
    spectrum = np.fft.rfft(arm_strain, n=length)
    crossings = np.fft.rfftfreq(length, d=1.0 / sample_rate) * (arm_length_m / SPEED_OF_LIGHT_M_S)

    strain = np.array(arm_strain, dtype=float, copy=True)
    for index, (node, weight) in enumerate(zip(nodes, interpolation, strict=True)):
        if tapered:
            response = finite_arm_residual(crossings, node, xp=np) - 1.0
            filtered = np.fft.irfft(spectrum * response, n=length)[:count]
        else:
            response = np.fft.rfft(kernels[index], n=length)
            filtered = np.fft.irfft(spectrum * response, n=length)[reach : reach + count]
        strain += weight * filtered
    return strain
