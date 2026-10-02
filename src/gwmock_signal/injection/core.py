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
"""Add simulated strain into a GWpy segment (time-domain superposition)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from astropy.units import second
from gwpy.timeseries import TimeSeries

from gwmock_signal.projection.resampling import resample_uniform_sinc
from gwmock_signal.sampling_grid import SamplingGrid

logger = logging.getLogger("gwmock_signal.injection")


def inject_strain(
    target: TimeSeries,
    injection: TimeSeries,
    *,
    interpolate_if_offset: bool = True,
) -> TimeSeries:
    """Return a new series equal to ``target`` plus ``injection`` on overlapping samples.

    Uses [`TimeSeries.is_compatible`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.is_compatible)
    to require matching sample spacing and units. The injection is cropped to the
    target span when needed. An injection off the target's sample lattice is
    resampled onto it with the band-limited windowed-sinc kernel the projection
    uses (see ``interpolate_if_offset``).

    Whether the injection is on the lattice is decided with an absolute tolerance
    derived from the float64 resolution of the GPS timestamps
    (:meth:`~gwmock_signal.sampling_grid.SamplingGrid.lattice_tolerance_samples`),
    so a fractional offset is never mistaken for an aligned one however far into
    the segment the injection starts.

    If nothing is added (no overlap, empty injection after crop, or offset skipped),
    returns a **copy** of ``target`` so the result is never the same object as
    ``target``.

    Args:
        target: Background segment (e.g. zeros or noise).
        injection: Strain to add (e.g. a projected waveform).
        interpolate_if_offset: If ``False`` and the injection start is not on a
            target sample boundary, return ``target.copy()`` without resampling.

    Returns:
        New [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/)
        with injected strain.

    Raises:
        ValueError: If GWpy compatibility checks fail (units, sample rate, etc.).
    """
    if not target.is_compatible(injection):
        raise ValueError("Injection is not compatible with target (sample rate, units, or epoch mismatch).")

    other = injection
    if (target.xunit == second) and (other.xspan[0] < target.xspan[0]):
        other = other.crop(start=target.xspan[0])
    if (target.xunit == second) and (other.xspan[1] > target.xspan[1]):
        other = other.crop(end=target.xspan[1])

    if len(other.times) == 0:
        logger.debug("Injection empty after crop; returning copy of target.")
        return target.copy()

    n_target = len(target.value)
    grid = SamplingGrid(epoch=float(target.t0.value), sampling_frequency=1.0 / float(target.dt.value))
    injection_start = float(other.t0.value)
    offset = float(grid.index_of(injection_start))

    if abs(offset - round(offset)) > grid.lattice_tolerance_samples(injection_start):
        if not interpolate_if_offset:
            logger.debug("Non-integer sample offset; not interpolating; returning copy of target.")
            return target.copy()

        logger.debug("Injecting with band-limited resampling (offset %.6f samples).", offset)
        # Target samples covered by the injection, i.e. whose position in the injection's own
        # sample index lies in [0, len - 1].
        start_idx = max(math.ceil(offset), 0)
        end_idx = min(math.floor(offset + len(other.value) - 1), n_target - 1)

        if start_idx > end_idx:
            logger.debug("No overlap after index search; returning copy of target.")
            return target.copy()

        # Positions from integer target indices minus one offset, rather than from per-sample GPS
        # timestamp arrays, each of which carries float64 rounding of a sizeable fraction of a sample.
        positions = np.arange(start_idx, end_idx + 1) - offset
        injected_data = target.value.copy()
        injected_data[start_idx : end_idx + 1] += resample_uniform_sinc(other.value, positions)
        return TimeSeries(
            injected_data,
            t0=target.t0,
            dt=target.dt,
            unit=target.unit,
        )

    start_idx = round(offset)
    end_idx = start_idx + len(other.value) - 1
    if start_idx < 0 or end_idx >= n_target or start_idx >= n_target:
        logger.warning(
            "Injection range [%s:%s] out of bounds for length %s; returning copy of target.",
            start_idx,
            end_idx,
            n_target,
        )
        return target.copy()

    injected_data = target.value.copy()
    inject_len = min(len(other.value), end_idx - start_idx + 1)
    injected_data[start_idx : start_idx + inject_len] += other.value[:inject_len]
    return TimeSeries(
        injected_data,
        t0=target.t0,
        dt=target.dt,
        unit=target.unit,
    )


def inject_strains_sequential(
    target: TimeSeries,
    injections: Sequence[TimeSeries],
    *,
    interpolate_if_offset: bool = True,
) -> TimeSeries:
    """Apply ``inject_strain`` to each series in order.

    Args:
        target: Initial segment.
        injections: Strain series applied in list order.
        interpolate_if_offset: Forwarded to each ``inject_strain`` call.

    Returns:
        Final [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).
        If ``injections`` is empty, returns ``target.copy()``.
    """
    if not injections:
        return target.copy()
    result = inject_strain(target, injections[0], interpolate_if_offset=interpolate_if_offset)
    for inj in injections[1:]:
        result = inject_strain(result, inj, interpolate_if_offset=interpolate_if_offset)
    return result
