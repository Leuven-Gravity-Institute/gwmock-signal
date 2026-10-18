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
"""Add simulated strain into a segment (time-domain superposition)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gwmock_signal.projection.resampling import resample_uniform_sinc
from gwmock_signal.timeseries import TimeSeries, _public_output, as_timeseries

if TYPE_CHECKING:
    import gwpy.timeseries

logger = logging.getLogger("gwmock_signal.injection")


def inject_strain(
    target: TimeSeries | gwpy.timeseries.TimeSeries,
    injection: TimeSeries | gwpy.timeseries.TimeSeries,
    *,
    interpolate_if_offset: bool = True,
) -> gwpy.timeseries.TimeSeries:
    """Return a new series equal to ``target`` plus ``injection`` on overlapping samples.

    The two must share a sample rate and a unit. Only the samples of ``injection`` that fall
    within ``target`` are added. An injection off the target's sample lattice is resampled onto
    it with the band-limited windowed-sinc kernel the projection uses (see
    ``interpolate_if_offset``), reading the whole injection, so a target sample near either end
    of ``target`` takes its kernel taps from injection samples beyond that end where there are
    any.

    Whether the injection is on the lattice is decided with an absolute tolerance
    derived from the float64 resolution of the GPS timestamps
    (:meth:`~gwmock_signal.sampling_grid.SamplingGrid.lattice_tolerance_samples`),
    so a fractional offset is never mistaken for an aligned one however far into
    the segment the injection starts.

    The target samples filled are those whose position in the injection's own sample index lies in
    ``[0, injection.n_samples - 1]``; a single off-lattice injection sample therefore fills none. If
    nothing is added (no overlap, or offset skipped), returns a **copy** of ``target`` so the result
    is never the same object as ``target``. Both series must be one-dimensional. The samples are
    combined on the host, so a JAX-backed input is copied there.

    Args:
        target: Background segment (e.g. zeros or noise): a
            :class:`~gwmock_signal.timeseries.TimeSeries` or a GWpy ``TimeSeries``.
        injection: Strain to add (e.g. a projected waveform), of either kind.
        interpolate_if_offset: If ``False`` and the injection start is not on a
            target sample boundary, return a copy of ``target`` without resampling.

    Returns:
        New [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/)
        with injected strain.

    Raises:
        ValueError: If either series is not one-dimensional, or the sample rates or units differ.
    """
    return _public_output(
        _inject(as_timeseries(target), as_timeseries(injection), interpolate_if_offset=interpolate_if_offset)
    )


def inject_strains_sequential(
    target: TimeSeries | gwpy.timeseries.TimeSeries,
    injections: Sequence[TimeSeries | gwpy.timeseries.TimeSeries],
    *,
    interpolate_if_offset: bool = True,
) -> gwpy.timeseries.TimeSeries:
    """Apply ``inject_strain`` to each series in order.

    Args:
        target: Initial segment, a :class:`~gwmock_signal.timeseries.TimeSeries` or a GWpy
            ``TimeSeries``.
        injections: Strain series applied in list order, of either kind.
        interpolate_if_offset: Forwarded to each ``inject_strain`` call.

    Returns:
        Final [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).
        If ``injections`` is empty, returns a copy of ``target``.
    """
    result = as_timeseries(target).copy()
    for injection in injections:
        result = _inject(result, as_timeseries(injection), interpolate_if_offset=interpolate_if_offset)
    return _public_output(result)


def _add_strain(background: TimeSeries, signal: np.ndarray) -> TimeSeries:
    """Return ``background`` plus ``signal``, a strain sampled on the background's own grid.

    For signals generated on the background's samples, so nothing is aligned or resampled.

    Raises:
        ValueError: If the background is not strain, or the lengths differ.
    """
    if background.unit != "strain":
        raise ValueError(f"the background must be strain to add a strain signal to; its unit is {background.unit!r}.")
    if background.n_samples != len(signal):
        raise ValueError(f"the signal has {len(signal)} samples but the background has {background.n_samples}.")
    return background.with_data(np.asarray(background.data, dtype=float) + signal)


def _inject(target: TimeSeries, injection: TimeSeries, *, interpolate_if_offset: bool) -> TimeSeries:
    """Return ``target`` plus the part of ``injection`` that falls within it, as a new series."""
    for role, series in (("target", target), ("injection", injection)):
        if len(series.data.shape) != 1:
            raise ValueError(f"the {role} must be one-dimensional; it has shape {tuple(series.data.shape)}.")
    if not target.is_compatible(injection):
        raise ValueError(
            "Injection is not compatible with target: sample rates "
            f"{injection.sample_rate} and {target.sample_rate} Hz, units {injection.unit!r} and {target.unit!r}."
        )
    n_target = target.n_samples
    values = np.asarray(injection.data)
    grid = target.grid
    offset = float(grid.index_of(injection.t0))
    if abs(offset - round(offset)) > grid.lattice_tolerance_samples(injection.t0):
        if not interpolate_if_offset:
            logger.debug("Non-integer sample offset; not interpolating; returning copy of target.")
            return target.copy()
        logger.debug("Injecting with band-limited resampling (offset %.6f samples).", offset)
        # Target samples covered by the injection, i.e. whose position in the injection's own
        # sample index lies in [0, len - 1].
        start_idx = max(math.ceil(offset), 0)
        end_idx = min(math.floor(offset + len(values) - 1), n_target - 1)
        if start_idx > end_idx:
            logger.debug("No overlap; returning copy of target.")
            return target.copy()
        # Positions from integer target indices minus one offset, rather than from per-sample GPS
        # timestamp arrays, each of which carries float64 rounding of a sizeable fraction of a sample.
        positions = np.arange(start_idx, end_idx + 1) - offset
        injected = np.array(target.data, copy=True)
        injected[start_idx : end_idx + 1] += resample_uniform_sinc(values, positions)
        return target.with_data(injected)
    start = round(offset)
    first = max(start, 0)
    stop = min(start + len(values), n_target)
    if first >= stop:
        logger.debug("No overlap; returning copy of target.")
        return target.copy()
    injected = np.array(target.data, copy=True)
    injected[first:stop] += values[first - start : stop - start]
    return target.with_data(injected)
