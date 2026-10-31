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
from gwmock_data.injection import places_whole_samples

from gwmock_signal.projection.resampling import OVERSAMPLING_FACTOR, oversample_fourier, resample_uniform_sinc
from gwmock_signal.timeseries import TimeSeries, as_timeseries

if TYPE_CHECKING:
    import gwpy.timeseries

logger = logging.getLogger("gwmock_signal.injection")


#: Beyond half a sample outside the target, neither rounding nor resampling places anything in it.
_HALF_SAMPLE = 0.5


def inject_strain(
    target: TimeSeries | gwpy.timeseries.TimeSeries,
    injection: TimeSeries | gwpy.timeseries.TimeSeries,
    *,
    interpolate_if_offset: bool = True,
    tapered: bool = False,
    on_grid: bool = False,
) -> TimeSeries:
    """Return a new series equal to ``target`` plus ``injection`` on overlapping samples.

    The two must share a sample rate and a unit. Only the samples of ``injection`` that fall
    within ``target`` are added. An injection off the target's sample lattice is resampled onto
    it with the band-limited windowed-sinc kernel the projection uses (see
    ``interpolate_if_offset``), reading the whole injection, so a target sample near either end
    of ``target`` takes its kernel taps from injection samples beyond that end where there are
    any.

    The injection is placed at the time it carries. It is added sample for sample only when
    rounding its offset to whole samples moves it by no more than LAL's no-op shift of 1e-4
    samples (``gwmock_data.injection.places_whole_samples``), as LALSimulation's
    ``XLALSimAddInjectionREAL8TimeSeries`` places an injection, or when the caller declares it
    built on the target's lattice (``on_grid``). A few float64 steps off the lattice is a genuine
    offset for a signal placed at its coalescence time: two steps are 0.24 µs at GPS 1e9.

    The target samples filled are those whose position in the injection's own sample index lies in
    ``[0, injection.n_samples - 1]``; a single off-lattice injection sample therefore fills none. If
    nothing is added (no overlap, or offset skipped), returns a **copy** of ``target`` so the result
    is never the same object as ``target``. Both series must be one-dimensional. The samples are
    combined on the host, so a JAX-backed input is copied there.

    Args:
        target: Background segment (e.g. zeros or noise): a
            :class:`~gwmock_signal.timeseries.TimeSeries` or a GWpy ``TimeSeries``.
        injection: Strain to add (e.g. a projected waveform), of either kind.
        interpolate_if_offset: If ``False``, nothing is resampled: an injection that would be
            resampled (one not placed by whole samples, above) is skipped and a copy of ``target``
            returned. One declared on the lattice but off it is refused, not skipped.
        tapered: Whether ``injection`` is a whole signal that falls smoothly to zero at both
            ends. If so, an off-lattice injection is first oversampled exactly, and placed
            accurately up to its Nyquist frequency rather than to 0.8 of it. Leave it ``False``
            for a piece cut from a longer signal: the exact interpolation is global, and the
            cut ends would ring across the whole piece.
        on_grid: Whether ``injection`` was built on the target's lattice, so that any fraction of
            a sample in its offset is the float64 rounding of its timestamps
            (:meth:`~gwmock_signal.sampling_grid.SamplingGrid.lattice_tolerance_samples`). It is
            then added sample for sample.

    Returns:
        New :class:`~gwmock_signal.timeseries.TimeSeries` with injected strain.

    Raises:
        ValueError: If either series is not one-dimensional, the sample rates or units differ, or
            ``injection`` is declared on the lattice but starts further off it than both its
            timestamps' float64 resolution and LAL's no-op shift allow.
    """
    return _inject(
        as_timeseries(target),
        as_timeseries(injection),
        interpolate_if_offset=interpolate_if_offset,
        tapered=tapered,
        on_grid=on_grid,
    )


def inject_strains_sequential(
    target: TimeSeries | gwpy.timeseries.TimeSeries,
    injections: Sequence[TimeSeries | gwpy.timeseries.TimeSeries],
    *,
    interpolate_if_offset: bool = True,
    tapered: bool = False,
    on_grid: bool = False,
) -> TimeSeries:
    """Apply ``inject_strain`` to each series in order.

    Args:
        target: Initial segment, a :class:`~gwmock_signal.timeseries.TimeSeries` or a GWpy
            ``TimeSeries``.
        injections: Strain series applied in list order, of either kind.
        interpolate_if_offset: Forwarded to each ``inject_strain`` call.
        tapered: Forwarded to each ``inject_strain`` call: whether every series in ``injections``
            is a whole signal tapered to zero at both ends.
        on_grid: Forwarded to each ``inject_strain`` call: whether every series in ``injections``
            was built on the target's lattice.

    Returns:
        Final :class:`~gwmock_signal.timeseries.TimeSeries`.
        If ``injections`` is empty, returns a copy of ``target``.
    """
    result = as_timeseries(target).copy()
    for injection in injections:
        result = _inject(
            result,
            as_timeseries(injection),
            interpolate_if_offset=interpolate_if_offset,
            tapered=tapered,
            on_grid=on_grid,
        )
    return result


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


def _inject(
    target: TimeSeries,
    injection: TimeSeries,
    *,
    interpolate_if_offset: bool,
    tapered: bool = False,
    on_grid: bool = False,
) -> TimeSeries:
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
    # An injection lying more than half a sample outside the target adds nothing however it would be
    # placed, rounded or resampled, so it is not located: a declaration it could not honour here is not
    # refused here.
    if offset > n_target - 1 + _HALF_SAMPLE or offset + len(values) - 1 < -_HALF_SAMPLE:
        logger.debug("No overlap; returning copy of target.")
        return target.copy()
    if not places_whole_samples(offset, grid.lattice_tolerance_samples(injection.t0), on_grid=on_grid):
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
        # timestamp arrays, each of which carries float64 rounding of a sizeable fraction of a sample;
        # the indices and the offset's fraction are passed apart, so the fraction keeps its precision
        # to the injection's last sample.
        whole = math.floor(offset)
        # A whole tapered signal is oversampled exactly first; the positions and the fraction then both
        # scale by the factor, exactly.
        factor = OVERSAMPLING_FACTOR if tapered else 1
        injected = np.array(target.data, copy=True)
        injected[start_idx : end_idx + 1] += resample_uniform_sinc(
            oversample_fourier(values) if tapered else values,
            factor * (np.arange(start_idx, end_idx + 1) - whole),
            fraction=factor * (whole - offset),
        )
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
