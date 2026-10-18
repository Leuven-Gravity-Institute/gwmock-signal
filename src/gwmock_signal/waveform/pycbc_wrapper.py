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
"""PyCBC time-domain waveforms as GWpy `TimeSeries` polarizations.

See `docs/user_guide/waveform.md` (examples) and `docs/api/waveform/index.md` (API reference).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from gwmock_signal.timeseries import TimeSeries, _public_output

if TYPE_CHECKING:
    import gwpy.timeseries


def get_td_waveform(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Import and call ``pycbc.waveform.get_td_waveform`` lazily."""
    return importlib.import_module("pycbc.waveform").get_td_waveform(*args, **kwargs)


def pycbc_waveform_wrapper(
    tc: float,
    sampling_frequency: float,
    minimum_frequency: float,
    waveform_model: str,
    **kwargs: Any,
) -> dict[str, gwpy.timeseries.TimeSeries]:
    """Generate plus and cross polarizations with PyCBC and return GWpy time series.

    Calls [`pycbc.waveform.get_td_waveform`](https://pycbc.org/pycbc/latest/html/waveform.html)
    with ``delta_t = 1/sampling_frequency``,
    ``f_lower = minimum_frequency``, and ``approximant=waveform_model``, then shifts both
    polarizations by ``tc`` in GPS seconds and converts them to GWpy.

    Args:
        tc: GPS time of coalescence (seconds); added to each series' ``start_time``.
        sampling_frequency: Sample rate in Hz for ``delta_t``.
        minimum_frequency: Low-frequency cutoff in Hz, passed to PyCBC as ``f_lower``.
        waveform_model: PyCBC time-domain approximant name (e.g. ``IMRPhenomD``).
        **kwargs: Additional arguments for ``get_td_waveform`` (masses, spins, distance,
            etc., depending on the approximant).

    Returns:
        Mapping whose keys are the strings ``plus`` and ``cross``; each value is a
        GWpy [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).

    Raises:
        ValueError: sampling_frequency must be > 0.

    Note:
        Invalid parameters may raise exceptions from PyCBC/LAL; see PyCBC waveform docs.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be > 0")

    hp, hc = get_td_waveform(
        approximant=waveform_model,
        delta_t=1 / sampling_frequency,
        f_lower=minimum_frequency,
        **kwargs,
    )
    hp.start_time += tc
    hc.start_time += tc
    return {
        # Copied, so the returned series do not share PyCBC's buffers.
        "plus": _public_output(TimeSeries.from_pycbc(hp).copy()),
        "cross": _public_output(TimeSeries.from_pycbc(hc).copy()),
    }
