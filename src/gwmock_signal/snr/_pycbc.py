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
"""Single-detector SNR computation using PyCBC as optional backend."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from gwpy.timeseries import TimeSeries

_PYCBC_IMPORT_ERROR = "pycbc is not installed. Run: pip install 'gwmock-signal[pycbc]'"


def optimal_snr(
    strain: TimeSeries,
    psd: Any,
    low_frequency_cutoff: float = 20.0,
    target_delta_f: float | None = None,
) -> float:
    """Return the optimal SNR of a gravitational-wave signal.

    Computes sqrt(<h|h>_PSD) via pycbc.filter.sigma. The strain is zero-padded
    in the time domain so that the FFT frequency resolution reaches target_delta_f
    (default: min(natural_delta_f, 1/32 Hz)) before calling sigma. This ensures
    pycbc's floor-truncated cutoff bin is at most ~0.03 Hz below f_low, reducing
    the over-estimate for short (high-mass) CBC from ~0.3% to < 0.02%.
    Padding only ever refines resolution: if the natural grid is already finer
    than target_delta_f the strain is used as-is. The supplied PSD is
    interpolated onto the padded grid via pycbc.psd.interpolate.

    Args:
        strain: Signal strain as a gwpy TimeSeries.
        psd: Noise PSD as a pycbc FrequencySeries.
        low_frequency_cutoff: Lower frequency bound in Hz.
        target_delta_f: Target frequency resolution in Hz. Default: min of the
            natural grid spacing and 1/32 Hz (never coarsens the grid).

    Returns:
        Optimal SNR as a non-negative float.

    Raises:
        ImportError: If pycbc is not installed.
    """
    try:
        pycbc_filter = importlib.import_module("pycbc.filter")
        pycbc_psd_mod = importlib.import_module("pycbc.psd")
        pycbc_types = importlib.import_module("pycbc.types")
    except ImportError as exc:
        raise ImportError(_PYCBC_IMPORT_ERROR) from exc

    td = strain.to_pycbc()
    natural_delta_f = 1.0 / float(td.duration)

    if target_delta_f is None:
        target_delta_f = min(natural_delta_f, 1.0 / 32.0)

    if natural_delta_f > target_delta_f:
        n_padded = round(float(td.sample_rate) / target_delta_f)
        td_padded = pycbc_types.TimeSeries(
            np.append(np.asarray(td), np.zeros(n_padded - len(td))),
            delta_t=td.delta_t,
            epoch=td.start_time,
        )
        htilde = td_padded.to_frequencyseries()
    else:
        htilde = td.to_frequencyseries()

    psd_interp = pycbc_psd_mod.interpolate(psd, htilde.delta_f, length=len(htilde))
    # Replace zero PSD values with inf (infinite noise → zero contribution to sigma).
    # pycbc.psd.interpolate fills out-of-range frequencies with its boundary value
    # (typically 0); leaving those as 0 causes division-by-zero in sigma.
    psd_data = np.asarray(psd_interp).copy()
    psd_data[psd_data == 0] = np.inf
    psd_interp = pycbc_types.FrequencySeries(psd_data, delta_f=htilde.delta_f)
    return float(pycbc_filter.sigma(htilde, psd=psd_interp, low_frequency_cutoff=low_frequency_cutoff))


def matched_filter_snr(
    data: TimeSeries,
    template: TimeSeries,
    psd: Any,
    low_frequency_cutoff: float = 20.0,
) -> TimeSeries:
    """Return the matched-filter SNR time series.

    Computes the SNR magnitude ``|<template|data>_PSD| / sigma`` via
    ``pycbc.filter.matched_filter``.

    Args:
        data: Data stream as a gwpy ``TimeSeries``.
        template: Template waveform as a gwpy ``TimeSeries``.
        psd: Noise PSD as a ``pycbc.types.FrequencySeries``.
        low_frequency_cutoff: Lower frequency bound in Hz.

    Returns:
        SNR magnitude as a gwpy ``TimeSeries``.

    Raises:
        ImportError: If pycbc is not installed.
    """
    try:
        pycbc_filter = importlib.import_module("pycbc.filter")
    except ImportError as exc:
        raise ImportError(_PYCBC_IMPORT_ERROR) from exc

    snr_pycbc = pycbc_filter.matched_filter(
        template.to_pycbc(),
        data.to_pycbc(),
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
    )
    return TimeSeries.from_pycbc(abs(snr_pycbc))
