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

from gwpy.timeseries import TimeSeries

_PYCBC_IMPORT_ERROR = "pycbc is not installed. Run: pip install 'gwmock-signal[pycbc]'"


def optimal_snr(
    strain: TimeSeries,
    psd: Any,
    low_frequency_cutoff: float = 20.0,
) -> float:
    """Return the optimal SNR of a gravitational-wave signal.

    Computes ``sqrt(<h|h>_PSD)`` via ``pycbc.filter.sigma``.

    Args:
        strain: Signal strain as a gwpy ``TimeSeries``.
        psd: Noise PSD as a ``pycbc.types.FrequencySeries``.
        low_frequency_cutoff: Lower frequency bound in Hz.

    Returns:
        Optimal SNR as a non-negative float.

    Raises:
        ImportError: If pycbc is not installed.
    """
    try:
        pycbc_filter = importlib.import_module("pycbc.filter")
    except ImportError as exc:
        raise ImportError(_PYCBC_IMPORT_ERROR) from exc

    htilde = strain.to_pycbc().to_frequencyseries()
    return float(pycbc_filter.sigma(htilde, psd=psd, low_frequency_cutoff=low_frequency_cutoff))


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
