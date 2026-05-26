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
"""Noise power spectral density utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pycbc.psd as _pycbc_psd
import pycbc.types

_DESIGN_PSD_MAP: dict[str, str] = {
    "aLIGO_design": "aLIGODesignSensitivityP1200087",
}


def load_design_psd(
    name: str,
    length: int,
    delta_f: float,
    f_low: float = 0.0,
) -> pycbc.types.FrequencySeries:
    """Return a pycbc FrequencySeries for a named design sensitivity curve.

    Args:
        name: Design PSD identifier. Supported values: ``"aLIGO_design"``.
        length: Number of frequency bins (= ``n // 2 + 1`` for a time series of length ``n``).
        delta_f: Frequency resolution in Hz.
        f_low: Low-frequency cutoff in Hz. Bins below this value are set to zero.

    Returns:
        :class:`pycbc.types.FrequencySeries` of one-sided PSD values.

    Raises:
        ValueError: If ``name`` is not a recognised design PSD.
    """
    if name not in _DESIGN_PSD_MAP:
        raise ValueError(f"Unknown design PSD {name!r}. Supported: {sorted(_DESIGN_PSD_MAP)}")
    return _pycbc_psd.from_string(_DESIGN_PSD_MAP[name], length, delta_f, f_low)


def from_numpy_psd(
    freqs: np.ndarray,
    psd_values: np.ndarray,
    length: int,
    delta_f: float,
    f_low: float = 0.0,
) -> pycbc.types.FrequencySeries:
    """Interpolate a tabulated PSD to a pycbc FrequencySeries.

    Args:
        freqs: Strictly positive frequencies in Hz, sorted ascending.
        psd_values: PSD values at ``freqs`` (same shape), all positive.
        length: Number of frequency bins in the output series.
        delta_f: Frequency resolution in Hz.
        f_low: Low-frequency cutoff in Hz. Bins below this value are set to zero.

    Returns:
        :class:`pycbc.types.FrequencySeries` interpolated from the tabulated values.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd_values = np.asarray(psd_values, dtype=float)
    return _pycbc_psd.from_numpy_arrays(freqs, psd_values, length, delta_f, f_low)


def evaluate_psd(psd: Callable[..., np.ndarray] | np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Evaluate a PSD specification on a frequency array, returning a numpy array.

    Used by :func:`~gwmock_signal.snr.core.noise_weighted_inner_product` and the
    ISS-003 matrix-valued cross-PSD path. Does **not** accept string names; use
    :func:`load_design_psd` for named design PSDs.

    Args:
        psd: One of:

            * callable — called with ``freqs`` and expected to return an array of the same shape.
            * :class:`numpy.ndarray` — must be the same length as ``freqs``; returned as a copy.

        freqs: Frequency array in Hz.

    Returns:
        Float array of PSD values with the same shape as ``freqs``.
        The bin where ``freqs == 0`` is always set to ``np.inf``.
    """
    freqs = np.asarray(freqs, dtype=float)

    if callable(psd):
        raw = psd(freqs)
        result = np.broadcast_to(np.asarray(raw, dtype=float), freqs.shape).copy()
    else:
        result = np.asarray(psd, dtype=float).copy()

    result[freqs == 0] = np.inf
    return result
