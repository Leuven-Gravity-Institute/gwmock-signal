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
"""Core matched-filter SNR functions."""

from __future__ import annotations

import numpy as np
import pycbc.filter
import pycbc.types
from gwpy.timeseries import TimeSeries

from gwmock_signal.snr.psd import evaluate_psd, load_design_psd


def _to_pycbc_psd(psd_spec: object, n: int, dt: float, f_low: float) -> pycbc.types.FrequencySeries:
    """Convert any PSD specification to a pycbc FrequencySeries on the rfft grid of n samples.

    Handles str (design PSD name), pycbc.types.FrequencySeries (pass-through),
    np.ndarray (wrap directly), and callable (evaluate on rfft grid then wrap).
    Inf-valued bins are converted to 0.0 (pycbc convention for excluded bins).
    """
    length = n // 2 + 1
    delta_f = 1.0 / (n * dt)
    if isinstance(psd_spec, pycbc.types.FrequencySeries):
        return psd_spec
    if isinstance(psd_spec, str):
        return load_design_psd(psd_spec, length, delta_f, f_low)
    if isinstance(psd_spec, np.ndarray):
        vals = psd_spec.copy()
    else:
        freqs = np.fft.rfftfreq(n, d=dt)
        vals = evaluate_psd(psd_spec, freqs)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return pycbc.types.FrequencySeries(vals, delta_f=delta_f)


def noise_weighted_inner_product(
    a_fft: np.ndarray,
    b_fft: np.ndarray,
    psd: np.ndarray,
    df: float,
) -> float:
    """Compute the noise-weighted inner product ``(a|b)``.

    Evaluates ``4 * Re[sum_{f > 0} conj(a_fft) * b_fft / psd] * df``.
    The DC bin (index 0) is excluded. Bins where ``psd == inf`` contribute zero.

    Custom numpy implementation — no pycbc equivalent for the general ``(a|b)`` case.
    Used as a low-level building block for ISS-003 matrix-valued cross-PSD.

    Args:
        a_fft: One-sided FFT of signal ``a``: ``np.fft.rfft(a) * dt``.
        b_fft: One-sided FFT of signal ``b``: ``np.fft.rfft(b) * dt``.
        psd: One-sided PSD array on the rfft frequency grid, same length as
            ``a_fft``. Use ``np.inf`` for bins that should be excluded.
        df: Frequency resolution in Hz.

    Returns:
        Real-valued inner product ``(a|b)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(
            np.isinf(psd[1:]),
            0.0 + 0.0j,
            np.conj(a_fft[1:]) * b_fft[1:] / psd[1:],
        )
    return float(4.0 * df * np.real(np.sum(integrand)))


def optimal_snr(
    strain: TimeSeries,
    psd: str | object,
    f_low: float = 20.0,
) -> float:
    """Return the optimal matched-filter SNR ``rho = sqrt((h|h))``.

    Delegates to :func:`pycbc.filter.sigma`.

    Args:
        strain: Detector strain timeseries ``h(t)``.
        psd: PSD specification: a ``str`` name (e.g. ``"aLIGO_design"``),
            a callable ``S_n(f) -> array``, a :class:`numpy.ndarray`, or a
            :class:`pycbc.types.FrequencySeries`.
        f_low: Low-frequency cutoff in Hz.

    Returns:
        Optimal SNR as a non-negative float.
    """
    data = np.asarray(strain.value, dtype=float)
    dt = float(strain.dt.value)
    n = len(data)
    pycbc_ts = pycbc.types.TimeSeries(data, delta_t=dt)
    pycbc_psd = _to_pycbc_psd(psd, n, dt, f_low)
    return float(pycbc.filter.sigma(pycbc_ts, psd=pycbc_psd, low_frequency_cutoff=f_low))


def matched_filter_snr(
    signal: TimeSeries,
    data: TimeSeries,
    psd: str | object,
    f_low: float = 20.0,
) -> TimeSeries:
    """Return the complex matched-filter SNR timeseries ``rho(t)``.

    Delegates to :func:`pycbc.filter.matched_filter`. When ``signal == data``
    the peak amplitude ``max|rho(t)|`` equals ``optimal_snr(signal, psd, f_low)``.

    Args:
        signal: Template strain timeseries ``h(t)``, same length as ``data``.
        data: Observed strain timeseries ``d(t)``.
        psd: PSD specification: a ``str`` name, a callable, a :class:`numpy.ndarray`,
            or a :class:`pycbc.types.FrequencySeries`.
        f_low: Low-frequency cutoff in Hz.

    Returns:
        Complex :class:`~gwpy.timeseries.TimeSeries` with the same ``t0``
        and sample rate as ``data``.
    """
    sig_arr = np.asarray(signal.value, dtype=float)
    dat_arr = np.asarray(data.value, dtype=float)
    dt = float(data.dt.value)
    n = len(dat_arr)

    pycbc_signal = pycbc.types.TimeSeries(sig_arr, delta_t=dt)
    pycbc_data = pycbc.types.TimeSeries(dat_arr, delta_t=dt)
    pycbc_psd = _to_pycbc_psd(psd, n, dt, f_low)

    sigma_sq = float(pycbc.filter.sigmasq(pycbc_signal, psd=pycbc_psd, low_frequency_cutoff=f_low))
    if not (sigma_sq > 0.0):  # catches zero, NaN (DC÷0 when f_low < delta_f), and negative
        return TimeSeries(np.zeros(n), t0=data.t0, dt=dt)

    snr_pc = pycbc.filter.matched_filter(pycbc_signal, pycbc_data, psd=pycbc_psd, low_frequency_cutoff=f_low)
    return TimeSeries(np.asarray(snr_pc.data), t0=data.t0, dt=dt)
