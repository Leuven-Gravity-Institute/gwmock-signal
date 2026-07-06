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
"""Shared frequency-domain -> time-domain conditioning for waveform backends.

Backends that evaluate a waveform in the *frequency domain* (with coalescence at
the FD phase reference, ``t = 0``) use these helpers to size the analysis segment
and place coalescence in the returned time series. Keeping the segment sizing and
placement here guarantees the LAL and ripple backends share one convention, so
the same source lands at the same sample in either backend and ``FFT(TD) == FD``
holds.
"""

from __future__ import annotations

import numpy as np

#: Solar mass in seconds (``G M_sun / c^3``). Matches ``lal.MTSUN_SI`` and
#: ``ripplegw.constants.MTSUN`` bit-for-bit, so the two backends auto-size to the
#: same segment length for a given source.
MTSUN_SI = 4.925490947641267e-06

#: Fraction of the analysis segment reserved *after* coalescence (ringdown + pad).
DEFAULT_RINGDOWN_FRACTION = 0.1
#: Extra inspiral headroom (seconds) beyond the leading-order chirp-time estimate.
SEGMENT_BUFFER_SECONDS = 2.0
#: Floor on the segment length (seconds) for very short signals.
MIN_SEGMENT_SECONDS = 1.0


def segment_sample_count(
    chirp_mass_solar: float,
    minimum_frequency: float,
    sampling_frequency: float,
    *,
    ringdown_fraction: float = DEFAULT_RINGDOWN_FRACTION,
    segment_duration: float | None = None,
) -> int:
    """Return an even sample count whose duration contains the full inspiral.

    The duration is rounded up to a power of two seconds so the inspiral (estimated
    from the leading-order post-Newtonian chirp time) fits in the pre-coalescence
    portion of the buffer without cyclic wraparound. A fixed ``segment_duration``
    overrides the estimate.
    """
    if segment_duration is not None:
        seconds = segment_duration
    else:
        mc_seconds = chirp_mass_solar * MTSUN_SI
        # Leading-order (Newtonian) chirp time from minimum_frequency to merger.
        tau0 = (5.0 / 256.0) * (np.pi * minimum_frequency) ** (-8.0 / 3.0) * mc_seconds ** (-5.0 / 3.0)
        seconds = max((tau0 + SEGMENT_BUFFER_SECONDS) / (1.0 - ringdown_fraction), MIN_SEGMENT_SECONDS)
    seconds_pow2 = float(2.0 ** np.ceil(np.log2(seconds)))
    n_samples = round(seconds_pow2 * sampling_frequency)
    if n_samples % 2:
        n_samples += 1
    return n_samples


def coalescence_placement(
    n_samples: int, sampling_frequency: float, ringdown_fraction: float = DEFAULT_RINGDOWN_FRACTION
) -> tuple[int, float]:
    """Return ``(merger_index, epoch)`` for placing coalescence in a segment.

    ``merger_index`` is the sample coalescence sits on after the time-domain roll
    (near the segment end, leaving a small ringdown pad); ``epoch`` is the time of
    the first sample relative to coalescence (negative), so a caller places
    coalescence at ``epoch + tc``.
    """
    merger_index = round((1.0 - ringdown_fraction) * n_samples)
    return merger_index, -merger_index / sampling_frequency


def condition_fd_to_td(
    hp_f: np.ndarray,
    hc_f: np.ndarray,
    n_samples: int,
    sampling_frequency: float,
    ringdown_fraction: float = DEFAULT_RINGDOWN_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inverse-FFT one-sided FD polarizations and place coalescence in the segment.

    ``hp_f``/``hc_f`` are one-sided spectra (coalescence at the FD phase reference,
    ``t = 0``). Returns ``(hp_t, hc_t, epoch)`` where ``epoch`` is the time of the
    first sample relative to coalescence (negative), so the caller places
    coalescence at ``epoch + tc``.
    """
    dt = 1.0 / sampling_frequency
    # Inverse real FFT: h(t) = irfft(h(f)) / dt (continuous-transform normalization).
    hp_t = np.fft.irfft(hp_f, n=n_samples) / dt
    hc_t = np.fft.irfft(hc_f, n=n_samples) / dt

    # With tc=0 coalescence lands at sample 0 and the inspiral wraps to the tail.
    # Roll it forward so coalescence sits near the segment end, leaving the inspiral
    # contiguous before it and a small ringdown pad after.
    merger_index, epoch = coalescence_placement(n_samples, sampling_frequency, ringdown_fraction)
    return np.roll(hp_t, merger_index), np.roll(hc_t, merger_index), epoch
