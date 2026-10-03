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
and place coalescence in the returned time series. Keeping the inspiral-duration
estimate and coalescence placement here means the LAL, gwsignal and ripple backends
promise the same inspiral headroom and use the same placement convention, so
``FFT(TD) == FD`` holds in each. They differ only in how a duration is rounded up
to a transform length: :func:`segment_sample_count` rounds to a power of two
seconds, ripple to a 5-smooth sample count.
"""

from __future__ import annotations

import numpy as np

#: Solar mass in seconds (``G M_sun / c^3``). Matches ``lal.MTSUN_SI`` and
#: ``ripplegw.constants.MTSUN`` bit-for-bit, so the two backends auto-size to the
#: same segment length for a given source.
MTSUN_SI = 4.925490947641267e-06

#: Fraction of the analysis segment reserved *after* coalescence (ringdown + pad).
DEFAULT_RINGDOWN_FRACTION = 0.1
#: Absolute headroom (seconds) added to the estimated inspiral duration, on top of the
#: proportional margin of :func:`inspiral_margin`.
SEGMENT_BUFFER_SECONDS = 2.0
#: Floor on the segment length (seconds) for very short signals.
MIN_SEGMENT_SECONDS = 1.0

#: Largest physical symmetric mass ratio, attained at equal masses.
MAXIMUM_ETA = 0.25

#: Minimum fractional headroom beyond the 1PN chirp time.
#:
#: A *proportional* margin, because the omitted terms scale with the duration. The flat
#: :data:`SEGMENT_BUFFER_SECONDS` alone left only 2.8% of headroom for a 10+1.4 system at 10 Hz,
#: less than the 4.9% the 1PN term contributes there, and that case wrapped its inspiral around the
#: buffer -- measurably, at 1.8% of peak amplitude in the region after the ringdown.
#:
#: This is a *floor*, not the whole margin -- see :func:`inspiral_margin`. A fixed fraction would
#: be indefensible wherever the PN series stops converging, and nothing here restricts callers to
#: the regime where it does: a 60+3 binary at 512 Hz is accepted, and its expansion parameter is
#: already about 0.79.
INSPIRAL_SAFETY_FRACTION = 0.10


def inspiral_margin(relative_correction: np.ndarray | float) -> np.ndarray | float:
    """Return the fractional headroom to add to the 1PN duration estimate, per event.

    At least :data:`INSPIRAL_SAFETY_FRACTION`, and never smaller than the 1PN term itself.

    The 1PN term is the last one *retained*. While the series converges the next term is smaller
    than it, so the 10% floor covers what is omitted. Where the term is large the series is not
    converging and the omitted terms are the same order as the one kept, so the margin has to grow
    with it. That makes the headroom self-scaling rather than resting on an unstated assumption
    about which masses and frequencies a caller will choose.

    Elementwise, so each event is sized against its own correction. Reducing to a single margin
    would apply one event's correction to another's duration, and the correction grows with total
    mass.

    Args:
        relative_correction: The 1PN term relative to the 0PN one, from :func:`inspiral_seconds`.

    Returns:
        The fractional margin(s), broadcast over the input: a plain ``float`` for a scalar or 0-d
        input, an array otherwise. Scalar input returns a ``float`` so callers can format the value
        into a message; a bare ndarray raises ``TypeError`` on ``:.1%``, which is a poor way to
        discover the return type.
    """
    margin = np.maximum(INSPIRAL_SAFETY_FRACTION, np.asarray(relative_correction, dtype=float))
    return float(margin) if margin.ndim == 0 else margin


def inspiral_seconds(
    chirp_mass_solar: np.ndarray | float,
    eta: np.ndarray | float,
    minimum_frequency: float,
    mtsun: float = MTSUN_SI,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inspiral duration from *minimum_frequency* to coalescence, and the 1PN term.

    The leading-order (Newtonian) chirp time with its 1PN correction,
    ``tau0 * (1 + (743/252 + 11 eta/3) (pi M f)^(2/3))``. That correction is *positive*, so the 0PN
    term alone always underestimates the duration, which is why it cannot size a buffer by itself.

    Spin is not included. Aligned spin lengthens the inspiral, but it enters at 1.5PN alongside the
    negative tail term, which outweighs it. Measured against the stationary-phase duration of
    LALSimulation's IMRPhenomXAS on 420 cases, this estimate is an upper bound in every case
    before any margin is added. The cases are primary masses 1.4, 3, 5, 10, 20 and 40 and
    secondary masses 1.2, 1.4, 2 and 5 solar masses, keeping only pairs with the secondary no
    heavier than the primary (21 of the 24), times equal aligned spins of 0, 0.5, 0.9 and 0.99 on
    both components, times cutoffs of 5, 7, 10, 15 and 20 Hz: 21 x 4 x 5 = 420.

    Args:
        chirp_mass_solar: Detector-frame chirp mass(es) in solar masses.
        eta: Symmetric mass ratio(es), ``m1 m2 / (m1 + m2)^2``, in ``(0, 0.25]``.
        minimum_frequency: Low-frequency cutoff in Hz.
        mtsun: Solar mass in seconds. The ripple backend passes ``ripplegw.constants.MTSUN`` so its
            sizing rests on its own library's constant; it equals :data:`MTSUN_SI` bit-for-bit.

    Returns:
        ``(duration, relative_correction)``, both broadcast over the inputs. The second is the
        1PN term's size relative to the 0PN one, which the caller needs: it is the last *retained*
        term, so where it is not small the omitted terms are not small either and a fixed margin
        would be meaningless.

    Raises:
        ValueError: If any mass or ratio is non-finite, non-positive, or outside ``(0, 0.25]``, or
            if ``minimum_frequency`` is not positive and finite.
    """
    chirp_mass_solar = np.asarray(chirp_mass_solar, dtype=float)
    eta = np.asarray(eta, dtype=float)
    # Rejected here, where the message can name the cause. `np.all`/`np.any` are vacuously true on
    # an empty array, so the checks below would pass and the caller would instead see numpy's
    # "zero-size array reduction has no identity" from the max in the sizing callers.
    if chirp_mass_solar.size == 0 or eta.size == 0:
        raise ValueError("chirp_mass_solar and eta must be non-empty; a grid cannot be sized for no events.")
    if not np.isfinite(minimum_frequency) or minimum_frequency <= 0.0:
        raise ValueError(f"minimum_frequency must be positive and finite; got {minimum_frequency}.")
    if not np.all(np.isfinite(chirp_mass_solar)) or np.any(chirp_mass_solar <= 0.0):
        raise ValueError("chirp_mass_solar must be positive and finite.")
    # eta > 0.25 is unphysical (0.25 is the equal-mass maximum); without this a bad value produces a
    # plausible-looking duration rather than an error.
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0.0) or np.any(eta > MAXIMUM_ETA):
        raise ValueError(f"eta must be finite and in (0, {MAXIMUM_ETA}].")

    chirp_mass_seconds = chirp_mass_solar * mtsun
    tau0 = (5.0 / 256.0) * (np.pi * minimum_frequency) ** (-8.0 / 3.0) * chirp_mass_seconds ** (-5.0 / 3.0)
    # M = Mc * eta^(-3/5): at fixed chirp mass a more asymmetric binary is heavier and so carries a
    # larger 1PN correction. That is why eta cannot be assumed equal-mass here, and why the
    # lightest chirp mass in a batch is not necessarily its longest inspiral.
    total_mass_seconds = chirp_mass_seconds * eta ** (-3.0 / 5.0)
    x = (np.pi * total_mass_seconds * minimum_frequency) ** (2.0 / 3.0)
    relative_correction = (743.0 / 252.0 + 11.0 * eta / 3.0) * x
    return tau0 * (1.0 + relative_correction), relative_correction


def inspiral_requirement_seconds(
    chirp_mass_solar: np.ndarray | float,
    eta: np.ndarray | float,
    minimum_frequency: float,
    mtsun: float = MTSUN_SI,
) -> np.ndarray:
    """Return the pre-coalescence seconds each event needs: its 1PN inspiral plus headroom.

    The duration from :func:`inspiral_seconds`, grown by its own :func:`inspiral_margin` and then
    by :data:`SEGMENT_BUFFER_SECONDS`. One definition for every backend, so the LAL, gwsignal and
    ripple buffers cannot disagree about how much inspiral they promise to hold -- they differ only
    in how they round the result up to a transform length.

    Each event gets *its own* margin, and a batched caller takes the maximum over the resulting
    requirements. Taking ``max(duration)`` and ``max(margin)`` separately would apply one event's
    1PN correction to another event's duration -- and since the correction grows with total mass, a
    heavy short event would inflate the grid chosen for a light long one.

    Args:
        chirp_mass_solar: Detector-frame chirp mass(es) in solar masses.
        eta: Symmetric mass ratio(es), aligned with ``chirp_mass_solar``.
        minimum_frequency: Frequency (Hz) the signal starts at.
        mtsun: Solar mass in seconds; see :func:`inspiral_seconds`.

    Returns:
        The required seconds before coalescence, broadcast over the inputs.

    Raises:
        ValueError: As :func:`inspiral_seconds`.
    """
    inspiral, relative_correction = inspiral_seconds(chirp_mass_solar, eta, minimum_frequency, mtsun)
    return np.asarray(inspiral, dtype=float) * (1.0 + inspiral_margin(relative_correction)) + SEGMENT_BUFFER_SECONDS


def segment_sample_count(  # noqa: PLR0913
    chirp_mass_solar: float,
    minimum_frequency: float,
    sampling_frequency: float,
    *,
    eta: float,
    ringdown_fraction: float = DEFAULT_RINGDOWN_FRACTION,
    segment_duration: float | None = None,
) -> int:
    """Return an even sample count whose duration contains the full inspiral.

    Sized from :func:`inspiral_requirement_seconds` -- the 1PN chirp time with a proportional
    margin and a flat pad -- then rounded up to a power of two seconds, so the inspiral fits in the
    pre-coalescence portion of the buffer without cyclic wraparound. A fixed ``segment_duration``
    overrides the estimate.

    Previously this used the *0PN* chirp time plus the flat pad alone, leaving the real headroom to
    whatever the power-of-two rounding happened to supply. Where that fell short the inspiral began
    before the buffer and wrapped around it: a 40+1.4 binary with aligned spins of 0.99 at 7 Hz
    needs 236.7 s and was given 230.4 s.

    Args:
        chirp_mass_solar: Detector-frame chirp mass in solar masses.
        minimum_frequency: Low-frequency cutoff in Hz.
        sampling_frequency: Sample rate in Hz.
        eta: Symmetric mass ratio, ``m1 m2 / (m1 + m2)^2``. Required rather than defaulted: an
            equal-mass default silently *underestimates* the duration for an asymmetric binary,
            which is the direction that wraps an inspiral around the buffer.
        ringdown_fraction: Fraction of the segment reserved after coalescence.
        segment_duration: Fixed segment length in seconds, overriding the estimate.

    Returns:
        An even sample count, a power of two in duration.
    """
    if segment_duration is not None:
        seconds = segment_duration
    else:
        required = float(np.max(inspiral_requirement_seconds(chirp_mass_solar, eta, minimum_frequency)))
        seconds = max(required / (1.0 - ringdown_fraction), MIN_SEGMENT_SECONDS)
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
