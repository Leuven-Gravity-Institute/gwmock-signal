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
"""Waveform backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from gwpy.timeseries import TimeSeries

_MISSING: Final[object] = object()


def _pop_alias(params: dict[str, object], canonical: str, *aliases: str, default: object = _MISSING) -> object:
    """Pop one canonical parameter or legacy alias from ``params``.

    Raises ``ValueError`` when multiple aliases are provided simultaneously so the
    caller cannot accidentally pass conflicting values.
    """
    names = (canonical, *aliases)
    present = [name for name in names if name in params]
    if len(present) > 1:
        joined = ", ".join(present)
        raise ValueError(f"Do not mix aliases for '{canonical}': {joined}")
    if present:
        return params.pop(present[0])
    if default is _MISSING:
        raise ValueError(f"Missing required parameter: '{canonical}'")
    return default


class WaveformBackend(ABC):
    """Abstract interface for time-domain waveform generators."""

    @abstractmethod
    def available_approximants(self) -> list[str]:
        """Return supported time-domain approximant names."""

    @abstractmethod
    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate ``plus`` and ``cross`` GWpy time series."""

    def pre_coalescence_duration(
        self,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> float | None:
        """Return how long before ``tc`` the generated waveform starts, in seconds.

        A caller placing a signal in segmented data needs this *before* generating: a compact
        binary's inspiral precedes its coalescence, so a buffer whose ``tc`` sits just past a
        segment boundary begins in an earlier segment. Deciding which segment claims an event
        without knowing this length means cropping the start away.

        **How much that costs depends on the low-frequency cutoff, and a figure quoted without one is
        not interpretable.** Measured with IMRPhenomD at 1024 Hz into a single detector (H1), ``tc``
        0.5 s past a segment boundary, 20 Hz cutoff against 30 Hz:

        =========================  ==============  ==============
        30+25 solar masses         20 Hz           30 Hz
        =========================  ==============  ==============
        buffer                     4.000 s         4.000 s
        dropped span               3.100 s (77.5%) 3.100 s (77.5%)
        dropped ``h**2``           **32.3%**       **0.91%**
        =========================  ==============  ==============

        For this binary the cutoff changes the *content* of the dropped span and not its geometry:
        the conditioning rounds both cutoffs to the same power of two, so the span is identical while
        the energy in it differs by a factor of 35 -- at 30 Hz those early samples lie below the
        cutoff and are near-silent, and at 20 Hz they also enlarge the total the fraction is taken
        against. Across offsets, energy lost at 20 Hz against 30 Hz: 99.9%/99.8% at 1 ms past the
        boundary, 72.8%/48.7% at 0.1 s, 54.2%/10.5% at 0.25 s, 32.3%/0.91% at 0.5 s, 2.9%/0.34% at
        1 s. The gap is widest in the middle, where the dropped span covers just the band between the
        two cutoffs.

        A binary neutron star is worse in absolute terms and differs in kind, because there the
        cutoff moves the geometry too -- the chirp time dominates the rounding:

        =========================  ==============  ==============
        1.4+1.35 solar masses      20 Hz           30 Hz
        =========================  ==============  ==============
        lead                       230.4 s         57.6 s
        buffer                     256.0 s         64.0 s
        dropped span               229.9 s (89.8%) 57.1 s (89.2%)
        dropped ``h**2``           **96.1%**       **93.1%**
        =========================  ==============  ==============

        The detector network shifts these by around a percentage point -- 32.9% rather than 32.3% at
        20 Hz for a three-detector ET triangle -- so it is worth naming alongside the cutoff, but it
        is not what makes the figures differ by orders of magnitude.

        These are unweighted ``h**2`` fractions -- a proxy, not a matched-filter SNR loss, which needs
        a detector PSD and frequency-domain weighting. **None is anchored against an external SNR
        tool.** They are this package's own measurement of its own output.

        Asked of the backend rather than computed by the caller on purpose. The length is a
        property of how each library conditions its output: the LAL backend sizes with
        :func:`~gwmock_signal.waveform.backends.conditioning.segment_sample_count`, the gwsignal
        backend inherits that because it overrides only the frequency-domain evaluation, ripple
        applies its own 5-smooth sizing, and PyCBC delegates to its library. A caller reproducing
        any of that would be a second implementation of a quantity that already exists, wrong
        differently per backend -- and wrong in the direction that silently truncates.

        **This is where the buffer starts, not where audible signal begins.** The buffer carries
        headroom beyond the estimated chirp time and is rounded up, so the first samples are
        near-silent: a 30+25 solar-mass binary reports 3.6 s while carrying roughly 1.1 s of
        inspiral. That is the safe direction for choosing a segment -- placing from this value
        never crops real signal -- but it is not a statement about signal duration.

        Returns:
            Seconds between the first sample and coalescence, always positive. ``None`` when this
            backend cannot say, which callers must treat as "unknown" rather than "zero": the
            default is deliberately unhelpful because a wrong number is worse than none. A caller
            that gets ``None`` should keep whatever conservative behaviour it had.

            Test for ``None`` explicitly. ``if duration:`` is a trap -- it is also false for
            ``0.0``, and treating an unknown length as zero places every event in the segment
            holding its coalescence, which is the behaviour this method exists to avoid.

            Of the backends here, only PyCBC returns ``None``.

        Args:
            approximant: The approximant that will be generated. Accepted because a backend may
                condition differently per family, even though the current ones do not.
            sampling_frequency: Sample rate in Hz, which sets the sample count.
            minimum_frequency: Low-frequency cutoff in Hz; the dominant term in the chirp time.
            **params: The source parameters that will be generated, in the same form
                ``generate_td_waveform`` takes.
        """
        del approximant, sampling_frequency, minimum_frequency, params
        return None
