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
        without knowing this length means cropping the start away -- measured at 32% of a
        30+25 solar-mass binary's strain-squared energy at 1024 Hz with 16-second segments, and
        99.998% for a binary neutron star, whose buffer can start before the run.

        Asked of the backend rather than computed by the caller on purpose. The length is a
        property of how each library conditions its output: this package's own frequency-domain
        backends size the buffer with
        :func:`~gwmock_signal.waveform.backends.conditioning.segment_sample_count`, ripple applies
        its own, and PyCBC and gwsignal delegate to theirs. A caller reproducing any of that would
        be a second implementation of a quantity that already exists, wrong differently per
        backend -- and wrong in the direction that silently truncates.

        Returns:
            Seconds between the first sample and coalescence, always positive. ``None`` when this
            backend cannot say, which callers must treat as "unknown" rather than "zero": the
            default is deliberately unhelpful because a wrong number is worse than none. A caller
            that gets ``None`` should keep whatever conservative behaviour it had.

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
