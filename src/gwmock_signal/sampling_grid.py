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
"""One canonical sample lattice for a batch of output segments.

Superposing a signal onto an output segment is *exact* when the two share a sample
lattice: the operation is an integer-offset add. When they do not, something has to
resample, and that resampling sets the accuracy of the whole pipeline.

That is not hypothetical. Coalescence times are continuous, so a signal buffer starting at
``epoch + coa_time`` essentially never lands on the output lattice, and
:func:`gwmock_signal.injection.inject_strain` then resamples with a cubic spline. Measured
against an analytic reference, at a half-sample offset:

| tone frequency | on-lattice | off-lattice |
|---|---|---|
| 0.1 x Nyquist | exact | 2.2e-4 |
| 0.5 x Nyquist | exact | 1.2e-1 |
| 0.8 x Nyquist | exact | 4.9e-1 |

A cubic cannot represent 2.5 samples per cycle, so at high frequency the error approaches
the signal itself. Meanwhile the device projection is accurate to ~1e-12, so without a shared
lattice the assembly step throws that away.

The fix is to declare the lattice up front and have the *device* place each event on it. The
device already resamples at ``t - tau(t)`` with a windowed-sinc kernel, so the fractional
lattice offset folds into the shift it is already applying: one exact resampling instead of an
exact one followed by a cubic one. Measured that way the alignment costs ~1e-12 across the
band, and assembly becomes an integer add again.

This module holds only the lattice arithmetic and its validation, so it imports neither JAX
nor GWpy and can be used from either side of the device boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Timestamp ULPs of slack allowed when deciding whether a time is on the lattice. The
#: tolerance has to be derived from float64 resolution rather than fixed: one ULP of a GPS
#: timestamp near 1.4e9 is about 2.4e-7 s, which at 2048 Hz is already 4.9e-4 samples. A fixed
#: 1e-6-sample tolerance -- the first thing written here -- was therefore some 500x *tighter*
#: than the timestamps can represent, and would have falsely rejected lattice times produced by
#: ordinary arithmetic. It survived initial testing only because 64 s at 2048 Hz happens to be
#: exact in binary; a non-power-of-two segment duration would have broken it.
_LATTICE_TOLERANCE_ULPS = 8.0

#: Smallest tolerance to use, in samples, so that a tiny epoch does not make the check
#: unreasonably strict.
_MINIMUM_TOLERANCE_SAMPLES = 1e-9

#: Largest tolerance to use, in samples. Well below the half-sample displacements the check
#: exists to catch, so no plausible ULP growth can make it accept a genuinely off-lattice time.
_MAXIMUM_TOLERANCE_SAMPLES = 1e-2


@dataclass(frozen=True)
class SamplingGrid:
    """A uniform sample lattice: sample ``k`` sits at ``epoch + k / sampling_frequency``.

    Args:
        epoch: GPS time of sample zero.
        sampling_frequency: Sample rate in Hz.

    Raises:
        ValueError: If ``sampling_frequency`` is not positive and finite, or ``epoch`` is not
            finite.
    """

    epoch: float
    sampling_frequency: float

    def __post_init__(self) -> None:
        """Validate the lattice definition."""
        if not np.isfinite(self.epoch):
            raise ValueError(f"epoch must be finite; got {self.epoch}.")
        if not np.isfinite(self.sampling_frequency) or self.sampling_frequency <= 0.0:
            raise ValueError(f"sampling_frequency must be positive and finite; got {self.sampling_frequency}.")

    @property
    def sample_spacing(self) -> float:
        """Seconds between samples."""
        return 1.0 / self.sampling_frequency

    def lattice_tolerance_samples(self, gps_time: np.ndarray | float) -> float:
        """Return the on-lattice tolerance, in samples, appropriate to these timestamps.

        Scaled by the float64 resolution of the times involved, because at GPS magnitudes that
        resolution is coarser than any fixed tolerance worth writing down: near 1.4e9 one ULP
        is already ~5e-4 samples at 2048 Hz.

        An empty input is answered from the epoch alone. A chunked or filtered caller can
        legitimately arrive with nothing to check, and reducing over an empty array would raise
        here instead of letting the (vacuously satisfied) lattice check return empty.
        """
        times = np.asarray(gps_time, dtype=float)
        magnitude = abs(float(self.epoch))
        if times.size:
            magnitude = max(magnitude, float(np.max(np.abs(times))))
        ulp_samples = float(np.spacing(magnitude)) * self.sampling_frequency
        return float(
            np.clip(_LATTICE_TOLERANCE_ULPS * ulp_samples, _MINIMUM_TOLERANCE_SAMPLES, _MAXIMUM_TOLERANCE_SAMPLES)
        )

    def index_of(self, gps_time: np.ndarray | float) -> np.ndarray:
        """Return the (generally fractional) lattice index of *gps_time*.

        The subtraction is done before the multiplication: GPS times are ~1.4e9 and the
        spacing is ~5e-4 s, so scaling first would lose the sub-sample part entirely.
        """
        return (np.asarray(gps_time, dtype=float) - self.epoch) * self.sampling_frequency

    def time_of(self, index: np.ndarray | int) -> np.ndarray:
        """Return the GPS time of lattice index *index*."""
        return self.epoch + np.asarray(index, dtype=float) * self.sample_spacing

    def split_index(self, gps_time: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Split *gps_time* into the preceding lattice index and a fractional remainder.

        Returns:
            ``(index, fraction)`` with ``fraction`` in ``[0, 1)``, such that
            ``gps_time == time_of(index) + fraction * sample_spacing``. The fraction is what
            the device must absorb for the sample to land on the lattice.
        """
        exact = self.index_of(gps_time)
        index = np.floor(exact)
        return index.astype(np.int64), exact - index

    def is_on_lattice(self, gps_time: np.ndarray | float) -> np.ndarray:
        """Return whether each *gps_time* coincides with a lattice sample."""
        exact = self.index_of(gps_time)
        return np.abs(exact - np.round(exact)) <= self.lattice_tolerance_samples(gps_time)

    def require_on_lattice(self, gps_time: np.ndarray | float, *, name: str) -> np.ndarray:
        """Return integer lattice indices, rejecting any time that is off-lattice.

        Rejected rather than silently rounded: rounding would move a signal by up to half a
        sample without telling anyone, and silently accepting an off-lattice time is exactly
        how the cubic-resampling error described in this module's docstring got in.

        Args:
            gps_time: Time(s) that must coincide with lattice samples.
            name: What is being checked, for the error message.

        Returns:
            Integer lattice indices, same shape as the input.

        Raises:
            ValueError: If any time is not on the lattice.
        """
        times = np.atleast_1d(np.asarray(gps_time, dtype=float))
        exact = self.index_of(times)
        offenders = np.abs(exact - np.round(exact)) > self.lattice_tolerance_samples(times)
        if np.any(offenders):
            first = int(np.argmax(offenders))
            raise ValueError(
                f"{name} must lie on the sampling grid (epoch={self.epoch!r}, "
                f"sampling_frequency={self.sampling_frequency!r}), but entry {first} at "
                f"{times[first]!r} is {exact[first] - round(float(exact[first])):+.6g} samples off. "
                f"Rounding it would displace the data by up to half a sample; choose segment "
                f"starts on the grid, or build the grid from them."
            )
        return np.round(exact).astype(np.int64).reshape(np.shape(gps_time))

    @classmethod
    def from_segment_starts(cls, segment_start_times: np.ndarray, sampling_frequency: float) -> SamplingGrid:
        """Build a grid anchored on the first segment start.

        Args:
            segment_start_times: GPS start times, which must themselves be mutually
                consistent with one lattice at this sample rate.
            sampling_frequency: Sample rate in Hz.

        Returns:
            A grid whose epoch is the first segment start.

        Raises:
            ValueError: If ``segment_start_times`` is empty, or the starts do not share one
                lattice — which would make a single canonical grid impossible.
        """
        starts = np.atleast_1d(np.asarray(segment_start_times, dtype=float))
        if starts.size == 0:
            raise ValueError("segment_start_times must not be empty.")
        grid = cls(epoch=float(starts[0]), sampling_frequency=sampling_frequency)
        grid.require_on_lattice(starts, name="segment_start_times")
        return grid
