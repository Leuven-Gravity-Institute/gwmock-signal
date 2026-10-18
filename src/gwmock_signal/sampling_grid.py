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
:func:`gwmock_signal.injection.inject_strain` then has to resample. It once did so with a cubic
spline, which measured against an analytic reference at a half-sample offset gave:

| tone frequency | on-lattice | off-lattice |
|---|---|---|
| 0.1 x Nyquist | exact | 2.2e-4 |
| 0.5 x Nyquist | exact | 1.2e-1 |
| 0.8 x Nyquist | exact | 4.9e-1 |

A cubic cannot represent 2.5 samples per cycle, so at high frequency the error approaches
the signal itself. ``inject_strain`` now resamples with the same windowed-sinc kernel as the
projection, but that is still a second resampling of a signal the device has already resampled
once, and it costs a pass over every injected sample.

The fix is to declare the lattice up front and have the *device* place each event on it. The
device already resamples at ``t - tau(t)`` with a windowed-sinc kernel, so the fractional
lattice offset folds into the shift it is already applying: one exact resampling instead of
two. Measured that way the alignment costs ~1e-12 across the band, and assembly becomes an
integer add again.

This module holds only the lattice arithmetic and its validation, so it imports neither JAX
nor GWpy and can be used from either side of the device boundary.

:class:`SamplingGrid` is defined in ``gwmock_data.sampling_grid`` and re-exported here.
"""

from __future__ import annotations

from gwmock_data.sampling_grid import SamplingGrid

__all__ = ["SamplingGrid"]
