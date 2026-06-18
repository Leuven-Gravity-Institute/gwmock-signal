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
"""Waveform backend interfaces and concrete implementations."""

from __future__ import annotations

from gwmock_signal.waveform.backends.base import WaveformBackend
from gwmock_signal.waveform.backends.lal import LALSimulationBackend
from gwmock_signal.waveform.backends.pycbc import PyCBCBackend
from gwmock_signal.waveform.backends.ripple import RippleBackend

__all__ = [
    "LALSimulationBackend",
    "PyCBCBackend",
    "RippleBackend",
    "WaveformBackend",
]
