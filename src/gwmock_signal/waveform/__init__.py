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
"""Waveform generation and backend abstractions."""

from __future__ import annotations

from importlib import import_module

from gwmock_signal.waveform.backends import LALSimulationBackend, PyCBCBackend, RippleBackend, WaveformBackend
from gwmock_signal.waveform.factory import WaveformFactory


def __getattr__(name: str):
    """Resolve optional waveform helpers lazily."""
    if name == "pycbc_waveform_wrapper":
        value = getattr(import_module("gwmock_signal.waveform.pycbc_wrapper"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LALSimulationBackend",
    "PyCBCBackend",
    "RippleBackend",
    "WaveformBackend",
    "WaveformFactory",
    "pycbc_waveform_wrapper",
]
