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
"""Top-level package for gwmock_signal."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gwmock_signal.version import __version__

_PUBLIC_SYMBOLS = {
    "CBCSimulator": ("gwmock_signal.simulator", "CBCSimulator"),
    "CustomDetector": ("gwmock_signal.detector", "CustomDetector"),
    "DetectorStrainStack": ("gwmock_signal.multichannel.stack", "DetectorStrainStack"),
    "GWSimulator": ("gwmock_signal.simulator", "GWSimulator"),
    "LALSimulationBackend": ("gwmock_signal.waveform.backends", "LALSimulationBackend"),
    "Network": ("gwmock_signal.network", "Network"),
    "TransientSimulator": ("gwmock_signal.simulator", "TransientSimulator"),
    "WaveformBackend": ("gwmock_signal.waveform.backends", "WaveformBackend"),
    "list_registered_source_types": ("gwmock_signal.registry", "list_registered_source_types"),
    "matched_filter_snr": ("gwmock_signal.snr._pycbc", "matched_filter_snr"),
    "optimal_snr": ("gwmock_signal.snr._pycbc", "optimal_snr"),
    "register_simulator_backend": ("gwmock_signal.registry", "register_simulator_backend"),
    "resolve_simulator_backend": ("gwmock_signal.registry", "resolve_simulator_backend"),
}


def __getattr__(name: str) -> Any:
    """Import public symbols lazily so optional dependencies stay optional."""
    try:
        module_name, attr_name = _PUBLIC_SYMBOLS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return standard module attributes plus lazy public exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "CBCSimulator",
    "CustomDetector",
    "DetectorStrainStack",
    "GWSimulator",
    "LALSimulationBackend",
    "Network",
    "TransientSimulator",
    "WaveformBackend",
    "__version__",
    "list_registered_source_types",
    "matched_filter_snr",
    "optimal_snr",
    "register_simulator_backend",
    "resolve_simulator_backend",
]
