"""I/O helpers for external detector and strain data formats."""

from __future__ import annotations

from gwmock_signal.io.hdf5 import read_hdf5, write_hdf5
from gwmock_signal.io.interferometer_format import (
    interferometer_config_to_custom_detector,
    read_interferometer_config,
    resolve_interferometer_config_path,
)

__all__ = [
    "interferometer_config_to_custom_detector",
    "read_hdf5",
    "read_interferometer_config",
    "resolve_interferometer_config_path",
    "write_hdf5",
]
