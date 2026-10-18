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
"""Read and write time series as HDF5, in the layout GWpy's ``TimeSeriesDict`` uses.

One dataset per series, named by its key and gzip-compressed, with the attributes GWpy writes and
reads: ``x0`` (the GPS time of the first sample), ``dx`` (the sample spacing, in seconds),
``xunit`` (``"s"``), ``unit`` and, for a series that has one, ``name``. A file written here therefore loads with
``gwpy.timeseries.TimeSeriesDict.read(path, format="hdf5")``, and a file GWpy wrote loads here.

Only ``h5py`` is needed, imported when a function is called; GWpy is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from gwmock_signal._optional import require
from gwmock_signal.timeseries import TimeSeries, as_timeseries

__all__ = ["read_hdf5", "write_hdf5"]


def write_hdf5(series: Mapping[str, TimeSeries], path: str | Path) -> None:
    """Write each series to ``path`` as one HDF5 dataset named by its key, replacing the file.

    Args:
        series: Mapping of dataset name to a one-dimensional time series (this package's or
            GWpy's).
        path: Output file.

    Raises:
        ValueError: If a series is not one-dimensional.
        ModuleNotFoundError: If ``h5py`` is not installed.
    """
    h5py = require("h5py", "hdf5", "HDF5 output")
    channels = {key: as_timeseries(value) for key, value in series.items()}
    for key, channel in channels.items():
        if len(channel.data.shape) != 1:
            raise ValueError(f"{key!r} must be one-dimensional to write; it has shape {tuple(channel.data.shape)}.")
    with h5py.File(Path(path), "w") as handle:
        for key, channel in channels.items():
            dataset = handle.create_dataset(key, data=np.asarray(channel.data), compression="gzip")
            dataset.attrs["x0"] = channel.t0
            dataset.attrs["dx"] = channel.dt
            dataset.attrs["xunit"] = "s"
            dataset.attrs["unit"] = channel.unit or ""
            if channel.name is not None:
                dataset.attrs["name"] = channel.name


def read_hdf5(path: str | Path) -> dict[str, TimeSeries]:
    """Read every dataset in ``path`` written in the layout :func:`write_hdf5` writes.

    Args:
        path: Input file.

    Returns:
        Mapping of dataset name to :class:`~gwmock_signal.timeseries.TimeSeries`, in file order.

    Raises:
        ValueError: If a dataset lacks the ``x0`` or ``dx`` attribute, or its time unit is not seconds.
        ModuleNotFoundError: If ``h5py`` is not installed.
    """
    h5py = require("h5py", "hdf5", "HDF5 input")
    result = {}
    with h5py.File(Path(path), "r") as handle:
        for key, dataset in handle.items():
            attrs = dataset.attrs
            if "x0" not in attrs or "dx" not in attrs:
                raise ValueError(f"dataset {key!r} has no x0 and dx attributes, so it is not a time series.")
            xunit = _text(attrs.get("xunit", "s"))
            if xunit not in ("s", "second"):
                raise ValueError(f"dataset {key!r} has a time axis in {xunit!r}; only seconds are supported.")
            name = _text(attrs["name"]) if "name" in attrs else None
            result[key] = TimeSeries(
                dataset[...],
                t0=float(attrs["x0"]),
                sample_rate=1.0 / float(attrs["dx"]),
                name=name,
                unit=_text(attrs.get("unit", "")) or None,
            )
    return result


def _text(value: object) -> str:
    """Return an HDF5 string attribute as ``str``, whether h5py read it as bytes or text."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
