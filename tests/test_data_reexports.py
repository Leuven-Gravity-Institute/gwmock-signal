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
"""The time series, its sampling grid and HDF5 I/O are gwmock-data's, re-exported unchanged.

They are the same objects rather than copies, so a series made by any gwmock package is an instance of
the class every other one checks for. The classes themselves are tested in gwmock-data.
"""

from __future__ import annotations

import gwmock_data.io
import gwmock_data.sampling_grid
import gwmock_data.timeseries

import gwmock_signal
import gwmock_signal.io
import gwmock_signal.io.hdf5
import gwmock_signal.sampling_grid
import gwmock_signal.timeseries


def test_the_time_series_comes_from_gwmock_data():
    """The class, its input helper and the pytree registration, here and at the package root."""
    assert gwmock_signal.timeseries.TimeSeries is gwmock_data.timeseries.TimeSeries
    assert gwmock_signal.timeseries.as_timeseries is gwmock_data.timeseries.as_timeseries
    assert gwmock_signal.timeseries.register_pytree is gwmock_data.timeseries.register_pytree
    assert gwmock_signal.TimeSeries is gwmock_data.timeseries.TimeSeries


def test_the_sampling_grid_comes_from_gwmock_data():
    """The lattice class, here and at the package root."""
    assert gwmock_signal.sampling_grid.SamplingGrid is gwmock_data.sampling_grid.SamplingGrid
    assert gwmock_signal.SamplingGrid is gwmock_data.sampling_grid.SamplingGrid


def test_hdf5_io_comes_from_gwmock_data():
    """Both I/O functions, from ``gwmock_signal.io`` and from ``gwmock_signal.io.hdf5``."""
    for module in (gwmock_signal.io, gwmock_signal.io.hdf5):
        assert module.write_hdf5 is gwmock_data.io.write_hdf5
        assert module.read_hdf5 is gwmock_data.io.read_hdf5
