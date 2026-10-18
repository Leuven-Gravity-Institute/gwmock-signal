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

Only ``h5py`` is needed, imported when a function is called; GWpy is not. The functions are
``gwmock_data.io``'s, re-exported here.
"""

from __future__ import annotations

from gwmock_data.io import read_hdf5, write_hdf5

__all__ = ["read_hdf5", "write_hdf5"]
