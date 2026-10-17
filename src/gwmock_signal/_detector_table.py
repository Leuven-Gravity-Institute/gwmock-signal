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
"""LAL's built-in detectors, shipped with the package so projection does not need ``lalsuite``.

Projection needs, for each detector, the seven numbers that fully determine its response tensor and
location (:data:`FIELDS`). For LAL's built-in interferometer codes (``H1``, ``V1``, ...) the package
reads them from ``data/lal_detectors.json``, a copy of LAL's registry written by
``scripts/lal_detector_table.py``. The test suite requires the copy to equal LAL's registry exactly
whenever ``lalsuite`` is installed, so a code resolves to the same geometry with or without it.

A code that is not in the table -- a detector registered with LAL at run time, including a
:class:`~gwmock_signal.detector.CustomDetector` -- is read from LAL's registry, which needs
``lalsuite``.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files

#: The table's fields, in the order :func:`gwmock_signal.projection.geometry.reconstructed_geometry`
#: consumes them.
FIELDS = (
    "vertex_longitude_rad",
    "vertex_latitude_rad",
    "vertex_elevation_m",
    "xarm_azimuth_rad",
    "yarm_azimuth_rad",
    "xarm_altitude_rad",
    "yarm_altitude_rad",
)


@cache
def builtin_detectors() -> dict[str, tuple[float, ...]]:
    """Return the defining fields of every built-in detector, keyed by its LAL code."""
    table = json.loads(files("gwmock_signal.data").joinpath("lal_detectors.json").read_text())
    return {code: tuple(float(entry[field]) for field in FIELDS) for code, entry in table.items()}
