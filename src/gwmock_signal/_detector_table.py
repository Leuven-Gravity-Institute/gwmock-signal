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

A code that is not in the table -- a detector registered with LAL at run time, for instance by
:meth:`~gwmock_signal.detector.CustomDetector.to_lal` -- is read from LAL's registry, which needs
``lalsuite``. Projecting a CustomDetector itself uses its own fields and needs neither.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType

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


def lal_storage_precision(fields: tuple[float, ...]) -> tuple[float, ...]:
    """Return ``fields`` as LAL stores them: elevation and arm angles in single precision.

    ``LALFrDetector`` keeps the vertex longitude and latitude as double and the elevation and the
    four arm angles as single precision, so a detector registered with LAL carries these values.

    Raises:
        OverflowError: If a finite value is beyond the single-precision range.
    """
    longitude, latitude, *single = fields
    return (float(longitude), float(latitude), *(_to_single(value) for value in single))


def _to_single(value: float) -> float:
    """Round ``value`` to single precision, refusing to turn a finite value into an infinity.

    ``struct`` itself is not enough: Python 3.12 packs such a value as an infinity, and only later
    versions raise.
    """
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        rounded = math.inf
    if math.isinf(rounded) and math.isfinite(value):
        raise OverflowError(f"{value!r} is beyond the single-precision range.")
    return rounded


@cache
def builtin_detectors() -> Mapping[str, tuple[float, ...]]:
    """Return the defining fields of every built-in detector, keyed by its LAL code.

    Read-only: the result is cached and shared by every caller, so it must not be mutable.
    """
    table = json.loads(files("gwmock_signal.data").joinpath("lal_detectors.json").read_text())
    return MappingProxyType({code: tuple(float(entry[field]) for field in FIELDS) for code, entry in table.items()})


def unknown_code_help(*, lal_available: bool) -> str:
    """Return what to use instead of an unknown detector code, after each caller's own first sentence.

    Projection and :class:`~gwmock_signal.network.Network` keep their established first sentences,
    which callers may match on, and share this explanation.
    """
    message = (
        f"Use a built-in interferometer code ({', '.join(sorted(builtin_detectors()))}), "
        "or a CustomDetector for other geometries."
    )
    if not lal_available:
        message += " Codes registered with LAL at run time also need lalsuite, which could not be imported."
    return message
