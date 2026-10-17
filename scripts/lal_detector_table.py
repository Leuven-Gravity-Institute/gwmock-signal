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


"""Write LAL's built-in detector registry to the table ``gwmock_signal`` ships.

Projection needs, for each detector, the seven numbers that fully determine its response tensor and
location: the vertex longitude, latitude and elevation, and each arm's azimuth and altitude. For
LAL's built-in interferometer codes (``H1``, ``V1``, ...) these come from ``lalsuite``. The package
ships them in ``src/gwmock_signal/data/lal_detectors.json`` so that projecting onto a built-in
detector does not need ``lalsuite`` installed; ``tests/unit/test_detector_table.py`` requires the
table to equal LAL's registry exactly whenever ``lalsuite`` is installed.

Run it to regenerate the table after a ``lalsuite`` release changes a built-in detector::

    uv run --no-project --with lalsuite python scripts/lal_detector_table.py

It lives in ``scripts/`` rather than in ``src/``: it is maintenance tooling and has no business in the
wheel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TABLE = Path(__file__).resolve().parents[1] / "src" / "gwmock_signal" / "data" / "lal_detectors.json"

#: ``FrDetector`` attribute for each field of the table, in the order projection consumes them.
FIELDS = {
    "vertex_longitude_rad": "vertexLongitudeRadians",
    "vertex_latitude_rad": "vertexLatitudeRadians",
    "vertex_elevation_m": "vertexElevation",
    "xarm_azimuth_rad": "xArmAzimuthRadians",
    "yarm_azimuth_rad": "yArmAzimuthRadians",
    "xarm_altitude_rad": "xArmAltitudeRadians",
    "yarm_altitude_rad": "yArmAltitudeRadians",
}


def registry_table(registry) -> dict[str, dict[str, object]]:
    """Return the table for every detector in ``registry`` (``lal.cached_detector_by_prefix``)."""
    table = {}
    for code in sorted(str(code) for code in registry):
        fr_detector = registry[code].frDetector
        table[code] = {"name": str(fr_detector.name)} | {
            field: float(getattr(fr_detector, attribute)) for field, attribute in FIELDS.items()
        }
    return table


def main() -> int:
    """Regenerate the table from the installed ``lalsuite``."""
    import lal  # noqa: PLC0415 -- the only reason to run this script

    # Four-space indent, as the repository's prettier configuration formats JSON.
    TABLE.write_text(json.dumps(registry_table(lal.cached_detector_by_prefix), indent=4) + "\n")
    print(f"wrote {TABLE.relative_to(Path.cwd()) if TABLE.is_relative_to(Path.cwd()) else TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
