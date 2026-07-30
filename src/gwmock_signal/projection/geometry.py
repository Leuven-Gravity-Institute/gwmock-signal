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
"""Detector geometry: the single source of truth for response tensors and locations.

These helpers reconstruct each detector's response tensor and Earth-fixed location
from LAL's cached detector registry. They are deliberately kept in one place (rather
than inlined in the projection code) so every consumer — the NumPy projection path
today, and a planned on-device JAX path — derives detector geometry from the same
constants instead of re-deriving them (a single source of truth).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, cast

import lal
import numpy as np
from astropy import coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix

if TYPE_CHECKING:
    from gwmock_signal.detector import CustomDetector

#: A detector named either by built-in LAL interferometer code or given explicitly.
DetectorSpec = "str | CustomDetector"


def resolve_detectors(detector_specs: Sequence[str | CustomDetector]) -> list[tuple[str, str]]:
    """Resolve detector specifications to ``(output_name, lal_lookup_key)`` pairs.

    Both the NumPy and the device projection paths need the same split, because the two are not
    interchangeable: a :class:`~gwmock_signal.detector.CustomDetector` is *looked up* by the
    two-character prefix it registers with LAL, but its output channel must be keyed by its own
    ``name``. Using one string for both -- as the device path did -- silently restricts that path
    to built-in interferometer codes, which is why ET presets could not reach it at all.

    Lives here rather than in either projection module so both derive the mapping from one
    implementation, consistent with this module being the single point of contact with LAL's
    registry.

    Args:
        detector_specs: Built-in LAL interferometer codes and/or ``CustomDetector`` instances.

    Returns:
        One ``(output_name, lal_lookup_key)`` pair per entry, in the order given. For a built-in
        code the two are the same string.

    Raises:
        TypeError: If an entry is neither a string nor a ``CustomDetector``.
        ValueError: If a string is not a detector LAL knows about.
    """
    from gwmock_signal.detector import CustomDetector  # noqa: PLC0415 — avoids an import cycle

    resolved: list[tuple[str, str]] = []
    for raw in detector_specs:
        if isinstance(raw, str):
            name = str(raw)
            # Resolved eagerly so an unknown code fails here, with the other detectors' names
            # still available for the message, rather than deep inside a jitted kernel.
            get_lal_detector(name)
            resolved.append((name, name))
        elif isinstance(raw, CustomDetector):
            resolved.append((raw.name, raw.to_lal().frDetector.prefix))
        else:
            raise TypeError(f"Unsupported detector specification type: {type(raw).__name__}")
    return resolved


def get_lal_detector(prefix: str) -> lal.Detector:
    """Return one detector from LAL's cached prefix registry."""
    try:
        return cast(lal.Detector, lal.cached_detector_by_prefix[prefix])
    except KeyError as exc:
        raise ValueError(
            f"Unknown or unsupported detector {prefix!r}. Use a valid LAL interferometer code (e.g. 'H1', 'L1', 'V1')."
        ) from exc


@cache
def reconstructed_geometry(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return detector response and location reconstructed from one LAL detector.

    Returns:
        A ``(response, location)`` tuple where ``response`` is the 3x3 detector
        response tensor and ``location`` is the Earth-fixed position (metres).
        Results are cached per detector prefix.
    """
    fr_detector = get_lal_detector(prefix).frDetector

    arm_response = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotation_longitude = rotation_matrix(-fr_detector.vertexLongitudeRadians * units.rad, "z")
    rotation_latitude = rotation_matrix(-(np.pi / 2.0 - fr_detector.vertexLatitudeRadians) * units.rad, "y")

    responses: list[np.ndarray] = []
    for azimuth, altitude in (
        (fr_detector.yArmAzimuthRadians, fr_detector.yArmAltitudeRadians),
        (fr_detector.xArmAzimuthRadians, fr_detector.xArmAltitudeRadians),
    ):
        rotation_azimuth = rotation_matrix(azimuth * units.rad, "z")
        rotation_altitude = rotation_matrix(-altitude * units.rad, "y")
        rotation = rotation_longitude @ rotation_latitude @ rotation_azimuth @ rotation_altitude
        responses.append(np.asarray(rotation @ arm_response @ rotation.T / 2.0, dtype=float))

    location = coordinates.EarthLocation.from_geodetic(
        fr_detector.vertexLongitudeRadians * units.rad,
        fr_detector.vertexLatitudeRadians * units.rad,
        height=fr_detector.vertexElevation * units.meter,
    )
    return responses[0] - responses[1], np.array([location.x.value, location.y.value, location.z.value], dtype=float)
