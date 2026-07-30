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

# LAL exposes detector geometry via ``lal.FrDetector`` plus
# ``lal.CreateDetector(...)``. This module wraps that API behind a validated
# dataclass and caches the constructed detector for reuse.

"""User-defined ground-based detector geometry."""

from __future__ import annotations

import math
import string
import threading
import uuid
from dataclasses import dataclass, field

import lal

_PREFIX_ALPHABET = string.ascii_uppercase + string.digits
_LAL_PREFIX_LENGTH = 2


#: Serialises prefix allocation and registration against LAL's registry.
#:
#: ``lal.cached_detector_by_prefix`` is process-global and mutable, and registering a detector is a
#: check-then-act: read the registry to find a free prefix, then write to it. Two threads doing that
#: concurrently can settle on the same prefix, both register, and the last write wins. After that,
#: ``resolve_detectors`` hands both channels the same lookup key and, with it, the same geometry. A
#: module-level lock rather than a per-instance one, because the contended state belongs to LAL, not
#: to any one detector.
_REGISTRY_LOCK = threading.Lock()


def _generate_detector_prefix() -> str:
    """Generate one unused two-character detector prefix for LAL registration.

    Callers must hold :data:`_REGISTRY_LOCK`, since the search reads the registry that another
    thread may be writing to.
    """
    max_prefixes = len(_PREFIX_ALPHABET) ** 2
    seed = uuid.uuid4().int % max_prefixes

    for offset in range(max_prefixes):
        value = (seed + offset) % max_prefixes
        prefix = _PREFIX_ALPHABET[value // len(_PREFIX_ALPHABET)] + _PREFIX_ALPHABET[value % len(_PREFIX_ALPHABET)]
        if prefix not in lal.cached_detector_by_prefix:
            return prefix

    raise RuntimeError("Failed to allocate a unique LAL detector prefix.")


@dataclass(frozen=True)
class CustomDetector:
    """A user-defined ground-based gravitational-wave detector.

    All angular fields are in radians; elevation is in metres.

    Attributes:
        name: Short identifier used as the key in strain output dicts.
            Must be unique within a network.
        latitude_rad: Geodetic latitude of the vertex in radians.
            Must be in ``[-pi/2, pi/2]``.
        longitude_rad: Geodetic longitude of the vertex in radians.
            Must be in ``[-pi, pi]``.
        elevation_m: Vertex elevation above the WGS-84 ellipsoid in metres.
            Must be in ``[-1e4, 1e5]``.
        xarm_azimuth_rad: Azimuthal angle of the x-arm measured from
            geodetic North in radians.
        yarm_azimuth_rad: Azimuthal angle of the y-arm measured from
            geodetic North in radians.
        xarm_tilt_rad: Altitude angle of the x-arm above the local
            horizon in radians.  Defaults to ``0.0``.
        yarm_tilt_rad: Altitude angle of the y-arm above the local
            horizon in radians.  Defaults to ``0.0``.
        prefix: Optional detector prefix used to register the detector in
            ``lal.cached_detector_by_prefix``. When omitted, one unused
            two-character prefix is generated automatically.
    """

    name: str
    latitude_rad: float
    longitude_rad: float
    elevation_m: float
    xarm_azimuth_rad: float
    yarm_azimuth_rad: float
    xarm_tilt_rad: float = 0.0
    yarm_tilt_rad: float = 0.0
    prefix: str = ""

    # Cache for the constructed LAL detector; not exposed as a public attribute.
    _lal_prefix: str = field(default="", init=False, repr=False, compare=False)
    _lal_detector: lal.Detector | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate geodetic parameters on construction."""
        _elev_min = -1e4
        _elev_max = 1e5
        half_pi = math.pi / 2.0

        if not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        if self.prefix:
            if not self.prefix.strip():
                raise ValueError("prefix must be a non-empty string when provided.")
            if len(self.prefix) != _LAL_PREFIX_LENGTH:
                raise ValueError(f"prefix must be exactly two characters; got {self.prefix!r}.")

        for field_name, value in (
            ("xarm_azimuth_rad", self.xarm_azimuth_rad),
            ("yarm_azimuth_rad", self.yarm_azimuth_rad),
            ("xarm_tilt_rad", self.xarm_tilt_rad),
            ("yarm_tilt_rad", self.yarm_tilt_rad),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite; got {value!r}.")

        if not (-half_pi <= self.latitude_rad <= half_pi):
            raise ValueError(f"latitude_rad must be in [-pi/2, pi/2]; got {self.latitude_rad!r}.")
        if not (-math.pi <= self.longitude_rad <= math.pi):
            raise ValueError(f"longitude_rad must be in [-pi, pi]; got {self.longitude_rad!r}.")
        if not (_elev_min <= self.elevation_m <= _elev_max):
            raise ValueError(f"elevation_m must be in [-1e4, 1e5] m; got {self.elevation_m!r}.")
        if self.prefix:
            object.__setattr__(self, "_lal_prefix", self.prefix)
        else:
            with _REGISTRY_LOCK:
                object.__setattr__(self, "_lal_prefix", _generate_detector_prefix())

    def to_lal(self) -> lal.Detector:
        """Return a cached :class:`lal.Detector` for this geometry.

        The first call constructs a detector from a ``lal.FrDetector`` and
        registers it in ``lal.cached_detector_by_prefix`` so the projection
        layer can resolve built-in and custom detectors through one lookup
        path. Subsequent calls return the cached object.

        Thread-safe: allocation, the clash check and registration happen together under
        :data:`_REGISTRY_LOCK`. Without that, two threads could each find the same prefix free and
        both register it, leaving one geometry reachable under a key two detectors believe they own.

        Returns:
            A :class:`lal.Detector` instance configured with
            this detector's geodetic coordinates and arm orientations.
        """
        # Unlocked fast path for the common case of repeated calls on a registered detector. Safe
        # because _lal_detector is only ever set once, under the lock, and never cleared.
        if self._lal_detector is not None:
            return self._lal_detector

        with _REGISTRY_LOCK:
            # Re-checked: another thread may have finished registering while this one waited.
            if self._lal_detector is not None:
                return self._lal_detector
            detector_prefix = self._lal_prefix
            if detector_prefix in lal.cached_detector_by_prefix:
                if self.prefix:
                    # User explicitly chose this prefix: a clash is a real error.
                    raise ValueError(
                        f"Detector prefix {detector_prefix!r} is already registered in LAL; "
                        "choose a unique prefix for CustomDetector."
                    )
                # Auto-generated prefix: it was allocated at construction against the
                # registry as it was then, but another detector (e.g. a sibling in the
                # same network) has since registered the same prefix. Allocate a fresh
                # one now, atomically with registration below, to close that window.
                detector_prefix = _generate_detector_prefix()
                object.__setattr__(self, "_lal_prefix", detector_prefix)
            fr_detector = lal.FrDetector()
            fr_detector.name = self.name
            fr_detector.prefix = detector_prefix
            fr_detector.vertexLongitudeRadians = self.longitude_rad
            fr_detector.vertexLatitudeRadians = self.latitude_rad
            fr_detector.vertexElevation = self.elevation_m
            fr_detector.xArmAzimuthRadians = self.xarm_azimuth_rad
            fr_detector.yArmAzimuthRadians = self.yarm_azimuth_rad
            fr_detector.xArmAltitudeRadians = self.xarm_tilt_rad
            fr_detector.yArmAltitudeRadians = self.yarm_tilt_rad

            detector = lal.CreateDetector(
                lal.Detector(),
                fr_detector,
                lal.LALDETECTORTYPE_IFODIFF,
            )
            if detector is None:
                raise RuntimeError(f"Failed to register detector {self.name!r} with LAL.")

            lal.cached_detector_by_prefix[detector_prefix] = detector
            object.__setattr__(self, "_lal_detector", detector)

        if self._lal_detector is None:
            raise RuntimeError(f"Failed to register detector {self.name!r} with LAL.")
        return self._lal_detector
