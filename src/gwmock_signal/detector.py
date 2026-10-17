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
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from gwmock_signal import _lalsuite
from gwmock_signal._detector_table import lal_storage_precision

if TYPE_CHECKING:
    import lal

#: What a missing ``lalsuite`` stops, as named in the error.
_LAL_FEATURE = "Registering a CustomDetector with LAL's detector registry"

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


def _generate_detector_prefix(lal: ModuleType) -> str:
    """Generate one unused two-character detector prefix for LAL registration.

    Callers must hold :data:`_REGISTRY_LOCK`, since the search reads the registry that another
    thread may be writing to. They pass ``lal`` in, resolved before taking the lock, so the lock is
    never held across the first import of the extension.
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
        xarm_azimuth_rad: Azimuthal angle of the x-arm measured
            clockwise from geodetic North (LAL convention) in radians.
        yarm_azimuth_rad: Azimuthal angle of the y-arm measured
            clockwise from geodetic North (LAL convention) in radians.
        xarm_tilt_rad: Altitude angle of the x-arm above the local
            horizon in radians.  Defaults to ``0.0``.
        yarm_tilt_rad: Altitude angle of the y-arm above the local
            horizon in radians.  Defaults to ``0.0``.
        prefix: Optional detector prefix used to register the detector in
            ``lal.cached_detector_by_prefix`` by :meth:`to_lal`. When omitted, an
            unused two-character prefix is generated when :meth:`to_lal` is first
            called. Projection does not register the detector, so it needs neither.
        precision: ``"double"`` (default) projects with the fields exactly as given.
            ``"single"`` rounds the elevation and the four arm angles to single
            precision first, as LAL stores them, so projection reproduces what a
            LAL-registered copy of this detector gives. Before ``precision`` existed,
            projection always went through LAL and so always used single precision;
            the double default changes the projected strain at that rounding, about
            1.5e-7 of peak for an ET-like detector.
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
    precision: Literal["double", "single"] = "double"

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
        self._check_precision()
        if self.prefix:
            object.__setattr__(self, "_lal_prefix", self.prefix)

    def _check_precision(self) -> None:
        """Reject an unknown precision, and single-precision fields that single precision cannot hold."""
        if self.precision not in ("double", "single"):
            raise ValueError(f"precision must be 'double' or 'single'; got {self.precision!r}.")
        if self.precision == "single":
            try:
                self.geometry_fields()
            except OverflowError as exc:
                raise ValueError(
                    "precision='single' needs the elevation and arm angles within the single-precision range."
                ) from exc

    def geometry_fields(self) -> tuple[float, ...]:
        """Return the fields that define this detector's projection geometry.

        In the order :func:`gwmock_signal.projection.geometry.reconstructed_geometry` takes them:
        vertex longitude, latitude and elevation, x- and y-arm azimuth, x- and y-arm altitude. With
        ``precision="single"`` the elevation and the arm angles are rounded to single precision,
        which LAL uses for those fields; the longitude and latitude stay double, as in LAL.
        """
        fields = (
            float(self.longitude_rad),
            float(self.latitude_rad),
            float(self.elevation_m),
            float(self.xarm_azimuth_rad),
            float(self.yarm_azimuth_rad),
            float(self.xarm_tilt_rad),
            float(self.yarm_tilt_rad),
        )
        return lal_storage_precision(fields) if self.precision == "single" else fields

    def to_lal(self) -> lal.Detector:
        """Return a cached :class:`lal.Detector` for this geometry.

        The first call allocates a prefix if none was given, constructs a detector
        from a ``lal.FrDetector`` and registers it in ``lal.cached_detector_by_prefix``,
        for code that needs a LAL detector; projection does not, and builds the
        geometry from :meth:`geometry_fields`. Subsequent calls return the cached
        object. Requires ``lalsuite``.

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

        lal = _lalsuite.require("lal", _LAL_FEATURE)
        with _REGISTRY_LOCK:
            # Re-checked: another thread may have finished registering while this one waited.
            if self._lal_detector is not None:
                return self._lal_detector
            detector_prefix = self._lal_prefix
            if detector_prefix and detector_prefix in lal.cached_detector_by_prefix:
                if self.prefix:
                    # User explicitly chose this prefix: a clash is a real error.
                    raise ValueError(
                        f"Detector prefix {detector_prefix!r} is already registered in LAL; "
                        "choose a unique prefix for CustomDetector."
                    )
                # An auto-generated prefix left over from a registration that failed, and since
                # taken by another detector: allocate afresh.
                detector_prefix = ""
            if not detector_prefix:
                # No explicit prefix: allocate one now, under the lock and atomically with the
                # registration below, so no other detector can claim it in between.
                detector_prefix = _generate_detector_prefix(lal)
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
