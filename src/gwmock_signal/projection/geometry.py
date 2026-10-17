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
from its defining fields: for LAL's built-in codes, from the table shipped with the package
(:mod:`gwmock_signal._detector_table`); for a detector registered with LAL at run time, from
LAL's registry. They are deliberately kept in one place (rather
than inlined in the projection code) so every consumer — the NumPy projection path
today, and a planned on-device JAX path — derives detector geometry from the same
constants instead of re-deriving them (a single source of truth).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, cast

import numpy as np

from gwmock_signal import _lalsuite
from gwmock_signal._detector_table import builtin_detectors, unknown_code_help
from gwmock_signal.detector import CustomDetector
from gwmock_signal.projection.kernels import detector_geometry

if TYPE_CHECKING:
    import lal

#: How a caller names a detector: a built-in LAL interferometer code, or an explicit geometry.
#: Defined here rather than in either projection module, so both paths mean the same thing by it.
#: ``CustomDetector`` is imported at runtime, not under ``TYPE_CHECKING``: a PEP 695 alias is
#: lazily evaluated, so a type-only import leaves ``DetectorSpec.__value__`` raising NameError for
#: anything that introspects it. ``gwmock_signal.detector`` imports only stdlib and ``gwmock_signal._lalsuite``, and the
#: package ``__init__`` resolves submodules lazily, so there is no cycle in either direction.
type DetectorSpec = str | CustomDetector


#: How projection looks a detector up: a detector code (built in, or registered with LAL at run time),
#: or -- for a :class:`~gwmock_signal.detector.CustomDetector` -- its defining fields. Hashable either
#: way, because the device path caches compiled kernels by it.
type DetectorKey = str | tuple[float, ...]


def resolve_detectors(detector_specs: Sequence[DetectorSpec]) -> list[tuple[str, DetectorKey]]:
    """Resolve detector specifications to ``(output_name, lookup_key)`` pairs.

    Both the NumPy and the device projection paths need the same split, because the two are not
    interchangeable: a :class:`~gwmock_signal.detector.CustomDetector` is *looked up* by its
    geometry, but its output channel must be keyed by its own ``name``.

    Lives here rather than in either projection module so both derive the mapping from one
    implementation, consistent with this module being the single point that turns a detector
    specification into geometry.

    Args:
        detector_specs: Detector codes and/or ``CustomDetector`` instances.

    Returns:
        One ``(output_name, lookup_key)`` pair per entry, in the order given: the code itself for a
        string, and :meth:`~gwmock_signal.detector.CustomDetector.geometry_fields` for a
        ``CustomDetector``, which is never registered with LAL to be projected.

    Raises:
        TypeError: If an entry is neither a string nor a ``CustomDetector``.
        ValueError: If a string is not a known detector code (see :func:`detector_fields`), or if
            two entries are the same detector.
    """
    resolved: list[tuple[str, DetectorKey]] = []
    for raw in detector_specs:
        if isinstance(raw, str):
            name = str(raw)
            # Resolved eagerly so an unknown code fails here, with the other detectors' names
            # still available for the message, rather than deep inside a jitted kernel.
            detector_fields(name)
            resolved.append((name, name))
        elif isinstance(raw, CustomDetector):
            resolved.append((raw.name, raw.geometry_fields()))
        else:
            raise TypeError(f"Unsupported detector specification type: {type(raw).__name__}")

    # Distinct channels must not be one detector: the network would then hold it twice under two
    # names. A code is identified by itself and a CustomDetector by its LAL prefix once it has one,
    # so the same detector given twice, or a CustomDetector given alongside its own registered
    # prefix, is caught -- while two distinct CustomDetectors may share a geometry.
    identities = [_identity(raw) for raw in detector_specs]
    duplicated = [identity for identity in dict.fromkeys(identities) if identities.count(identity) > 1]
    if duplicated:
        colliding = {
            _describe(identity): [
                name for (name, _), other in zip(resolved, identities, strict=True) if other == identity
            ]
            for identity in duplicated
        }
        raise ValueError(
            f"Detectors resolve to the same detector: {colliding}. Each channel must be a "
            f"distinct detector; a CustomDetector and its own registered prefix are the same one."
        )
    return resolved


def _identity(spec: DetectorSpec) -> object:
    """Return what makes two detector specifications the same detector."""
    if isinstance(spec, CustomDetector):
        return spec._lal_prefix or ("CustomDetector", id(spec))
    return str(spec)


def _describe(identity: object) -> str:
    """Return a readable name for a detector identity, for error messages."""
    return identity if isinstance(identity, str) else "the same CustomDetector"


def _fields(key: DetectorKey) -> tuple[float, ...]:
    """Return the defining fields a lookup key stands for."""
    return key if isinstance(key, tuple) else detector_fields(key)


def get_lal_detector(prefix: str) -> lal.Detector:
    """Return one detector from LAL's cached prefix registry."""
    lal = _lalsuite.require("lal", "Projecting onto a detector from LAL's registry")
    try:
        return cast("lal.Detector", lal.cached_detector_by_prefix[prefix])
    except KeyError as exc:
        raise ValueError(
            f"Unknown or unsupported detector {prefix!r}. Use a valid LAL interferometer code (e.g. 'H1', 'L1', 'V1')."
        ) from exc


def detector_fields(code: str) -> tuple[float, ...]:
    """Return the fields that define the detector named by ``code`` (see :func:`_defining_fields`).

    LAL's built-in codes come from the table shipped with the package, so they need no ``lalsuite``
    and cannot be shadowed by a later change to LAL's mutable registry; the table equals LAL's
    registry exactly whenever ``lalsuite`` is installed. Any other code -- a detector registered
    with LAL at run time, such as a :class:`~gwmock_signal.detector.CustomDetector` -- is read from
    LAL's registry.

    Raises:
        ValueError: If the code is neither built in nor registered with LAL, including when
            ``lalsuite`` cannot be imported to hold a run-time registration.
    """
    fields = builtin_detectors().get(code)
    if fields is not None:
        return fields
    lal = _lalsuite.find("lal")
    try:
        # One lookup: the registry is process-global and mutable, so a membership test followed by
        # a separate read could see the entry removed in between.
        registered = lal.cached_detector_by_prefix[code] if lal is not None else None
    except KeyError:
        registered = None
    if registered is None:
        raise ValueError(
            f"Unknown or unsupported detector {code!r}. {unknown_code_help(lal_available=lal is not None)}"
        )
    return _defining_fields(registered.frDetector)


def reconstructed_geometry(key: DetectorKey) -> tuple[np.ndarray, np.ndarray]:
    """Return the response tensor and location of the detector ``key`` stands for.

    Args:
        key: A lookup key from :func:`resolve_detectors` -- a detector code, or a
            :class:`~gwmock_signal.detector.CustomDetector`'s defining fields.

    Returns:
        A ``(response, location)`` tuple where ``response`` is the 3x3 detector
        response tensor and ``location`` is the Earth-fixed position (metres): the
        WGS-84 conversion of the stored radians, computed as ERFA's ``eraGd2gce``
        does (see :func:`gwmock_signal.projection.kernels.detector_geometry`). It can
        differ from astropy's ``EarthLocation`` of the same fields by about one ulp,
        because astropy converts the angles through degrees and wraps the longitude.

    Raises:
        ValueError: If the code is not a known detector; see :func:`detector_fields`.
    """
    # Cached on the geometry itself rather than on the prefix. LAL's registry is process-global
    # and mutable, so a prefix is not a stable identity: freeing one and re-registering it for a
    # detector elsewhere on Earth would otherwise return the first detector's tensor for the
    # second, silently and in a way that still looks like a plausible network. The test harness
    # used to clear this cache after any test that registered a detector, which is how the hazard
    # is known to be real; that hook is gone because keying on the geometry removes the need.
    return _geometry_from_fields(_fields(key))


def _defining_fields(fr_detector: object) -> tuple[float, ...]:
    """Return the values that fully determine a detector's response and location."""
    return (
        float(fr_detector.vertexLongitudeRadians),
        float(fr_detector.vertexLatitudeRadians),
        float(fr_detector.vertexElevation),
        float(fr_detector.xArmAzimuthRadians),
        float(fr_detector.yArmAzimuthRadians),
        float(fr_detector.xArmAltitudeRadians),
        float(fr_detector.yArmAltitudeRadians),
    )


@cache
def _geometry_from_fields(fields: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``(response, location)`` from the defining geodetic fields.

    Args:
        fields: The tuple returned by :func:`_defining_fields`.

    Returns:
        A ``(response, location)`` tuple where ``response`` is the 3x3 detector response
        tensor and ``location`` is the Earth-fixed position in metres.
    """
    return detector_geometry(fields, xp=np)
