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
"""Named detector network catalog for ground-based GW observatories."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

import lal

if TYPE_CHECKING:
    from gwmock_signal.detector import CustomDetector

# Convenience presets for common multi-detector networks.  These are *not* the
# source of truth for which detector codes exist — LAL is.  Any code returned
# by ``lal.cached_detector_by_prefix`` is valid independently of what is listed
# here.  New LAL detectors are automatically usable via
# :meth:`Network.from_detectors` without touching this file.
_NETWORK_PRESETS: dict[str, tuple[str, ...]] = {
    # LIGO Hanford + LIGO Livingston
    "H1L1": ("H1", "L1"),
    # LIGO Hanford + LIGO Livingston + Virgo
    "H1L1V1": ("H1", "L1", "V1"),
    # LIGO Hanford + LIGO Livingston + Virgo + KAGRA
    "HLVK": ("H1", "L1", "V1", "K1"),
    # Einstein Telescope triangle configuration (three 60° arms)
    "ET-triangle": ("E1", "E2", "E3"),
    # Einstein Telescope L-shaped configuration
    "ET-L": ("E0",),
}

_FILE_BACKED_PRESET_ALIASES: dict[str, str] = {
    "ET-Triangle-Sardinia": "et-triangle-sardinia.yaml",
    "ET-Triangle-EMR": "et-triangle-emr.yaml",
    "ET-2L-Aligned": "et-2l-aligned.yaml",
    "ET-2L-Misaligned": "et-2l-misaligned.yaml",
    # Compatibility aliases from the migration roadmap.
    "ET-Sardinia": "et-triangle-sardinia.yaml",
    "ET-EMR": "et-triangle-emr.yaml",
}

_PRESET_PACKAGE = "gwmock_signal.data.detectors"

# Keys whose presence in an entry signals a full custom geometry rather than a
# plain detector alias string.  Both *_deg and *_rad variants are included so that
# either convention triggers CustomDetector construction.
_geometry_keys: frozenset[str] = frozenset(
    {
        "latitude_deg",
        "latitude_rad",
        "longitude_deg",
        "longitude_rad",
        "elevation_m",
        "xarm_azimuth_deg",
        "xarm_azimuth_rad",
        "yarm_azimuth_deg",
        "yarm_azimuth_rad",
    }
)


def _parse_angle(entry: dict, base: str, *, required: bool, default: float = 0.0) -> float:
    """Return an angle in radians from *entry*, accepting ``{base}_deg`` or ``{base}_rad``.

    Args:
        entry: Detector config mapping.
        base: Angle parameter base name (e.g. ``"latitude"``).
        required: If ``True`` raise :class:`ValueError` when neither key is present.
        default: Value returned when both keys are absent and ``required`` is ``False``.

    Raises:
        ValueError: If both ``{base}_deg`` and ``{base}_rad`` are present simultaneously.
        ValueError: If ``required`` is ``True`` and neither key is present.
    """
    deg_key = f"{base}_deg"
    rad_key = f"{base}_rad"
    has_deg = deg_key in entry
    has_rad = rad_key in entry
    if has_deg and has_rad:
        raise ValueError(
            f"Conflicting angle specification for '{base}': provide either '{deg_key}' or '{rad_key}', not both."
        )
    if has_deg:
        return math.radians(float(entry[deg_key]))
    if has_rad:
        return float(entry[rad_key])
    if required:
        raise ValueError(f"Missing required angle '{base}': provide either '{deg_key}' or '{rad_key}'.")
    return default


def _load_data(path: Path) -> object:
    """Load YAML or JSON from *path*; raise ``ValueError`` for other extensions."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415

        with path.open() as fh:
            return yaml.safe_load(fh)
    if suffix == ".json":
        with path.open() as fh:
            return json.load(fh)
    raise ValueError(
        f"Unsupported file extension {path.suffix!r}. Supported extensions: .yaml, .yml, .json, .interferometer"
    )


def _preset_resource(alias: str):
    """Return the importlib resource for one bundled detector preset."""
    try:
        resource_name = _FILE_BACKED_PRESET_ALIASES[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {alias!r}. Known presets: {sorted(_FILE_BACKED_PRESET_ALIASES)}") from exc
    return files(_PRESET_PACKAGE).joinpath(resource_name)


def list_lal_detectors() -> list[str]:
    """Return every detector code available in the installed LAL bindings."""
    return sorted(str(code) for code in lal.cached_detector_by_prefix)


def _detector_from_entry(entry: dict) -> str | CustomDetector:
    """Convert one detector dict entry to a str alias or :class:`CustomDetector`.

    Args:
        entry: Detector config mapping (must include ``"name"``).

    Returns:
        The LAL detector code string when no geometry keys are present,
        or a :class:`CustomDetector` instance.

    Raises:
        ValueError: If a required angle is absent or if both ``*_deg`` and
            ``*_rad`` variants of the same angle are present simultaneously.
    """
    from gwmock_signal.detector import CustomDetector  # noqa: PLC0415

    det_name: str = entry["name"]
    if not _geometry_keys & set(entry.keys()):
        return det_name

    if "elevation_m" not in entry:
        raise ValueError(f"Missing required geometry field: 'elevation_m' for detector {det_name!r}.")

    try:
        return CustomDetector(
            name=det_name,
            prefix=str(entry.get("prefix", "")),
            latitude_rad=_parse_angle(entry, "latitude", required=True),
            longitude_rad=_parse_angle(entry, "longitude", required=True),
            elevation_m=float(entry["elevation_m"]),
            xarm_azimuth_rad=_parse_angle(entry, "xarm_azimuth", required=True),
            yarm_azimuth_rad=_parse_angle(entry, "yarm_azimuth", required=True),
            xarm_tilt_rad=_parse_angle(entry, "xarm_tilt", required=False, default=0.0),
            yarm_tilt_rad=_parse_angle(entry, "yarm_tilt", required=False, default=0.0),
        )
    except ValueError as exc:
        raise ValueError(f"Detector {det_name!r}: {exc}") from exc


@dataclass(frozen=True)
class Network:
    """A detector network: a named, ordered collection of detector specs.

    Attributes:
        name: Human-readable label for the network.
        detector_names: Ordered tuple of LAL detector prefix strings (e.g.
            ``"H1"``) or
            :class:`~gwmock_signal.detector.CustomDetector` instances.
    """

    name: str
    detector_names: tuple[str | CustomDetector, ...]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_detectors(
        cls,
        detectors: Sequence[str | CustomDetector],
        *,
        name: str = "",
    ) -> Network:
        """Construct a :class:`Network` from any sequence of detector specs.

        String entries are validated against LAL's runtime detector list
        (``lal.cached_detector_by_prefix``), so any detector code that the
        installed LAL bindings know about works here without hard-coding.
        :class:`~gwmock_signal.detector.CustomDetector` instances are accepted
        as-is for non-standard geometries.

        Args:
            detectors: Sequence of LAL detector code strings (e.g. ``"H1"``)
                or :class:`~gwmock_signal.detector.CustomDetector` instances.
            name: Optional human-readable label.  Defaults to the
                comma-joined detector names.

        Returns:
            A :class:`Network` containing exactly the specified detectors.

        Raises:
            ValueError: If any string entry is not in LAL's available
                detector set, or if *detectors* is empty.
        """
        from gwmock_signal.detector import CustomDetector  # noqa: PLC0415

        if not detectors:
            raise ValueError("detectors must be a non-empty sequence.")

        lal_codes: set[str] = set(list_lal_detectors())
        validated: list[str | CustomDetector] = []
        for det in detectors:
            if isinstance(det, CustomDetector):
                validated.append(det)
            else:
                code = str(det)
                if code not in lal_codes:
                    raise ValueError(
                        f"Unknown LAL detector code {code!r}. "
                        f"Run Network.list_lal_detectors() to see available codes, "
                        "or use a CustomDetector for non-standard geometries."
                    )
                validated.append(code)

        auto_name = name or ",".join(d if isinstance(d, str) else d.name for d in validated)
        return cls(name=auto_name, detector_names=tuple(validated))

    @classmethod
    def from_name(cls, alias: str) -> Network:
        """Construct a :class:`Network` from a named preset.

        Args:
            alias: One of the pre-defined network names returned by
                :meth:`list_names`.

        Returns:
            A :class:`Network` whose ``detector_names`` are the detector codes
            for that preset.

        Raises:
            ValueError: If *alias* is not in the named presets.
        """
        if alias in _NETWORK_PRESETS:
            return cls(name=alias, detector_names=_NETWORK_PRESETS[alias])
        if alias in _FILE_BACKED_PRESET_ALIASES:
            return cls.from_preset(alias)
        raise ValueError(f"Unknown network {alias!r}. Known networks: {cls.list_names()}")

    @classmethod
    def from_preset(cls, alias: str) -> Network:
        """Construct a :class:`Network` from a bundled YAML/JSON detector preset.

        Bundled presets live under ``gwmock_signal.data.detectors`` and are
        resolved with :mod:`importlib.resources`, so they work from source trees
        and installed wheels alike.

        Args:
            alias: One of the bundled preset aliases returned by
                :meth:`list_names`, for example ``"ET-Triangle-Sardinia"``.

        Returns:
            A :class:`Network` loaded from the packaged detector geometry file.

        Raises:
            ValueError: If *alias* does not map to a bundled preset file.
            FileNotFoundError: If the mapped packaged preset file is missing.
        """
        resource = _preset_resource(alias)
        if not resource.is_file():
            raise FileNotFoundError(f"Bundled detector preset for {alias!r} is missing.")
        with as_file(resource) as path:
            net = cls.from_file(path)
        return cls.from_detectors(net.detector_names, name=net.name)

    @classmethod
    def from_file(cls, path: str | Path) -> Network:
        """Load a :class:`Network` from a YAML, JSON, or deprecated ``.interferometer`` file.

        The file must have a top-level ``name`` key (str) and a ``detectors``
        key containing a non-empty list of detector entries.  Each entry must
        have a ``name`` field. If any geometry key is present a
        :class:`~gwmock_signal.detector.CustomDetector` is constructed;
        otherwise the name is treated as a plain LAL detector code string.

        ``.interferometer`` files are treated as Bilby compatibility shims for
        a single custom detector and emit a :class:`DeprecationWarning`. Migrate
        those files to the YAML detector preset/network format instead.

        **Angle conventions** — every angle parameter accepts either a degrees
        variant (``*_deg``) or a radians variant (``*_rad``), but *not both* for
        the same parameter:

        +-----------------------+-------------------+-----------+
        | Parameter             | Degrees key       | Radians key |
        +=======================+===================+=============+
        | Geodetic latitude     | ``latitude_deg``  | ``latitude_rad`` |
        | Geodetic longitude    | ``longitude_deg`` | ``longitude_rad`` |
        | x-arm azimuth         | ``xarm_azimuth_deg`` | ``xarm_azimuth_rad`` |
        | y-arm azimuth         | ``yarm_azimuth_deg`` | ``yarm_azimuth_rad`` |
        | x-arm tilt (optional) | ``xarm_tilt_deg`` | ``xarm_tilt_rad`` |
        | y-arm tilt (optional) | ``yarm_tilt_deg`` | ``yarm_tilt_rad`` |
        +-----------------------+-------------------+-----------+

        ``elevation_m`` is always in metres.  Tilt fields default to ``0.0``
        when absent.

        Args:
            path: Path to a ``.yaml``, ``.yml``, ``.json``, or
                deprecated ``.interferometer`` file.

        Returns:
            A :class:`Network` whose ``detector_names`` contains str entries
            for LAL detector aliases and
            :class:`~gwmock_signal.detector.CustomDetector` entries for
            user-defined geometries.

        Raises:
            ValueError: If the file extension is unsupported, a required
                angle is absent, both ``*_deg`` and ``*_rad`` are given for
                the same angle, or a geometry value is out of range.
        """
        path = Path(path)
        if path.suffix.lower() == ".interferometer":
            from gwmock_signal.io.interferometer_format import interferometer_config_to_custom_detector  # noqa: PLC0415

            detector = interferometer_config_to_custom_detector(path)
            return cls.from_detectors((detector,), name=detector.name)

        data = _load_data(path)

        if not isinstance(data, dict):
            raise TypeError("Network config must be a YAML/JSON mapping at the top level.")
        if "name" not in data:
            raise ValueError("Missing required field: 'name'")
        if "detectors" not in data:
            raise ValueError("Missing required field: 'detectors'")

        detectors_raw = data["detectors"]
        if not isinstance(detectors_raw, list) or len(detectors_raw) == 0:
            raise ValueError("'detectors' must be a non-empty list.")

        detectors: list[str | CustomDetector] = []
        for entry in detectors_raw:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError("Each detector entry must be a mapping with a 'name' field.")
            detectors.append(_detector_from_entry(entry))

        return cls.from_detectors(detectors, name=data["name"])

    # ------------------------------------------------------------------
    # Listing helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_names(cls) -> list[str]:
        """Return a sorted list of all named network and bundled preset aliases."""
        return sorted(set(_NETWORK_PRESETS) | set(_FILE_BACKED_PRESET_ALIASES))

    @classmethod
    def list_lal_detectors(cls) -> list[str]:
        """Return every detector code available in the installed LAL, sorted.

        Returns:
            Sorted list of LAL detector prefix strings
            (e.g. ``['E0', 'E1', 'H1', 'L1', 'V1', ...]``).
        """
        return list_lal_detectors()
