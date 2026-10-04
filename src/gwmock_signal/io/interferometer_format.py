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

"""Compatibility shim for Bilby ``.interferometer`` detector configs."""

from __future__ import annotations

import math
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Any

_WARNED_INTERFEROMETER_PATHS: set[Path] = set()


def resolve_interferometer_config_path(config_file: str | Path) -> Path:
    """Resolve and validate one Bilby ``.interferometer`` config path."""
    config_path = Path(config_file).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_file!s} not found.")
    return config_path.resolve()


def _warn_interferometer_deprecation(config_path: Path) -> None:
    """Emit one deprecation warning per resolved config file path."""
    if config_path in _WARNED_INTERFEROMETER_PATHS:
        return

    warnings.warn(
        (
            f"Bilby '.interferometer' support is deprecated and will be removed in the next major release. "
            f"Migrate {config_path.name!r} to the YAML detector preset/network format described in roadmap step 06."
        ),
        DeprecationWarning,
        stacklevel=3,
    )
    _WARNED_INTERFEROMETER_PATHS.add(config_path)


def _read_interferometer_config(config_path: Path, *, encoding: str = "utf-8") -> dict[str, Any]:
    """Read one resolved ``.interferometer`` file into a Bilby-style mapping."""
    bilby_params: dict[str, Any] = {}

    with config_path.open(encoding=encoding) as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in raw_line:
                raise ValueError(
                    f"Invalid .interferometer assignment at {config_path}:{line_number}: expected 'key = value'."
                )

            key, value_text = raw_line.split("=", maxsplit=1)
            key = key.strip()
            if key == "power_spectral_density":
                continue

            try:
                bilby_params[key] = literal_eval(value_text.strip())
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid literal value for {key!r} at {config_path}:{line_number}.") from exc

    return bilby_params


def read_interferometer_config(config_file: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Read one ``.interferometer`` file into its Bilby-style parameter mapping."""
    resolved_config_file = resolve_interferometer_config_path(config_file)
    _warn_interferometer_deprecation(resolved_config_file)
    return _read_interferometer_config(resolved_config_file, encoding=encoding)


def _required_float(mapping: dict[str, Any], key: str) -> float:
    """Return one required float field from *mapping*."""
    try:
        return float(mapping[key])
    except KeyError as exc:
        raise ValueError(f"Missing required field {key!r} in .interferometer config.") from exc


def _bilby_azimuth_to_lal_rad(azimuth_deg: float) -> float:
    """Convert a Bilby arm azimuth to LAL's convention, in radians.

    Bilby measures arm azimuths in degrees North of East (counter-clockwise from East), whereas
    :class:`CustomDetector` follows LAL and measures them clockwise from North.
    """
    return math.radians((90.0 - azimuth_deg) % 360.0)


def interferometer_config_to_custom_detector(config_file: str | Path, encoding: str = "utf-8"):
    """Convert one ``.interferometer`` file into a :class:`CustomDetector`.

    Arm azimuths are converted from Bilby's convention (degrees North of East) to LAL's
    (clockwise from North), so the detector arms keep their physical orientation.
    """
    from gwmock_signal.detector import CustomDetector  # noqa: PLC0415

    resolved_config_file = resolve_interferometer_config_path(config_file)
    _warn_interferometer_deprecation(resolved_config_file)
    bilby_params = _read_interferometer_config(resolved_config_file, encoding=encoding)

    try:
        name = str(bilby_params["name"])
    except KeyError as exc:
        raise ValueError("Missing required field 'name' in .interferometer config.") from exc

    return CustomDetector(
        name=name,
        latitude_rad=math.radians(_required_float(bilby_params, "latitude")),
        longitude_rad=math.radians(_required_float(bilby_params, "longitude")),
        elevation_m=_required_float(bilby_params, "elevation"),
        xarm_azimuth_rad=_bilby_azimuth_to_lal_rad(_required_float(bilby_params, "xarm_azimuth")),
        yarm_azimuth_rad=_bilby_azimuth_to_lal_rad(_required_float(bilby_params, "yarm_azimuth")),
        xarm_tilt_rad=float(bilby_params.get("xarm_tilt", 0.0)),
        yarm_tilt_rad=float(bilby_params.get("yarm_tilt", 0.0)),
    )
