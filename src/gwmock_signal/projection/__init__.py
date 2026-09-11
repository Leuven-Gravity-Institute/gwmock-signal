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
"""Projection of GW polarizations onto detector networks."""

from __future__ import annotations

from gwmock_signal.projection.network import (
    MAX_LINEAR_SIDEREAL_SPAN_SECONDS,
    PROJECTION_BACKENDS,
    project_polarizations_to_network,
    validate_projection_backend,
)

__all__ = [
    "MAX_LINEAR_SIDEREAL_SPAN_SECONDS",
    "PROJECTION_BACKENDS",
    "project_polarizations_to_network",
    "validate_projection_backend",
]
