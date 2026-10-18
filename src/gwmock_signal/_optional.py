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
"""Import an optional package, naming the extra that installs it when it is missing."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str, feature: str) -> ModuleType:
    """Return ``module``, importing it; say which extra installs it if it is missing.

    Only ``module`` itself missing, or a package it sits in, is reported that way. Any other
    missing module -- one it needs, or a part of an installed package that fails to import -- is
    re-raised unchanged, since installing the extra would not help.

    Args:
        module: Dotted module name, such as ``"gwpy.timeseries"``.
        extra: The ``gwmock-signal`` extra that installs it, such as ``"gwpy"``.
        feature: What needs it, for the message, such as ``"Frame (GWF) output"``.

    Raises:
        ModuleNotFoundError: If the package is not installed.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        parts = module.split(".")
        enclosing = {".".join(parts[: index + 1]) for index in range(len(parts))}
        if exc.name not in enclosing:
            raise
        root = parts[0]
        raise ModuleNotFoundError(
            f"{feature} needs {root}, which is not installed. Install it with: pip install 'gwmock-signal[{extra}]'",
            name=exc.name,
        ) from exc
