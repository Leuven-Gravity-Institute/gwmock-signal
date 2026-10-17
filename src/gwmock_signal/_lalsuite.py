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
"""Access to ``lalsuite`` at the point of use, with an actionable error when it is missing.

``lalsuite`` provides LAL and LALSimulation: the detector registry that
:class:`~gwmock_signal.detector.CustomDetector` registers with (LAL's built-in detectors are also
shipped with the package, so they resolve without it), and the LAL-backed waveform generators. It is a C extension whose wheels follow new
CPython releases with a delay, so ``gwmock-signal`` installs it only on the Python versions it
publishes wheels for.

Modules that need it therefore resolve it through :func:`require` when a LAL feature is used, not
when they are imported, so the rest of the package stays importable without it. A missing
``lalsuite`` then surfaces as an :class:`ImportError` that names the feature, the running Python and
what to do, rather than as a bare ``No module named 'lal'`` from an unrelated import.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def missing_lalsuite_message(feature: str) -> str:
    """Return the message for using ``feature`` without ``lalsuite`` installed."""
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"{feature} requires lalsuite, which is not installed in this environment (Python {python}). "
        "gwmock-signal installs lalsuite only on the Python versions lalsuite publishes wheels for. "
        "Install it with `pip install lalsuite` if your Python version is supported, or use a Python "
        "version that lalsuite supports."
    )


def _is_missing_lalsuite(exc: ModuleNotFoundError, module: str) -> bool:
    """Whether ``exc`` means the requested module is absent, as opposed to something it needs.

    Only the requested module's own absence counts. ``import lal`` failing because a module *it*
    imports is missing -- another lalsuite module included -- is a broken installation, not a missing
    one, and keeps its own error.
    """
    return exc.name == module.partition(".")[0]


def find(module: str) -> ModuleType | None:
    """Import and return ``module``, or ``None`` if it cannot be imported in this process.

    For callers that only *add* to a result they can produce without LAL -- the detector codes LAL
    registered at run time, on top of the built-in table. Any import failure, a broken installation
    included, means ``None`` there: a ``lal`` this process cannot import cannot hold a run-time
    registration either. Callers that *need* LAL use :func:`require`, which reports a broken
    installation as such.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def require(module: str, feature: str) -> ModuleType:
    """Import and return ``module`` (``"lal"``, ``"lalsimulation"``, ...) for ``feature``.

    Raises:
        ImportError: If ``lalsuite`` is not installed, with a message naming ``feature``, the running
            Python and the remedy; the original error is chained. Any other import failure -- an
            installed lalsuite whose shared libraries fail to load, or a dependency it cannot find --
            propagates unchanged, since "not installed" would be the wrong diagnosis.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if not _is_missing_lalsuite(exc, module):
            raise
        raise ImportError(missing_lalsuite_message(feature)) from exc
