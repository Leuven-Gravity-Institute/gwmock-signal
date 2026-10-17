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

``lalsuite`` provides LAL and LALSimulation: the detector registry behind
:class:`~gwmock_signal.detector.CustomDetector`, :mod:`gwmock_signal.network` and the projection
geometry, and the LAL-backed waveform generators. It is a C extension whose wheels follow new
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

#: The top-level modules lalsuite provides. Only their absence is "lalsuite is not installed".
_LALSUITE_MODULES = frozenset({"lal", "lalsimulation", "lalpulsar", "lalframe", "lalinspiral", "lalburst"})


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
    """Whether ``exc`` means a lalsuite module is absent, as opposed to something it needs."""
    return exc.name in _LALSUITE_MODULES | {module.partition(".")[0]}


def find(module: str) -> ModuleType | None:
    """Import and return ``module``, or ``None`` if ``lalsuite`` is not installed.

    For callers with a fallback that does not need LAL. A broken installation, or a missing
    dependency of ``lalsuite``, still raises, for the reason given in :func:`require`.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if not _is_missing_lalsuite(exc, module):
            raise
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
