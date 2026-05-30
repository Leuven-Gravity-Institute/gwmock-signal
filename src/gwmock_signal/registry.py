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
"""Public source-type registry for gwmock-signal simulator backends."""

from __future__ import annotations

from gwmock_signal.simulator import CBCSimulator, GWSimulator

_SOURCE_TYPE_REGISTRY: dict[str, type[GWSimulator]] = {}


def _normalize_source_type(source_type: str) -> str:
    """Return the canonical registry key for a source type string."""
    if not isinstance(source_type, str):
        raise TypeError("source_type must be a string")
    normalized = source_type.strip().lower()
    if not normalized:
        raise ValueError("source_type must be a non-empty string")
    return normalized


def register_simulator_backend(source_type: str, backend: type[GWSimulator]) -> None:
    """Register a simulator backend class for a gwmock-pop ``source_type``.

    Registration is intentionally class-based rather than instance-based so the
    downstream lookup contract stays stable even when different backends require
    different constructor arguments.

    Args:
        source_type: gwmock-pop source-family key such as ``"bbh"``.
        backend: Concrete ``GWSimulator`` subclass implementing that source type.

    Raises:
        TypeError: If *backend* is not a ``GWSimulator`` subclass.
        ValueError: If *source_type* is empty or already registered to a
            different backend.
    """
    if not issubclass(backend, GWSimulator):
        raise TypeError("backend must be a GWSimulator subclass")

    normalized = _normalize_source_type(source_type)
    existing = _SOURCE_TYPE_REGISTRY.get(normalized)
    if existing is not None and existing is not backend:
        raise ValueError(
            f"source_type {normalized!r} is already registered to {existing.__name__}; "
            f"refusing to replace it with {backend.__name__}"
        )

    _SOURCE_TYPE_REGISTRY[normalized] = backend


def resolve_simulator_backend(source_type: str) -> type[GWSimulator]:
    """Resolve the registered simulator backend class for a source type.

    Args:
        source_type: gwmock-pop source-family key such as ``"bbh"``.

    Returns:
        The registered concrete ``GWSimulator`` subclass.

    Raises:
        KeyError: If no backend is registered for *source_type*.
        ValueError: If *source_type* is empty.
    """
    normalized = _normalize_source_type(source_type)
    try:
        return _SOURCE_TYPE_REGISTRY[normalized]
    except KeyError as exc:
        raise KeyError(f"No simulator backend is registered for source_type={normalized!r}") from exc


def list_registered_source_types() -> tuple[str, ...]:
    """Return the registered source-type keys in sorted order."""
    return tuple(sorted(_SOURCE_TYPE_REGISTRY))


register_simulator_backend("bbh", CBCSimulator)
register_simulator_backend("bns", CBCSimulator)
register_simulator_backend("nsbh", CBCSimulator)


__all__ = [
    "list_registered_source_types",
    "register_simulator_backend",
    "resolve_simulator_backend",
]
