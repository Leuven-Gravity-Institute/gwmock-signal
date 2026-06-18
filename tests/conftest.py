"""Configuration and fixtures for pytest."""

from __future__ import annotations

import lal
import pytest

from gwmock_signal.projection.geometry import reconstructed_geometry


@pytest.fixture(autouse=True)
def _restore_lal_detector_registry():
    """Undo CustomDetector registrations so they do not leak across tests.

    ``CustomDetector.to_lal`` registers detectors in LAL's process-global
    ``lal.cached_detector_by_prefix``. Without cleanup these accumulate for the
    whole test session, which makes tests order-dependent: prefixes can clash and
    the cached ``reconstructed_geometry`` can return stale geometry if a freed
    prefix is later reused. This autouse fixture snapshots the registry before each
    test and removes anything the test added, clearing the geometry cache whenever
    a prefix is freed.
    """
    before = set(lal.cached_detector_by_prefix)
    yield
    added = set(lal.cached_detector_by_prefix) - before
    for prefix in added:
        del lal.cached_detector_by_prefix[prefix]
    if added:
        reconstructed_geometry.cache_clear()


@pytest.fixture
def some_name() -> str:
    """Provide a string name.

    Returns:
        A string name.

    """
    return "developer"
