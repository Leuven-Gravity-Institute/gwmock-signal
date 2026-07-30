"""Configuration and fixtures for pytest."""

from __future__ import annotations

import lal
import pytest


@pytest.fixture(autouse=True)
def _restore_lal_detector_registry():
    """Undo CustomDetector registrations so they do not leak across tests.

    ``CustomDetector.to_lal`` registers detectors in LAL's process-global
    ``lal.cached_detector_by_prefix``. Without cleanup these accumulate for the whole
    test session, which makes tests order-dependent because prefixes can clash.

    This no longer clears a geometry cache. It used to, because
    ``reconstructed_geometry`` was cached *by prefix* and would return stale geometry
    when a freed prefix was reused -- a hazard this fixture papered over for tests
    while production code had no equivalent protection. The cache is now keyed on the
    geometry itself, so prefix reuse cannot alias two detectors and no invalidation
    hook is needed. See ``gwmock_signal.projection.geometry``.
    """
    before = set(lal.cached_detector_by_prefix)
    yield
    for prefix in set(lal.cached_detector_by_prefix) - before:
        del lal.cached_detector_by_prefix[prefix]


@pytest.fixture
def some_name() -> str:
    """Provide a string name.

    Returns:
        A string name.

    """
    return "developer"
