"""Configuration and fixtures for pytest."""

from __future__ import annotations

import pytest

try:
    import lal
except ModuleNotFoundError:  # Python 3.15 and later: lalsuite publishes no wheel yet
    lal = None


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
    if lal is None:
        yield
        return
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


def _installed(module: str) -> bool:
    """Whether ``module`` imports; a broken installation raises rather than reading as absent."""
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Skip the tests that need h5py, GWpy or lalsuite where that package is not installed.

    They are marked ``requires_h5py``, ``requires_gwpy`` or ``requires_lalsuite``.

    Python 3.15 is the case: h5py, GWpy and lalsuite publish no wheel for it yet, and the dev group and
    the package install them only below it. Everything else still runs there.
    """
    for marker, module in (("h5py", "h5py"), ("gwpy", "gwpy"), ("lalsuite", "lalsimulation")):
        if _installed(module):
            continue
        skip = pytest.mark.skip(reason=f"{marker} is not installed")
        for item in items:
            if item.get_closest_marker(f"requires_{marker}") is not None:
                item.add_marker(skip)
