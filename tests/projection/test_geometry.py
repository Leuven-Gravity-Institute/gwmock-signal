"""Detector geometry must not be cached against a mutable identity.

LAL's ``cached_detector_by_prefix`` is process-global and mutable, so a two-character prefix is
not a stable identity for a detector. ``reconstructed_geometry`` was cached *by prefix*, which
meant freeing a prefix and re-registering it for a detector elsewhere on Earth returned the first
detector's response tensor for the second -- silently, and looking like a plausible network.

``tests/conftest.py`` used to clear that cache after any test which registered a detector. A
workaround in the test harness for a hazard the production code still had is evidence the hazard
is real, so these tests pin the fix instead.
"""

from __future__ import annotations

import lal
import numpy as np
import pytest

from gwmock_signal.detector import CustomDetector
from gwmock_signal.projection.geometry import reconstructed_geometry, resolve_detectors


def _detector(name: str, prefix: str, longitude: float) -> CustomDetector:
    """A detector at a chosen longitude, registered under an explicit prefix."""
    return CustomDetector(
        name=name,
        latitude_rad=0.7615,
        longitude_rad=longitude,
        elevation_m=51.884,
        xarm_azimuth_rad=0.3387,
        yarm_azimuth_rad=1.3861,
        prefix=prefix,
    )


def test_reused_prefix_does_not_return_stale_geometry() -> None:
    """Re-registering a freed prefix elsewhere must not return the first geometry.

    This is the exact sequence the old prefix-keyed cache got wrong. It goes through the public
    entry point, and the two longitudes are ~1.4 rad apart so the locations differ by thousands
    of kilometres -- a mismatch no tolerance choice could hide.
    """
    first = _detector("FIRST", "Q1", longitude=0.1833)
    first.to_lal()
    _, location_first = reconstructed_geometry("Q1")

    del lal.cached_detector_by_prefix["Q1"]

    second = _detector("SECOND", "Q1", longitude=-1.2)
    second.to_lal()
    _, location_second = reconstructed_geometry("Q1")

    separation_km = float(np.linalg.norm(location_first - location_second)) / 1e3
    assert separation_km > 1000.0, (
        f"the same prefix returned geometry only {separation_km:.1f} km apart after being "
        f"re-registered at a different longitude; the cache is keyed on a mutable identity"
    )


def test_identical_geometry_still_shares_one_cache_entry() -> None:
    """Two prefixes with identical geometry should hit the same cached reconstruction.

    Keying on the geometry rather than the prefix is what makes the fix safe; this pins that it
    is still a cache and not an accidental recomputation on every call.
    """
    left = _detector("LEFT", "Q2", longitude=0.1833)
    right = _detector("RIGHT", "Q3", longitude=0.1833)
    left.to_lal()
    right.to_lal()

    response_left, location_left = reconstructed_geometry("Q2")
    response_right, location_right = reconstructed_geometry("Q3")
    # The same object, not merely equal values: identical inputs must reach one cache entry.
    assert response_left is response_right
    assert location_left is location_right


def test_unknown_prefix_is_rejected() -> None:
    """A prefix LAL does not know is a caller error, named as such."""
    with pytest.raises(ValueError, match="Unknown or unsupported detector"):
        reconstructed_geometry("ZZ")


def test_duplicate_lookup_keys_are_rejected_by_the_resolver() -> None:
    """Two channels resolving to one LAL key would silently duplicate a geometry.

    Passing a custom detector alongside its own registered prefix gives two distinct output names
    but one lookup key. Left unchecked the network would hold the same detector twice under
    different names, which is not a network anyone means to request -- and it is the observable
    form the prefix-registration race would take.
    """
    detector = _detector("Q4_NAMED", "Q4", longitude=0.1833)
    prefix = detector.to_lal().frDetector.prefix
    with pytest.raises(ValueError, match="resolve to the same LAL detector"):
        resolve_detectors([detector, prefix])
