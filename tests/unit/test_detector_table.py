"""LAL's built-in detectors ship with the package and must equal LAL's registry exactly.

Projection resolves a built-in code (``H1``, ``V1``, ...) from the table in
``gwmock_signal/data/lal_detectors.json``, so it needs no ``lalsuite``. That is only safe while the
table holds the very doubles LAL's registry holds: a field one ulp off would move a detector, and
every projection onto it, without any error. Equality is therefore exact, never a tolerance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from lal_detector_table import TABLE, registry_table

from gwmock_signal import _detector_table
from gwmock_signal.detector import CustomDetector
from gwmock_signal.projection.geometry import _defining_fields, detector_fields, reconstructed_geometry


def test_the_table_equals_lal_registry_field_by_field():
    """Every built-in code carries exactly LAL's values."""
    lal = pytest.importorskip("lal")

    for code, fields in _detector_table.builtin_detectors().items():
        assert fields == _defining_fields(lal.cached_detector_by_prefix[code].frDetector), code


def test_the_table_holds_every_code_lal_ships():
    """Read from a fresh interpreter, so run-time registrations by other tests are not counted."""
    pytest.importorskip("lal")
    code = "import lal; print(','.join(sorted(lal.cached_detector_by_prefix)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip().split(",") == sorted(_detector_table.builtin_detectors())


def test_the_shipped_file_is_what_the_script_writes():
    """The JSON on disk is exactly the script's output for the installed LAL, names included."""
    lal = pytest.importorskip("lal")
    builtin_codes = set(_detector_table.builtin_detectors())
    registry = {code: lal.cached_detector_by_prefix[code] for code in builtin_codes}

    assert json.loads(TABLE.read_text()) == registry_table(registry)


def test_the_table_fields_are_in_projection_order():
    """The table's field order is the order the geometry reconstruction unpacks."""
    from lal_detector_table import FIELDS

    assert tuple(FIELDS) == _detector_table.FIELDS


def test_a_built_in_code_is_not_shadowed_by_a_changed_lal_registry():
    """LAL's registry is mutable; a built-in code keeps the table's geometry regardless."""
    pytest.importorskip("lal")
    expected = _detector_table.builtin_detectors()["H1"]

    with patch.dict("lal.cached_detector_by_prefix", {"H1": object()}):
        assert detector_fields("H1") == expected


def test_a_code_registered_at_run_time_still_resolves_through_lal():
    """A CustomDetector's prefix is not built in, and is read from LAL's registry.

    LAL stores the elevation and the arm angles in single precision, so the values here are exact
    in float32 and come back unchanged.
    """
    pytest.importorskip("lal")
    detector = CustomDetector("custom", 0.7, -1.2, 100.0, 0.25, 1.875, prefix="Q7")
    detector.to_lal()

    assert "Q7" not in _detector_table.builtin_detectors()
    assert detector_fields("Q7") == (-1.2, 0.7, 100.0, 0.25, 1.875, 0.0, 0.0)


def test_an_unknown_code_is_a_value_error():
    """Neither built in nor registered: rejected as unknown."""
    with pytest.raises(ValueError, match="Unknown or unsupported detector 'ZZ9'"):
        detector_fields("ZZ9")


#: Without ``lal``: the built-in geometry, listing and validation work, and an unknown code is a
#: ValueError that says why run-time codes are unavailable. Prints one result per line.
_PROBE = r"""
import sys
sys.modules["lal"] = None
sys.modules["lalsimulation"] = None
from gwmock_signal.network import Network, list_lal_detectors
from gwmock_signal.projection.geometry import reconstructed_geometry

response, location = reconstructed_geometry("H1")
print("H1:" + ",".join(float(x).hex() for x in [*response.ravel(), *location]))
print("codes:" + ",".join(list_lal_detectors()))
print("network:" + ",".join(str(d) for d in Network.from_detectors(["H1", "V1"]).detector_names))
try:
    reconstructed_geometry("ZZ9")
except ValueError as exc:
    print("unknown:" + str(exc))
"""


def test_without_lalsuite_built_in_detectors_resolve_to_the_same_geometry():
    """The geometry of a built-in code is bit-identical with and without lalsuite installed."""
    pytest.importorskip("lal")
    response, location = reconstructed_geometry("H1")
    with_lal = ",".join(float(x).hex() for x in [*response.ravel(), *location])

    source_root = Path(_detector_table.__file__).resolve().parents[1]
    search_path = [str(source_root), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(search_path)}
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, check=False)  # noqa: S603

    assert result.returncode == 0, result.stderr
    lines = dict(line.split(":", 1) for line in result.stdout.splitlines())
    assert lines["H1"] == with_lal
    assert lines["codes"].split(",") == sorted(_detector_table.builtin_detectors())
    assert lines["network"] == "H1,V1"
    assert "Unknown or unsupported detector 'ZZ9'" in lines["unknown"]
    assert "lalsuite, which is not installed" in lines["unknown"]
