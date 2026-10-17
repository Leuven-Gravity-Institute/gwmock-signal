"""LAL's built-in detectors ship with the package and must equal LAL's registry exactly.

Projection resolves a built-in code (``H1``, ``V1``, ...) from the table in
``gwmock_signal/data/lal_detectors.json``, so it needs no ``lalsuite``. That is only safe while the
table holds the very doubles LAL's registry holds: a field one ulp off would move a detector, and
every projection onto it, without any error. Equality is therefore exact, never a tolerance.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from lal_detector_table import TABLE, render

from gwmock_signal import _detector_table, _lalsuite
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
    # -I: no PYTHONPATH or user site, so the child imports the installed lal and nothing else.
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip().split(",") == sorted(_detector_table.builtin_detectors())


def test_the_shipped_file_is_what_the_script_writes():
    """The file on disk is byte-for-byte the script's output for the installed LAL, names included."""
    lal = pytest.importorskip("lal")
    builtin_codes = set(_detector_table.builtin_detectors())
    registry = {code: lal.cached_detector_by_prefix[code] for code in builtin_codes}

    assert TABLE.read_text() == render(registry)


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
    """Neither built in nor registered: rejected as unknown, by projection and by Network alike."""
    from gwmock_signal.network import Network

    rejections = {
        "Unknown or unsupported detector 'ZZ9'": lambda: detector_fields("ZZ9"),
        "Unknown LAL detector code 'ZZ9'": lambda: Network.from_detectors(["ZZ9"]),
    }
    for first_sentence, reject in rejections.items():
        with pytest.raises(ValueError, match=first_sentence) as excinfo:
            reject()
        assert "Use a built-in interferometer code (A1, " in str(excinfo.value)
        if _lalsuite.find("lal") is not None:
            assert "could not be imported" not in str(excinfo.value)


def test_the_cached_table_cannot_be_modified():
    """Every caller shares one cached table, so it is read-only."""
    with pytest.raises(TypeError):
        _detector_table.builtin_detectors()["H1"] = (0.0,) * 7  # type: ignore[index]


def test_a_broken_lal_still_lists_and_validates_the_built_in_codes(monkeypatch):
    """A lal that fails to load holds no run-time registrations; the built-in codes still work."""
    from gwmock_signal.network import Network, list_lal_detectors

    real_import = _lalsuite.importlib.import_module

    def broken(name):
        if name == "lal":
            raise ImportError("liblal.so.20: cannot open shared object file")
        return real_import(name)

    monkeypatch.setattr(_lalsuite.importlib, "import_module", broken)

    assert list_lal_detectors() == sorted(_detector_table.builtin_detectors())
    assert Network.from_detectors(["H1"]).detector_names == ("H1",)
    assert detector_fields("H1") == _detector_table.builtin_detectors()["H1"]


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
try:
    Network.from_detectors(["ZZ9"])
except ValueError as exc:
    print("network_unknown:" + str(exc))
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
    assert "Unknown LAL detector code 'ZZ9'" in lines["network_unknown"]
    for key in ("unknown", "network_unknown"):
        assert "need lalsuite, which could not be imported" in lines[key]
