"""Without ``lalsuite``, the package imports and every LAL feature says what is missing and why.

``lalsuite`` is installed only on the Python versions it publishes wheels for. Where it is absent,
importing ``gwmock_signal`` modules must still work, and using a LAL feature must raise an
:class:`ImportError` that names the feature, the running Python and the remedy -- not a bare
``No module named 'lal'`` from an unrelated import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gwmock_signal import _lalsuite


def test_the_message_names_the_feature_the_python_and_the_remedy():
    """The error is actionable: what failed, on which Python, and what to do."""
    message = _lalsuite.missing_lalsuite_message("The LAL waveform backend")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert message.startswith("The LAL waveform backend requires lalsuite")
    assert f"Python {python}" in message
    assert "pip install lalsuite" in message


def test_a_missing_module_raises_the_actionable_error_with_the_cause_chained(monkeypatch):
    """The original import failure stays available as ``__cause__`` for debugging."""
    monkeypatch.setitem(sys.modules, "lal", None)  # makes ``import lal`` raise ImportError

    with pytest.raises(ImportError, match=r"^Listing detectors requires lalsuite") as excinfo:
        _lalsuite.require("lal", "Listing detectors")

    assert isinstance(excinfo.value.__cause__, ImportError)


def test_an_installed_module_is_returned():
    """With lalsuite present, ``require`` is a plain import."""
    lal = pytest.importorskip("lal")

    assert _lalsuite.require("lal", "anything") is lal


#: Run in a fresh interpreter with ``lal`` and ``lalsimulation`` made unimportable. Each probe either
#: imports a module (must succeed) or uses a LAL feature (must raise the actionable ImportError).
_PROBE = r"""
import sys
sys.modules["lal"] = None
sys.modules["lalsimulation"] = None

import gwmock_signal.detector, gwmock_signal.network, gwmock_signal.projection.geometry
import gwmock_signal.waveform.backends.lal, gwmock_signal.waveform.backends.gwsignal
import gwmock_signal.waveform.factory
print("imports: ok")

from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import list_lal_detectors
from gwmock_signal.projection.geometry import get_lal_detector
from gwmock_signal.waveform.backends.gwsignal import GWSignalBackend
from gwmock_signal.waveform.backends.lal import LALSimulationBackend
from gwmock_signal.waveform.factory import WaveformFactory

probes = {
    "factory": WaveformFactory,
    "lal_backend": LALSimulationBackend,
    "gwsignal_backend": GWSignalBackend,
    "list_detectors": list_lal_detectors,
    "registry_lookup": lambda: get_lal_detector("H1"),
    "custom_detector": lambda: CustomDetector("X", 0.0, 0.0, 0.0, 0.0, 1.5),
}
for name, probe in probes.items():
    try:
        probe()
    except ImportError as exc:
        print(f"{name}: {exc}")
    else:
        print(f"{name}: NO ERROR")
"""


def test_without_lalsuite_modules_import_and_lal_features_explain_themselves():
    """The import-time failure is gone, and each LAL entry point raises the actionable error."""
    source_root = Path(_lalsuite.__file__).resolve().parents[1]
    search_path = [str(source_root), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(search_path)}
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, check=False)  # noqa: S603

    assert result.returncode == 0, result.stderr
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines())
    assert lines.pop("imports") == "ok"
    expected_feature = {
        "factory": "The LAL waveform backend",
        "lal_backend": "The LAL waveform backend",
        "gwsignal_backend": "The gwsignal waveform backend",
        "list_detectors": "Listing LAL's built-in detectors",
        "registry_lookup": "Projecting onto a detector from LAL's registry",
        "custom_detector": "Registering a CustomDetector with LAL's detector registry",
    }
    assert set(lines) == set(expected_feature)
    for name, feature in expected_feature.items():
        assert lines[name].startswith(f"{feature} requires lalsuite"), (name, lines[name])
        assert "pip install lalsuite" in lines[name]
