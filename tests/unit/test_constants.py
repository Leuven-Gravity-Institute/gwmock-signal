"""The built-in constants must match LAL's bit-for-bit, and must not need LAL to import.

LALSimulation converts with its own constants internally, so a built-in value that differed in the
last bit would put a one-ulp error into every waveform generated through it. Equality is therefore
exact (``==``), never a tolerance.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gwmock_signal import constants
from gwmock_signal.constants import _builtin


def _lal_reference(lal) -> dict[str, float]:
    """Each constant's reference value from ``lal`` itself, independently of ``constants._lal``.

    LAL defines no megaparsec; ``PC_SI * 1e6`` is the expression LAL-based codes use for it.
    """
    return {
        "MSUN_SI": lal.MSUN_SI,
        "MTSUN_SI": lal.MTSUN_SI,
        "PC_SI": lal.PC_SI,
        "MPC_SI": lal.PC_SI * 1e6,
    }


@pytest.mark.parametrize("name", constants.CONSTANT_NAMES)
def test_builtin_constant_equals_lal(name):
    """Each built-in value equals its LAL reference to the last bit.

    LAL defines ``MSUN_SI``, ``MTSUN_SI`` and ``PC_SI``; ``MPC_SI`` is compared with ``lal.PC_SI * 1e6``.
    """
    lal = pytest.importorskip("lal")

    assert getattr(_builtin, name) == _lal_reference(lal)[name]


@pytest.mark.parametrize("name", constants.CONSTANT_NAMES)
def test_lal_provider_reads_lal(name):
    """The LAL provider is what the comparison workflow trusts, so it must return LAL's values."""
    lal = pytest.importorskip("lal")
    from gwmock_signal.constants import _lal

    assert getattr(_lal, name) == _lal_reference(lal)[name]


def test_every_provider_defines_exactly_the_listed_names():
    """A constant added to one provider but not the other would escape the comparison above."""
    pytest.importorskip("lal")
    from gwmock_signal.constants import _lal

    for provider in (_builtin, _lal):
        defined = {name for name in vars(provider) if name.isupper()}
        assert defined == set(constants.CONSTANT_NAMES), provider.__name__


@pytest.mark.parametrize(
    ("ours", "ripple_value"),
    [
        pytest.param("MSUN_SI", lambda c: c.MSUN, id="MSUN_SI-MSUN"),
        pytest.param("MTSUN_SI", lambda c: c.MTSUN, id="MTSUN_SI-MTSUN"),
        pytest.param("PC_SI", lambda c: c.MPC / 1e6, id="PC_SI-MPC/1e6"),
        pytest.param("MPC_SI", lambda c: c.MPC, id="MPC_SI-MPC"),
    ],
)
def test_builtin_constant_equals_ripple(ours, ripple_value):
    """Ripple writes LAL's constants out independently; the doubles must agree with ours.

    ripple defines no parsec, but its ``MPC / 1e6`` round-trips exactly to LAL's ``PC_SI``.
    """
    ripple_constants = pytest.importorskip("ripplegw.constants")

    assert getattr(constants, ours) == ripple_value(ripple_constants)


def test_public_names_are_the_builtin_objects():
    """The names the backends have always exported are the built-in values, not a second copy."""
    from gwmock_signal.stochastic import spectrum
    from gwmock_signal.waveform.backends import conditioning, lal

    assert lal.MSUN is constants.MSUN_SI
    assert lal.MPC is constants.MPC_SI
    assert conditioning.MTSUN_SI is constants.MTSUN_SI
    assert spectrum.MPC_SI is constants.MPC_SI


def test_importing_the_constants_does_not_import_lal():
    """The point of the built-in provider: converting units must not require ``lalsuite``.

    The subprocess is pointed at the same source tree this test imported, so it cannot pick up a
    different installed copy of the package.
    """
    # constants/__init__.py -> constants -> gwmock_signal -> the source root.
    source_root = Path(constants.__file__).resolve().parents[2]
    # Drop an empty inherited PYTHONPATH: an empty entry would put the working directory on the path.
    search_path = [str(source_root), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(search_path)}
    code = "import sys, gwmock_signal.constants as c; print(c.__file__); print('lal' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)  # noqa: S603

    imported_from, lal_imported = result.stdout.splitlines()
    assert Path(imported_from).resolve().is_relative_to(source_root)
    assert lal_imported == "False"
