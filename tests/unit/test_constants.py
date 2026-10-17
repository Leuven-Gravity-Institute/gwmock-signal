"""The built-in constants must equal LAL's bit-for-bit, and must not need LAL to import.

LALSimulation converts with its own constants internally, so a built-in value that differed in the
last bit would put a one-ulp error into every waveform generated through it. Equality is therefore
exact (``==``), never a tolerance.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from gwmock_signal import constants
from gwmock_signal.constants import _builtin


@pytest.mark.parametrize("name", constants.CONSTANT_NAMES)
def test_builtin_constant_equals_lal(name):
    """Each built-in value is the double LAL defines, to the last bit."""
    pytest.importorskip("lal")
    from gwmock_signal.constants import _lal

    assert getattr(_builtin, name) == getattr(_lal, name)


def test_every_provider_defines_exactly_the_listed_names():
    """A constant added to one provider but not the other would escape the comparison above."""
    pytest.importorskip("lal")
    from gwmock_signal.constants import _lal

    for provider in (_builtin, _lal):
        defined = {name for name in vars(provider) if name.isupper()}
        assert defined == set(constants.CONSTANT_NAMES), provider.__name__


@pytest.mark.parametrize(
    ("ours", "ripple_name"),
    [("MSUN_SI", "MSUN"), ("MTSUN_SI", "MTSUN"), ("MPC_SI", "MPC")],
)
def test_builtin_constant_equals_ripple(ours, ripple_name):
    """The ripple backend sizes and scales with its own constants; they must be the same doubles."""
    ripple_constants = pytest.importorskip("ripplegw.constants")

    assert getattr(constants, ours) == getattr(ripple_constants, ripple_name)


def test_public_names_resolve_to_the_builtin_values():
    """The names the backends have always exported are the built-in values, not a second copy."""
    from gwmock_signal.waveform.backends import conditioning, lal

    assert lal.MSUN == constants.MSUN_SI
    assert lal.MPC == constants.MPC_SI
    assert conditioning.MTSUN_SI == constants.MTSUN_SI


def test_importing_the_constants_does_not_import_lal():
    """The point of the built-in provider: converting units must not require ``lalsuite``."""
    code = "import sys, gwmock_signal.constants; print('lal' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip() == "False"
