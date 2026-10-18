"""Test importing the python-gwmock_signal package and its modules."""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import gwmock_signal


def get_all_submodules(package):
    """Discover all submodules in the package.

    Args:
        package: The package to inspect.

    """
    submodules = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(mod_name)
    return submodules


def test_import_main_package():
    """Test that the main gwmock_signal package can be imported."""
    assert hasattr(gwmock_signal, "__version__")
    assert gwmock_signal.__version__ is not None


def test_import_main_package_without_pycbc() -> None:
    """Importing the top-level package must not require PyCBC."""
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import builtins

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("pycbc"):
        raise ModuleNotFoundError("No module named 'pycbc'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

import gwmock_signal

assert hasattr(gwmock_signal, "__version__")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


#: Submodules that import lalsuite at module level, so they can only be imported where it is installed.
_NEEDS_LALSUITE = {"gwmock_signal.constants._lal"}


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(name, marks=pytest.mark.requires_lalsuite) if name in _NEEDS_LALSUITE else name
        for name in get_all_submodules(gwmock_signal)
    ],
)
def test_import_submodule(module_name):
    """Test that all submodules can be imported."""
    __import__(module_name)
