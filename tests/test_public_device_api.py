"""The on-device entry points must be reachable as advertised public API.

A consumer that has to import ``gwmock_signal.jax_batch`` is coupled to this package's internal
layout rather than to an interface. gwmock's orchestration needs the batched path to run a GPU
end-to-end simulation, so the names it needs are exported here and pinned by these tests.

The export table is lazy, and that is load-bearing rather than incidental: ``[jax]`` is an optional
extra, so importing this package must not import JAX. The test below asserts that, because it is the
kind of property that regresses silently the moment someone adds a convenience top-level import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import gwmock_signal

#: The device surface gwmock needs. Named individually rather than derived from the module, so that
#: removing one is a test failure rather than an invisible narrowing of the API.
_DEVICE_SYMBOLS = (
    "BatchedDetectorStrain",
    "SamplingGrid",
    "assemble_segments",
    "RippleBackend",
    "simulate_cbc_batch",
    "simulate_cbc_catalogue",
)


@pytest.mark.parametrize("name", _DEVICE_SYMBOLS)
def test_device_symbol_is_exported(name: str) -> None:
    """Each device entry point resolves from the package root."""
    assert hasattr(gwmock_signal, name), f"{name} is not reachable as gwmock_signal.{name}"


@pytest.mark.parametrize("name", _DEVICE_SYMBOLS)
def test_device_symbol_is_advertised(name: str) -> None:
    """Each is listed in ``__all__`` and ``dir()``, not merely resolvable by accident."""
    assert name in gwmock_signal.__all__
    assert name in dir(gwmock_signal)


@pytest.mark.parametrize("name", _DEVICE_SYMBOLS)
def test_the_exported_object_is_the_submodule_object(name: str) -> None:
    """The export must alias the implementation, not a second copy of it.

    A re-implementation behind the public name would drift from the one the tests exercise.
    """
    # No importorskip: importing ``jax_batch`` does not import JAX, so this runs in a base
    # installation too -- which is the mode the lazy table exists to support, and where an earlier
    # version of this test was silently skipped and therefore vacuous.
    import importlib

    module = {
        "SamplingGrid": "gwmock_signal.sampling_grid",
        "RippleBackend": "gwmock_signal.waveform.backends",
    }.get(name, "gwmock_signal.jax_batch")
    assert getattr(gwmock_signal, name) is getattr(importlib.import_module(module), name)


def test_importing_the_package_does_not_import_jax() -> None:
    """``[jax]`` is optional, so a plain import must not pull it in.

    Run in a subprocess: this test process has already imported JAX via other tests, so checking
    ``sys.modules`` in-process would pass regardless of what the import does.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import sys, gwmock_signal; print('jax' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "importing gwmock_signal imported JAX; the [jax] extra is no longer optional"
    )


def test_resolving_a_device_symbol_does_not_import_jax_either() -> None:
    """Touching the name resolves the module, whose own JAX imports are function-local.

    Not a requirement of the design -- only the previous test is -- but it is a real property worth
    pinning, since it lets a caller introspect the API without a GPU stack present.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, gwmock_signal; gwmock_signal.simulate_cbc_catalogue; print('jax' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_an_unknown_symbol_still_raises_attribute_error() -> None:
    """The lazy ``__getattr__`` must not turn a typo into something other than AttributeError."""
    with pytest.raises(AttributeError):
        gwmock_signal.simulate_cbc_catalog  # noqa: B018 - the access is the assertion


def test_recommend_chunk_size_is_deliberately_not_root_api() -> None:
    """The memory heuristic stays out of the package root, but remains reachable.

    Its model is calibrated from a single A100 measurement and exists to turn an opaque allocation
    failure into an actionable one, so it is not a promise worth making at the root. Pinned in both
    directions so neither the omission nor the availability is lost by accident.
    """
    assert "recommend_chunk_size" not in gwmock_signal.__all__
    from gwmock_signal.jax_batch import recommend_chunk_size

    assert callable(recommend_chunk_size)


def test_the_missing_extra_failure_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the ``[jax]`` extra, *using* the device path must say how to fix it.

    This is the likely first experience of the feature for anyone on a base install, and it was the
    one path the review could not execute -- resolving an exported name stays lazy and succeeds, so
    the failure only appears on use. Simulated by making the ripple import fail rather than by
    uninstalling anything.
    """
    import importlib

    from gwmock_signal.waveform.backends.ripple import RippleBackend

    real_import = importlib.import_module

    def _fail_for_ripple(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("ripplegw") or name == "jax":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fail_for_ripple)
    with pytest.raises(ImportError) as raised:
        RippleBackend()
    message = str(raised.value)
    assert "gwmock-signal[jax]" in message, f"the error does not name the install command: {message}"
