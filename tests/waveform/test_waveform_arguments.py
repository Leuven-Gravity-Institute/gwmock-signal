"""Tests for the waveform_arguments extra-parameter mapping (LAL backend)."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import GWSignalBackend, LALSimulationBackend, PyCBCBackend, RippleBackend
from gwmock_signal.waveform.backends.ripple import (
    _ALLOWED_WAVEFORM_ARGUMENTS,
    _RESERVED_WAVEFORM_ARGUMENTS,
)

CANONICAL_PARAMS = {
    "tc": 1_126_259_462.4,
    "sampling_frequency": 2048.0,
    "minimum_frequency": 20.0,
    "detector_frame_mass_1": 40.0,
    "detector_frame_mass_2": 20.0,
    "luminosity_distance": 410.0,
    "spin_1z": 0.3,
    "spin_2z": -0.2,
    "inclination": 0.8,
    "coa_phase": 1.2,
}


def test_mode_array_restricts_higher_modes() -> None:
    """Restricting IMRPhenomXHM to (2,2) changes the waveform."""
    backend = LALSimulationBackend()
    full = backend.generate_td_waveform("IMRPhenomXHM", **CANONICAL_PARAMS)
    dominant = backend.generate_td_waveform(
        "IMRPhenomXHM", waveform_arguments={"ModeArray": [(2, 2), (2, -2)]}, **CANONICAL_PARAMS
    )
    assert full["plus"].value.shape == dominant["plus"].value.shape
    assert not np.array_equal(full["plus"].value, dominant["plus"].value)


def test_scalar_argument_is_applied() -> None:
    """A scalar LAL-dictionary option (PhenomXHMMband threshold) is honoured."""
    backend = LALSimulationBackend()
    default = backend.generate_td_waveform("IMRPhenomXHM", **CANONICAL_PARAMS)
    tweaked = backend.generate_td_waveform(
        "IMRPhenomXHM", waveform_arguments={"PhenomXHMThresholdMband": 0.0}, **CANONICAL_PARAMS
    )
    assert not np.array_equal(default["plus"].value, tweaked["plus"].value)


def test_empty_arguments_are_a_no_op() -> None:
    """An empty mapping produces the same waveform as omitting it."""
    backend = LALSimulationBackend()
    without = backend.generate_td_waveform("IMRPhenomD", **CANONICAL_PARAMS)
    with_empty = backend.generate_td_waveform("IMRPhenomD", waveform_arguments={}, **CANONICAL_PARAMS)
    np.testing.assert_array_equal(without["plus"].value, with_empty["plus"].value)


def test_unknown_argument_raises() -> None:
    """Keys without a matching LAL setter fail loudly."""
    with pytest.raises(ValueError, match="SimInspiralWaveformParamsInsertNotAThing"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD", waveform_arguments={"NotAThing": 1}, **CANONICAL_PARAMS
        )


@pytest.mark.parametrize("reserved", ["TidalLambda1", "TidalLambda2"])
def test_tidal_arguments_are_reserved(reserved: str) -> None:
    """Tidal deformabilities must go through the canonical lambda_1/lambda_2."""
    with pytest.raises(ValueError, match="lambda_1/lambda_2"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD", waveform_arguments={reserved: 100.0}, **CANONICAL_PARAMS
        )


@pytest.mark.parametrize("bad_value", ["not-a-list", [(2,)], [2, 2], None])
def test_malformed_mode_array_raises(bad_value: object) -> None:
    """Malformed ModeArray values get a descriptive error, not a bare unpack failure."""
    with pytest.raises(ValueError, match=r"ModeArray must be an iterable of \(l, m\) pairs"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomXHM", waveform_arguments={"ModeArray": bad_value}, **CANONICAL_PARAMS
        )


def test_non_dict_arguments_raise() -> None:
    """waveform_arguments must be a mapping with string keys."""
    with pytest.raises(ValueError, match="dict with string keys"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD", waveform_arguments=[("ModeArray", [(2, 2)])], **CANONICAL_PARAMS
        )


@pytest.mark.parametrize(
    ("approximant", "arguments"),
    [
        ("IMRPhenomXHM", {"ModeArray": [(2, 2), (2, -2)]}),
        ("IMRPhenomXHM", {"PhenomXHMThresholdMband": 0.0}),
        ("IMRPhenomXPHM", {"PhenomXPrecVersion": 102}),
    ],
)
def test_gwsignal_backend_applies_arguments_via_lal_path(approximant: str, arguments: dict) -> None:
    """Extras route through the LAL path bit-identically.

    gwsignal's public dictionary cannot carry them faithfully; see the
    backend docstring.
    """
    pytest.importorskip("lalsimulation.gwsignal", reason="lalsimulation.gwsignal not available")
    gws = GWSignalBackend().generate_td_waveform(approximant, waveform_arguments=arguments, **CANONICAL_PARAMS)
    lal = LALSimulationBackend().generate_td_waveform(approximant, waveform_arguments=arguments, **CANONICAL_PARAMS)
    np.testing.assert_array_equal(gws["plus"].value, lal["plus"].value)
    np.testing.assert_array_equal(gws["cross"].value, lal["cross"].value)


def test_gwsignal_backend_arguments_change_the_waveform() -> None:
    """The options actually take effect through the gwsignal backend entry point."""
    pytest.importorskip("lalsimulation.gwsignal", reason="lalsimulation.gwsignal not available")
    backend = GWSignalBackend()
    full = backend.generate_td_waveform("IMRPhenomXHM", **CANONICAL_PARAMS)
    dominant = backend.generate_td_waveform(
        "IMRPhenomXHM", waveform_arguments={"ModeArray": [(2, 2), (2, -2)]}, **CANONICAL_PARAMS
    )
    assert not np.array_equal(full["plus"].value, dominant["plus"].value)


# --- PyCBC backend --------------------------------------------------------
#
# ``_resolve_waveform_arguments`` is a staticmethod, so its input validation is
# exercised without importing PyCBC. The pass-through / locking behaviour of
# ``generate_td_waveform`` is driven with PyCBC's import faked and the wrapper
# mocked, so it runs (and is coverage-counted) even where PyCBC is not
# installed -- no CI job installs the [pycbc] extra.


def _fake_result() -> dict[str, TimeSeries]:
    ts = TimeSeries(np.zeros(10), t0=0.0, dt=1.0 / 4096)
    return {"plus": ts, "cross": ts}


@contextmanager
def _pycbc_backend(mock_wrapper: MagicMock):
    """Yield a PyCBCBackend with PyCBC's import faked and the wrapper mocked.

    ``PyCBCBackend.__init__`` imports ``pycbc.waveform`` and
    ``generate_td_waveform`` imports the local ``pycbc_wrapper`` module; both go
    through this module's ``importlib.import_module``. Faking only the former
    lets the method run without a real PyCBC install while the wrapper is
    patched to capture the forwarded kwargs.
    """
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "pycbc.waveform":
            fake = MagicMock()
            fake.td_approximants.return_value = ["IMRPhenomD", "IMRPhenomXHM"]
            return fake
        return real_import(name)

    with (
        patch("gwmock_signal.waveform.backends.pycbc.importlib.import_module", side_effect=fake_import),
        patch("gwmock_signal.waveform.pycbc_wrapper.pycbc_waveform_wrapper", mock_wrapper),
    ):
        yield PyCBCBackend()


def test_pycbc_non_dict_arguments_raise() -> None:
    """waveform_arguments must be a mapping with string keys."""
    with pytest.raises(ValueError, match="dict with string keys"):
        PyCBCBackend._resolve_waveform_arguments([("mode_array", [[2, 2]])])


def test_pycbc_non_string_keys_raise() -> None:
    """Non-string keys are rejected."""
    with pytest.raises(ValueError, match="dict with string keys"):
        PyCBCBackend._resolve_waveform_arguments({1: "x"})


@pytest.mark.parametrize("reserved", ["mass1", "spin1z", "lambda1", "approximant", "delta_t", "f_lower"])
def test_pycbc_reserved_arguments_raise(reserved: str) -> None:
    """Keys the backend derives or manages itself cannot be overridden via extras."""
    with pytest.raises(ValueError, match="not waveform_arguments"):
        PyCBCBackend._resolve_waveform_arguments({reserved: 0.0})


def test_pycbc_non_reserved_arguments_pass_validation() -> None:
    """Approximant-specific options survive validation unchanged."""
    args = {"mode_array": [[2, 2], [3, 3]], "f_ref": 30.0}
    assert PyCBCBackend._resolve_waveform_arguments(args) == args


def test_pycbc_arguments_forwarded_to_get_td_waveform() -> None:
    """Extras are merged into the kwargs passed to the PyCBC wrapper."""
    mock_wrapper = MagicMock(return_value=_fake_result())
    with _pycbc_backend(mock_wrapper) as backend:
        backend.generate_td_waveform(
            "IMRPhenomXHM",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=40.0,
            detector_frame_mass_2=30.0,
            luminosity_distance=410.0,
            waveform_arguments={"mode_array": [[2, 2]], "f_ref": 30.0},
        )
    kw = mock_wrapper.call_args.kwargs
    assert kw["mode_array"] == [[2, 2]]
    assert kw["f_ref"] == 30.0
    assert kw["mass1"] == 40.0
    assert kw["lambda1"] == 0.0
    assert "waveform_arguments" not in kw


def test_pycbc_empty_arguments_are_a_no_op() -> None:
    """An empty mapping forwards no extra kwargs."""
    mock_wrapper = MagicMock(return_value=_fake_result())
    with _pycbc_backend(mock_wrapper) as backend:
        backend.generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=40.0,
            detector_frame_mass_2=30.0,
            luminosity_distance=410.0,
            waveform_arguments={},
        )
    kw = mock_wrapper.call_args.kwargs
    assert "waveform_arguments" not in kw
    assert "mode_array" not in kw


def test_pycbc_unsupported_top_level_parameter_raises() -> None:
    """Extras must go through waveform_arguments, not as flat top-level kwargs."""
    mock_wrapper = MagicMock(return_value=_fake_result())
    with (
        _pycbc_backend(mock_wrapper) as backend,
        pytest.raises(ValueError, match="Unsupported PyCBC waveform parameters: f_ref"),
    ):
        backend.generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=40.0,
            detector_frame_mass_2=30.0,
            luminosity_distance=410.0,
            f_ref=30.0,
        )
    mock_wrapper.assert_not_called()


# --- ripple backend -------------------------------------------------------
#
# ripple options are *constructor* kwargs of the preset, so waveform_arguments is
# a narrow whitelist. ``_resolve_waveform_arguments`` is a staticmethod, so the
# validation runs without JAX/ripple; behaviour and the interface-hardening
# checks need the real ripple install (the test-jax CI job).

# sampling_frequency is deliberately high (Nyquist 2048 Hz): the NRTidal taper
# no_taper toggles acts near the ~kHz merger/contact frequency, so at a lower
# Nyquist it would be truncated and the option would look like a no-op. The
# higher minimum_frequency keeps the analysis segment (and test runtime) small.
_RIPPLE_BNS_PARAMS = {
    "tc": 1_126_259_462.4,
    "sampling_frequency": 4096.0,
    "minimum_frequency": 40.0,
    "detector_frame_mass_1": 1.5,
    "detector_frame_mass_2": 1.4,
    "luminosity_distance": 100.0,
    "lambda_1": 1500.0,
    "lambda_2": 1500.0,
}


def test_ripple_non_dict_arguments_raise() -> None:
    """waveform_arguments must be a mapping with string keys."""
    with pytest.raises(ValueError, match="dict with string keys"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", [("no_taper", True)])


def test_ripple_non_string_keys_raise() -> None:
    """Non-string keys are rejected."""
    with pytest.raises(ValueError, match="dict with string keys"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", {1: True})


def test_ripple_allowed_option_passes_validation() -> None:
    """A whitelisted option for the approximant survives validation unchanged."""
    args = {"no_taper": True}
    assert RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", args) == args


def test_ripple_f_ref_is_reserved() -> None:
    """f_ref is backend-owned and cannot be set via waveform_arguments."""
    with pytest.raises(ValueError, match="f_ref is configured on the backend"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", {"f_ref": 30.0})


def test_ripple_use_lambda_tildes_is_reserved() -> None:
    """use_lambda_tildes conflicts with the canonical lambda_1/lambda_2 mapping."""
    with pytest.raises(ValueError, match="use_lambda_tildes is not supported"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", {"use_lambda_tildes": True})


def test_ripple_unknown_option_raises() -> None:
    """A key ripple's preset does not accept fails early, not as an opaque TypeError."""
    with pytest.raises(ValueError, match="does not accept waveform_arguments: not_a_thing"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD_NRTidalv2", {"not_a_thing": 1})


def test_ripple_option_rejected_for_wrong_approximant() -> None:
    """no_taper is only valid for NRTidal models; other approximants reject it."""
    with pytest.raises(ValueError, match=r"does not accept waveform_arguments: no_taper.*Allowed.*\(none\)"):
        RippleBackend._resolve_waveform_arguments("IMRPhenomD", {"no_taper": True})


def test_ripple_no_taper_changes_the_waveform() -> None:
    """The whitelisted no_taper option actually takes effect through the backend."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    default = backend.generate_td_waveform("IMRPhenomD_NRTidalv2", **_RIPPLE_BNS_PARAMS)
    no_taper = backend.generate_td_waveform(
        "IMRPhenomD_NRTidalv2", waveform_arguments={"no_taper": True}, **_RIPPLE_BNS_PARAMS
    )
    assert default["plus"].value.shape == no_taper["plus"].value.shape
    assert not np.array_equal(default["plus"].value, no_taper["plus"].value)


def test_ripple_empty_arguments_are_a_no_op() -> None:
    """An empty mapping produces the same waveform as omitting it."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    without = backend.generate_td_waveform("IMRPhenomD_NRTidalv2", **_RIPPLE_BNS_PARAMS)
    with_empty = backend.generate_td_waveform("IMRPhenomD_NRTidalv2", waveform_arguments={}, **_RIPPLE_BNS_PARAMS)
    np.testing.assert_array_equal(without["plus"].value, with_empty["plus"].value)


def test_ripple_batch_path_rejects_waveform_arguments() -> None:
    """The batch path does not support waveform_arguments yet and says so."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    with pytest.raises(ValueError, match="not supported in the batch path"):
        backend.generate_fd_polarizations_batch(
            "IMRPhenomD_NRTidalv2",
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            parameters={
                "detector_frame_mass_1": np.array([1.5]),
                "detector_frame_mass_2": np.array([1.4]),
                "luminosity_distance": np.array([100.0]),
                "lambda_1": np.array([400.0]),
                "lambda_2": np.array([300.0]),
                "waveform_arguments": {"no_taper": True},
            },
        )


# Interface-hardening: assert this backend's whitelist/reserve assumptions match
# ripple's real constructor signatures. ripple is pre-1.0; if a version bump adds,
# renames, or removes one of these options, these tests fail in the bump PR rather
# than letting waveform_arguments route to the wrong place silently.


def test_ripple_whitelisted_options_exist_in_constructors() -> None:
    """Every whitelisted option is a real keyword of that approximant's preset."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    for approximant, options in _ALLOWED_WAVEFORM_ARGUMENTS.items():
        signature = inspect.signature(ripplegw.waveform_preset[approximant].__init__)
        for option in options:
            assert option in signature.parameters, f"{approximant} preset no longer accepts {option!r}"


def test_ripple_reserved_use_lambda_tildes_still_exists() -> None:
    """use_lambda_tildes is still a ripple option (so reserving it stays meaningful)."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    assert "use_lambda_tildes" in _RESERVED_WAVEFORM_ARGUMENTS
    signature = inspect.signature(ripplegw.waveform_preset["IMRPhenomD_NRTidalv2"].__init__)
    assert "use_lambda_tildes" in signature.parameters


def test_ripple_no_taper_absent_from_non_nrtidal_constructor() -> None:
    """no_taper is genuinely NRTidal-only, justifying the per-approximant whitelist."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    signature = inspect.signature(ripplegw.waveform_preset["IMRPhenomD"].__init__)
    assert "no_taper" not in signature.parameters
