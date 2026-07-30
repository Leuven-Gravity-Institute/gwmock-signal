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
# ripple options are *constructor* kwargs of the waveform, so waveform_arguments is
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
    """A key ripple's constructor does not accept fails early, not as an opaque TypeError."""
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


def _ripple_batch_parameters() -> dict[str, object]:
    """Two-event NRTidal batch on the high-Nyquist grid the taper effect needs."""
    return {
        "detector_frame_mass_1": np.array([1.5, 1.6]),
        "detector_frame_mass_2": np.array([1.4, 1.3]),
        "luminosity_distance": np.array([100.0, 120.0]),
        "lambda_1": np.array([1500.0, 1200.0]),
        "lambda_2": np.array([1500.0, 1400.0]),
    }


def test_ripple_batch_no_taper_changes_the_waveform() -> None:
    """The batch-wide no_taper keyword takes effect through the vmapped path."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    common = {"sampling_frequency": 4096.0, "minimum_frequency": 40.0, "parameters": _ripple_batch_parameters()}
    default = backend.generate_fd_polarizations_batch("IMRPhenomD_NRTidalv2", **common)
    no_taper = backend.generate_fd_polarizations_batch(
        "IMRPhenomD_NRTidalv2", waveform_arguments={"no_taper": True}, **common
    )
    assert np.asarray(default.plus).shape == np.asarray(no_taper.plus).shape
    assert not np.array_equal(np.asarray(default.plus), np.asarray(no_taper.plus))


def test_ripple_batch_none_and_empty_arguments_are_a_no_op() -> None:
    """Omitting the keyword, passing None, and passing {} all agree."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    common = {"sampling_frequency": 4096.0, "minimum_frequency": 40.0, "parameters": _ripple_batch_parameters()}
    omitted = backend.generate_fd_polarizations_batch("IMRPhenomD_NRTidalv2", **common)
    explicit_none = backend.generate_fd_polarizations_batch("IMRPhenomD_NRTidalv2", waveform_arguments=None, **common)
    empty = backend.generate_fd_polarizations_batch("IMRPhenomD_NRTidalv2", waveform_arguments={}, **common)
    np.testing.assert_array_equal(np.asarray(omitted.plus), np.asarray(explicit_none.plus))
    np.testing.assert_array_equal(np.asarray(omitted.plus), np.asarray(empty.plus))


def test_ripple_batch_validates_waveform_arguments() -> None:
    """The batch keyword goes through the same validation as the per-event path."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    common = {"sampling_frequency": 4096.0, "minimum_frequency": 40.0, "parameters": _ripple_batch_parameters()}
    with pytest.raises(ValueError, match="use_lambda_tildes is not supported"):
        backend.generate_fd_polarizations_batch(
            "IMRPhenomD_NRTidalv2", waveform_arguments={"use_lambda_tildes": True}, **common
        )
    with pytest.raises(ValueError, match=r"does not accept waveform_arguments: no_taper"):
        backend.generate_fd_polarizations_batch("IMRPhenomD", waveform_arguments={"no_taper": True}, **common)


def test_ripple_batch_rejects_waveform_arguments_inside_parameters() -> None:
    """waveform_arguments must be its own keyword, not smuggled into parameters."""
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    backend = RippleBackend()
    parameters = {**_ripple_batch_parameters(), "waveform_arguments": {"no_taper": True}}
    with pytest.raises(ValueError, match="its own keyword argument"):
        backend.generate_fd_polarizations_batch(
            "IMRPhenomD_NRTidalv2",
            sampling_frequency=4096.0,
            minimum_frequency=40.0,
            parameters=parameters,
        )


# Interface-hardening: assert this backend's whitelist/reserve assumptions match
# ripple's real constructor signatures. ripple is pre-1.0; if a version bump adds,
# renames, or removes one of these options, these tests fail in the bump PR rather
# than letting waveform_arguments route to the wrong place silently.


def test_ripple_whitelisted_options_exist_in_constructors() -> None:
    """Every whitelisted option is a real keyword of that approximant's constructor."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    for approximant, options in _ALLOWED_WAVEFORM_ARGUMENTS.items():
        # The registry maps a name to the *class*; 0.3.0 replaced the ``waveform_preset``
        # mapping with a ``waveform(name, **config)`` factory, so the class is reached this way.
        signature = inspect.signature(ripplegw.WAVEFORM_REGISTRY[approximant].__init__)
        for option in options:
            assert option in signature.parameters, f"{approximant} constructor no longer accepts {option!r}"


def test_ripple_reserved_use_lambda_tildes_still_exists() -> None:
    """use_lambda_tildes is still a ripple option (so reserving it stays meaningful)."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    assert "use_lambda_tildes" in _RESERVED_WAVEFORM_ARGUMENTS
    signature = inspect.signature(ripplegw.WAVEFORM_REGISTRY["IMRPhenomD_NRTidalv2"].__init__)
    assert "use_lambda_tildes" in signature.parameters


def test_ripple_no_taper_absent_from_non_nrtidal_constructor() -> None:
    """no_taper is genuinely NRTidal-only, justifying the per-approximant whitelist."""
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import inspect

    signature = inspect.signature(ripplegw.WAVEFORM_REGISTRY["IMRPhenomD"].__init__)
    assert "no_taper" not in signature.parameters


# The ripplegw dependency carries no upper bound, so a newer release installs without a change
# here. These tests are half of what replaces the bound -- they fail in the bump PR rather than at
# generation time. The other half is `_require_ripple_interface`, checked at backend construction.


def test_ripple_exposes_the_interface_the_backend_calls() -> None:
    """Every attribute production code reaches for must be present.

    Deliberately the *production* surface only. ``list_waveforms`` and ``get_waveform_metadata``
    are used by tests below but are not required at construction, because refusing to build the
    backend over something it never calls would reject an otherwise compatible release -- the
    opposite of the point of leaving the dependency unbounded above.
    """
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import importlib

    from gwmock_signal.waveform.backends.ripple import _REQUIRED_RIPPLE_INTERFACE

    for module_path, attribute in _REQUIRED_RIPPLE_INTERFACE:
        module = importlib.import_module(module_path)
        assert getattr(module, attribute, None) is not None, f"ripple no longer provides {module_path}.{attribute}"


def test_registry_is_a_mapping_for_the_constructor_option_tests() -> None:
    """``WAVEFORM_REGISTRY`` is read by the signature tests, so it must stay a mapping.

    Checked as ``Mapping`` rather than ``dict``: a custom mapping would serve those tests fine,
    and production does not touch the registry at all.
    """
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    from collections.abc import Mapping

    assert isinstance(getattr(ripplegw, "WAVEFORM_REGISTRY", None), Mapping)


def test_incompatible_ripple_is_rejected_when_the_backend_is_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must fire from ``RippleBackend()``, not merely exist as a function.

    An earlier version of this test called ``_require_ripple_interface`` directly, which does not
    pin the placement it claims: moving the call out of ``__init__`` would leave the test green.
    The module is patched so construction itself has to raise.
    """
    pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import importlib

    from gwmock_signal.waveform.backends import ripple as ripple_module

    real_import = importlib.import_module

    class _Crippled:
        """ripplegw with the factory removed, standing in for an incompatible release."""

        __version__ = "9.9.9"

        def __getattr__(self, name: str) -> object:
            if name == "waveform":
                raise AttributeError(name)
            return getattr(real_import("ripplegw"), name)

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        return _Crippled() if name == "ripplegw" else real_import(name, *args, **kwargs)

    monkeypatch.setattr(ripple_module.importlib, "import_module", _fake_import)
    with pytest.raises(RuntimeError, match=r"9\.9\.9") as raised:
        ripple_module.RippleBackend()
    assert "ripplegw.waveform" in str(raised.value)


def test_the_guard_lists_only_what_is_actually_missing() -> None:
    """The message must name the absent attributes and not the present ones.

    Asserting merely that ``"waveform"`` appears is not enough: the explanatory sentence mentions
    the factory regardless, so that assertion passes even when the missing-name list is wrong.
    This checks the list itself.
    """
    from gwmock_signal.waveform.backends.ripple import _require_ripple_interface

    class _Module:
        __version__ = "9.9.9"

    present = _Module()
    present.waveform = lambda *a, **k: None  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError) as raised:
        _require_ripple_interface(
            {"ripplegw": present, "ripplegw.conversions": _Module(), "ripplegw.constants": _Module()}
        )
    listed = str(raised.value).split(", which this backend")[0]
    assert "ripplegw.conversions.ms_to_Mc_eta" in listed
    assert "ripplegw.constants.MTSUN" in listed
    assert "ripplegw.waveform" not in listed, "an attribute that is present was reported missing"


def test_a_rejected_constructor_option_becomes_a_compatibility_error() -> None:
    """A whitelisted option the installed ripple refuses must not surface as a bare TypeError.

    The whitelist is static and the dependency is unbounded above, so this is the path a renamed
    upstream option takes.
    """
    from gwmock_signal.waveform.backends.ripple import _build_ripple_waveform

    def _factory(name: str, **config: object) -> object:
        raise TypeError(f"__init__() got an unexpected keyword argument 'no_taper' for {name}")

    with pytest.raises(RuntimeError, match="rejected the constructor arguments") as raised:
        _build_ripple_waveform(_factory, "IMRPhenomD_NRTidalv2", f_ref=20.0, options={"no_taper": True})
    message = str(raised.value)
    assert "IMRPhenomD_NRTidalv2" in message
    assert "no_taper" in message


def test_ripple_waveform_factory_accepts_the_name_positionally_and_forwards_config() -> None:
    """``waveform(name, f_ref=...)`` is the exact call shape both call sites use.

    Checked by *calling* it rather than only inspecting parameter kinds, and by confirming the
    configuration reaches the constructed object -- a factory that accepted ``f_ref`` and dropped
    it would pass a signature-only check while silently ignoring the reference frequency.
    """
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    pytest.importorskip("jax", reason="jax not installed")

    constructed = ripplegw.waveform("IMRPhenomD", f_ref=31.0)
    assert constructed is not None
    stored = [
        value
        for value in vars(constructed).values()
        if isinstance(value, (int, float)) and float(value) == pytest.approx(31.0)
    ]
    assert stored, f"f_ref=31.0 was accepted but is not stored on the waveform: {sorted(vars(constructed))}"


@pytest.mark.parametrize(
    ("approximant", "extra"),
    [
        ("IMRPhenomD", {}),
        ("TaylorF2", {"lambda_1": 200.0, "lambda_2": 150.0}),
        ("IMRPhenomXPHM", {"s1_x": 0.1, "s1_y": 0.05, "s2_x": -0.02, "s2_y": 0.03}),
    ],
)
def test_ripple_polarizations_are_returned_under_p_and_c(approximant: str, extra: dict) -> None:
    """Evaluated waveforms must return finite plus/cross under the keys this backend reads.

    Both the batched device kernel and the single-waveform path index ``["p"]`` and ``["c"]``, so a
    rename is a ``KeyError`` deep inside generation. All three parameter families are covered,
    because the tidal and precessing branches pass different parameter sets.
    """
    pytest.importorskip("jax", reason="jax not installed")
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import numpy as np

    frequencies = np.arange(20.0, 60.0, 1.0)
    parameters = {
        "M_c": 25.0,
        "eta": 0.24,
        "s1_z": 0.1,
        "s2_z": -0.05,
        "d_L": 400.0,
        "phase_c": 0.3,
        "iota": 0.6,
        **extra,
    }
    evaluated = ripplegw.waveform(approximant, f_ref=20.0)(frequencies, parameters)
    for key in ("p", "c"):
        assert key in evaluated, f"{approximant} returned keys {sorted(evaluated)}, expected {key!r}"
        values = np.asarray(evaluated[key])
        assert values.shape == frequencies.shape, f"{approximant} {key!r} has shape {values.shape}"
        assert np.all(np.isfinite(values)), f"{approximant} {key!r} contains non-finite values"
        assert np.iscomplexobj(values), f"{approximant} {key!r} is no longer complex"
    assert np.any(np.asarray(evaluated["p"]) != 0.0)


def test_ripple_rejects_a_renamed_parameter_key_loudly() -> None:
    """A renamed required parameter must raise, not silently fall back to a default.

    This is what makes the unbounded dependency tolerable for the ``params`` contract, which the
    construction-time guard cannot see. Measured: ripple raises ``KeyError`` for a missing required
    key, and ignores unknown extra keys without changing the output.
    """
    pytest.importorskip("jax", reason="jax not installed")
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")
    import numpy as np

    frequencies = np.arange(20.0, 40.0, 1.0)
    good = {
        "M_c": 25.0,
        "eta": 0.24,
        "s1_z": 0.1,
        "s2_z": -0.05,
        "d_L": 400.0,
        "phase_c": 0.3,
        "iota": 0.6,
    }
    waveform = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
    reference = np.asarray(waveform(frequencies, good)["p"])

    renamed = {key: value for key, value in good.items() if key != "M_c"} | {"chirp_mass": 25.0}
    with pytest.raises(KeyError):
        waveform(frequencies, renamed)

    # An unknown *extra* key is ignored rather than rejected, and must not perturb the output.
    with_extra = np.asarray(waveform(frequencies, good | {"not_a_parameter": 1.0})["p"])
    assert np.array_equal(with_extra, reference)


def test_spin_and_tidal_classification_agrees_with_ripple_metadata() -> None:
    """This backend's model groupings must match the metadata ripple publishes.

    ``_ALIGNED_SPIN_MODELS`` / ``_TIDAL_MODELS`` / ``_PRECESSING_MODELS`` encode which spin and
    tidal parameters each approximant takes. Since 0.3 ripple publishes the same facts through
    ``get_waveform_metadata``, so the classification exists in two places and can drift -- and a
    drift would route parameters to the wrong model silently, not loudly. Rather than delete the
    local tuples (they also fix the *order* of the public approximant list and gate validation
    messages), this pins the two against each other.

    Both directions are asserted for every group, so a model marked as *both* tidal and precessing
    upstream is caught rather than passing whichever group it happens to be listed in.
    """
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")

    from gwmock_signal.waveform.backends.ripple import (
        _ALIGNED_SPIN_MODELS,
        _PRECESSING_MODELS,
        _TIDAL_MODELS,
    )

    expected = {
        **dict.fromkeys(_ALIGNED_SPIN_MODELS, (False, False)),
        **dict.fromkeys(_TIDAL_MODELS, (True, False)),
        **dict.fromkeys(_PRECESSING_MODELS, (False, True)),
    }
    for approximant, (is_tidal, is_precessing) in expected.items():
        metadata = ripplegw.get_waveform_metadata(approximant)
        assert bool(metadata["is_tidal"]) is is_tidal, (
            f"{approximant}: ripple says is_tidal={metadata['is_tidal']}, this backend assumes {is_tidal}"
        )
        assert bool(metadata["is_precessing"]) is is_precessing, (
            f"{approximant}: ripple says is_precessing={metadata['is_precessing']}, "
            f"this backend assumes {is_precessing}"
        )


def test_every_supported_approximant_is_registered_with_ripple() -> None:
    """No approximant this backend advertises may be missing from ripple's registry.

    ``available_approximants`` is a promise; an unregistered name would only fail at generation
    time, after a catalogue had been configured around it.
    """
    ripplegw = pytest.importorskip("ripplegw", reason="ripple (JAX) not installed")

    from gwmock_signal.waveform.backends.ripple import _SUPPORTED_APPROXIMANTS

    registered = set(ripplegw.list_waveforms())
    missing = sorted(set(_SUPPORTED_APPROXIMANTS) - registered)
    assert not missing, f"advertised but not registered with ripple: {missing}"
