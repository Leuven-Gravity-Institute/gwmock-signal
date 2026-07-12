"""Tests for the waveform_arguments extra-parameter mapping (LAL backend)."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.waveform.backends import GWSignalBackend, LALSimulationBackend

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


def test_gwsignal_backend_falls_back_and_matches_lal() -> None:
    """Until gwsignal translation lands, extras route through the LAL path bit-identically."""
    pytest.importorskip("lalsimulation.gwsignal", reason="lalsimulation.gwsignal not available")
    arguments = {"ModeArray": [(2, 2), (2, -2)]}
    gws = GWSignalBackend().generate_td_waveform("IMRPhenomXHM", waveform_arguments=arguments, **CANONICAL_PARAMS)
    lal = LALSimulationBackend().generate_td_waveform("IMRPhenomXHM", waveform_arguments=arguments, **CANONICAL_PARAMS)
    np.testing.assert_array_equal(gws["plus"].value, lal["plus"].value)
