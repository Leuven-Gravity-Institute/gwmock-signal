"""Tests for the gwsignal waveform backend."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import GWSignalBackend, LALSimulationBackend
from gwmock_signal.waveform.factory import WaveformFactory

pytest.importorskip("lalsimulation.gwsignal", reason="lalsimulation.gwsignal not available")

CANONICAL_PARAMS = {
    "tc": 1_126_259_462.4,
    "sampling_frequency": 2048.0,
    "minimum_frequency": 20.0,
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "luminosity_distance": 410.0,
    "spin_1z": 0.3,
    "spin_2z": -0.2,
    "inclination": 0.4,
    "coa_phase": 1.2,
}


def test_available_approximants_match_lal_backend() -> None:
    """The gwsignal backend advertises the same catalogue as the LAL backend."""
    assert GWSignalBackend().available_approximants() == LALSimulationBackend().available_approximants()


def test_generates_timeseries_dict() -> None:
    """A minimal gwsignal waveform call returns GWpy time series."""
    result = GWSignalBackend().generate_td_waveform("IMRPhenomD", **CANONICAL_PARAMS)
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)
    assert isinstance(result["cross"], TimeSeries)


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXPHM"])
def test_fd_native_matches_lal_backend_exactly(approximant: str) -> None:
    """FD-native approximants are bit-identical to the LAL backend.

    Masses and distances are converted with LAL's SI constants, so no
    unit-conversion drift is tolerated.
    """
    gws = GWSignalBackend().generate_td_waveform(approximant, **CANONICAL_PARAMS)
    lal = LALSimulationBackend().generate_td_waveform(approximant, **CANONICAL_PARAMS)
    for pol in ("plus", "cross"):
        assert gws[pol].t0 == lal[pol].t0
        assert gws[pol].dt == lal[pol].dt
        np.testing.assert_array_equal(gws[pol].value, lal[pol].value)


def test_td_native_falls_back_to_lal_path() -> None:
    """TD-native approximants delegate to the parent SimInspiralFD path.

    gwsignal's FD output discards the conditioning epoch, so the fallback is
    what keeps the coalescence convention intact; the result must equal the
    LAL backend's output exactly.
    """
    approximant = "SEOBNRv4"
    gws = GWSignalBackend().generate_td_waveform(approximant, **CANONICAL_PARAMS)
    lal = LALSimulationBackend().generate_td_waveform(approximant, **CANONICAL_PARAMS)
    for pol in ("plus", "cross"):
        assert gws[pol].t0 == lal[pol].t0
        np.testing.assert_array_equal(gws[pol].value, lal[pol].value)


def test_tidal_parameters_accepted() -> None:
    """lambda_1/lambda_2 flow through the gwsignal parameter dictionary."""
    params = dict(CANONICAL_PARAMS)
    params.update(
        detector_frame_mass_1=1.5,
        detector_frame_mass_2=1.3,
        spin_1z=0.02,
        spin_2z=0.01,
        lambda_1=400.0,
        lambda_2=450.0,
        sampling_frequency=4096.0,
    )
    gws = GWSignalBackend().generate_td_waveform("IMRPhenomPv2_NRTidalv2", **params)
    lal = LALSimulationBackend().generate_td_waveform("IMRPhenomPv2_NRTidalv2", **params)
    np.testing.assert_array_equal(gws["plus"].value, lal["plus"].value)


def test_unsupported_parameter_raises() -> None:
    """Parameter validation is inherited from the LAL backend."""
    params = dict(CANONICAL_PARAMS)
    params["not_a_parameter"] = 1.0
    with pytest.raises(ValueError, match="Unsupported LAL waveform parameters"):
        GWSignalBackend().generate_td_waveform("IMRPhenomD", **params)


def test_factory_integration() -> None:
    """The factory builds its registry from the gwsignal backend."""
    factory = WaveformFactory(backend=GWSignalBackend())
    model = factory.get_model("IMRPhenomD")
    result = model(**CANONICAL_PARAMS)
    assert set(result) == {"plus", "cross"}
