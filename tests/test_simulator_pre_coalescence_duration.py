"""Asking a simulator how long before ``coa_time`` its output starts.

The backend already answers this; the point of exposing it on the simulator is that a caller
placing signals into segmented data holds a simulator, not a backend, and reaching the backend
through ``_waveform_factory._backend`` would be two private attributes across a package boundary.

The property under test is agreement: the number a simulator promises must be where the buffer it
generates actually starts. A test recomputing the sizing formula would pass against a simulator
whose generation had drifted from what it reports, which is the failure that matters.
"""

from __future__ import annotations

import pytest

from gwmock_signal.simulator import CBCSimulator

_TC = 1419724820.0

#: Complete source mappings, exactly as a caller passes them -- including the projection keys the
#: waveform backend must never see.
#:
#: They matter. The first version of these tests fed the prediction a *reduced* mapping and
#: generation a separately augmented one, so both paths passed while the prediction was in fact
#: unusable: it forwarded `coa_time`, `right_ascension`, `declination` and `polarization_angle` to a
#: backend that rejects them, raising `ValueError` on the very input generation accepts. Tailoring
#: the input per path is what hid it, so both now receive the same mapping.
_BBH = {
    "mass1": 30.0,
    "mass2": 25.0,
    "distance": 400.0,
    "inclination": 0.4,
    "coa_time": _TC,
    "right_ascension": 1.1,
    "declination": 0.3,
    "polarization_angle": 0.2,
}
_BNS = {**_BBH, "mass1": 1.4, "mass2": 1.35, "distance": 100.0}


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
@pytest.mark.parametrize("sampling_frequency", [1024.0, 2048.0])
def test_the_simulator_predicts_where_its_own_output_starts(params, sampling_frequency):
    """Promise equals delivery, through the same entry point a caller uses."""
    simulator = CBCSimulator(waveform_model="IMRPhenomD")

    predicted = simulator.pre_coalescence_duration(params, sampling_frequency, 20.0)
    hp, _ = simulator.generate_polarizations(params, sampling_frequency, 20.0)
    actual = _TC - float(hp.t0.value)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / sampling_frequency)


def test_a_custom_registered_model_answers_unknown():
    """A registered callable never reaches the backend, so the backend's sizing cannot describe it.

    Returning the backend's number here would be worse than returning nothing: it would look
    authoritative while describing a waveform the simulator is not going to generate, and a caller
    would place the event using it.
    """
    import numpy as np
    from gwpy.timeseries import TimeSeries

    def _flat(**kwargs):
        del kwargs
        n = 128
        return {
            "plus": TimeSeries(np.ones(n), t0=0.0, sample_rate=128.0),
            "cross": TimeSeries(np.zeros(n), t0=0.0, sample_rate=128.0),
        }

    simulator = CBCSimulator(waveform_model="MyFlatModel")
    simulator.register_waveform_model("MyFlatModel", _flat)

    assert simulator.pre_coalescence_duration(_BBH, 1024.0, 20.0) is None


def test_a_factory_registration_shadowing_an_approximant_answers_unknown():
    """The exclusion is by identity, not by name, so shadowing cannot slip through.

    Exercised through :class:`WaveformFactory` rather than the simulator, because the two layers
    differ and the difference is worth recording: ``CBCSimulator.register_waveform_model`` refuses a
    name that already exists, while the factory's overwrites it. The factory is public API, so the
    identity check is not defending against a hypothetical -- and a name-based check would hand the
    backend's sizing to a waveform it no longer generates.
    """
    import numpy as np
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.simulator import _waveform_parameters
    from gwmock_signal.waveform.factory import WaveformFactory

    def _flat(**kwargs):
        del kwargs
        return {
            "plus": TimeSeries(np.ones(128), t0=0.0, sample_rate=128.0),
            "cross": TimeSeries(np.zeros(128), t0=0.0, sample_rate=128.0),
        }

    # Backend-shaped parameters here, not the caller's mapping: stripping the projection-owned keys
    # is the simulator's contract, and the factory sits below it.
    backend_params = _waveform_parameters(_BBH)

    factory = WaveformFactory()
    assert factory.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **backend_params) is not None

    factory.register_model("IMRPhenomD", _flat)

    assert factory.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **backend_params) is None


def test_the_simulator_refuses_to_shadow_an_existing_model():
    """Recorded because it is why the test above uses the factory, not the simulator."""
    import numpy as np
    from gwpy.timeseries import TimeSeries

    def _flat(**kwargs):
        del kwargs
        return {
            "plus": TimeSeries(np.ones(128), t0=0.0, sample_rate=128.0),
            "cross": TimeSeries(np.zeros(128), t0=0.0, sample_rate=128.0),
        }

    simulator = CBCSimulator(waveform_model="IMRPhenomD")

    with pytest.raises(ValueError, match="already registered"):
        simulator.register_waveform_model("IMRPhenomD", _flat)


def test_an_unknown_model_raises_like_generation_would():
    """Unknown and unanswerable are different: one is a caller error, the other a capability gap."""
    simulator = CBCSimulator(waveform_model="NotAnApproximant")

    with pytest.raises(ValueError, match="not found"):
        simulator.pre_coalescence_duration(_BBH, 1024.0, 20.0)


def test_the_query_accepts_exactly_what_generation_accepts():
    """Neither path may require a mapping the other rejects.

    The bug this pins: generation strips the projection-owned keys before reaching the backend and
    the query did not, so a complete CBC mapping raised `ValueError: Unsupported LAL waveform
    parameters: coa_time, declination, polarization_angle, right_ascension` from the query while
    generation succeeded. The query was therefore unusable by its only intended caller, and every
    other test passed because it was handed a reduced mapping.

    Asserted as a symmetry rather than by listing the keys, so adding a projection parameter cannot
    reintroduce the divergence.
    """
    simulator = CBCSimulator(waveform_model="IMRPhenomD")

    predicted = simulator.pre_coalescence_duration(_BBH, 1024.0, 20.0)
    hp, _ = simulator.generate_polarizations(_BBH, 1024.0, 20.0)

    assert predicted == pytest.approx(_TC - float(hp.t0.value), abs=0.5 / 1024.0)
    # And the mapping really does carry the keys that used to break it, or this proves nothing.
    assert {"coa_time", "right_ascension", "declination", "polarization_angle"} <= set(_BBH)
