"""The simulators take the package's own time series as well as GWpy's, and return the same stacks.

A background of either kind gives identical strain. A registered waveform model may return either
kind; anything else is refused.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock_signal.injection.core import _add_strain
from gwmock_signal.simulator import CBCSimulator
from gwmock_signal.stochastic import StochasticBackgroundSimulator
from gwmock_signal.timeseries import TimeSeries
from gwmock_signal.timeseries import TimeSeries as NativeTimeSeries
from gwmock_signal.waveform.backends import LALSimulationBackend

_TC = 1_126_259_462.4
_PARAMS = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "distance": 410.0,
    "inclination": 0.0,
    "coa_phase": 0.0,
    "coa_time": _TC,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0,
}
_FS = 512.0


def _toy_simulator(model) -> CBCSimulator:
    backend = MagicMock(spec=LALSimulationBackend)
    backend.available_approximants.return_value = []
    simulator = CBCSimulator(waveform_model="toy", waveform_backend=backend)
    simulator.register_waveform_model("toy", model)
    return simulator


def _native_toy(*, tc: float, sampling_frequency: float, **_: object) -> tuple[TimeSeries, TimeSeries]:
    t = np.arange(64) / sampling_frequency
    return (
        TimeSeries(np.sin(2 * np.pi * 40.0 * t), t0=tc, sample_rate=sampling_frequency),
        TimeSeries(np.cos(2 * np.pi * 40.0 * t), t0=tc, sample_rate=sampling_frequency),
    )


def _gwpy_toy(**kwargs: object) -> tuple[GWpyTimeSeries, GWpyTimeSeries]:
    plus, cross = _native_toy(**kwargs)
    return plus.to_gwpy(), cross.to_gwpy()


@pytest.mark.parametrize("model", [_native_toy, _gwpy_toy], ids=["native-model", "gwpy-model"])
def test_a_cbc_background_of_either_kind_gives_the_same_strain(model):
    """Native and GWpy backgrounds, with a model returning either kind, inject identically."""
    simulator = _toy_simulator(model)
    background = {name: TimeSeries(np.zeros(256), t0=_TC - 0.125, sample_rate=_FS) for name in ("H1", "L1")}
    kwargs = {"sampling_frequency": _FS, "minimum_frequency": 20.0}

    native = simulator.simulate(_PARAMS, ["H1", "L1"], background, **kwargs)
    gwpy = simulator.simulate(_PARAMS, ["H1", "L1"], {k: v.to_gwpy() for k, v in background.items()}, **kwargs)

    assert np.array_equal(native.data, gwpy.data)
    assert np.any(native.data != 0.0)
    assert isinstance(native["H1"], NativeTimeSeries)


def test_a_registered_model_returning_anything_else_is_refused():
    """A bare array has no timing, so it is not a polarization."""
    simulator = _toy_simulator(lambda **_: (np.zeros(8), np.zeros(8)))

    with pytest.raises(TypeError, match=r"must return gwmock_signal\.TimeSeries or GWpy TimeSeries"):
        simulator.generate_polarizations(_PARAMS, sampling_frequency=_FS, minimum_frequency=20.0)


def test_a_registered_model_s_own_series_are_returned():
    """``generate_polarizations`` hands back what the model returned, native here."""
    plus, cross = _toy_simulator(_native_toy).generate_polarizations(
        _PARAMS, sampling_frequency=_FS, minimum_frequency=20.0
    )

    assert isinstance(plus, TimeSeries)
    assert isinstance(cross, TimeSeries)


def test_a_stochastic_background_of_either_kind_gives_the_same_strain():
    """The same seed and a background of either kind give identical strain."""
    pytest.importorskip("gwmock_noise")
    background = {
        "H1": TimeSeries(np.ones(128), t0=100.0, sample_rate=128.0, unit="strain"),
        "L1": TimeSeries(np.full(128, 2.0), t0=100.0, sample_rate=128.0, unit="strain"),
    }
    results = []
    for kind in (background, {k: v.to_gwpy() for k, v in background.items()}):
        simulator = StochasticBackgroundSimulator(duration=1.0, seed=4, overlap_reduction={("H1", "L1"): np.zeros(65)})
        results.append(
            simulator.simulate(
                {"omega_ref": 1.0e30}, ["H1", "L1"], background=kind, sampling_frequency=128.0, minimum_frequency=0.0
            )
        )

    assert np.array_equal(results[0].data, results[1].data)
    assert results[0].t0 == 100.0


class TestAddStrain:
    """Adding a signal generated on the background's own samples."""

    def test_the_background_must_be_strain(self):
        """A dimensionless background is refused, as adding a strain to it always was."""
        with pytest.raises(ValueError, match="must be strain"):
            _add_strain(TimeSeries(np.zeros(4), t0=0.0, sample_rate=1.0), np.ones(4))

    def test_the_lengths_must_match(self):
        """The signal is one value per background sample."""
        with pytest.raises(ValueError, match="3 samples but the background has 4"):
            _add_strain(TimeSeries(np.zeros(4), t0=0.0, sample_rate=1.0, unit="strain"), np.ones(3))

    def test_the_result_keeps_the_background_s_timing_and_name(self):
        """Only the samples change."""
        background = TimeSeries(np.ones(4), t0=5.0, sample_rate=2.0, unit="strain", name="H1")

        result = _add_strain(background, np.arange(4.0))

        assert result.data.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert (result.t0, result.sample_rate, result.name, result.unit) == (5.0, 2.0, "H1", "strain")
        assert background.data.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_the_cbc_path_does_not_precess_the_source_direction():
    """Compact-binary searches use the source direction as given; only the continuous-wave path precesses."""
    from unittest.mock import patch

    from gwmock_signal.projection.network import _project

    simulator = _toy_simulator(_native_toy)
    with patch("gwmock_signal.simulator._project", side_effect=_project) as spy:
        simulator.simulate(_PARAMS, ["H1"], sampling_frequency=_FS, minimum_frequency=20.0)

    assert spy.call_count == 1
    assert spy.call_args.kwargs["precess_source_direction"] is False
