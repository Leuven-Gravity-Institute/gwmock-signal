"""Protocol smoke tests for the public ``GWSimulator.simulate`` contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal import CBCSimulator, GWSimulator
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.timeseries import as_timeseries


def _minimal_params() -> dict[str, float]:
    return {
        "detector_frame_mass_1": 36.0,
        "detector_frame_mass_2": 29.0,
        "coa_time": 1_126_259_462.4,
        "distance": 410.0,
        "inclination": 0.0,
        "right_ascension": 1.375,
        "declination": -1.211,
        "polarization_angle": 0.0,
    }


class _ConstantBackgroundSimulator(GWSimulator):
    @property
    def required_params(self) -> frozenset[str]:
        return frozenset({"amplitude"})

    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[str],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = False,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        del minimum_frequency, earth_rotation, interpolate_if_offset

        self._validate_params(params)
        amplitude = float(params["amplitude"])

        if background is None:
            strains = {
                name: TimeSeries(np.full(8, amplitude), t0=0.0, sample_rate=sampling_frequency)
                for name in detector_names
            }
        else:
            channels = {name: as_timeseries(background[name]) for name in detector_names}
            strains = {
                name: TimeSeries(
                    np.asarray(channel.data, dtype=float) + amplitude,
                    t0=channel.t0,
                    sample_rate=sampling_frequency,
                )
                for name, channel in channels.items()
            }

        return DetectorStrainStack.from_mapping(detector_names, strains)


def test_non_transient_simulator_returns_detector_strain_stack() -> None:
    """A minimal ``GWSimulator`` subclass does not need ``generate_polarizations``."""
    simulator = _ConstantBackgroundSimulator()
    detector_names = ["H1", "L1"]

    result = simulator.simulate(
        {"amplitude": 2.5},
        detector_names,
        background=None,
        sampling_frequency=16.0,
        minimum_frequency=1.0,
    )

    assert isinstance(result, DetectorStrainStack)
    assert result.detector_names == tuple(detector_names)
    assert result.data.shape == (2, 8)
    assert np.allclose(result["H1"].value, 2.5)
    assert np.allclose(result["L1"].value, 2.5)


def test_transient_simulator_accepts_background_none() -> None:
    """CBC simulators can return projected strain directly when no background is given."""
    simulator = CBCSimulator("IMRPhenomD")

    result = simulator.simulate(
        _minimal_params(),
        ["H1"],
        background=None,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
    )

    assert isinstance(result, DetectorStrainStack)
    assert result.detector_names == ("H1",)
    assert result["H1"].n_samples > 0
