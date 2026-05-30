#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""LALSimulation-backed time-domain waveform generation."""

from __future__ import annotations

import lal
import lalsimulation
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

MSUN = lal.MSUN_SI
MPC = lal.PC_SI * 1e6


class LALSimulationBackend(WaveformBackend):
    """Time-domain waveform backend implemented with LALSimulation."""

    def available_approximants(self) -> list[str]:
        """Return all implemented LAL time-domain approximants."""
        return [
            lalsimulation.GetStringFromApproximant(i)
            for i in range(lalsimulation.NumApproximants)
            if lalsimulation.SimInspiralImplementedTDApproximants(i)
        ]

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations with ``SimInspiralChooseTDWaveform``."""
        remaining = dict(params)
        mass1 = float(_pop_alias(remaining, "detector_frame_mass_1", "mass1"))
        mass2 = float(_pop_alias(remaining, "detector_frame_mass_2", "mass2"))
        distance = float(_pop_alias(remaining, "luminosity_distance", "distance"))
        spin_1x = float(_pop_alias(remaining, "spin_1x", "spin1x", default=0.0))
        spin_1y = float(_pop_alias(remaining, "spin_1y", "spin1y", default=0.0))
        spin_1z = float(_pop_alias(remaining, "spin_1z", "spin1z", default=0.0))
        spin_2x = float(_pop_alias(remaining, "spin_2x", "spin2x", default=0.0))
        spin_2y = float(_pop_alias(remaining, "spin_2y", "spin2y", default=0.0))
        spin_2z = float(_pop_alias(remaining, "spin_2z", "spin2z", default=0.0))
        inclination = float(_pop_alias(remaining, "inclination", default=0.0))
        coa_phase = float(_pop_alias(remaining, "coa_phase", default=0.0))
        lambda_1 = float(_pop_alias(remaining, "lambda_1", "tidal_1", default=0.0))
        lambda_2 = float(_pop_alias(remaining, "lambda_2", "tidal_2", default=0.0))
        if remaining:
            extras = ", ".join(sorted(remaining))
            raise ValueError(f"Unsupported LAL waveform parameters: {extras}")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be > 0")
        if lambda_1 < 0:
            raise ValueError("lambda_1 must be >= 0")
        if lambda_2 < 0:
            raise ValueError("lambda_2 must be >= 0")

        approx_enum = lalsimulation.GetApproximantFromString(approximant)
        lal_params = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda_1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, lambda_2)
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
            mass1 * MSUN,
            mass2 * MSUN,
            spin_1x,
            spin_1y,
            spin_1z,
            spin_2x,
            spin_2y,
            spin_2z,
            distance * MPC,
            inclination,
            coa_phase,
            0.0,
            0.0,
            0.0,
            1.0 / sampling_frequency,
            minimum_frequency,
            minimum_frequency,
            lal_params,
            approx_enum,
        )
        t0 = float(hp.epoch) + tc
        dt = hp.deltaT
        return {
            "plus": TimeSeries(hp.data.data, t0=t0, dt=dt),
            "cross": TimeSeries(hc.data.data, t0=t0, dt=dt),
        }
