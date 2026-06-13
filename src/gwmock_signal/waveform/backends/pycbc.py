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
"""PyCBC-backed time-domain waveform generation."""

from __future__ import annotations

import importlib

from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

_PYCBC_IMPORT_ERROR = "pycbc is not installed. Run: pip install 'gwmock-signal[pycbc]'"


class PyCBCBackend(WaveformBackend):
    """Time-domain waveform backend implemented with PyCBC."""

    def __init__(self) -> None:
        """Require PyCBC only when this backend is instantiated."""
        try:
            self._pycbc_waveform = importlib.import_module("pycbc.waveform")
        except ImportError as exc:
            raise ImportError(_PYCBC_IMPORT_ERROR) from exc

    def available_approximants(self) -> list[str]:
        """Return all PyCBC time-domain approximants."""
        return list(self._pycbc_waveform.td_approximants())

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations through ``pycbc_waveform_wrapper``."""
        pycbc_waveform_wrapper = importlib.import_module("gwmock_signal.waveform.pycbc_wrapper").pycbc_waveform_wrapper
        remaining = dict(params)
        translated = {
            "mass1": _pop_alias(remaining, "detector_frame_mass_1", "mass1"),
            "mass2": _pop_alias(remaining, "detector_frame_mass_2", "mass2"),
            "distance": _pop_alias(remaining, "luminosity_distance", "distance"),
            "spin1x": _pop_alias(remaining, "spin_1x", "spin1x", default=0.0),
            "spin1y": _pop_alias(remaining, "spin_1y", "spin1y", default=0.0),
            "spin1z": _pop_alias(remaining, "spin_1z", "spin1z", default=0.0),
            "spin2x": _pop_alias(remaining, "spin_2x", "spin2x", default=0.0),
            "spin2y": _pop_alias(remaining, "spin_2y", "spin2y", default=0.0),
            "spin2z": _pop_alias(remaining, "spin_2z", "spin2z", default=0.0),
            "inclination": _pop_alias(remaining, "inclination", default=0.0),
            "coa_phase": _pop_alias(remaining, "coa_phase", default=0.0),
            "lambda1": _pop_alias(remaining, "lambda_1", "lambda1", "tidal_1", default=0.0),
            "lambda2": _pop_alias(remaining, "lambda_2", "lambda2", "tidal_2", default=0.0),
        }
        translated.update(remaining)
        return pycbc_waveform_wrapper(
            tc=tc,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            waveform_model=approximant,
            **translated,
        )
