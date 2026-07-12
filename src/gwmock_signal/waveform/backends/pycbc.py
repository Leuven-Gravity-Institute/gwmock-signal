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
from typing import Final

from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

_PYCBC_IMPORT_ERROR = "pycbc is not installed. Run: pip install 'gwmock-signal[pycbc]'"

#: PyCBC ``get_td_waveform`` keyword names this backend derives from canonical
#: parameters or manages itself. They must not be set through
#: ``waveform_arguments`` (they would silently override the translated value or
#: the wrapper-controlled grid), so passing one there is rejected.
_RESERVED_WAVEFORM_ARGUMENTS: Final[frozenset[str]] = frozenset(
    {
        "mass1",
        "mass2",
        "distance",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "inclination",
        "coa_phase",
        "lambda1",
        "lambda2",
        "approximant",
        "delta_t",
        "f_lower",
    }
)


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

    @staticmethod
    def _resolve_waveform_arguments(value: object) -> dict[str, object]:
        """Validate the optional extra-argument mapping forwarded to PyCBC.

        Keys are approximant-specific ``get_td_waveform`` options (e.g.
        ``mode_array``, ``f_ref``, ``numerical_relativity_file``). Reserved keys
        that this backend derives from canonical parameters or manages itself are
        rejected so extras cannot silently override them.
        """
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("waveform_arguments must be a dict with string keys")
        reserved = sorted(key for key in value if key in _RESERVED_WAVEFORM_ARGUMENTS)
        if reserved:
            joined = ", ".join(reserved)
            raise ValueError(f"Pass these as canonical parameters, not waveform_arguments: {joined}")
        return dict(value)

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations through ``pycbc_waveform_wrapper``.

        Canonical CBC parameters (masses, spins, distance, orientation, tidal
        deformabilities) are translated to PyCBC's native names. Any other
        ``get_td_waveform`` option must be passed inside a ``waveform_arguments``
        mapping; unrecognised top-level parameters are rejected.
        """
        pycbc_waveform_wrapper = importlib.import_module("gwmock_signal.waveform.pycbc_wrapper").pycbc_waveform_wrapper
        remaining = dict(params)
        waveform_arguments = self._resolve_waveform_arguments(_pop_alias(remaining, "waveform_arguments", default={}))
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
        if remaining:
            extras = ", ".join(sorted(remaining))
            raise ValueError(f"Unsupported PyCBC waveform parameters: {extras}")
        translated.update(waveform_arguments)
        return pycbc_waveform_wrapper(
            tc=tc,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            waveform_model=approximant,
            **translated,
        )
