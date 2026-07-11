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
"""Waveform backend built on the LVK ``gwsignal`` interface.

``lalsimulation.gwsignal`` is the collaboration's forward-looking waveform
API, shipped inside the ``lalsuite`` wheel this package already depends on.
This backend evaluates frequency-domain-native approximants through
``GenerateFDWaveform`` and shares all segment sizing, phase-reference
handling, and frequency-to-time conditioning with
:class:`~gwmock_signal.waveform.backends.lal.LALSimulationBackend`, so both
backends place the same source at the same sample.

Two deliberate scope limits:

- Time-domain-native LAL approximants fall back to the parent's
  ``SimInspiralFD`` path: gwsignal's FD output hardcodes ``epoch=0`` (see
  ``to_gwpy_Series`` in ``gwsignal.core.waveform``), discarding the
  conditioning epoch that the ``dt = 1/deltaF + epoch`` phase-reference
  correction requires. The waveform data are identical either way; only the
  epoch metadata differs, and the parent path retains it.
- External Python models (e.g. SEOBNRv5 via ``pyseobnr``) are not exposed
  yet: their FD conditioning goes through gwsignal's own (unreviewed)
  routines with the same epoch loss, so their coalescence placement cannot
  currently be reconciled with this package's convention. Support requires
  conditioning their time-domain output directly.

Masses and distances are converted with LAL's SI constants (``lal.MSUN_SI``,
``lal.PC_SI``) rather than astropy's, which keeps the output bit-identical
to the direct LALSimulation calls (astropy's solar mass differs at the
1e-7 level).
"""

from __future__ import annotations

import importlib

import lalsimulation
import numpy as np

from gwmock_signal.waveform.backends.lal import (
    MPC,
    MSUN,
    LALSimulationBackend,
    _FrequencyGrid,
    _ResolvedParameters,
)


class GWSignalBackend(LALSimulationBackend):
    """Time-domain waveform backend implemented with ``lalsimulation.gwsignal``.

    Accepts the same constructor arguments, canonical parameters, and
    approximant names as :class:`LALSimulationBackend`, and produces
    numerically identical waveforms; it differs only in routing FD-native
    approximants through the gwsignal interface.
    """

    def _evaluate_fd(
        self, approximant: str, p: _ResolvedParameters, grid: _FrequencyGrid
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate FD-native approximants via gwsignal; defer the rest to LAL."""
        approx_enum = lalsimulation.GetApproximantFromString(approximant)
        if not lalsimulation.SimInspiralImplementedFDApproximants(approx_enum):
            # gwsignal's FD path discards the conditioning epoch needed to
            # re-reference TD-native approximants; the parent keeps it.
            return super()._evaluate_fd(approximant, p, grid)

        # Deferred: importing astropy and gwsignal is not free, and the LAL
        # fallback above must work even if they were somehow unavailable.
        u = importlib.import_module("astropy.units")
        gws_waveform = importlib.import_module("lalsimulation.gwsignal.core.waveform")
        gws_models = importlib.import_module("lalsimulation.gwsignal.models")

        dimensionless = u.dimensionless_unscaled
        parameters = {
            "mass1": p.mass1 * MSUN * u.kg,
            "mass2": p.mass2 * MSUN * u.kg,
            "spin1x": p.spin_1x * dimensionless,
            "spin1y": p.spin_1y * dimensionless,
            "spin1z": p.spin_1z * dimensionless,
            "spin2x": p.spin_2x * dimensionless,
            "spin2y": p.spin_2y * dimensionless,
            "spin2z": p.spin_2z * dimensionless,
            "distance": p.distance * MPC * u.m,
            "inclination": p.inclination * u.rad,
            "phi_ref": p.coa_phase * u.rad,
            "longAscNodes": 0.0 * u.rad,
            "eccentricity": 0.0 * dimensionless,
            "meanPerAno": 0.0 * u.rad,
            "lambda1": p.lambda_1 * dimensionless,
            "lambda2": p.lambda_2 * dimensionless,
            "deltaF": grid.delta_f * u.Hz,
            "f22_start": grid.minimum_frequency * u.Hz,
            "f22_ref": grid.f_ref * u.Hz,
            "f_max": grid.f_max * u.Hz,
            # FD-native evaluation only: no time-domain conditioning, so the
            # output is already referenced to the FD phase (epoch shift 0).
            "condition": 0,
        }
        generator = gws_models.gwsignal_get_waveform_generator(approximant)
        hp, hc = gws_waveform.GenerateFDWaveform(parameters, generator)
        return np.asarray(hp.value), np.asarray(hc.value), 0.0
