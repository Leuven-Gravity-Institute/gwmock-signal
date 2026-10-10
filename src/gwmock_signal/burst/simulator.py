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
"""Simulator for unmodelled gravitational-wave bursts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.burst.waveforms import PARAMETRIC_MODELS, read_waveform_file
from gwmock_signal.simulator import TransientSimulator

#: Keys every burst event carries, whatever its model. The model-specific ones are checked when the
#: model is generated, because they differ between families.
_REQUIRED: frozenset[str] = frozenset(
    {"burst_model", "peak_time", "right_ascension", "declination", "polarization_angle"}
)


class BurstSimulator(TransientSimulator):
    """Burst injections: the ad hoc LVK families and numerical waveforms read from file.

    Each event names its morphology in ``burst_model``, so one simulator serves a mixed injection
    set. The event keys follow the per-event record of established burst injection pipelines --
    peak time, distance, sky position, polarization angle, inclination, model -- under this
    package's canonical names:

    ========================  ====================================================================
    ``burst_model``           ``"sine_gaussian"``, ``"gaussian"``, ``"white_noise_burst"``, or a
                              name given in ``waveform_files``
    ``peak_time``             GPS time of the waveform's centre at the geocentre, s
    ``right_ascension``       rad
    ``declination``           rad
    ``polarization_angle``    rad
    ``hrss``                  ad hoc families: ``sqrt(int (h_+^2 + h_x^2) dt)`` at the Earth
    ``distance``              file waveforms: source distance, Mpc
    ========================  ====================================================================

    plus the family's own shape parameters, listed on each function in
    :mod:`gwmock_signal.burst.waveforms`. Keys a model does not use are ignored, so an event may
    carry labels such as which injection set it belongs to.

    A file waveform is scaled by ``reference_distance / distance`` and otherwise used as written:
    the file holds ``h_+`` and ``h_x`` for one viewing direction, and this class does not apply an
    inclination to it.

    Args:
        waveform_files: Model name -> text file of ``t h_+ h_x`` columns (see
            :func:`~gwmock_signal.burst.waveforms.read_waveform_file`). Each file is read once, on
            first use.
        reference_distance: Distance, in Mpc, at which the files' strains are given. Required
            when ``waveform_files`` is not empty, and shared by all of them.

    Raises:
        ValueError: If a file model reuses an ad hoc family's name, or ``reference_distance`` is
            missing or not finite and positive while files are given.
    """

    def __init__(
        self,
        *,
        waveform_files: Mapping[str, str | Path] | None = None,
        reference_distance: float | None = None,
    ) -> None:
        """Initialise with the optional numerical waveform files."""
        files = {str(name): Path(path) for name, path in (waveform_files or {}).items()}
        shadowed = sorted(set(files) & set(PARAMETRIC_MODELS))
        if shadowed:
            raise ValueError(f"waveform_files may not reuse the ad hoc model names {shadowed}.")
        if files and (reference_distance is None or not math.isfinite(reference_distance) or reference_distance <= 0):
            raise ValueError("reference_distance must be finite and positive (Mpc) when waveform_files are given.")
        self._waveform_files = files
        self.reference_distance = reference_distance
        self._file_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def required_params(self) -> frozenset[str]:
        """Return the keys every burst event must carry."""
        return _REQUIRED

    @property
    def burst_models(self) -> tuple[str, ...]:
        """Return every ``burst_model`` this instance can generate."""
        return (*PARAMETRIC_MODELS, *self._waveform_files)

    def generate_polarizations(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> tuple[TimeSeries, TimeSeries]:
        """Return ``(h_plus, h_cross)`` for one burst event, centred on its ``peak_time``.

        Args:
            params: One burst event; see the class docstring.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Unused. A burst's band is set by its own parameters, and cutting
                it here would change the hrss the event records.

        Returns:
            Tuple of ``(hp, hc)`` GWpy ``TimeSeries`` objects.

        Raises:
            ValueError: If ``burst_model`` is unknown or the event lacks one of its parameters.
        """
        model = params["burst_model"]
        if model in PARAMETRIC_MODELS:
            return PARAMETRIC_MODELS[model](params, sampling_frequency)
        if model in self._waveform_files:
            return self._file_polarizations(model, params, sampling_frequency)
        raise ValueError(f"Unknown burst_model {model!r}; this simulator knows {list(self.burst_models)}.")

    def _file_polarizations(
        self, model: str, params: Mapping[str, Any], sampling_frequency: float
    ) -> tuple[TimeSeries, TimeSeries]:
        """Return a file waveform placed at ``peak_time``, scaled to ``distance``, at the target rate."""
        if "distance" not in params:
            raise ValueError(f"burst_model {model!r} needs parameters ['distance'].")
        distance = float(params["distance"])
        if not (math.isfinite(distance) and distance > 0):
            raise ValueError(f"distance must be finite and positive, got {params['distance']!r}.")
        if model not in self._file_cache:
            self._file_cache[model] = read_waveform_file(self._waveform_files[model])
        times, plus, cross = self._file_cache[model]

        scale = self.reference_distance / distance
        t0 = float(params["peak_time"]) + times[0]
        dt = times[1] - times[0]
        series = [
            TimeSeries(scale * strain, t0=t0, dt=dt, name=name) for strain, name in ((plus, "plus"), (cross, "cross"))
        ]
        if not math.isclose(1.0 / dt, sampling_frequency, rel_tol=1e-9):
            series = [s.resample(sampling_frequency) for s in series]
        return series[0], series[1]
