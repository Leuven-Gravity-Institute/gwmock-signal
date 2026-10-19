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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gwmock_data.resampling import resample_uniform_sinc

from gwmock_signal.burst.waveforms import PARAMETRIC_MODELS, read_waveform_file
from gwmock_signal.simulator import TransientSimulator
from gwmock_signal.timeseries import TimeSeries

#: Keys every burst event carries, whatever its model. The model-specific ones are checked when the
#: model is generated, because they differ between families.
_REQUIRED: frozenset[str] = frozenset(
    {"burst_model", "peak_time", "right_ascension", "declination", "polarization_angle"}
)

InclinationDependence = Literal["none", "axisymmetric", "rotating"]


#: Largest fraction of a waveform file's power allowed above the new Nyquist frequency when it is
#: downsampled, since the shared kernel does not low-pass.
_ALIAS_TOLERANCE = 1e-12


def _resample(values: np.ndarray, rate: float, new_rate: float) -> np.ndarray:
    """Resample a waveform file's samples with the shared band-limited kernel.

    ``resample_uniform_sinc`` interpolates; it does not low-pass. A file being
    downsampled must therefore already be band-limited below the new Nyquist
    frequency, and one that is not is refused rather than aliased.

    Args:
        values: Samples at ``rate``.
        rate: Sample rate of ``values`` in Hz.
        new_rate: Sample rate to return, in Hz.

    Returns:
        The samples at ``new_rate`` over the same span.

    Raises:
        ValueError: If downsampling and more than ``_ALIAS_TOLERANCE`` of the
            waveform's power lies above the new Nyquist frequency.
    """
    if new_rate < rate:
        # A Hann window keeps the periodogram's leakage (about 1e-5 for an untapered tone) far below
        # the tolerance, so only content that is really above the new Nyquist frequency is counted.
        power = np.abs(np.fft.rfft(values * np.hanning(values.shape[0]))) ** 2
        above = power[np.fft.rfftfreq(values.shape[0], 1.0 / rate) > new_rate / 2.0].sum()
        if above > _ALIAS_TOLERANCE * power.sum():
            raise ValueError(
                f"The waveform file carries {above / power.sum():.2e} of its power above {new_rate / 2.0:g} Hz, "
                f"the Nyquist frequency of {new_rate:g} Hz; low-pass it before resampling."
            )
    n_new = max(1, int(np.floor((values.shape[0] - 1) * new_rate / rate)) + 1)
    return resample_uniform_sinc(values, np.arange(n_new) * (rate / new_rate))


@dataclass(frozen=True)
class WaveformFile:
    """A numerical burst waveform on disk and how to place it.

    Attributes:
        path: Text file of ``t h_+ h_x`` columns (see
            :func:`~gwmock_signal.burst.waveforms.read_waveform_file`).
        reference_distance: Distance, in Mpc, at which the file's strains are given.
        inclination_dependence: How the strain varies with the event's ``inclination``:

            * ``"none"`` -- used as written, whatever the inclination;
            * ``"axisymmetric"`` -- an axisymmetric (``m = 0``) quadrupole, given edge-on:
              ``h_+`` is scaled by ``sin^2(iota)`` and ``h_x`` is zero;
            * ``"rotating"`` -- a rotating (``m = +-2``) quadrupole, given face-on: ``h_+`` is
              scaled by ``(1 + cos^2 iota) / 2`` and ``h_x`` by ``cos(iota)``.

    Raises:
        ValueError: If ``reference_distance`` is not finite and positive, or
            ``inclination_dependence`` is not one of the three above.
    """

    path: str | Path
    reference_distance: float
    inclination_dependence: InclinationDependence = "none"

    def __post_init__(self) -> None:
        """Refuse a distance or an inclination dependence the simulator could not apply."""
        if not (math.isfinite(self.reference_distance) and self.reference_distance > 0):
            raise ValueError(f"reference_distance must be finite and positive (Mpc), got {self.reference_distance!r}.")
        if self.inclination_dependence not in _INCLINATION_FACTORS:
            raise ValueError(
                f"inclination_dependence must be one of {sorted(_INCLINATION_FACTORS)}, "
                f"got {self.inclination_dependence!r}."
            )


#: ``(h_+ factor, h_x factor)`` as functions of the inclination, for each inclination dependence.
_INCLINATION_FACTORS = {
    "none": lambda iota: (1.0, 1.0),
    "axisymmetric": lambda iota: (math.sin(iota) ** 2, 0.0),
    "rotating": lambda iota: ((1.0 + math.cos(iota) ** 2) / 2.0, math.cos(iota)),
}


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
    ``inclination``           rad; ad hoc families that take it, and file waveforms whose
                              ``inclination_dependence`` is not ``"none"``
    ========================  ====================================================================

    plus the family's own shape parameters, listed on each function in
    :mod:`gwmock_signal.burst.waveforms`. Keys a model does not use are ignored, so an event may
    carry labels such as which injection set it belongs to.

    A file waveform is scaled by its ``reference_distance / distance`` and by the inclination
    factors of its :class:`WaveformFile`.

    Args:
        waveform_files: Model name -> :class:`WaveformFile`. Each file is read once, on first use.

    Raises:
        ValueError: If a file model reuses an ad hoc family's name, or a value is not a
            :class:`WaveformFile`.
    """

    def __init__(self, *, waveform_files: Mapping[str, WaveformFile] | None = None) -> None:
        """Initialise with the optional numerical waveform files."""
        files = dict(waveform_files or {})
        shadowed = sorted(set(files) & set(PARAMETRIC_MODELS))
        if shadowed:
            raise ValueError(f"waveform_files may not reuse the ad hoc model names {shadowed}.")
        wrong = sorted(name for name, entry in files.items() if not isinstance(entry, WaveformFile))
        if wrong:
            raise ValueError(f"waveform_files values must be WaveformFile instances; {wrong} are not.")
        self._waveform_files: dict[str, WaveformFile] = files
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
            Tuple of ``(hp, hc)`` series.

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
        entry = self._waveform_files[model]
        needed = ["distance"] + ([] if entry.inclination_dependence == "none" else ["inclination"])
        missing = [key for key in needed if key not in params]
        if missing:
            raise ValueError(f"burst_model {model!r} needs parameters {missing}.")
        distance = float(params["distance"])
        if not (math.isfinite(distance) and distance > 0):
            raise ValueError(f"distance must be finite and positive, got {params['distance']!r}.")
        inclination = float(params.get("inclination", 0.0))
        if not math.isfinite(inclination):
            raise ValueError(f"inclination must be finite, got {params['inclination']!r}.")
        if model not in self._file_cache:
            self._file_cache[model] = read_waveform_file(entry.path)
        times, plus, cross = self._file_cache[model]

        scale = entry.reference_distance / distance
        plus_factor, cross_factor = _INCLINATION_FACTORS[entry.inclination_dependence](inclination)
        t0 = float(params["peak_time"]) + times[0]
        file_rate = 1.0 / (times[1] - times[0])
        resample = not math.isclose(file_rate, sampling_frequency, rel_tol=1e-9)
        series = []
        for strain, factor, name in ((plus, plus_factor, "plus"), (cross, cross_factor, "cross")):
            values = scale * factor * np.asarray(strain, dtype=float)
            if resample:
                values = _resample(values, file_rate, sampling_frequency)
            rate = sampling_frequency if resample else file_rate
            series.append(TimeSeries(values, t0=t0, sample_rate=rate, name=name))
        return series[0], series[1]
