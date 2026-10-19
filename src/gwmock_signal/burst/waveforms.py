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
"""Burst polarizations: the ad hoc families and numerical waveforms read from file.

The ad hoc families -- sine-Gaussian, Gaussian and band-limited white-noise burst -- are generated
by LALSimulation's own burst routines, the ones the LIGO-Virgo-KAGRA burst injections use, so this
module encodes no second opinion about their shapes. It adds three things LALSimulation leaves to
its caller:

* **Placement.** LALSimulation centres every burst on ``t = 0``; it is moved to ``peak_time``.
* **Inclination.** The sine-Gaussian and white-noise generators take a polarisation-ellipse
  eccentricity. A source with inclination ``iota`` has ``h_x / h_+ = 2 cos(iota) / (1 + cos^2 iota)``,
  which is eccentricity ``e = sin^2(iota) / (1 + cos^2 iota)``; the sign of ``cos(iota)`` (the
  handedness, which ``e`` cannot carry) is applied to the cross polarization.
* **Discrete hrss.** Every family is scaled so that its samples carry the requested
  ``hrss = sqrt(sum (h_+^2 + h_x^2) dt)``. For a well-sampled sine-Gaussian or Gaussian this
  changes nothing measurable (about 1e-16); it corrects a pulse the samples do not resolve (a
  0.1 ms Gaussian at 8192 Hz by 1.3e-3) and the white-noise burst, which LALSimulation normalises
  by ``int hdot^2 dt``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_signal import _lalsuite
from gwmock_signal.timeseries import TimeSeries

#: What a missing ``lalsuite`` stops, as named in the error.
_LAL_FEATURE = "The burst waveforms"

#: Ad hoc model names, as ``burst_model`` takes them.
SINE_GAUSSIAN = "sine_gaussian"
GAUSSIAN = "gaussian"
WHITE_NOISE_BURST = "white_noise_burst"

#: LALSimulation refuses a white-noise burst whose time-frequency area is below this.
_MIN_WNB_TIME_BANDWIDTH = 2.0 / math.pi
#: Columns of a waveform file: time relative to ``peak_time``, then ``h_+`` and ``h_x``.
_WAVEFORM_FILE_COLUMNS = 3
#: Relative spread of the sample spacing a waveform file may have and still count as uniform.
_UNIFORM_SPACING_RTOL = 1e-6


def _require(params: Mapping[str, Any], keys: tuple[str, ...], model: str) -> None:
    """Raise ``ValueError`` naming every key of *keys* that *params* lacks for *model*."""
    missing = [key for key in keys if key not in params]
    if missing:
        raise ValueError(f"burst_model {model!r} needs parameters {sorted(missing)}.")


def _positive(params: Mapping[str, Any], key: str) -> float:
    """Return ``params[key]`` as a float, refusing anything not finite and positive."""
    value = float(params[key])
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{key} must be finite and positive, got {params[key]!r}.")
    return value


def _below_nyquist(frequency: float, sampling_frequency: float, key: str) -> None:
    """Refuse a frequency the sample rate cannot represent."""
    if frequency >= sampling_frequency / 2:
        raise ValueError(
            f"{key}={frequency} Hz is not below the Nyquist frequency {sampling_frequency / 2} Hz "
            f"of sampling_frequency={sampling_frequency} Hz."
        )


def _eccentricity_and_handedness(inclination: float) -> tuple[float, float]:
    """Return the LAL ellipse eccentricity for *inclination* and the sign of the cross term."""
    if not math.isfinite(inclination):
        raise ValueError(f"inclination must be finite, got {inclination!r}.")
    cos_iota = math.cos(inclination)
    eccentricity = (1.0 - cos_iota**2) / (1.0 + cos_iota**2)
    return eccentricity, (-1.0 if cos_iota < 0 else 1.0)


def _to_timeseries(hp: Any, hc: Any, peak_time: float, cross_sign: float = 1.0) -> tuple[TimeSeries, TimeSeries]:
    """Turn LAL polarizations centred on ``t = 0`` into series centred on *peak_time*."""
    t0 = float(peak_time) + float(hp.epoch)
    rate = 1.0 / hp.deltaT
    plus = TimeSeries(np.array(hp.data.data, dtype=float), t0=t0, sample_rate=rate, name="plus")
    cross = TimeSeries(cross_sign * np.array(hc.data.data, dtype=float), t0=t0, sample_rate=rate, name="cross")
    return plus, cross


def _at_hrss(hp: Any, hc: Any, hrss: float, peak_time: float, cross_sign: float = 1.0) -> tuple[TimeSeries, TimeSeries]:
    """Scale LAL polarizations so their discrete ``h_rss`` is *hrss*, and place them at *peak_time*.

    LAL normalises the continuous waveform, which the samples reproduce only when they resolve it: a
    0.1 ms Gaussian at 8192 Hz is under one sample wide and overshoots its label by 1.3e-3. Scaling
    by ``hrss / MeasureHrss`` makes the label describe the injected samples for every family.
    """
    lalsimulation = _lalsuite.require("lalsimulation", _LAL_FEATURE)
    scale = hrss / lalsimulation.MeasureHrss(hp, hc)
    plus, cross = _to_timeseries(hp, hc, peak_time, cross_sign)
    return plus.with_data(plus.value * scale), cross.with_data(cross.value * scale)


def sine_gaussian(params: Mapping[str, Any], sampling_frequency: float) -> tuple[TimeSeries, TimeSeries]:
    """Return an elliptically polarized sine-Gaussian centred on ``peak_time``.

    Args:
        params: ``peak_time``, ``hrss``, ``frequency`` (central frequency, Hz),
            ``quality_factor`` and ``inclination`` (rad; ``0`` is circular, ``pi/2`` linear).
        sampling_frequency: Sample rate in Hz.

    Returns:
        ``(h_plus, h_cross)``.
    """
    _require(params, ("peak_time", "hrss", "frequency", "quality_factor", "inclination"), SINE_GAUSSIAN)
    frequency = _positive(params, "frequency")
    _below_nyquist(frequency, sampling_frequency, "frequency")
    eccentricity, cross_sign = _eccentricity_and_handedness(float(params["inclination"]))
    lalsimulation = _lalsuite.require("lalsimulation", _LAL_FEATURE)
    hp, hc = lalsimulation.SimBurstSineGaussian(
        _positive(params, "quality_factor"),
        frequency,
        _positive(params, "hrss"),
        eccentricity,
        0.0,
        1.0 / sampling_frequency,
    )
    return _at_hrss(hp, hc, _positive(params, "hrss"), params["peak_time"], cross_sign)


def gaussian(params: Mapping[str, Any], sampling_frequency: float) -> tuple[TimeSeries, TimeSeries]:
    """Return a linearly polarized Gaussian pulse centred on ``peak_time``.

    The cross polarization is identically zero, as in LALSimulation and the LVK injection sets.

    Args:
        params: ``peak_time``, ``hrss`` and ``duration`` (the standard deviation of ``h_+(t)``, s).
        sampling_frequency: Sample rate in Hz.

    Returns:
        ``(h_plus, h_cross)``.
    """
    _require(params, ("peak_time", "hrss", "duration"), GAUSSIAN)
    lalsimulation = _lalsuite.require("lalsimulation", _LAL_FEATURE)
    hp, hc = lalsimulation.SimBurstGaussian(
        _positive(params, "duration"),
        _positive(params, "hrss"),
        1.0 / sampling_frequency,
    )
    return _at_hrss(hp, hc, _positive(params, "hrss"), params["peak_time"])


def white_noise_burst(params: Mapping[str, Any], sampling_frequency: float) -> tuple[TimeSeries, TimeSeries]:
    """Return a band- and time-limited white-noise burst centred on ``peak_time``.

    The band is set by its lower edge and width. LALSimulation
    shapes both the band and the time envelope smoothly rather than with hard edges, so the content
    is concentrated in, not confined to, ``[low_frequency, low_frequency + bandwidth]``.

    Args:
        params: ``peak_time``, ``hrss``, ``low_frequency`` (Hz), ``bandwidth`` (Hz), ``duration``
            (s), ``inclination`` (rad) and ``seed`` -- an integer that fixes the noise realisation,
            so the same parameters always give the same waveform.
        sampling_frequency: Sample rate in Hz.

    Returns:
        ``(h_plus, h_cross)``.
    """
    _require(
        params,
        ("peak_time", "hrss", "low_frequency", "bandwidth", "duration", "inclination", "seed"),
        WHITE_NOISE_BURST,
    )
    bandwidth = _positive(params, "bandwidth")
    duration = _positive(params, "duration")
    centre = _positive(params, "low_frequency") + bandwidth / 2
    _below_nyquist(centre + bandwidth / 2, sampling_frequency, "low_frequency + bandwidth")
    if duration * bandwidth < _MIN_WNB_TIME_BANDWIDTH:
        raise ValueError(
            f"duration * bandwidth = {duration * bandwidth:g} is below 2/pi; "
            "a white-noise burst cannot be that compact in both time and frequency."
        )
    seed = int(params["seed"])
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {params['seed']!r}.")
    eccentricity, cross_sign = _eccentricity_and_handedness(float(params["inclination"]))
    lalsimulation = _lalsuite.require("lalsimulation", _LAL_FEATURE)
    lal = _lalsuite.require("lal", _LAL_FEATURE)
    hp, hc = lalsimulation.GenerateBandAndTimeLimitedWhiteNoiseBurst(
        duration, centre, bandwidth, eccentricity, 0.0, 1.0, 1.0 / sampling_frequency, lal.gsl_rng("ranlux", seed)
    )
    return _at_hrss(hp, hc, _positive(params, "hrss"), params["peak_time"], cross_sign)


def read_waveform_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a numerical burst waveform: three whitespace-separated columns ``t h_+ h_x``.

    ``t`` is in seconds relative to the waveform's reference instant (placed at ``peak_time``), and
    must be uniformly sampled; the strains are at the file's ``reference_distance``.

    Args:
        path: Text file to read.

    Returns:
        ``(times, h_plus, h_cross)``.

    Raises:
        ValueError: If the file does not have three columns, has fewer than two rows, holds a
            non-finite value, or is not uniformly sampled in increasing time.
    """
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != _WAVEFORM_FILE_COLUMNS or data.shape[0] < 2:  # noqa: PLR2004
        raise ValueError(f"{path}: expected at least two rows of three columns 't h_plus h_cross', got {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: contains non-finite values.")
    times, plus, cross = data.T
    steps = np.diff(times)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=_UNIFORM_SPACING_RTOL, atol=0.0):
        raise ValueError(f"{path}: time column must be uniformly sampled and increasing.")
    return times, plus, cross


#: The ad hoc families, keyed by ``burst_model``.
PARAMETRIC_MODELS = {
    SINE_GAUSSIAN: sine_gaussian,
    GAUSSIAN: gaussian,
    WHITE_NOISE_BURST: white_noise_burst,
}
