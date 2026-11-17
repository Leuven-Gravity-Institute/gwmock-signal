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
"""Draw a burst population of standard candles: Poisson in time, uniform in volume.

Every source radiates the same energy ``E_GW`` in gravitational waves. Sources occur at a constant
rate per unit volume and time, so their number over a span is Poisson, their times are uniform,
and their distances are uniform in Euclidean volume out to ``max_distance``. Each event's ``hrss``
follows from its energy, distance and central frequency through the narrow-band relation of the
LVK all-sky short-burst searches (O3: Phys. Rev. D 104, 122004, Eq. (2)):

    E_GW^iso = (pi^2 c^3 / G) r^2 f0^2 hrss^2.

A white-noise burst is taken to radiate isotropically. An elliptical sine-Gaussian is a rotating
source, ``E_GW^rot = (2/5) E_GW^iso`` evaluated at its face-on hrss (ibid., Sec. IV), and its
observed hrss follows the rotating pattern ``((1 + cos^2 i) / 2)^2 + cos^2 i``. Gaussians are
broadband, the relation does not hold for them, and they are refused.

This is the population the O3 search's rate-density upper limits describe: a rate density at, or
below, a limit for a stated energy is a population consistent with O3. The specification is in
the user guide, *Burst injections*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gwmock_signal.burst.waveforms import SINE_GAUSSIAN, WHITE_NOISE_BURST
from gwmock_signal.constants import MPC_SI, MTSUN_SI

#: Value of every drawn event's ``injection_set`` key.
STANDARD_CANDLE_LABEL = "standard_candle"

#: The elliptical sine-Gaussians and white-noise bursts of the test set, the families the O3
#: search's rate-density upper limits are given for (Sec. IV); its Gaussians are left out.
STANDARD_CANDLE_FAMILIES: tuple[Mapping[str, Any], ...] = (
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 3.0},
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 9.0},
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 100.0},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 100.0, "bandwidth": 100.0, "duration": 0.1},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 250.0, "bandwidth": 100.0, "duration": 0.1},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 700.0, "bandwidth": 100.0, "duration": 0.1},
)

#: Largest ``max_distance``, Mpc. The volume is Euclidean and redshift is ignored; at 100 Mpc
#: (z of about 0.02) the luminosity distance and the volume already differ from Euclidean by a few
#: per cent, so a farther population needs a cosmological draw this function does not make.
MAX_DISTANCE_LIMIT = 100.0

#: The speed of light, m/s: exact, by the SI definition of the metre.
_C_SI = 299_792_458.0
#: Seconds in a Julian year, the year of a rate density in Gpc^-3 yr^-1.
_YEAR_S = 365.25 * 86_400.0
_GPC3_PER_MPC3 = 1.0e-9
#: Largest seed handed to the white-noise generator (an unsigned 32-bit integer).
_MAX_SEED = 2**32 - 1


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return value


def standard_candle_hrss(
    *, energy: float, distance: float, frequency: float, inclination: float, elliptical: bool
) -> float:
    """Return the hrss at the Earth of a standard candle.

    Args:
        energy: Energy radiated in gravitational waves, in solar masses times c^2.
        distance: Distance to the source, Mpc.
        frequency: Central frequency of the burst, Hz.
        inclination: Angle between the source's rotation axis and the line of sight, rad. Used
            only when *elliptical*.
        elliptical: True for a rotating source (an elliptical sine-Gaussian), False for an
            isotropic one (a white-noise burst).

    Returns:
        ``sqrt(G E / (pi^2 c^3)) / (r f0)`` for an isotropic source. For a rotating source, that
        times ``sqrt(5/2)`` face on, and times ``sqrt(5/4) * sqrt(((1 + cos^2 i) / 2)^2 + cos^2 i)``
        at inclination ``i``, so that the average of hrss^2 over orientations is the isotropic
        value and the energy is the one stated.

    Raises:
        ValueError: If *energy*, *distance* or *frequency* is not finite and positive, or
            *inclination* is not finite.
    """
    energy = _positive_finite(energy, "energy")
    distance = _positive_finite(distance, "distance")
    frequency = _positive_finite(frequency, "frequency")
    if not math.isfinite(inclination):
        raise ValueError(f"inclination must be finite, got {inclination!r}.")
    # G E / c^3 = energy * MTSUN_SI * c^2, with E = energy * MSUN * c^2 and MTSUN = G MSUN / c^3.
    isotropic = math.sqrt(energy * MTSUN_SI) * _C_SI / (math.pi * distance * MPC_SI * frequency)
    if not elliptical:
        return isotropic
    cos_iota = math.cos(inclination)
    pattern = ((1.0 + cos_iota**2) / 2.0) ** 2 + cos_iota**2
    return isotropic * math.sqrt(1.25 * pattern)


def _central_frequency(event: Mapping[str, Any]) -> float:
    if event["burst_model"] == SINE_GAUSSIAN:
        return float(event["frequency"])
    return float(event["low_frequency"]) + 0.5 * float(event["bandwidth"])


def draw_standard_candle_population(  # noqa: PLR0913
    *,
    start_time: float,
    end_time: float,
    rate_density: float,
    energy: float,
    max_distance: float,
    rng: np.random.Generator,
    families: Sequence[Mapping[str, Any]] = STANDARD_CANDLE_FAMILIES,
    sine_gaussian_frequency_range: tuple[float, float] = (30.0, 1800.0),
) -> list[dict[str, Any]]:
    """Return a population of standard-candle bursts in ``[start_time, end_time)``.

    The number of events is Poisson with mean ``rate_density * (4/3) pi max_distance^3 * T``, with
    ``T`` the span in Julian years. Each event takes a family chosen uniformly from *families*
    and draws:

    * ``peak_time`` uniform in the span, ``distance`` uniform in volume out to *max_distance*;
    * sky position isotropic, polarization angle uniform in ``[0, pi)``, ``inclination`` uniform
      in ``cos(iota)``;
    * ``frequency`` log-uniform in *sine_gaussian_frequency_range* for sine-Gaussians, and
      ``seed`` for white-noise bursts;

    and its ``hrss`` is :func:`standard_candle_hrss` of *energy*, its distance, its central
    frequency (a white-noise burst's is the middle of its band) and its inclination. A burst
    whose peak lies near either end of the span may run past it.

    Args:
        start_time: GPS start of the span, s.
        end_time: GPS end of the span, s.
        rate_density: Rate per unit volume and time, Gpc^-3 yr^-1, summed over *families*.
        energy: Energy each source radiates, in solar masses times c^2.
        max_distance: Farthest source, Mpc; at most :data:`MAX_DISTANCE_LIMIT`.
        rng: Source of randomness; pass a seeded generator for a reproducible population.
        families: Family templates, each with ``burst_model`` (sine-Gaussian or white-noise
            burst) and its fixed parameters.
        sine_gaussian_frequency_range: ``(low, high)`` Hz for sine-Gaussian central frequencies.

    Returns:
        One event mapping per source, in time order, each ready for
        :meth:`~gwmock_signal.burst.BurstSimulator.simulate`, carrying its ``distance`` (Mpc),
        its ``energy`` and ``injection_set="standard_candle"``.

    Raises:
        ValueError: If the span is empty, a rate, energy, distance or frequency range is not
            positive, *max_distance* exceeds :data:`MAX_DISTANCE_LIMIT`, or *families* is empty
            or holds a family without ``burst_model`` or with a model other than a sine-Gaussian
            or a white-noise burst.
    """
    if not (math.isfinite(start_time) and math.isfinite(end_time) and end_time > start_time):
        raise ValueError(f"end_time must be after start_time, got [{start_time}, {end_time}).")
    rate_density = float(rate_density)
    if not (math.isfinite(rate_density) and rate_density >= 0):
        raise ValueError(f"rate_density must be finite and non-negative, got {rate_density!r}.")
    energy = _positive_finite(energy, "energy")
    max_distance = _positive_finite(max_distance, "max_distance")
    if max_distance > MAX_DISTANCE_LIMIT:
        raise ValueError(
            f"max_distance={max_distance!r} Mpc is beyond {MAX_DISTANCE_LIMIT:g} Mpc, where a Euclidean "
            "volume without redshift no longer describes the population."
        )
    if not families:
        raise ValueError("families must not be empty.")
    if any("burst_model" not in family for family in families):
        raise ValueError("every family must name its burst_model.")
    unsupported = sorted(
        {str(f["burst_model"]) for f in families if f["burst_model"] not in (SINE_GAUSSIAN, WHITE_NOISE_BURST)}
    )
    if unsupported:
        raise ValueError(
            f"families {unsupported} are not supported: the energy-to-hrss relation holds for "
            f"narrow-band {SINE_GAUSSIAN!r} and {WHITE_NOISE_BURST!r} families only."
        )
    low, high = (float(b) for b in sine_gaussian_frequency_range)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(
            f"sine_gaussian_frequency_range must be finite with 0 < low <= high, got {sine_gaussian_frequency_range!r}."
        )
    log_frequency = (math.log(low), math.log(high))

    volume_gpc3 = (4.0 / 3.0) * math.pi * max_distance**3 * _GPC3_PER_MPC3
    mean = rate_density * volume_gpc3 * (end_time - start_time) / _YEAR_S
    count = int(rng.poisson(mean))
    times = np.sort(rng.uniform(start_time, end_time, size=count))

    events = []
    for peak_time in times:
        family = dict(families[rng.integers(len(families))])
        model = family["burst_model"]
        event: dict[str, Any] = {
            "peak_time": float(peak_time),
            "right_ascension": rng.uniform(0.0, 2 * math.pi),
            "declination": math.asin(rng.uniform(-1.0, 1.0)),
            "polarization_angle": rng.uniform(0.0, math.pi),
            "inclination": math.acos(rng.uniform(-1.0, 1.0)),
            # 1 - U is in (0, 1], so no source sits at zero distance.
            "distance": max_distance * (1.0 - rng.uniform(0.0, 1.0)) ** (1.0 / 3.0),
            "energy": energy,
        }
        if model == SINE_GAUSSIAN:
            event["frequency"] = math.exp(rng.uniform(*log_frequency))
        else:
            event["seed"] = int(rng.integers(_MAX_SEED + 1))
        event.update(family)
        event["hrss"] = standard_candle_hrss(
            energy=energy,
            distance=event["distance"],
            frequency=_central_frequency(event),
            inclination=event["inclination"],
            elliptical=model == SINE_GAUSSIAN,
        )
        event["injection_set"] = STANDARD_CANDLE_LABEL
        events.append(event)
    return events
