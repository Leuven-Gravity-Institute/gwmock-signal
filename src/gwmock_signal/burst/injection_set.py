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
"""Draw a burst *test* injection set: fixed rate, isotropic sky, log-uniform hrss.

This is a sensitivity test set, not an astrophysical population -- the rate is chosen to measure a
search's efficiency, not drawn from any source rate -- and every event it returns says so in its
``injection_set`` key. The specification it implements is in the user guide, *Burst injections*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gwmock_signal.burst.waveforms import GAUSSIAN, SINE_GAUSSIAN, WHITE_NOISE_BURST

#: Value of every drawn event's ``injection_set`` key.
TEST_SET_LABEL = "test"

#: The draft families: the ad hoc morphologies of the LVK O3 all-sky short-burst search
#: (Phys. Rev. D 104, 122004, Table I) with the sine-Gaussian central frequency drawn rather than
#: gridded. Sine-Gaussians and white-noise bursts take a drawn inclination (elliptical, uniform in
#: ``cos(iota)``); Gaussians are linearly polarized and take none.
DRAFT_BURST_FAMILIES: tuple[Mapping[str, Any], ...] = (
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 3.0},
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 9.0},
    {"burst_model": SINE_GAUSSIAN, "quality_factor": 100.0},
    {"burst_model": GAUSSIAN, "duration": 1.0e-4},
    {"burst_model": GAUSSIAN, "duration": 2.5e-3},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 100.0, "bandwidth": 100.0, "duration": 0.1},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 250.0, "bandwidth": 100.0, "duration": 0.1},
    {"burst_model": WHITE_NOISE_BURST, "low_frequency": 750.0, "bandwidth": 100.0, "duration": 0.1},
)

#: Families whose polarization depends on the inclination.
_INCLINED = frozenset({SINE_GAUSSIAN, WHITE_NOISE_BURST})
#: Largest seed handed to the white-noise generator (an unsigned 32-bit integer).
_MAX_SEED = 2**32 - 1


def _log_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    """Return the logarithms of *bounds*, after checking they are a positive range."""
    low, high = (float(b) for b in bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(f"{name} must be finite with 0 < low <= high, got {bounds!r}.")
    return math.log(low), math.log(high)


def draw_burst_injection_set(  # noqa: PLR0913
    *,
    start_time: float,
    end_time: float,
    interval: float,
    hrss_range: tuple[float, float],
    rng: np.random.Generator,
    families: Sequence[Mapping[str, Any]] = DRAFT_BURST_FAMILIES,
    sine_gaussian_frequency_range: tuple[float, float] = (30.0, 1800.0),
) -> list[dict[str, Any]]:
    """Return one burst event every *interval* seconds in ``[start_time, end_time)``.

    Events sit at ``start_time + (k + 1/2) * interval``, so none straddles either end. Each event
    takes a family chosen uniformly from *families*, then draws whatever the family does not fix:

    * sky position isotropic, polarization angle uniform in ``[0, pi)``;
    * ``hrss`` log-uniform in *hrss_range*;
    * ``inclination`` uniform in ``cos(iota)`` for sine-Gaussians and white-noise bursts;
    * ``frequency`` log-uniform in *sine_gaussian_frequency_range* for sine-Gaussians;
    * ``seed`` for white-noise bursts, so each event's noise realisation is reproducible.

    Args:
        start_time: GPS start of the span, s.
        end_time: GPS end of the span, s.
        interval: Seconds between consecutive events; the fixed test-set rate is ``1 / interval``.
        hrss_range: ``(low, high)`` bounds of the log-uniform hrss distribution. No default:
            the range must straddle the detection threshold of the network being tested.
        rng: Source of randomness; pass a seeded generator for a reproducible set.
        families: Family templates, each a mapping with ``burst_model`` and any fixed parameters.
        sine_gaussian_frequency_range: ``(low, high)`` Hz for sine-Gaussian central frequencies.

    Returns:
        One event mapping per injection, in time order, each ready for
        :meth:`~gwmock_signal.burst.BurstSimulator.simulate` and carrying
        ``injection_set="test"``.

    Raises:
        ValueError: If the span or interval is not positive, a range is not positive, or
            *families* is empty or has an entry without ``burst_model``.
    """
    if not (math.isfinite(start_time) and math.isfinite(end_time) and end_time > start_time):
        raise ValueError(f"end_time must be after start_time, got [{start_time}, {end_time}).")
    if not (math.isfinite(interval) and interval > 0):
        raise ValueError(f"interval must be finite and positive, got {interval!r}.")
    if not families:
        raise ValueError("families must not be empty.")
    if any("burst_model" not in family for family in families):
        raise ValueError("every family must name its burst_model.")
    log_hrss = _log_bounds(hrss_range, "hrss_range")
    log_frequency = _log_bounds(sine_gaussian_frequency_range, "sine_gaussian_frequency_range")

    count = math.floor((end_time - start_time) / interval)
    events = []
    for k in range(count):
        family = dict(families[rng.integers(len(families))])
        model = family["burst_model"]
        event: dict[str, Any] = {
            "peak_time": start_time + (k + 0.5) * interval,
            "right_ascension": rng.uniform(0.0, 2 * math.pi),
            "declination": math.asin(rng.uniform(-1.0, 1.0)),
            "polarization_angle": rng.uniform(0.0, math.pi),
            "hrss": math.exp(rng.uniform(*log_hrss)),
        }
        if model in _INCLINED:
            event["inclination"] = math.acos(rng.uniform(-1.0, 1.0))
        if model == SINE_GAUSSIAN:
            event["frequency"] = math.exp(rng.uniform(*log_frequency))
        if model == WHITE_NOISE_BURST:
            event["seed"] = int(rng.integers(_MAX_SEED + 1))
        event.update(family)
        event["injection_set"] = TEST_SET_LABEL
        events.append(event)
    return events
