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
"""Batched, on-device CBC simulation for catalogue-scale generation.

Generates a whole catalogue of compact-binary signals on device: ripple
frequency-domain waveforms under ``jax.vmap`` (one shared, worst-case grid), then
the JAX antenna pattern + geocenter delay + inverse FFT per event and detector.
The result is raw strain arrays plus timing metadata; injecting those signals into
fixed-duration data-segment files (including signals spanning several segments) is
a separate assembly step.

Requires the optional ``[jax]`` extra (via :class:`RippleBackend`). JAX is imported
lazily so the package still imports without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gwmock_signal.projection.geometry import reconstructed_geometry
from gwmock_signal.projection.jax_projection import (
    antenna_pattern,
    gmst_rad,
    project_polarizations_fd,
    time_delay_from_geocenter,
)
from gwmock_signal.waveform.backends.ripple import RippleBackend

if TYPE_CHECKING:
    from jax import Array


@dataclass(frozen=True)
class BatchedDetectorStrain:
    """Catalogue-scale detector strain as raw arrays plus timing metadata.

    ``strain`` has shape ``(n_events, n_detectors, n_samples)`` and is a JAX array
    (on device). Each event/detector row is a time series with sample spacing
    ``1 / sampling_frequency`` whose first sample is at GPS time
    ``epoch + coa_time[event]`` (coalescence sits ``-epoch`` seconds from the start,
    near the segment end). The signals are not yet placed on a shared timeline or
    segmented into files — that assembly step injects them into fixed-duration data
    segments and is handled separately.
    """

    strain: Array
    detector_names: tuple[str, ...]
    coa_time: np.ndarray
    epoch: float
    sampling_frequency: float


def simulate_cbc_batch(  # noqa: PLR0913
    approximant: str,
    detector_names: Sequence[str],
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    parameters: Mapping[str, object],
    backend: RippleBackend | None = None,
) -> BatchedDetectorStrain:
    """Simulate a catalogue of CBC signals on device, one strain per event and detector.

    Evaluates ripple frequency-domain waveforms for the whole catalogue under
    ``jax.vmap`` (a single grid sized worst-case for the longest inspiral), then
    projects each event onto each detector with the JAX antenna pattern and
    geocenter delay and inverse-FFTs to strain. The antenna pattern and delay are
    evaluated once per event at the **segment midpoint** (``earth_rotation=False``),
    matching :func:`gwmock_signal.projection.network.project_polarizations_to_network`.

    Args:
        approximant: A supported ripple approximant name.
        detector_names: Detector codes (e.g. ``"H1"``, ``"L1"``).
        sampling_frequency: Sample rate in Hz.
        minimum_frequency: Low-frequency cutoff in Hz.
        parameters: Mapping of **canonical** gwmock-pop parameter names (no aliases)
            to equal-length 1-D arrays. In addition to the waveform parameters
            (masses, spins, distance, inclination, coa_phase) this must include
            ``right_ascension``, ``declination``, ``polarization_angle`` and
            ``coa_time``.
        backend: Optional configured :class:`RippleBackend` (e.g. with a fixed
            ``segment_duration`` or ``f_ref``). Defaults to ``RippleBackend()``.

    Returns:
        A :class:`BatchedDetectorStrain` with the ``(n_events, n_detectors,
        n_samples)`` strain and per-event timing metadata.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import
    import jax.numpy as jnp  # noqa: PLC0415

    backend = backend or RippleBackend()
    fd = backend.generate_fd_polarizations_batch(
        approximant,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        parameters=parameters,
    )
    n_samples = fd.n_samples
    dt = 1.0 / sampling_frequency
    merger_index, epoch = backend.coalescence_placement(n_samples, sampling_frequency)

    right_ascension = jnp.asarray(_required(parameters, "right_ascension"), dtype=jnp.float64)
    declination = jnp.asarray(_required(parameters, "declination"), dtype=jnp.float64)
    polarization_angle = jnp.asarray(_required(parameters, "polarization_angle"), dtype=jnp.float64)
    coa_time = np.asarray(_required(parameters, "coa_time"), dtype=float)

    # earth_rotation=False reference time: the midpoint of each event's placed segment.
    midpoint_offset = epoch + 0.5 * (n_samples - 1) * dt
    gmst = gmst_rad(jnp.asarray(coa_time, dtype=jnp.float64) + midpoint_offset)

    def _project_event(plus: Array, cross: Array, f_plus: Array, f_cross: Array, time_delay: Array) -> Array:
        strain = project_polarizations_fd(
            fd.frequencies,
            plus,
            cross,
            f_plus=f_plus,
            f_cross=f_cross,
            time_delay=time_delay,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
        )
        return jnp.roll(strain, merger_index)  # place coalescence near the segment end

    per_detector = []
    for name in detector_names:
        response, location = reconstructed_geometry(name)
        f_plus, f_cross = antenna_pattern(
            response,
            gmst,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )
        time_delay = time_delay_from_geocenter(location, gmst, right_ascension=right_ascension, declination=declination)
        per_detector.append(jax.vmap(_project_event)(fd.plus, fd.cross, f_plus, f_cross, time_delay))

    strain = jnp.stack(per_detector, axis=1)  # (n_events, n_detectors, n_samples)
    return BatchedDetectorStrain(
        strain=strain,
        detector_names=tuple(detector_names),
        coa_time=coa_time,
        epoch=epoch,
        sampling_frequency=sampling_frequency,
    )


def _required(parameters: Mapping[str, object], name: str) -> object:
    """Return a required batch parameter or raise a clear error."""
    if name not in parameters:
        raise ValueError(f"Missing required batch parameter: {name!r}")
    return parameters[name]
