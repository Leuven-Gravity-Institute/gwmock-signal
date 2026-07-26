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
from gwpy.timeseries import TimeSeries

from gwmock_signal.injection import inject_strains_sequential
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.projection.geometry import reconstructed_geometry
from gwmock_signal.projection.jax_projection import (
    antenna_pattern,
    gmst_rad,
    project_polarizations_fd,
    project_polarizations_td_rotating,
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
    earth_rotation: bool = True,
) -> BatchedDetectorStrain:
    """Simulate a catalogue of CBC signals on device, one strain per event and detector.

    Evaluates ripple frequency-domain waveforms for the whole catalogue under
    ``jax.vmap`` (a single grid sized worst-case for the longest inspiral), then
    projects each event onto each detector with the JAX antenna pattern and geocenter
    delay and inverse-FFTs to strain. The antenna pattern and delay are evaluated per
    sample by default and once per event at the **segment midpoint** when
    ``earth_rotation=False``, matching the two branches of
    :func:`gwmock_signal.projection.network.project_polarizations_to_network`.

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
        earth_rotation: If ``True`` (default, matching
            :func:`~gwmock_signal.projection.network.project_polarizations_to_network`),
            evaluate the antenna pattern and geocenter delay per sample and resample the
            polarizations at the delayed times. If ``False``, evaluate both once at the
            segment midpoint and apply the delay as an exact frequency-domain phase
            shift, which is cheaper but only valid for signals short compared with an
            hour. A binary neutron star in the Einstein Telescope band occupies 2048 s
            at 10 Hz and 16384 s at 5 Hz, over which the detector sweeps tens of
            degrees, so ``False`` is not appropriate for that population.

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

    if earth_rotation:
        strain = _project_rotating(
            fd,
            detector_names,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
            merger_index=merger_index,
            segment_start=jnp.asarray(coa_time, dtype=jnp.float64) + epoch,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )
        return BatchedDetectorStrain(
            strain=strain,
            detector_names=tuple(detector_names),
            coa_time=coa_time,
            epoch=epoch,
            sampling_frequency=sampling_frequency,
        )

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

    # jit-compile the vmapped projection so each detector's batch fuses into one
    # kernel; compiled once and reused across detectors (shared shapes).
    project_batch = jax.jit(jax.vmap(_project_event))

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
        per_detector.append(project_batch(fd.plus, fd.cross, f_plus, f_cross, time_delay))

    strain = jnp.stack(per_detector, axis=1)  # (n_events, n_detectors, n_samples)
    return BatchedDetectorStrain(
        strain=strain,
        detector_names=tuple(detector_names),
        coa_time=coa_time,
        epoch=epoch,
        sampling_frequency=sampling_frequency,
    )


def _project_rotating(  # noqa: PLR0913
    fd: object,
    detector_names: Sequence[str],
    *,
    n_samples: int,
    sampling_frequency: float,
    merger_index: int,
    segment_start: Array,
    right_ascension: Array,
    declination: Array,
    polarization_angle: Array,
) -> Array:
    """Project a batch with a time-dependent antenna pattern, on device.

    A time-varying response is not a frequency-domain multiply, so unlike the static
    path this one inverse-FFTs the polarizations first, places coalescence in the
    segment, and then applies the per-sample response in the time domain via
    :func:`~gwmock_signal.projection.jax_projection.project_polarizations_td_rotating`.

    Args:
        fd: Frequency-domain polarizations from the ripple backend.
        detector_names: LAL detector prefixes.
        n_samples: Samples per event segment.
        sampling_frequency: Sample rate in Hz.
        merger_index: Sample index coalescence is rolled to.
        segment_start: GPS time of each event segment's first sample, shape ``(n_events,)``.
        right_ascension: Per-event right ascension in radians.
        declination: Per-event declination in radians.
        polarization_angle: Per-event polarization angle in radians.

    Returns:
        Strain of shape ``(n_events, n_detectors, n_samples)``.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    def _to_time_domain(spectrum: Array) -> Array:
        # Same scaling and placement as the static path, so the two differ only in how
        # the detector response is applied.
        return jnp.roll(jnp.fft.irfft(spectrum, n=n_samples) * sampling_frequency, merger_index)

    plus_td = jax.vmap(_to_time_domain)(fd.plus)
    cross_td = jax.vmap(_to_time_domain)(fd.cross)

    per_detector = []
    for name in detector_names:
        response, location = reconstructed_geometry(name)

        def _one(  # noqa: PLR0913, PLR0917 - vmapped over six per-event arrays
            plus: Array,
            cross: Array,
            start: Array,
            ra: Array,
            dec: Array,
            psi: Array,
            response: Array = response,
            location: Array = location,
        ) -> Array:
            return project_polarizations_td_rotating(
                plus,
                cross,
                response=response,
                location=location,
                start_time=start,
                sampling_frequency=sampling_frequency,
                n_samples=n_samples,
                right_ascension=ra,
                declination=dec,
                polarization_angle=psi,
            )

        per_detector.append(
            jax.jit(jax.vmap(_one))(plus_td, cross_td, segment_start, right_ascension, declination, polarization_angle)
        )

    return jnp.stack(per_detector, axis=1)


def assemble_segments(
    batch: BatchedDetectorStrain,
    *,
    segment_duration: float,
    segment_start_times: Sequence[float],
    backgrounds: Sequence[Mapping[str, TimeSeries]] | None = None,
    interpolate_if_offset: bool = True,
) -> list[DetectorStrainStack]:
    """Scatter the batched signals into fixed-duration data segments (in memory).

    Each output segment spans ``[start, start + segment_duration)``. Every signal
    that overlaps a segment is injected into it with
    :func:`~gwmock_signal.injection.inject_strains_sequential`, which crops the
    signal to the segment span — so a signal longer than ``segment_duration``
    contributes its overlapping part to each of the consecutive segments it spans.

    Args:
        batch: Batched per-event/detector strain from :func:`simulate_cbc_batch`.
        segment_duration: Duration of every output segment, in seconds.
        segment_start_times: GPS start time of each output segment (typically a
            contiguous tiling, e.g. ``start + k * segment_duration``).
        backgrounds: Optional per-segment backgrounds, aligned with
            ``segment_start_times``; each maps detector name to a background
            ``TimeSeries`` to inject into. When ``None`` (default), zero-noise
            segments are created.
        interpolate_if_offset: Forwarded to ``inject_strains_sequential`` for
            signals whose start is not on a segment-sample boundary.

    Returns:
        One :class:`~gwmock_signal.multichannel.stack.DetectorStrainStack` per
        entry in ``segment_start_times`` (same order), with channels in
        ``batch.detector_names`` order.
    """
    if backgrounds is not None and len(backgrounds) != len(segment_start_times):
        raise ValueError("backgrounds must be aligned one-to-one with segment_start_times.")

    strain = np.asarray(batch.strain)
    _, _, n_samples = strain.shape
    sampling_frequency = batch.sampling_frequency
    dt = 1.0 / sampling_frequency
    signal_start = batch.epoch + np.asarray(batch.coa_time, dtype=float)
    signal_end = signal_start + n_samples * dt
    n_segment_samples = round(segment_duration * sampling_frequency)
    detectors = batch.detector_names

    segments: list[DetectorStrainStack] = []
    for k, raw_start in enumerate(segment_start_times):
        seg_start = float(raw_start)
        seg_end = seg_start + segment_duration
        overlapping = np.nonzero((signal_start < seg_end) & (signal_end > seg_start))[0]
        channels: dict[str, TimeSeries] = {}
        for d, name in enumerate(detectors):
            if backgrounds is not None:
                background = backgrounds[k][name]
            else:
                background = TimeSeries(np.zeros(n_segment_samples), t0=seg_start, sample_rate=sampling_frequency)
            injections = [TimeSeries(strain[i, d], t0=float(signal_start[i]), dt=dt) for i in overlapping]
            channels[name] = inject_strains_sequential(
                background, injections, interpolate_if_offset=interpolate_if_offset
            )
        segments.append(DetectorStrainStack.from_mapping(detectors, channels))
    return segments


def simulate_cbc_catalogue(  # noqa: PLR0913
    approximant: str,
    detector_names: Sequence[str],
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    parameters: Mapping[str, object],
    segment_duration: float,
    start_time: float,
    end_time: float,
    backend: RippleBackend | None = None,
    n_chirp_mass_bins: int = 1,
    chunk_size: int | None = None,
    interpolate_if_offset: bool = True,
) -> list[DetectorStrainStack]:
    """Generate a catalogue on device and assemble it into fixed-duration segments.

    Convenience wrapper that runs :func:`simulate_cbc_batch` and then
    :func:`assemble_segments`, tiling ``[start_time, end_time)`` into contiguous
    zero-noise segments of ``segment_duration``. Signals are placed at their
    ``coa_time`` and split across the segments they span; signals outside the span
    simply do not appear. For non-zero backgrounds use the two-step API
    (:func:`simulate_cbc_batch` then :func:`assemble_segments`) so you can supply a
    background per segment.

    Two independent memory controls (composable):

    - ``chunk_size`` bounds the *peak* generation memory by processing at most that
      many events per batched call. All chunks of a bin share that bin's grid, so
      chunking is **output-identical** to processing the whole bin at once.
    - ``n_chirp_mass_bins`` bounds the *buffer length* by generating heavier events
      on shorter grids. Because each bin uses a different frequency resolution,
      binning is **not** bit-identical to a single grid (see below).

    Args:
        approximant: A supported ripple approximant name.
        detector_names: Detector codes (e.g. ``"H1"``, ``"L1"``).
        sampling_frequency: Sample rate in Hz.
        minimum_frequency: Low-frequency cutoff in Hz.
        parameters: Canonical catalogue parameters as struct-of-arrays (see
            :func:`simulate_cbc_batch`).
        segment_duration: Duration of every output segment, in seconds.
        start_time: GPS start of the first segment.
        end_time: GPS time the tiling must cover up to; the final segment is the
            first one whose span reaches or passes ``end_time``.
        backend: Optional configured :class:`RippleBackend`. If it pins a
            ``segment_duration`` that grid is used for every bin (binning then
            saves no buffer memory but the run stays output-identical).
        n_chirp_mass_bins: Number of chirp-mass groups generated separately, each on
            its own worst-case grid (lightest first), injected on top of the
            earlier bins. ``1`` (default) uses a single grid sized for the
            lowest-mass event. Binned output agrees with a single-grid run only at
            the per-event grid discretization level (a fraction of a percent in
            overlap) — the resolution the per-event path uses.
        chunk_size: If set, generate at most this many events per batched call
            (within each bin). Output-identical to ``None``; only bounds peak memory.
        interpolate_if_offset: Forwarded to :func:`assemble_segments`.

    Returns:
        One :class:`~gwmock_signal.multichannel.stack.DetectorStrainStack` per
        segment, in time order.
    """
    if segment_duration <= 0:
        raise ValueError("segment_duration must be > 0")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if n_chirp_mass_bins < 1:
        raise ValueError("n_chirp_mass_bins must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    backend = backend or RippleBackend()
    n_segments = int(np.ceil((end_time - start_time) / segment_duration))
    segment_start_times = start_time + np.arange(n_segments) * segment_duration

    segments: list[DetectorStrainStack] | None = None
    for bin_indices in _chirp_mass_bins(parameters, n_chirp_mass_bins):
        # Pin the grid to this bin's worst case so every chunk of the bin shares it
        # (chunking stays output-identical). A user-pinned backend is left as-is.
        bin_backend = _bin_backend(backend, parameters, bin_indices, minimum_frequency, sampling_frequency)
        for chunk_indices in _count_chunks(bin_indices, chunk_size):
            chunk_parameters = {key: np.asarray(values)[chunk_indices] for key, values in parameters.items()}
            batch = simulate_cbc_batch(
                approximant,
                detector_names,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                parameters=chunk_parameters,
                backend=bin_backend,
            )
            # Chain groups: inject each on top of the segments built from earlier ones.
            backgrounds = [stack.to_dict() for stack in segments] if segments is not None else None
            segments = assemble_segments(
                batch,
                segment_duration=segment_duration,
                segment_start_times=segment_start_times,
                backgrounds=backgrounds,
                interpolate_if_offset=interpolate_if_offset,
            )
    return segments if segments is not None else []


def _chirp_mass(parameters: Mapping[str, object]) -> np.ndarray:
    """Return the (detector-frame) chirp mass for every event in ``parameters``."""
    mass1 = np.asarray(_required(parameters, "detector_frame_mass_1"), dtype=float)
    mass2 = np.asarray(_required(parameters, "detector_frame_mass_2"), dtype=float)
    return (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2


def _chirp_mass_bins(parameters: Mapping[str, object], n_bins: int) -> list[np.ndarray]:
    """Split event indices into ``n_bins`` contiguous chirp-mass groups (lightest first).

    Empty bins (when ``n_bins`` exceeds the event count) are dropped.
    """
    order = np.argsort(_chirp_mass(parameters))
    return [group for group in np.array_split(order, n_bins) if group.size > 0]


def _count_chunks(indices: np.ndarray, chunk_size: int | None) -> list[np.ndarray]:
    """Split ``indices`` into consecutive chunks of at most ``chunk_size`` (or one chunk)."""
    if chunk_size is None:
        return [indices]
    return [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]


def _bin_backend(
    backend: RippleBackend,
    parameters: Mapping[str, object],
    bin_indices: np.ndarray,
    minimum_frequency: float,
    sampling_frequency: float,
) -> RippleBackend:
    """Return a backend pinned to the bin's worst-case grid (or ``backend`` if already pinned)."""
    if backend.segment_duration is not None:
        return backend
    lightest = float(np.min(_chirp_mass(parameters)[bin_indices]))
    return backend.with_segment_duration(backend.segment_duration_for(lightest, minimum_frequency, sampling_frequency))


def _required(parameters: Mapping[str, object], name: str) -> object:
    """Return a required batch parameter or raise a clear error."""
    if name not in parameters:
        raise ValueError(f"Missing required batch parameter: {name!r}")
    return parameters[name]
