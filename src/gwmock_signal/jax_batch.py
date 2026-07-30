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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.injection import inject_strains_sequential
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.projection.geometry import DetectorSpec, reconstructed_geometry, resolve_detectors
from gwmock_signal.projection.jax_projection import (
    antenna_pattern,
    project_polarizations_fd,
    project_polarizations_td_rotating,
    time_delay_from_geocenter,
)
from gwmock_signal.projection.resampling import (
    require_shift_within_padding,
    require_terrestrial_location,
)
from gwmock_signal.projection.sidereal import gmst_anchor_and_rate, gmst_rad_astropy
from gwmock_signal.sampling_grid import SamplingGrid
from gwmock_signal.waveform.backends.ripple import FrequencyDomainPolarizations, RippleBackend

if TYPE_CHECKING:
    from jax import Array


@dataclass(frozen=True)
class BatchedDetectorStrain:
    """Catalogue-scale detector strain as raw arrays plus timing metadata.

    ``strain`` has shape ``(n_events, n_detectors, n_samples)`` and is a JAX array
    (on device). Each event/detector row is a time series with sample spacing
    ``1 / sampling_frequency``; coalescence sits ``-epoch`` seconds from the start of the
    buffer, near its end.

    Where the buffer *begins* depends on whether it was aligned to an output lattice:

    - **Aligned** (``grid`` and ``start_index`` set): the first sample is at
      ``grid.time_of(start_index)``, exactly on the lattice, so superposing the signal onto a
      segment of that grid is an integer-offset add.
    - **Unaligned** (both ``None``): the first sample is at ``epoch + coa_time[event]``, an
      arbitrary time, and a consumer must resample to place it -- which is accurate only for
      heavily oversampled strain.

    The signals are not yet placed on a shared timeline or segmented into files; that assembly
    step is handled separately.
    """

    strain: Array
    detector_names: tuple[str, ...]
    coa_time: np.ndarray
    epoch: float
    sampling_frequency: float
    #: Lattice index of each event's first sample, when the batch was generated against an
    #: output grid. Set means every event starts exactly on that grid, so superposing it is an
    #: integer-offset add; ``None`` means the starts are arbitrary and the consumer has to
    #: resample, which is accurate only for heavily oversampled signals.
    start_index: np.ndarray | None = None
    #: The grid the indices refer to, or ``None`` when unaligned.
    grid: SamplingGrid | None = None


#: Peak device buffers, in units of one ``n_events x n_samples`` float64 array, needed by
#: batched waveform generation itself — the part that does not scale with the number of
#: detectors. Calibrated from a single measurement: an IMRPhenomXPHM batch of 16384 events
#: at 8192 samples on 3 detectors asked XLA for 85.4 GiB, which is 85.4 such units, and
#: :data:`_PROJECTION_BUFFERS_PER_DETECTOR` accounts for the rest.
#:
#: This is one data point, not a calibration curve. It is used only to size chunks and to
#: turn an opaque out-of-memory abort into an actionable error, so it is deliberately on
#: the pessimistic side: over-estimating costs a smaller chunk, under-estimating costs a
#: crashed production run.
_GENERATION_BUFFERS = 73.4

#: Additional peak buffers per detector, same units: the projected strain plus the inverse
#: FFT and stacking temporaries that accompany it.
_PROJECTION_BUFFERS_PER_DETECTOR = 4.0

#: Multiplier applied when ``earth_rotation=True``. The rotating path inverse-FFTs the
#: polarizations, holds per-sample sidereal time, delay and antenna-pattern arrays, and
#: gathers through a windowed-sinc kernel, so it needs more live arrays per detector than
#: the frequency-domain path. Not separately measured -- flagged in the docstring.
_ROTATION_BUFFER_MULTIPLIER = 1.6

#: Fraction of device memory a batch may be sized to occupy. The rest is headroom for
#: allocator fragmentation and anything else resident.
_DEFAULT_MEMORY_FRACTION = 0.6


#: How close ``segment_duration * sampling_frequency`` must be to a whole number for the
#: segments to share one lattice. A segment boundary landing mid-sample cannot be honoured by
#: any integer-offset scheme.
_WHOLE_SAMPLE_TOLERANCE = 1e-6

#: Distinct projection-kernel configurations kept compiled. Each holds one XLA executable.
_KERNEL_CACHE_SIZE = 32


def _resolve_detector_specs(
    detector_names: Sequence[DetectorSpec],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split detector specifications into output channel names and LAL lookup keys.

    Thin wrapper over :func:`~gwmock_signal.projection.geometry.resolve_detectors` that returns
    tuples, because the keys are used as ``lru_cache`` arguments for the compiled kernels and the
    names go into a frozen dataclass -- both need to be hashable.

    Args:
        detector_names: Built-in LAL interferometer codes and/or ``CustomDetector`` instances.

    Returns:
        ``(output_names, lookup_keys)``, parallel and in the order given.

    Raises:
        ValueError: If no detectors are given, or a name is duplicated -- either would silently
            produce a strain array whose detector axis does not correspond to the requested
            channels.
    """
    resolved = resolve_detectors(detector_names)
    if not resolved:
        raise ValueError("At least one detector is required.")
    output_names = tuple(name for name, _ in resolved)
    if len(set(output_names)) != len(output_names):
        duplicated = sorted({name for name in output_names if output_names.count(name) > 1})
        raise ValueError(
            f"Detector names must be unique; got duplicates {duplicated}. Two CustomDetector "
            f"instances may share a geometry but not a name, since the name keys the output."
        )
    return output_names, tuple(key for _, key in resolved)


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _static_projection_kernel(n_samples: int, sampling_frequency: float, merger_index: int) -> Callable[..., Array]:
    """Return a cached jitted kernel for the midpoint-only (``earth_rotation=False``) path.

    The frequency grid is taken as an argument rather than rebuilt here: it already exists on
    the ripple output, and recomputing it would be the same quantity derived in two places.

    Args:
        n_samples: Samples per event segment.
        sampling_frequency: Sample rate in Hz.
        merger_index: Sample index coalescence is rolled to.

    Returns:
        A callable over ``(frequencies, plus, cross, f_plus, f_cross, time_delay)``, batched
        over events and unmapped over the shared frequency grid.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import
    import jax.numpy as jnp  # noqa: PLC0415

    def _project_event(  # noqa: PLR0913, PLR0917 - one vmapped argument per projection input
        frequencies: Array,
        plus: Array,
        cross: Array,
        f_plus: Array,
        f_cross: Array,
        time_delay: Array,
    ) -> Array:
        strain = project_polarizations_fd(
            frequencies,
            plus,
            cross,
            f_plus=f_plus,
            f_cross=f_cross,
            time_delay=time_delay,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
        )
        return jnp.roll(strain, merger_index)  # place coalescence near the segment end

    return jax.jit(jax.vmap(_project_event, in_axes=(None, 0, 0, 0, 0, 0)))


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _time_domain_kernel(n_samples: int, sampling_frequency: float, merger_index: int) -> Callable[[Array], Array]:
    """Return a cached jitted inverse FFT that also places coalescence in the segment."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    def _to_time_domain(spectrum: Array) -> Array:
        # Same scaling and placement as the static path, so the two differ only in how the
        # detector response is applied.
        return jnp.roll(jnp.fft.irfft(spectrum, n=n_samples) * sampling_frequency, merger_index)

    return jax.jit(jax.vmap(_to_time_domain))


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _rotating_projection_kernel(n_samples: int, sampling_frequency: float) -> Callable[..., Array]:
    """Return a cached jitted kernel for the rotating (``earth_rotation=True``) path.

    Detector geometry and the sidereal rate are unmapped *arguments*, not closed-over values:
    the geometry so one compiled kernel serves the whole network, and the rate because it is
    derived from Astropy per call and would otherwise make every distinct epoch a new cache
    key -- reintroducing exactly the per-call recompilation this cache removes.

    Args:
        n_samples: Samples per event segment.
        sampling_frequency: Sample rate in Hz.

    Returns:
        A callable over ``(plus, cross, ra, dec, psi, gmst_start, gmst_rate, response,
        location)``, batched over events.
    """
    import jax  # noqa: PLC0415

    def _one(  # noqa: PLR0913, PLR0917 - vmapped per-event arrays plus unmapped geometry
        plus: Array,
        cross: Array,
        ra: Array,
        dec: Array,
        psi: Array,
        gmst_start: Array,
        extra_shift: Array,
        gmst_rate: Array,
        response: Array,
        location: Array,
    ) -> Array:
        return project_polarizations_td_rotating(
            plus,
            cross,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            right_ascension=ra,
            declination=dec,
            polarization_angle=psi,
            gmst_start=gmst_start,
            gmst_rate=gmst_rate,
            extra_shift_samples=extra_shift,
        )

    return jax.jit(jax.vmap(_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None)))


def estimate_batch_memory_bytes(
    n_events: int,
    n_detectors: int,
    n_samples: int,
    *,
    earth_rotation: bool = True,
) -> int:
    """Estimate peak device memory for one :func:`simulate_cbc_batch` call.

    A vmapped batch holds far more than the strain it returns: the measured peak for an
    IMRPhenomXPHM batch was about 28x its own output. The estimate is therefore
    ``n_events * n_samples * 8 * (generation + per_detector * n_detectors)``, with the
    coefficients above.

    !!! warning "One calibration point"

        The coefficients come from a single A100 measurement with IMRPhenomXPHM, and the
        split between detector-independent and per-detector buffers is assumed rather than
        measured. Treat this as an order-of-magnitude guard that produces a useful error
        message, not as an accurate predictor. Approximants with smaller graphs than
        IMRPhenomXPHM will be over-estimated, which only costs a smaller chunk.

    Args:
        n_events: Events in the batch.
        n_detectors: Detectors projected onto.
        n_samples: Samples per event segment.
        earth_rotation: Whether the rotating projection is used, which needs more
            simultaneous buffers per detector.

    Returns:
        Estimated peak bytes.
    """
    if min(n_events, n_detectors, n_samples) < 1:
        raise ValueError("n_events, n_detectors and n_samples must all be >= 1")
    per_detector = _PROJECTION_BUFFERS_PER_DETECTOR * (_ROTATION_BUFFER_MULTIPLIER if earth_rotation else 1.0)
    buffers = _GENERATION_BUFFERS + per_detector * n_detectors
    return int(n_events * n_samples * 8 * buffers)


def available_device_memory_bytes() -> int | None:
    """Return the memory limit of the default JAX device, or ``None`` if unknown.

    CPU devices do not report a limit, and neither do some older backends, so callers must
    treat ``None`` as "cannot check" rather than as "no limit".
    """
    try:
        import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import
    except ImportError:
        return None
    devices = jax.devices()
    if not devices:
        return None
    stats = getattr(devices[0], "memory_stats", lambda: None)()
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    return int(limit) if limit else None


def recommend_chunk_size(
    n_detectors: int,
    n_samples: int,
    *,
    earth_rotation: bool = True,
    memory_fraction: float = _DEFAULT_MEMORY_FRACTION,
    available_bytes: int | None = None,
) -> int | None:
    """Return the largest event count expected to fit, or ``None`` if unknown.

    Args:
        n_detectors: Detectors projected onto.
        n_samples: Samples per event segment.
        earth_rotation: Whether the rotating projection is used.
        memory_fraction: Fraction of device memory the batch may occupy; must be in ``(0, 1]``.
        available_bytes: Device memory limit; queried from JAX when omitted.

    Returns:
        A chunk size of at least 1, or ``None`` when the device limit is unknown.

    Raises:
        ValueError: If ``memory_fraction`` is outside ``(0, 1]``.
    """
    # A fraction above 1 would recommend a chunk larger than the device, i.e. it would hand
    # back exactly the out-of-memory abort this function exists to prevent.
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"memory_fraction must be in (0, 1]; got {memory_fraction}.")
    limit = available_device_memory_bytes() if available_bytes is None else available_bytes
    if not limit:
        return None
    per_event = estimate_batch_memory_bytes(1, n_detectors, n_samples, earth_rotation=earth_rotation)
    return max(1, int(limit * memory_fraction // per_event))


def _check_batch_fits(n_events: int, n_detectors: int, n_samples: int, *, earth_rotation: bool) -> None:
    """Raise a useful error when a batch is not expected to fit in device memory.

    XLA's own failure for this is a bare ``RESOURCE_EXHAUSTED`` naming a number of GiB,
    with nothing about which knob to turn. This names the estimate, the limit, and a chunk
    size that should work.
    """
    limit = available_device_memory_bytes()
    if not limit:
        return
    estimate = estimate_batch_memory_bytes(n_events, n_detectors, n_samples, earth_rotation=earth_rotation)
    if estimate <= limit:
        return
    suggestion = recommend_chunk_size(n_detectors, n_samples, earth_rotation=earth_rotation, available_bytes=limit)
    raise MemoryError(
        f"This batch is estimated to need {estimate / 2**30:.1f} GiB of device memory but the "
        f"device reports {limit / 2**30:.1f} GiB: {n_events} events x {n_detectors} detectors x "
        f"{n_samples} samples, earth_rotation={earth_rotation}. Generate the catalogue through "
        f"simulate_cbc_catalogue(chunk_size={suggestion}), or reduce n_samples by raising "
        f"minimum_frequency. The estimate is approximate (see estimate_batch_memory_bytes); "
        f"pass a larger chunk_size explicitly if you believe it is pessimistic."
    )


def simulate_cbc_batch(  # noqa: PLR0913
    approximant: str,
    detector_names: Sequence[DetectorSpec],
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    parameters: Mapping[str, object],
    backend: RippleBackend | None = None,
    earth_rotation: bool = True,
    output_grid: SamplingGrid | None = None,
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
        detector_names: Built-in LAL interferometer codes (e.g. ``"H1"``, ``"L1"``) and/or
            :class:`~gwmock_signal.detector.CustomDetector` instances. A custom detector is
            resolved through the prefix it registers with LAL, but its output channel is keyed
            by its own ``name``.
        sampling_frequency: Sample rate in Hz.
        minimum_frequency: Low-frequency cutoff in Hz.
        parameters: Mapping of **canonical** gwmock-pop parameter names (no aliases)
            to equal-length 1-D arrays. In addition to the waveform parameters
            (masses, spins, distance, inclination, coa_phase) this must include
            ``right_ascension``, ``declination``, ``polarization_angle`` and
            ``coa_time``.
        backend: Optional configured :class:`RippleBackend` (e.g. with a fixed
            ``segment_duration`` or ``f_ref``). Defaults to ``RippleBackend()``.
        output_grid: Sample lattice the returned strain should start on. When given, each
            event's first sample is placed exactly on the grid and the sub-sample remainder is
            absorbed into the shift the projection already applies -- an exact resampling
            rather than a second, cruder one downstream. Superposition then becomes an
            integer-offset add. When omitted, buffers start at the arbitrary time
            ``epoch + coa_time`` and the consumer must resample; see
            :mod:`gwmock_signal.sampling_grid` for what that costs.
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
    import jax.numpy as jnp  # noqa: PLC0415

    backend = backend or RippleBackend()
    # Resolved before anything expensive: an unknown detector should fail here, not after a
    # catalogue has been generated. The two halves are used for different things -- lookup_keys
    # index LAL's registry, output_names key the result -- and conflating them is what limited
    # this path to built-in interferometer codes.
    output_names, lookup_keys = _resolve_detector_specs(detector_names)

    # Before generating anything: the estimate includes the waveform-generation buffers, so a
    # check placed after generation could never fire for a batch that exhausts memory *during*
    # generation -- which is most of the estimate. The grid length does not need the waveform,
    # only the lightest chirp mass (or a pinned segment duration, which _segment_samples
    # handles), so it can be sized up front.
    _check_batch_fits(
        len(np.atleast_1d(np.asarray(_required(parameters, "coa_time")))),
        len(lookup_keys),
        _planned_n_samples(backend, parameters, minimum_frequency, sampling_frequency),
        earth_rotation=earth_rotation,
    )

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

    # Split each event's desired start into a lattice index and the sub-sample remainder the
    # projection must absorb. Without a grid there is nothing to align to and the remainder is
    # zero, which leaves both branches exactly as they were.
    if output_grid is None:
        start_index = None
        alignment_shift = np.zeros_like(coa_time)
    else:
        if output_grid.sampling_frequency != sampling_frequency:
            raise ValueError(
                f"output_grid.sampling_frequency ({output_grid.sampling_frequency}) must equal "
                f"sampling_frequency ({sampling_frequency})."
            )
        start_index, alignment_shift = output_grid.split_index(coa_time + epoch)

    if earth_rotation:
        strain = _project_rotating(
            fd,
            lookup_keys,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
            merger_index=merger_index,
            # The aligned buffer starts one fractional sample earlier than requested, so the
            # sidereal anchor must move with it or F(t) and tau(t) are evaluated up to a full
            # sample after the samples they multiply.
            segment_start_gps=coa_time + epoch - alignment_shift / sampling_frequency,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
            alignment_shift=alignment_shift,
        )
        return BatchedDetectorStrain(
            strain=strain,
            detector_names=output_names,
            coa_time=coa_time,
            epoch=epoch,
            sampling_frequency=sampling_frequency,
            start_index=start_index,
            grid=output_grid,
        )

    # earth_rotation=False reference time: the midpoint of each event's placed segment.
    midpoint_offset = epoch + 0.5 * (n_samples - 1) * dt
    # Astropy is the single implementation of the sidereal model for both branches and
    # both projection paths; see gwmock_signal.projection.sidereal. The alignment shift moves
    # the buffer, so the midpoint reference time moves with it, as in the rotating branch.
    aligned_midpoint = coa_time + midpoint_offset - alignment_shift / sampling_frequency
    gmst = jnp.asarray(gmst_rad_astropy(aligned_midpoint), dtype=jnp.float64)

    project_batch = _static_projection_kernel(n_samples, sampling_frequency, merger_index)

    per_detector = []
    for key in lookup_keys:
        response, location = reconstructed_geometry(key)
        f_plus, f_cross = antenna_pattern(
            response,
            gmst,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )
        time_delay = time_delay_from_geocenter(location, gmst, right_ascension=right_ascension, declination=declination)
        # The alignment offset is a pure time shift, so it rides on the same exact phase
        # factor as the geocenter delay: no interpolation is involved on this branch at all.
        per_detector.append(
            project_batch(
                fd.frequencies,
                fd.plus,
                fd.cross,
                f_plus,
                f_cross,
                time_delay + jnp.asarray(alignment_shift, dtype=jnp.float64) / sampling_frequency,
            )
        )

    strain = jnp.stack(per_detector, axis=1)  # (n_events, n_detectors, n_samples)
    return BatchedDetectorStrain(
        strain=strain,
        detector_names=output_names,
        coa_time=coa_time,
        epoch=epoch,
        sampling_frequency=sampling_frequency,
        start_index=start_index,
        grid=output_grid,
    )


def _project_rotating(  # noqa: PLR0913
    fd: FrequencyDomainPolarizations,
    lookup_keys: Sequence[str],
    *,
    n_samples: int,
    sampling_frequency: float,
    merger_index: int,
    segment_start_gps: np.ndarray,
    right_ascension: Array,
    declination: Array,
    polarization_angle: Array,
    alignment_shift: np.ndarray,
) -> Array:
    """Project a batch with a time-dependent antenna pattern, on device.

    A time-varying response is not a frequency-domain multiply, so unlike the static
    path this one inverse-FFTs the polarizations first, places coalescence in the
    segment, and then applies the per-sample response in the time domain via
    :func:`~gwmock_signal.projection.jax_projection.project_polarizations_td_rotating`.

    Args:
        fd: Frequency-domain polarizations from the ripple backend.
        lookup_keys: LAL registry keys, one per detector -- a built-in interferometer code, or
            the prefix a ``CustomDetector`` registered itself under. Not the output channel
            names, which for a custom detector differ.
        n_samples: Samples per event segment.
        sampling_frequency: Sample rate in Hz.
        merger_index: Sample index coalescence is rolled to.
        segment_start_gps: GPS time of each event segment's first sample, shape
            ``(n_events,)``. Used on the host to anchor sidereal time per event.
        right_ascension: Per-event right ascension in radians.
        declination: Per-event declination in radians.
        polarization_angle: Per-event polarization angle in radians.
        alignment_shift: Per-event sub-sample offset, in samples, to absorb so the output
            starts on the caller's lattice. Folded into the resampling the projection already
            performs, so it is one exact operation rather than two.

    Returns:
        Strain of shape ``(n_events, n_detectors, n_samples)``.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    to_time_domain = _time_domain_kernel(n_samples, sampling_frequency, merger_index)
    plus_td = to_time_domain(fd.plus)
    cross_td = to_time_domain(fd.cross)

    # One Astropy evaluation per event on the host, plus one shared rate: the kernel then
    # needs only a multiply-add for sidereal time, and no second sidereal implementation.
    # Checked on the host, where these are concrete: inside the jitted kernel the geometry and
    # the shift are traced, so the padding assumptions cannot be validated there.
    require_shift_within_padding(alignment_shift, name="alignment_shift")
    for key in lookup_keys:
        require_terrestrial_location(reconstructed_geometry(key)[1], name=f"location of {key}")

    gmst_anchors, gmst_rate = gmst_anchor_and_rate(segment_start_gps)
    gmst_anchors = jnp.asarray(gmst_anchors, dtype=jnp.float64)
    # Explicit dtype rather than the weakly-typed Python float Astropy hands back. Differing
    # values do not retrace (measured), but a weak type and a concrete float64 are two
    # distinct signatures, so anything later passing an array scalar would pay a second
    # compilation. Pinning it removes that possibility.
    gmst_rate = jnp.asarray(gmst_rate, dtype=jnp.float64)

    project_batch = _rotating_projection_kernel(n_samples, sampling_frequency)

    per_detector = []
    for key in lookup_keys:
        response, location = reconstructed_geometry(key)
        per_detector.append(
            project_batch(
                plus_td,
                cross_td,
                right_ascension,
                declination,
                polarization_angle,
                gmst_anchors,
                jnp.asarray(alignment_shift, dtype=jnp.float64),
                gmst_rate,
                jnp.asarray(response, dtype=jnp.float64),
                jnp.asarray(location, dtype=jnp.float64),
            )
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

    Each output segment spans ``[start, start + segment_duration)``. A signal longer than
    ``segment_duration`` contributes its overlapping part to each of the consecutive segments
    it spans.

    When ``batch`` was generated against a :class:`~gwmock_signal.sampling_grid.SamplingGrid`
    -- see ``output_grid`` on :func:`simulate_cbc_batch` -- every signal already starts on the
    output lattice and superposition is an exact integer-offset add. The segment starts are
    then required to lie on that same grid, and are rejected rather than rounded if they do
    not. Otherwise signals fall between samples and
    :func:`~gwmock_signal.injection.inject_strains_sequential` resamples them, which is only
    accurate for heavily oversampled strain.

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
            signals whose start is not on a segment-sample boundary. Unused when the batch
            carries a sampling grid, because then nothing needs interpolating.

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
    aligned = batch.start_index is not None and batch.grid is not None
    if aligned:
        # Where the buffer actually begins, which differs from epoch + coa_time by the
        # fractional remainder the device absorbed. Using the requested time here would
        # misattribute overlap for events sitting within a fraction of a sample of a boundary.
        signal_start = np.asarray(batch.grid.time_of(batch.start_index), dtype=float)
    else:
        signal_start = batch.epoch + np.asarray(batch.coa_time, dtype=float)
    signal_end = signal_start + n_samples * dt
    n_segment_samples = round(segment_duration * sampling_frequency)
    detectors = batch.detector_names

    if aligned:
        # The catalogue wrapper checks this, but a direct caller reaches here too, and integer
        # overlap arithmetic on a rounded length would silently describe a different interval
        # from the one requested.
        exact_samples = segment_duration * sampling_frequency
        if abs(exact_samples - round(exact_samples)) > _WHOLE_SAMPLE_TOLERANCE:
            raise ValueError(
                f"segment_duration * sampling_frequency must be a whole number of samples for a "
                f"grid-aligned batch; {segment_duration} x {sampling_frequency} = {exact_samples}."
            )
        segment_index = batch.grid.require_on_lattice(
            np.asarray(segment_start_times, dtype=float), name="segment_start_times"
        )
        event_index = np.asarray(batch.start_index, dtype=np.int64)
        # Checked up front rather than per channel: a mismatched background is a caller error, and
        # discovering it on the last segment after assembling every earlier one wastes the work.
        _require_backgrounds_match_segments(
            backgrounds,
            detectors=detectors,
            segment_index=segment_index,
            n_segment_samples=n_segment_samples,
            grid=batch.grid,
            sampling_frequency=sampling_frequency,
        )

    segments: list[DetectorStrainStack] = []
    for k, raw_start in enumerate(segment_start_times):
        seg_start = float(raw_start)
        seg_end = seg_start + segment_duration
        if aligned:
            # Integer lattice arithmetic, not reconstructed GPS times. Both are on one lattice
            # by construction here, so comparing sample counts is exact and implements the
            # half-open convention [start, start + duration) without float round-off deciding
            # whether a signal ending exactly on a boundary belongs to the next segment.
            offset = event_index - int(segment_index[k])
            overlapping = np.nonzero((offset < n_segment_samples) & (offset + n_samples > 0))[0]
        else:
            overlapping = np.nonzero((signal_start < seg_end) & (signal_end > seg_start))[0]
        channels: dict[str, TimeSeries] = {}
        for d, name in enumerate(detectors):
            if aligned:
                channels[name] = _aligned_channel(
                    background=None if backgrounds is None else backgrounds[k][name],
                    strain=strain[:, d],
                    offsets=offset,
                    overlapping=overlapping,
                    n_segment_samples=n_segment_samples,
                    segment_start=seg_start,
                    sampling_frequency=sampling_frequency,
                )
                continue
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


def _require_backgrounds_match_segments(  # noqa: PLR0913
    backgrounds: Sequence[Mapping[str, TimeSeries]] | None,
    *,
    detectors: tuple[str, ...],
    segment_index: np.ndarray,
    n_segment_samples: int,
    grid: SamplingGrid,
    sampling_frequency: float,
) -> None:
    """Check every supplied background against the segment it is paired with.

    A no-op when ``backgrounds`` is ``None``, so the caller needs no branch.

    Args:
        backgrounds: Per-segment mapping of detector name to background, or ``None``.
        detectors: Detector names, in batch order.
        segment_index: Lattice index of each segment's first sample.
        n_segment_samples: Samples each segment must contain.
        grid: Lattice the batch and the segment starts share.
        sampling_frequency: Sample rate in Hz the batch was generated at.

    Raises:
        ValueError: If any background does not match its segment.
    """
    if backgrounds is None:
        return
    for k, index in enumerate(segment_index):
        for name in detectors:
            _require_background_matches_segment(
                backgrounds[k][name],
                name=name,
                segment_index=int(index),
                n_segment_samples=n_segment_samples,
                grid=grid,
                sampling_frequency=sampling_frequency,
            )


def _require_background_matches_segment(  # noqa: PLR0913
    background: TimeSeries,
    *,
    name: str,
    segment_index: int,
    n_segment_samples: int,
    grid: SamplingGrid,
    sampling_frequency: float,
) -> None:
    """Reject a background that does not describe the segment it will be added to.

    The signals are placed by integer lattice offsets derived from ``segment_start_times``, so a
    background describing a *different* interval cannot be reconciled with them -- and the failure
    is silent in either direction. Taking the background's own epoch for the result would label
    the output with one interval while the data sits at another; taking the segment's, as this
    code does, silently discards what the caller asked for. Both were measured: a background
    offset by 100 s, one at twice the sample rate, and one of half the length were all accepted
    and produced quietly wrong output, the last of them a half-length segment.

    Rejecting instead follows the same rule as the sampling grid itself -- a mismatch is a caller
    error worth naming, not something to round away.

    Args:
        background: Caller-supplied background for one segment and detector.
        name: Detector name, for the error message.
        segment_index: Lattice index of the segment's first sample.
        n_segment_samples: Samples the segment must contain.
        grid: Lattice the batch and the segment starts share.
        sampling_frequency: Sample rate in Hz the batch was generated at.

    Raises:
        ValueError: If the background's length, sample rate or epoch does not match the segment.
    """
    if len(background) != n_segment_samples:
        raise ValueError(
            f"background for {name} at segment index {segment_index} has {len(background)} "
            f"samples, but the segment is {n_segment_samples}. A shorter background silently "
            f"yields a shorter segment, so it is rejected."
        )
    background_rate = float(background.sample_rate.value)
    # Relative comparison: a rate that is not a power of two does not round-trip exactly through
    # gwpy's dt, so exact equality would reject a background the caller built correctly.
    if not np.isclose(background_rate, sampling_frequency, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"background for {name} at segment index {segment_index} is sampled at "
            f"{background_rate} Hz, but the batch is at {sampling_frequency} Hz."
        )
    background_index = int(grid.require_on_lattice(float(background.t0.value), name=f"t0 of background for {name}"))
    if background_index != segment_index:
        raise ValueError(
            f"background for {name} starts at lattice index {background_index}, but the segment "
            f"it is paired with starts at {segment_index} "
            f"({(background_index - segment_index) / sampling_frequency:+g} s away). The signals "
            f"are placed by integer offsets from the segment start, so a background describing a "
            f"different interval cannot be combined with them."
        )


def _aligned_channel(  # noqa: PLR0913
    *,
    background: TimeSeries | None,
    strain: np.ndarray,
    offsets: np.ndarray,
    overlapping: np.ndarray,
    n_segment_samples: int,
    segment_start: float,
    sampling_frequency: float,
) -> TimeSeries:
    """Superpose one segment and detector, wrapping the result exactly once.

    The accumulator is a bare array that becomes a ``TimeSeries`` only at the end. Building the
    accumulator *as* a ``TimeSeries`` cost two full copies of every segment -- one to obtain a
    writable buffer, one from gwpy's copy-on-construct -- and the scatter runs at ~1.1x the
    memory-bandwidth bound for the traffic it must do, so a redundant traversal of the segment is
    a first-order cost rather than bookkeeping.

    Args:
        background: Segment to add into, or ``None`` for a zero background. Never mutated.
        strain: Per-event strain for one detector, shape ``(n_events, n_samples)``.
        offsets: Start offset of each event in samples relative to the segment start.
        overlapping: Indices of events that overlap this segment.
        n_segment_samples: Samples in the segment.
        segment_start: GPS time of the segment's first sample.
        sampling_frequency: Sample rate in Hz.

    Returns:
        The background plus every overlapping signal.
    """
    if background is None:
        data = np.zeros(n_segment_samples, dtype=float)
        unit = None
    else:
        data = np.array(background.value, dtype=float, copy=True)
        unit = background.unit
    _superpose_on_lattice(data, strain, offsets, overlapping)
    # copy=False hands gwpy the buffer just built here, which nothing else holds a reference to.
    return TimeSeries(data, t0=segment_start, sample_rate=sampling_frequency, unit=unit, copy=False)


def _superpose_on_lattice(
    data: np.ndarray,
    strain: np.ndarray,
    offsets: np.ndarray,
    overlapping: np.ndarray,
) -> None:
    """Add lattice-aligned signals into one segment, in place, with integer slicing only.

    Exact by construction: every signal already starts on the segment's own lattice, so this
    is a slice addition with no resampling. The alternative -- letting a signal land between
    samples and interpolating -- reaches 12% error at half Nyquist and 49% at 0.8 Nyquist,
    which would discard the accuracy the device projection was built for.

    Args:
        data: Accumulator for one segment and detector, shape ``(n_segment_samples,)``,
            **modified in place**. The caller owns it and wraps it afterwards.
        strain: Per-event strain for one detector, shape ``(n_events, n_samples)``.
        offsets: Start offset of each event in samples relative to the segment start.
        overlapping: Indices of events that overlap this segment.
    """
    n_segment_samples = data.shape[0]
    n_signal = strain.shape[1]
    for i in overlapping:
        offset = int(offsets[i])
        source_start = max(0, -offset)
        source_stop = min(n_signal, n_segment_samples - offset)
        if source_stop <= source_start:
            continue
        target_start = offset + source_start
        data[target_start : target_start + (source_stop - source_start)] += strain[i, source_start:source_stop]


def simulate_cbc_catalogue(  # noqa: PLR0913
    approximant: str,
    detector_names: Sequence[DetectorSpec],
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    parameters: Mapping[str, object],
    segment_duration: float,
    start_time: float,
    end_time: float,
    backend: RippleBackend | None = None,
    earth_rotation: bool = True,
    n_chirp_mass_bins: int = 1,
    chunk_size: int | None = None,
    memory_fraction: float = _DEFAULT_MEMORY_FRACTION,
    align_to_output_grid: bool = True,
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
        detector_names: Built-in LAL interferometer codes (e.g. ``"H1"``, ``"L1"``) and/or
            :class:`~gwmock_signal.detector.CustomDetector` instances, as for
            :func:`simulate_cbc_batch`.
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
        earth_rotation: Forwarded to :func:`simulate_cbc_batch`. Defaults to ``True``, and
            also makes the automatic chunk size smaller, because the rotating path holds
            more buffers per detector.
        n_chirp_mass_bins: Number of chirp-mass groups generated separately, each on
            its own worst-case grid (lightest first), injected on top of the
            earlier bins. ``1`` (default) uses a single grid sized for the
            lowest-mass event. Binned output agrees with a single-grid run only at
            the per-event grid discretization level (a fraction of a percent in
            overlap) — the resolution the per-event path uses.
        chunk_size: Generate at most this many events per batched call (within each bin).
            Output-identical whatever the value; it only bounds peak memory. When omitted,
            a size is chosen from the device memory limit and the grid actually selected
            (see :func:`recommend_chunk_size`) — previously the default was no chunking at
            all, which is what made a large catalogue abort with a bare XLA
            out-of-memory error. Pass an explicit value to override the estimate.
        memory_fraction: Fraction of device memory an automatically chosen chunk may
            occupy. Ignored when ``chunk_size`` is given.
        align_to_output_grid: Generate every batch on the lattice defined by this function's
            own segment starts, so superposition is an exact integer-offset add. Defaults to
            ``True`` because the alternative resamples each signal with a cubic spline, which
            reaches 12% error at half Nyquist. Set ``False`` only to reproduce that older
            behaviour deliberately; it is a legacy mode, not a fallback.
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
    # Resolved once here purely so an unusable detector is reported before any generation. Each
    # chunk resolves again inside simulate_cbc_batch, which is cheap: a CustomDetector caches its
    # LAL detector on first use, and a built-in code is a dictionary lookup.
    _resolve_detector_specs(detector_names)
    n_segments = int(np.ceil((end_time - start_time) / segment_duration))
    segment_start_times = start_time + np.arange(n_segments) * segment_duration

    output_grid: SamplingGrid | None = None
    if align_to_output_grid:
        samples_per_segment = segment_duration * sampling_frequency
        if abs(samples_per_segment - round(samples_per_segment)) > _WHOLE_SAMPLE_TOLERANCE:
            raise ValueError(
                f"segment_duration * sampling_frequency must be a whole number of samples for the "
                f"segments to share one lattice; {segment_duration} x {sampling_frequency} = "
                f"{samples_per_segment}. Adjust one of them, or pass align_to_output_grid=False to "
                f"accept resampled superposition."
            )
        # One grid for the whole catalogue, so every chunk and every chirp-mass bin lands on the
        # same lattice and can be superposed with integer offsets. The starts are then rebuilt
        # *from integer sample indices* rather than kept as start_time + k * segment_duration:
        # over a long span the repeated float multiplication accumulates representation error,
        # so a start intended to be on the lattice can drift off it even when the spacing is a
        # whole number of samples.
        output_grid = SamplingGrid(epoch=float(start_time), sampling_frequency=sampling_frequency)
        segment_start_times = output_grid.time_of(np.arange(n_segments, dtype=np.int64) * round(samples_per_segment))

    segments: list[DetectorStrainStack] | None = None
    for bin_indices in _chirp_mass_bins(parameters, n_chirp_mass_bins):
        # Pin the grid to this bin's worst case so every chunk of the bin shares it
        # (chunking stays output-identical). A user-pinned backend is left as-is.
        bin_backend = _bin_backend(backend, parameters, bin_indices, minimum_frequency, sampling_frequency)
        # Size the chunk from this bin's own grid: bins differ in n_samples by design, so a
        # single catalogue-wide chunk size would be wrong for all but one of them.
        effective_chunk = chunk_size
        if effective_chunk is None:
            bin_samples = _bin_n_samples(bin_backend, parameters, bin_indices, minimum_frequency, sampling_frequency)
            effective_chunk = recommend_chunk_size(
                len(tuple(detector_names)),
                bin_samples,
                earth_rotation=earth_rotation,
                memory_fraction=memory_fraction,
            )
        for chunk_indices in _count_chunks(bin_indices, effective_chunk):
            chunk_parameters = {key: np.asarray(values)[chunk_indices] for key, values in parameters.items()}
            batch = simulate_cbc_batch(
                approximant,
                detector_names,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                parameters=chunk_parameters,
                backend=bin_backend,
                earth_rotation=earth_rotation,
                output_grid=output_grid,
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


def _symmetric_mass_ratio(parameters: Mapping[str, object]) -> np.ndarray:
    """Return the symmetric mass ratio for every event in ``parameters``.

    Needed alongside the chirp mass because the 1PN term in the inspiral duration depends on the
    total mass, and at fixed chirp mass a more asymmetric binary is heavier and lasts longer. Sizing
    a grid from the lightest chirp mass alone therefore does not identify the worst case.
    """
    mass1 = np.asarray(_required(parameters, "detector_frame_mass_1"), dtype=float)
    mass2 = np.asarray(_required(parameters, "detector_frame_mass_2"), dtype=float)
    return mass1 * mass2 / (mass1 + mass2) ** 2


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
    return backend.with_segment_duration(
        backend.segment_duration_for(
            _chirp_mass(parameters)[bin_indices],
            minimum_frequency,
            sampling_frequency,
            eta=_symmetric_mass_ratio(parameters)[bin_indices],
        )
    )


def _planned_n_samples(
    backend: RippleBackend,
    parameters: Mapping[str, object],
    minimum_frequency: float,
    sampling_frequency: float,
) -> int:
    """Return the shared grid length the batch will allocate, before generating it.

    The batched ripple path sizes one grid from the longest inspiral present (or from a pinned
    segment duration), so the dominant term in the memory estimate is knowable in advance. That is
    what lets the preflight run before any large allocation happens.

    Every event is passed, exactly as the backend does when it generates: sizing this from a proxy
    such as the lightest chirp mass could disagree with the grid actually allocated, which would
    make the preflight estimate wrong in the direction that matters.
    """
    return int(
        backend._segment_samples(
            _chirp_mass(parameters),
            minimum_frequency,
            sampling_frequency,
            eta=_symmetric_mass_ratio(parameters),
        )
    )


def _bin_n_samples(
    backend: RippleBackend,
    parameters: Mapping[str, object],
    bin_indices: np.ndarray,
    minimum_frequency: float,
    sampling_frequency: float,
) -> int:
    """Return the grid length the bin will use, without generating any waveform.

    Needed before the batch runs so the chunk can be sized from the grid that will
    actually be allocated: bins deliberately differ in ``n_samples``, so one
    catalogue-wide chunk size would be wrong for all but one of them.
    """
    return int(
        backend._segment_samples(
            _chirp_mass(parameters)[bin_indices],
            minimum_frequency,
            sampling_frequency,
            eta=_symmetric_mass_ratio(parameters)[bin_indices],
        )
    )


def _required(parameters: Mapping[str, object], name: str) -> object:
    """Return a required batch parameter or raise a clear error."""
    if name not in parameters:
        raise ValueError(f"Missing required batch parameter: {name!r}")
    return parameters[name]
