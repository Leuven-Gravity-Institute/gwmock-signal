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
"""Project GW polarizations onto ground-based detectors."""

from __future__ import annotations

import functools
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from gwmock_signal.projection import finite_arm, solar_system
from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.geometry import (
    DetectorKey,
    DetectorSpec,
    arm_lengths,
    detector_fields,
    reconstructed_geometry,
    resolve_detectors,
)
from gwmock_signal.projection.kernels import (
    antenna_pattern,
    arm_directions,
    finite_arm_residual,
    time_delay_from_geocenter,
)
from gwmock_signal.projection.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    EARTH_RADIUS_M,
    OVERSAMPLING_FACTOR,
    SPEED_OF_LIGHT_M_S,
    edge_padding,
    oversample_fourier,
    require_shift_within_padding,
    require_terrestrial_location,
    resample_uniform_sinc,
)
from gwmock_signal.projection.sidereal import gmst_rad_astropy, precessed_sky_anchor_and_rate
from gwmock_signal.projection.solar_system import SolarSystemEphemeris
from gwmock_signal.timeseries import TimeSeries, as_timeseries

if TYPE_CHECKING:
    import gwpy.timeseries

    from gwmock_signal.sampling_grid import SamplingGrid

logger = logging.getLogger("gwmock_signal")

#: Conservative fractional strain error per second of signal, for ``earth_rotation=False``.
#:
#: The constant-pattern branch evaluates the antenna pattern once, at the midpoint of the span it
#: is given, so its error grows with how far Earth turns across that span.
#:
#: Measured against the rotating branch over 48 combinations of detector (H1, L1), sky position,
#: polarization angle and frequency at a 300 s span, interior samples only: the worst per-sample
#: deviation relative to peak grew at between 1.1e-5 and 2.9e-4 per second, median 5.9e-5. The
#: spread is a factor of five, because how fast the antenna pattern turns depends on where the
#: source sits relative to the detector's response. This is the **maximum** observed, so what it
#: produces is an upper bound for a typical source rather than a prediction for a given one.
_CONSTANT_PATTERN_ERROR_PER_SECOND = 2.9e-4

#: Span at or beyond which ``earth_rotation=False`` is worth warning about, in seconds.
#:
#: Where the conservative rate above reaches one percent. The boundary is inclusive: a span of
#: exactly this length warns. That is the safer way round for a guard against an error nothing else
#: reveals, and it costs only a message on a span already at the one-percent mark. That separates the cases cleanly in
#: practice: a few-second compact-binary merger stays silent, a minutes-long inspiral does not.
_CONSTANT_PATTERN_WARN_SECONDS = 30.0

#: Estimated error above which a percentage stops meaning anything and saturation is named instead.
#:
#: The linear rate is a local model near the threshold. Extrapolated to a day it would claim more
#: than 1000%, when the real deviation saturates around the signal amplitude itself.
_CONSTANT_PATTERN_SATURATION = 0.3


def _validate_polarizations(
    polarizations: Mapping[str, TimeSeries | gwpy.timeseries.TimeSeries],
) -> tuple[TimeSeries, TimeSeries]:
    """Validate ``plus``/``cross`` share one time grid, returning them as :class:`TimeSeries`."""
    if not isinstance(polarizations, Mapping):
        raise TypeError("polarizations must be a mapping with 'plus' and 'cross' keys.")
    if "plus" not in polarizations or "cross" not in polarizations:
        raise ValueError("polarizations must contain both 'plus' and 'cross' keys.")
    try:
        hp = as_timeseries(polarizations["plus"])
        hc = as_timeseries(polarizations["cross"])
    except TypeError as exc:
        raise TypeError(
            "polarizations['plus'] and polarizations['cross'] must be gwmock_signal.TimeSeries or "
            "gwpy.timeseries.TimeSeries."
        ) from exc
    if len(hp.data.shape) != 1 or len(hc.data.shape) != 1:
        raise ValueError("plus and cross must each be one-dimensional.")
    if hp.n_samples != hc.n_samples:
        raise ValueError(f"plus and cross must have the same number of samples; got {hp.n_samples} and {hc.n_samples}.")
    if hp.sample_rate != hc.sample_rate:
        raise ValueError("plus and cross must have the same sample rate.")
    hp_times = np.asarray(hp.times, dtype=float)
    hc_times = np.asarray(hc.times, dtype=float)
    dt = hp.dt
    if not np.allclose(hp_times, hc_times, rtol=0.0, atol=max(np.finfo(float).eps, 0.5 * dt)):
        raise ValueError("plus and cross must share the same time samples (same t0, dt, and length).")
    return hp, hc


def _padded_for_delay(
    hp: TimeSeries, hc: TimeSeries, *, sinc_taps: int, kaiser_beta: float
) -> tuple[TimeSeries, TimeSeries]:
    """Return the polarizations zero-padded so no delayed sample falls outside the projection.

    Without this every detector series was exactly as long as the input while the delay moved up
    to ~21 ms of signal past an end: a short burst vanished entirely on the rotating branch and
    wrapped to the wrong end of the buffer on the phase-shift branch. Sized by
    :func:`~gwmock_signal.projection.resampling.edge_padding` -- the delay bound for any ground-based
    detector plus the kernel's half-width -- rather than by this network's largest delay, so every
    series shares one start and length and the device kernel's shape does not depend on the sky
    position. ``t0`` moves back by whole samples, so the input's samples keep their times.
    """
    pad = edge_padding(hp.sample_rate, sinc_taps, kaiser_beta)
    return hp.pad(pad, pad), hc.pad(pad, pad)


def _gmst_accurate(t_gps: float) -> float:
    """Return Greenwich mean sidereal time in radians using Astropy."""
    return float(gmst_rad_astropy(float(t_gps)))


def _gmst_accurate_array(t_gps: np.ndarray) -> np.ndarray:
    """Return GMST in radians for an array of GPS times in one vectorized call."""
    return gmst_rad_astropy(t_gps)


def _make_detectors(detector_specs: Sequence[DetectorSpec]) -> list[tuple[str, DetectorKey]]:
    """Resolve detector names to one lookup key per output channel.

    Delegates to :func:`~gwmock_signal.projection.geometry.resolve_detectors` so this path and
    the device path cannot diverge on what a detector specification means.
    """
    return resolve_detectors(detector_specs)


def _warn_if_constant_pattern_is_stretched(time_array: np.ndarray, *, earth_rotation: bool) -> None:
    """Warn when ``earth_rotation=False`` is applied to a span long enough to matter.

    Args:
        time_array: Sample times of the polarizations being projected, in GPS seconds.
        earth_rotation: The caller's setting; nothing is warned about when it is ``True``.
    """
    if earth_rotation:
        return
    span = float(time_array[-1] - time_array[0])
    if span < _CONSTANT_PATTERN_WARN_SECONDS:
        return

    estimate = span * _CONSTANT_PATTERN_ERROR_PER_SECOND
    scale = (
        f"up to about {100.0 * estimate:.0f}% of peak strain"
        if estimate <= _CONSTANT_PATTERN_SATURATION
        else "of order the signal amplitude itself"
    )
    logger.warning(
        "earth_rotation=False evaluates the antenna pattern once, at the midpoint of the %.1f s "
        "span given here, so the error grows with how far Earth turns across it -- %s. The "
        "approximation is intended for signals short enough that the pattern is effectively "
        "constant (below ~%.0f s). Two further consequences worth knowing: the result depends on "
        "how the data was split into segments, because the midpoint does; and nothing in the "
        "output reveals either effect. Use earth_rotation=True for a signal this long.",
        span,
        scale,
        _CONSTANT_PATTERN_WARN_SECONDS,
    )


#: Compiled-kernel cache bound, matching ``jax_batch._KERNEL_CACHE_SIZE``. An unbounded cache
#: is a process-lifetime leak of XLA executables and their device memory: the justification
#: "a run uses one or two segment shapes" is a per-run claim, and this cache outlives a run,
#: so a long-lived worker projecting varied lengths would accumulate kernels without end.
_KERNEL_CACHE_SIZE = 32

#: Longest span the device path's linear sidereal model is accepted for, in seconds.
#:
#: Set by an error budget rather than by where the validation table ends. The linear model's
#: deviation from Astropy, converted to the quantity that actually enters the projection -- a
#: geocenter delay -- is 1.2e-15 s over 8192 s, 6.2e-11 s over a day, and 1.2e-7 s over 90 days.
#: The projection still carries a known 8.7e-07 s worst-case disagreement with
#: ``lalpulsar.Barycenter``, from omitting nutation where LAL includes it, so a day-long span sits
#: four orders of magnitude below a systematic this code knowingly accepts. (It was 1.8e-04 s and
#: six orders before the source direction was precessed into the frame of date; the budget tightened
#: with the fix, and 8192 s and one day both still clear it.)
#:
#: One day is therefore not a physical limit but the point beyond which nothing has been measured
#: and a single segment stops being a plausible way to use this. Consecutive segments are
#: unaffected at any run length: each one re-anchors against Astropy.
#:
#: Public because a caller that *configures* the backend needs it before projecting: gwmock reads
#: it to refuse a segment longer than this while validating a configuration, rather than letting
#: the run reach the projection and fail there. A caller that hard-coded the number instead would
#: drift from it silently the first time the error budget above is revisited.
MAX_LINEAR_SIDEREAL_SPAN_SECONDS = 86400.0

#: The two implementations of the rotating projection, by name.
PROJECTION_BACKENDS: frozenset[str] = frozenset({"numpy", "jax"})


def validate_projection_backend(value: str, *, parameter: str = "backend") -> str:
    """Return *value* if it names a projection implementation, else raise.

    Shared so the name is checked identically wherever it is accepted -- this function, and the
    simulators that take it at construction time and only reach the projection later. Checking it
    in each place separately is how one of them ends up accepting a name the projection does not.

    Args:
        value: The candidate backend name.
        parameter: What to call it in the error message, since callers spell the argument
            differently (``backend`` here, ``projection_backend`` on a simulator).

    Returns:
        *value* unchanged.

    Raises:
        ValueError: If *value* is not one of :data:`PROJECTION_BACKENDS`.
    """
    if value not in PROJECTION_BACKENDS:
        raise ValueError(f"{parameter} must be 'numpy' or 'jax', got {value!r}.")
    return value


@functools.lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _compiled_rotating_projection(  # noqa: PLR0913, PLR0917
    sampling_frequency: float,
    n_samples: int,
    sinc_taps: int,
    kaiser_beta: float,
    tapered: bool = False,
    arm_lengths: tuple[float, float] | None = None,
    solar_components: tuple[bool, bool, bool] | None = None,
    barycentric: bool = False,
):
    """Return a jitted rotating projection for one segment shape, compiled once and reused.

    Without this the device path runs **eagerly**, dispatching every operation in the per-sample
    kernel separately, and the cost is dominated by that dispatch rather than by the arithmetic.
    Measured on an A100 before this wrapper existed, fitting device time to ``a + b * n``: 3.7 s
    fixed against 3.3e-6 s per sample, so a 256 s segment spent 90% of its time in dispatch. Two
    machines confirmed the cause rather than merely the size -- an A30 node whose CPU was 2.6x
    faster than the A100 node's showed a fixed cost 2.6x smaller, tracking the *host* speed, which
    device work would not do.

    Detector geometry stays traced, so one compilation serves every detector in a network. Making
    ``response`` and ``location`` static would specialise the kernel per detector and pay the
    compile again for each; :mod:`~gwmock_signal.projection.jax_batch` avoids that the same way
    and for the same reason.

    Cached on the shape rather than rebuilt per call because ``jax.jit`` re-traces a fresh wrapper
    every time one is constructed, which would reintroduce the cost this exists to remove. The
    cache is keyed on four scalars and bounded at ``_KERNEL_CACHE_SIZE``, because a long-lived
    process projecting varied shapes would otherwise retain XLA executables for its lifetime.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    from gwmock_signal.projection.jax_projection import (  # noqa: PLC0415
        SolarArguments,
        project_polarizations_td_rotating,
    )

    def _project(  # noqa: PLR0913, PLR0917
        plus,
        cross,
        response,
        location,
        right_ascension,
        declination,
        right_ascension_rate,
        declination_rate,
        polarization_angle,
        gmst_start,
        gmst_rate,
        arms,
        solar_arrays,
        solar_start_offset,
        solar_absorbed,
        solar_offset_samples,
        extra_shift,
    ):
        solar = (
            None
            if solar_components is None
            else SolarArguments(
                solar_arrays,
                solar_components,
                solar_start_offset,
                None if barycentric else solar_absorbed,
                solar_offset_samples,
            )
        )
        return project_polarizations_td_rotating(
            plus,
            cross,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            right_ascension=right_ascension,
            declination=declination,
            right_ascension_rate=right_ascension_rate,
            declination_rate=declination_rate,
            polarization_angle=polarization_angle,
            gmst_start=gmst_start,
            gmst_rate=gmst_rate,
            sinc_taps=sinc_taps,
            kaiser_beta=kaiser_beta,
            tapered=tapered,
            arms=None if arm_lengths is None else arms,
            arm_lengths=arm_lengths,
            solar=solar,
            extra_shift_samples=extra_shift,
        )

    return jax.jit(_project)


def _project_rotating_on_device(  # noqa: PLR0913
    hp: TimeSeries,
    hc: TimeSeries,
    detectors: Sequence[tuple[str, DetectorKey]],
    *,
    time_array: np.ndarray,
    right_ascension: float,
    declination: float,
    right_ascension_rate: float,
    declination_rate: float,
    polarization_angle: float,
    sinc_taps: int,
    kaiser_beta: float,
    tapered: bool = False,
    arm_lengths: Mapping[str, tuple[float, float]] | None = None,
    timing: _SolarTiming | None = None,
    lattice_shift: float = 0.0,
) -> dict[str, TimeSeries]:
    """Run the rotating projection through the JAX primitive, one detector at a time.

    The device function is the same algorithm step for step -- per-sample delay, per-sample
    antenna pattern, one resampling of the polarizations at the delayed times -- and both call
    :func:`~gwmock_signal.projection.resampling.edge_padding` and the same kernel constants, so
    neither can pad or window differently from the other.

    Sidereal time is the one place the two differ in form rather than arithmetic. The host path
    asks Astropy for GMST at every sample; the device path takes a host-computed anchor and rate
    and extrapolates linearly, because Astropy cannot be traced. That is not an approximation at
    these lengths: the linear model deviates by 6.4e-14 rad over 2048 s and 5.6e-14 rad over
    8192 s, worth 1.2e-15 s of geocenter delay, which is orders below the kernel's own error.
    See :mod:`~gwmock_signal.projection.sidereal` for the table.

    Detectors are looped rather than vmapped. ``jax_batch`` vmaps this primitive over *events*,
    where the arrays are per-event and independent; here there is one pair of polarizations and
    a handful of detectors, so a vmap would add a compiled kernel per network size for a saving
    on three iterations of a loop whose body is already the whole cost.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import

    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate  # noqa: PLC0415

    # Refused rather than allowed to degrade. Without x64 every `jnp.asarray(..., float64)` in
    # the device path truncates to float32 and the projection still returns a plausible-looking
    # series -- measured 1.1e-06 to 1.4e-06 of peak over 256 s to 4096 s against this function's
    # own host path, where the 64-bit device path is within 5.5e-13. (Before the resampling
    # position was passed as a whole sample and a fraction, the float32 position alone made it
    # 2.7e-03 over 256 s and 1.8e-02 over 1024 s.) A warning would not do: nothing downstream
    # can tell such a series from a correct one.
    # Importing `ripplegw` happens to enable x64, which is why the compact-binary and
    # continuous-wave paths have never hit this; that is an accident of an unrelated import.
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "backend='jax' requires JAX in 64-bit mode; call "
            'jax.config.update("jax_enable_x64", True) before projecting. In 32-bit mode the '
            "delays and sidereal angles lose the precision this projection depends on and the "
            "result is wrong by about 1e-6 of peak, over a million times the 64-bit error, while "
            "still looking like strain."
        )

    hp_vals = np.asarray(hp.data, dtype=float)
    hc_vals = np.asarray(hc.data, dtype=float)
    sampling_frequency = hp.sample_rate
    n_samples = len(hp_vals)
    # The linear sidereal model is fitted per call, so a run of many consecutive segments never
    # accumulates error -- each gets a fresh Astropy anchor. What it cannot absorb is one
    # unusually long segment. Measured against Astropy at this epoch: 1.5e-10 rad over 8 hours,
    # 2.9e-9 over a day, 5.8e-6 over 90 days. The tabulated validation in `sidereal` stops at
    # 8192 s, which is where this refuses, rather than letting the public API accept a span the
    # model was never checked against and returning a quietly degraded answer.
    span = float(time_array[-1] - time_array[0])
    if span > MAX_LINEAR_SIDEREAL_SPAN_SECONDS:
        raise ValueError(
            f"backend='jax' accepts spans up to {MAX_LINEAR_SIDEREAL_SPAN_SECONDS:.0f} s "
            f"and this one is {span:.0f} s. The device path extrapolates sidereal time linearly "
            f"from one anchor, which is exact to 1.2e-15 s of delay over the validated range and "
            f"degrades beyond it. Project in segments, or use backend='numpy', which asks Astropy "
            f"for every sample."
        )
    anchors, rate = gmst_anchor_and_rate(float(time_array[0]))

    solar_components = None if timing is None else timing.model.components
    if timing is None:
        solar = ((np.zeros(1),), 0.0, np.zeros(3), 0.0)
    else:
        # Relative to the model's epoch, so the device never handles a GPS epoch.
        absorbed = np.zeros(3) if timing.absorbed is None else np.array(timing.absorbed, dtype=float)
        solar = (
            timing.model.arrays(),
            float(time_array[0]) - timing.model.epoch,
            absorbed,
            float(timing.offset_samples),
        )
    lengths = {name: tuple(float(length) for length in value) for name, value in (arm_lengths or {}).items()}

    strains: dict[str, TimeSeries] = {}
    for name, prefix in detectors:
        response, location = reconstructed_geometry(prefix)
        # Same precondition the host path asserts, and for the same reason: the padding width
        # assumes a terrestrial delay bound. Checked here too, because on the device `location`
        # is a traced argument and the check could not run there.
        require_terrestrial_location(location, name=f"location of {prefix}")
        arm_length = lengths.get(name) if arm_lengths is not None else None
        terms = _response_terms(prefix, arm_length, sampling_frequency)
        arms = np.stack([term.arm for term in terms]) if arm_length is not None else np.zeros((2, 3))
        compiled = _compiled_rotating_projection(
            sampling_frequency,
            n_samples,
            sinc_taps,
            kaiser_beta,
            tapered,
            arm_length,
            solar_components,
            timing is not None and timing.absorbed is None,
        )
        projected = compiled(
            hp_vals,
            hc_vals,
            response,
            location,
            right_ascension,
            declination,
            right_ascension_rate,
            declination_rate,
            polarization_angle,
            float(np.atleast_1d(anchors)[0]),
            float(rate),
            arms,
            *solar,
            lattice_shift,
        )
        strains[name] = TimeSeries(
            np.asarray(projected, dtype=float),
            t0=float(time_array[0]),
            sample_rate=hp.sample_rate,
            name=name,
        )
    return strains


def project_polarizations_to_network(  # noqa: PLR0913
    polarizations: Mapping[str, TimeSeries | gwpy.timeseries.TimeSeries],
    detector_names: Sequence[DetectorSpec],
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    effects: ProjectionEffects | None = None,
    reference_time: float | None = None,
    ephemeris: SolarSystemEphemeris | None = None,
    backend: str = "numpy",
    sinc_taps: int = DEFAULT_SINC_TAPS,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
    tapered: bool = False,
    earth_rotation: bool | None = None,
    precess_source_direction: bool | None = None,
    finite_arm_length: bool | None = None,
    output_grid: SamplingGrid | None = None,
) -> dict[str, TimeSeries]:
    """Project tensor plus/cross strains onto detectors using detector geometry.

    Built-in and custom detector codes are resolved through the LAL cached
    detector registry. For ``earth_rotation=False``, the constant
    geocenter->detector delay is applied via an exact frequency-domain phase
    shift (``h(t-tau) <-> H(f)*exp(-2*pi*i*f*tau)``), which is lossless at all
    frequencies. For ``earth_rotation=True``, the polarizations are resampled at the
    time-dependent delayed times with the Kaiser-windowed sinc kernel in
    :mod:`gwmock_signal.projection.resampling`, gathered from a zero-padded copy so kernel
    taps reaching past either end read zero rather than repeating the endpoint.

    **Each detector series is longer than the input, and starts earlier.** Delaying the signal
    by the geocentre-to-detector light-travel time -- up to Earth's radius over the speed of
    light, about 21 ms -- moves part of it past the end, or before the start, of the
    polarization buffer. The polarizations are therefore zero-padded by
    :func:`~gwmock_signal.projection.resampling.edge_padding` samples at each end before they
    are projected, and every returned series covers that padded span: it starts that many
    samples before the input and has twice that many more samples, on the input's own sample
    lattice. The padding is sized for the largest delay any ground-based detector can have, plus
    the resampling kernel's half-width, so it is the same for every detector, sky position and
    backend: all returned series share one start time and length. Inside the input's span the
    samples are what a same-length projection gives, up to rounding; outside it they hold the
    delayed signal that a same-length projection would have dropped (``earth_rotation=True``) or
    wrapped round to the other end of the buffer (``earth_rotation=False``). A caller that needs
    the input's span exactly can crop to it, at the cost of that content.

    Args:
        polarizations: Mapping containing ``plus`` and ``cross`` time series, each a
            :class:`~gwmock_signal.timeseries.TimeSeries` or a GWpy ``TimeSeries``, on a common
            grid. Sample times are the lattice ``t0 + k * dt`` of the series' start time and
            spacing, not a GWpy ``times`` array, which at GPS magnitudes and a rate that is not a
            power of two drifts from it (by 0.22 samples within one second at 3000 Hz).
        detector_names: Sequence of IFO codes (e.g. ``H1``, ``L1``, ``V1``) or
            :class:`~gwmock_signal.detector.CustomDetector` instances, or a mix
            of both.
        right_ascension: Source right ascension in radians.
        declination: Source declination in radians.
        polarization_angle: Polarization angle psi in radians (tensor modes).
        effects: Which physical effects to include; see
            :class:`~gwmock_signal.projection.effects.ProjectionEffects`. All are on by default:
            Earth rotation, the finite-arm response, source precession and the Roemer, Einstein and
            Shapiro delays. ``ProjectionEffects.long_wavelength()`` turns them all off.
        reference_time: GPS time at which a transient's solar-system delay is absorbed: its value and
            rate there are taken out, so that time keeps its meaning at the geocentre. The coalescence
            time of a compact binary, the peak time of a burst. ``None``, the default, takes the time of
            the polarizations' peak power, which for a compact binary is the merger, milliseconds from
            coalescence; absorbing there rather than at coalescence changes the delay by about 1e-12 s.
        ephemeris: The Earth and Sun tables for the solar-system delays; the LALPulsar DE405 tables by
            default, downloaded and cached on first use.
        backend: Which implementation evaluates the ``earth_rotation=True`` branch.
            ``"numpy"`` (the default) runs on the host. ``"jax"`` delegates to
            :func:`~gwmock_signal.projection.jax_projection.project_polarizations_td_rotating`,
            which runs on whatever JAX backend is configured and is several times faster even
            on a CPU because the whole per-sample kernel fuses into one compiled loop. The two
            agree to 1e-10 of peak, pinned by ``test_rotating_projection_matches_numpy_path``;
            the difference is floating-point reassociation, not a different model. Selected
            rather than automatic: an implicit switch on whether JAX imports would make the
            numerical output depend on what happens to be installed.
        sinc_taps: Taps in the band-limited resampling kernel used by the
            ``earth_rotation=True`` branch. More taps cost arithmetic and buy accuracy.
        kaiser_beta: Kaiser window shape parameter for that kernel.
        tapered: Whether the polarizations are a whole signal that falls smoothly to zero at both
            ends, such as a generated compact-binary waveform. If so, the ``earth_rotation=True``
            branch oversamples them exactly before the kernel reads them, and is accurate up to
            their Nyquist frequency rather than to 0.8 of it. Leave it ``False`` for a piece cut
            from a longer signal: the exact interpolation is global, and the cut ends would ring
            across the whole piece. The ``earth_rotation=False`` branch is unaffected.
        earth_rotation: Deprecated; ``effects=ProjectionEffects(earth_rotation=...)``. Overrides that
            field when given.
        precess_source_direction: Deprecated; ``effects=ProjectionEffects(source_precession=...)``.
            Overrides that field when given. Which value suits which analysis is described on
            ``source_precession``.
        finite_arm_length: Deprecated; ``effects=ProjectionEffects(finite_arm_length=...)``. Overrides
            that field when given.
        output_grid: A lattice to put the returned series on, at the inputs' sample rate, such as
            the segment the strain will be added to. Each series then starts at the lattice sample
            preceding the padded input's start, up to one sample earlier, and its samples are the
            detector strain at those lattice times: the sub-sample offset joins the delay in the one
            resampling the projection makes. Adding the result to a series on the same lattice is
            then an integer-offset add. Placing a series that is off the lattice needs a second
            resampling instead, of strain that need not be band-limited at its ends, and
            resampling a finished buffer cannot reproduce what the band-limited interpolant of the
            polarizations puts past them. ``None``, the default, keeps the inputs' own lattice.

    Returns:
            Mapping from each detector name to the projected strain as a
            :class:`~gwmock_signal.timeseries.TimeSeries`, at the inputs' sample rate and on their
            sample lattice, or on ``output_grid`` when given, padded by
            ``edge_padding(sample_rate, sinc_taps, kaiser_beta)`` samples at each end as described above.

    Raises:
        TypeError: If ``polarizations`` is not a mapping of time series as required.
        ValueError: If keys are missing, time grids disagree, a detector name is not
            recognized, ``backend`` is unknown, ``backend="jax"`` is combined with
            ``earth_rotation=False``, or the resampling kernel (``sinc_taps``, ``kaiser_beta``)
            is invalid -- checked on both branches, since it sizes the padding on both, or the
            finite-arm response is asked of a detector with no arm length, or a solar-system delay of
            a span the ephemeris does not cover (2000 to 2040 for the default tables), or ``output_grid``
            has another sample rate.
    """
    return _project(
        polarizations,
        detector_names,
        right_ascension=right_ascension,
        declination=declination,
        polarization_angle=polarization_angle,
        effects=_resolve_effects(
            effects,
            earth_rotation=earth_rotation,
            precess_source_direction=precess_source_direction,
            finite_arm_length=finite_arm_length,
        ),
        backend=backend,
        sinc_taps=sinc_taps,
        kaiser_beta=kaiser_beta,
        tapered=tapered,
        ephemeris=ephemeris,
        reference_time=reference_time,
        output_grid=output_grid,
    )


def _arm_lengths(detector_specs: Sequence[DetectorSpec]) -> dict[str, tuple[float, float]]:
    """Return each detector's x- and y-arm lengths, keyed by its output name, for the finite-arm response.

    Raises:
        ValueError: If a detector has no arm length recorded.
    """
    lengths = {}
    for spec in detector_specs:
        name = spec if isinstance(spec, str) else spec.name
        recorded = arm_lengths(spec)
        if recorded is None:
            raise ValueError(
                f"The finite-arm response needs every detector's arm length, and {name!r} has none: give a "
                f"CustomDetector its arm_length_m, or turn finite_arm_length off in ProjectionEffects. LAL "
                f"records none for its resonant bars."
            )
        lengths[name] = recorded
    return lengths


@dataclass(frozen=True)
class _ResponseTerm:
    """One part of a detector's response: a tensor read at one point.

    The long-wavelength response is one term, the detector's response tensor at its vertex. The finite-arm
    response is two, one per arm: half the outer product of the arm's direction with itself, read at the
    arm's midpoint with the residual of its length applied, the y arm's subtracted from the x arm's.
    """

    sign: float
    tensor: np.ndarray
    location: np.ndarray
    arm: np.ndarray | None = None
    length: float | None = None


def _response_terms(
    prefix: DetectorKey, lengths: tuple[float, float] | None, sample_rate: float
) -> list[_ResponseTerm]:
    """Return a detector's response terms: its vertex, or with ``lengths`` its two arms.

    Shared by the host, device and batched projections, so each checks the same geometry: the vertex
    within Earth's radius, and each arm's midpoint within what the padding covers, by its distance from
    the geocentre, which bounds its delay for any sky position.

    Raises:
        ValueError: If the vertex or a midpoint lies beyond what the edge padding covers.
    """
    response, location = reconstructed_geometry(prefix)
    require_terrestrial_location(location, name=f"location of {prefix}")
    if lengths is None:
        return [_ResponseTerm(1.0, response, location)]
    fields = prefix if isinstance(prefix, tuple) else detector_fields(prefix)
    terms = [
        _ResponseTerm(sign, 0.5 * np.outer(arm, arm), location + 0.5 * length * arm, arm, length)
        for sign, arm, length in zip((1.0, -1.0), arm_directions(fields, xp=np), lengths, strict=True)
    ]
    for term in terms:
        _require_midpoint_within_padding(float(np.linalg.norm(term.location)) / SPEED_OF_LIGHT_M_S, sample_rate)
    return terms


@dataclass(frozen=True)
class _SolarTiming:
    """The solar-system delay for one projection, as each detector sample's signal time needs it.

    ``absorbed`` is ``(reference time, delay, rate)`` for a transient, whose delay enters only through
    how it changes from that time, the time measured from the model's epoch; ``None`` for a source whose
    polarizations are given in SSB time, which takes the delay in full. ``offset_samples`` is then the
    whole number of samples by which the polarizations' SSB time runs ahead of their labels.

    Times are passed as seconds since the model's epoch, never as GPS times: a GPS time near 1e9 s holds
    only 1e-7 s, which the delay's rate of up to 1e-4 would turn into an error of 1e-11 s.
    """

    model: solar_system.DelayModel
    absorbed: tuple[float, float, float] | None
    offset_samples: int = 0

    def delay(self, since_epoch: np.ndarray) -> np.ndarray:
        """Return the delay to add to a signal time at the geocentre, given in seconds since the epoch."""
        value = self.model.evaluate(since_epoch)
        if self.absorbed is None:
            return value
        reference, delay, rate = self.absorbed
        return value - delay - rate * (since_epoch - reference)


def _solar_timing(  # noqa: PLR0913
    effects: ProjectionEffects,
    ephemeris: SolarSystemEphemeris,
    *,
    span: tuple[float, float],
    right_ascension: float,
    declination: float,
    reference_time: float,
    barycentric_offset_samples: int | None,
) -> _SolarTiming | None:
    """Return the solar-system timing the effects ask for, or ``None`` when they include no part of it.

    ``right_ascension`` and ``declination`` are the source's J2000 position: the Earth's orbit is in that
    frame, whatever the projection does to the direction for the Earth's rotation.
    """
    if not effects.solar_system:
        return None
    components = {"roemer": effects.roemer_delay, "einstein": effects.einstein_delay, "shapiro": effects.shapiro_delay}
    # Wide enough for any geocentre-to-detector delay at either end.
    margin = 0.1
    model = solar_system.DelayModel.build(
        span[0] - margin, span[1] + margin, right_ascension, declination, ephemeris, **components
    )
    if barycentric_offset_samples is not None:
        return _SolarTiming(model, None, barycentric_offset_samples)
    value, rate = solar_system.delay(np.array([reference_time]), right_ascension, declination, ephemeris, **components)
    return _SolarTiming(model, (float(reference_time) - model.epoch, float(value[0]), float(rate[0])))


def _kernel_input(
    hp_vals: np.ndarray, hc_vals: np.ndarray, pad: int, *, tapered: bool
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return what the rotating branch's kernel reads: ``(pad, factor, plus, cross)``.

    Both polarizations zero-padded as the device path pads them, and -- for a whole tapered signal --
    oversampled exactly by ``factor``, so the kernel stays accurate up to their Nyquist frequency rather
    than to 0.8 of it. Neither depends on the detector, so it is done once for all of them.
    """
    plus, cross = np.pad(hp_vals, (pad, pad)), np.pad(hc_vals, (pad, pad))
    if not tapered:
        return pad, 1, plus, cross
    return pad, OVERSAMPLING_FACTOR, oversample_fourier(plus), oversample_fourier(cross)


def _require_midpoint_within_padding(midpoint_delay: np.ndarray | float, sample_rate: float) -> None:
    """Reject a response term's delay, or a bound on it, that the edge padding does not cover.

    The padding covers the largest vertex delay, Earth's radius over ``c``, rounded up to whole samples,
    and one sample more (see :func:`~gwmock_signal.projection.resampling.edge_padding`). An arm's midpoint
    lies up to half an arm further from the geocentre, which that sample covers for arms shorter than
    ``2 c / fs``: 73 km at 8192 Hz.

    Raises:
        ValueError: If the delay, in samples, exceeds that allowance.
    """
    allowed = math.ceil(EARTH_RADIUS_M / SPEED_OF_LIGHT_M_S * sample_rate) + 1
    largest = float(np.max(np.abs(midpoint_delay)))
    if largest * sample_rate > allowed:
        raise ValueError(
            f"An arm midpoint's delay from the geocentre reaches {largest * 1e3:.4f} ms, beyond the "
            f"{allowed / sample_rate * 1e3:.4f} ms the edge padding covers at {sample_rate:g} Hz: an arm far longer "
            f"than any ground-based detector's."
        )


def _resolve_effects(
    effects: ProjectionEffects | None,
    *,
    earth_rotation: bool | None = None,
    precess_source_direction: bool | None = None,
    finite_arm_length: bool | None = None,
    caller: str = "project_polarizations_to_network",
) -> ProjectionEffects:
    """Return the effects to use: ``effects`` (all on by default), with any deprecated flag applied.

    ``earth_rotation``, ``precess_source_direction`` and ``finite_arm_length`` were separate arguments before
    :class:`~gwmock_signal.projection.effects.ProjectionEffects` gathered every effect in one place. Given,
    each overrides the corresponding field and warns.
    """
    resolved = ProjectionEffects() if effects is None else effects
    for old, new, value in (
        ("earth_rotation", "earth_rotation", earth_rotation),
        ("precess_source_direction", "source_precession", precess_source_direction),
        ("finite_arm_length", "finite_arm_length", finite_arm_length),
    ):
        if value is not None:
            warnings.warn(
                f"{caller}({old}=...) is deprecated; pass effects=ProjectionEffects({new}=...) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            resolved = replace(resolved, **{new: bool(value)})
    return resolved


def _require_backend_supports(backend: str, effects: ProjectionEffects) -> None:
    """Reject a backend name, or a combination of effects the chosen backend does not implement."""
    validate_projection_backend(backend)
    if backend == "jax" and not effects.earth_rotation:
        # The constant-pattern branch is a frequency-domain phase shift, and no device
        # counterpart exists. Refused rather than silently served from the host path, which
        # would report a backend that did not run.
        raise ValueError(
            "backend='jax' is only available with earth_rotation=True. The constant-pattern "
            "branch applies one frequency-domain phase shift for the whole span and has no "
            "device implementation; it is also cheap, being the branch that skips the resampler."
        )


def _project(  # noqa: PLR0913
    polarizations: Mapping[str, TimeSeries | gwpy.timeseries.TimeSeries],
    detector_names: Sequence[DetectorSpec],
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    effects: ProjectionEffects,
    backend: str,
    sinc_taps: int,
    kaiser_beta: float,
    keep_delayed_samples: bool = True,
    tapered: bool = False,
    ephemeris: SolarSystemEphemeris | None = None,
    reference_time: float | None = None,
    barycentric_offset_samples: int | None = None,
    output_grid: SamplingGrid | None = None,
) -> dict[str, TimeSeries]:
    """Project as :func:`project_polarizations_to_network` does, returning :class:`TimeSeries`.

    ``keep_delayed_samples=False`` returns series exactly as long as the input instead, dropping
    whatever the delay moves past either end. Only for a caller that has already generated the
    polarizations with that margin and trims it off itself -- the continuous-wave simulator,
    whose signal carries on beyond the buffer rather than ending in it.

    ``output_grid`` puts the returned series on that lattice: each starts up to one sample before the
    padded input, at the lattice sample preceding it, and the sub-sample difference joins the delay in
    the one resampling the projection already makes, as ``simulate_cbc_batch`` does for its
    ``output_grid``.

    ``barycentric_offset_samples`` marks polarizations given in SSB time, as the continuous-wave
    simulator generates them: sample ``j``, labelled ``t0 + j / fs``, is the signal at SSB time
    ``t0 + (j + barycentric_offset_samples) / fs``. They then take the solar-system delays in full rather
    than relative to ``reference_time``.
    """
    _require_backend_supports(backend, effects)
    hp, hc = _validate_polarizations(polarizations)
    normalized_names = [d if isinstance(d, str) else d.name for d in detector_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("detector_names must not contain duplicates.")
    detectors = _make_detectors(list(detector_names))
    # Looked up before anything is computed, so a detector without an arm length fails at once.
    lengths = _arm_lengths(list(detector_names)) if effects.finite_arm_length else {}

    # The caller's span, not the padded one, for the two things that describe the input rather than
    # the projection grid: the duration warning, and where the sky-position line is anchored.
    input_start = float(hp.t0)
    input_end = input_start + (hp.n_samples - 1) * hp.dt
    _warn_if_constant_pattern_is_stretched(np.array([input_start, input_end]), earth_rotation=effects.earth_rotation)
    if reference_time is None:
        # A transient's peak power, on the unpadded input, which `input_start` labels: for a compact
        # binary the merger, within milliseconds of coalescence, where absorbing the solar-system delay
        # instead changes it by ~1e-12 s.
        power = np.asarray(hp.data, dtype=float) ** 2 + np.asarray(hc.data, dtype=float) ** 2
        reference_time = input_start + int(np.argmax(power)) * hp.dt
    if keep_delayed_samples:
        # Padded before anything reads the time axis, so both branches and both backends project
        # the padded span as if it were the input.
        hp, hc = _padded_for_delay(hp, hc, sinc_taps=sinc_taps, kaiser_beta=kaiser_beta)

    # The output's sample times, and how many samples they lie before the input's. Off a lattice the two
    # coincide. On one, every quantity below that depends on time -- sidereal angle, sky line, delays,
    # pattern -- is evaluated at the output's own samples, and only the resampling reads the input.
    lattice_shift = 0.0
    time_array = np.asarray(hp.times, dtype=float)
    if output_grid is not None:
        if output_grid.sampling_frequency != hp.sample_rate:
            raise ValueError(
                f"output_grid.sampling_frequency ({output_grid.sampling_frequency}) must equal the "
                f"polarizations' sample rate ({hp.sample_rate})."
            )
        start_index, fraction = output_grid.split_index(float(hp.t0))
        require_shift_within_padding(fraction, name="alignment shift")
        lattice_shift = float(fraction)
        time_array = output_grid.time_of(int(start_index) + np.arange(hp.n_samples, dtype=np.int64))
    reference = float(0.5 * (time_array[0] + time_array[-1]))
    # The J2000 position, before any precession below: the Earth's orbit is in that frame.
    timing = _solar_timing(
        effects,
        SolarSystemEphemeris() if ephemeris is None else ephemeris,
        span=(float(time_array[0]), float(time_array[-1])),
        right_ascension=right_ascension,
        declination=declination,
        reference_time=reference_time,
        barycentric_offset_samples=barycentric_offset_samples,
    )

    # Resolved once, before anything else reads the sky position. Every branch below -- host scalar,
    # host array, and the device kernel, for the delay *and* the antenna pattern -- derives its
    # geometry from these numbers, so deciding here rather than at each site is what makes it
    # impossible for one path to precess and another not to.
    #
    # Anchored at the caller's *first* sample, before any padding; the device kernel's
    # `sample_offsets` count from its own first sample, so the line is carried there below, and both
    # backends read the same line.
    #
    # A position and a *rate*, not a position alone. Freezing it per segment looks harmless -- the
    # angles move 2306 arcseconds per century, so across 4096 s they drift 3e-6 arcseconds -- but that
    # argument is about drift *within* a segment and says nothing about the step *between* two of them.
    # The step broke continuous-wave phase coherence at 1.6e-08 of peak against a 1e-09 tolerance.
    # Linear in absolute time removes it: two abutting segments evaluate the same line at the same
    # absolute time, so they agree exactly where they meet whatever their anchors are.
    #
    # Zero rates when not precessing, rather than a separate code path: the same multiply-add then
    # holds the position fixed exactly, so the two conventions cannot diverge in anything but the
    # numbers they put in.
    if effects.source_precession:
        # Python floats on both branches, which `precessed_sky_anchor_and_rate` guarantees: 0-d
        # NumPy arrays here would be a distinct `jax.jit` signature from the other branch's weakly
        # typed floats, so a process using both conventions at one segment shape would compile the
        # same device kernel twice.
        (right_ascension, declination), (d_right_ascension, d_declination) = precessed_sky_anchor_and_rate(
            right_ascension, declination, input_start
        )
    else:
        d_right_ascension = d_declination = 0.0

    # Dispatched here, before any of the host branch's preparation. Everything below -- two
    # rffts, the frequency grid, and per-sample Astropy GMST with its sines and cosines -- serves
    # only the NumPy branches, and the device path recomputes what it needs from an anchor and a
    # rate. Left after the dispatch it was pure waste, and not a small one: at 4096 s and 512 Hz
    # the Astropy call alone is 4.3 s on this machine and the unused arrays are ~120 MiB, so the
    # device route was neither as fast nor as independent of host scaling as it claimed.
    if backend == "jax":
        return _project_rotating_on_device(
            hp,
            hc,
            detectors,
            time_array=time_array,
            # The device kernel counts its offsets from its own first sample, the padded one, so the
            # line is carried there from the caller's first sample, where it was fitted. Fitting it
            # at the padded start instead would draw a slightly different line.
            right_ascension=right_ascension + d_right_ascension * (float(time_array[0]) - input_start),
            declination=declination + d_declination * (float(time_array[0]) - input_start),
            right_ascension_rate=d_right_ascension,
            declination_rate=d_declination,
            polarization_angle=polarization_angle,
            sinc_taps=sinc_taps,
            kaiser_beta=kaiser_beta,
            tapered=tapered,
            arm_lengths=lengths if effects.finite_arm_length else None,
            timing=timing,
            lattice_shift=lattice_shift,
        )

    return _project_on_host(
        hp,
        hc,
        detectors,
        time_array=time_array,
        reference_time=reference,
        sky_line=(input_start, right_ascension, declination, d_right_ascension, d_declination),
        polarization_angle=polarization_angle,
        earth_rotation=effects.earth_rotation,
        sinc_taps=sinc_taps,
        kaiser_beta=kaiser_beta,
        tapered=tapered,
        arm_lengths=lengths if effects.finite_arm_length else None,
        timing=timing,
        lattice_shift=lattice_shift,
    )


def _project_on_host(  # noqa: PLR0913
    hp: TimeSeries,
    hc: TimeSeries,
    detectors: Sequence[tuple[str, DetectorKey]],
    *,
    time_array: np.ndarray,
    reference_time: float,
    sky_line: tuple[float, float, float, float, float],
    polarization_angle: float,
    earth_rotation: bool,
    sinc_taps: int,
    kaiser_beta: float,
    tapered: bool,
    arm_lengths: Mapping[str, tuple[float, float]] | None,
    timing: _SolarTiming | None,
    lattice_shift: float = 0.0,
) -> dict[str, TimeSeries]:
    """Project on the host, for :func:`_project`, once the polarizations are padded and the sky resolved.

    ``sky_line`` is ``(input_start, right_ascension, declination, d_right_ascension, d_declination)``: the
    source position at the caller's first sample and its rates, as :func:`_project` resolved them.
    ``arm_lengths`` holds each detector's arm lengths for the finite-arm response, ``None`` for the
    long-wavelength one; ``timing`` the solar-system delay, ``None`` for none. ``time_array`` holds the
    output's sample times, which lie ``lattice_shift`` samples before the input's (:func:`_project`).

    Every detector is a sum of response terms (:func:`_response_terms`), each its own pattern times the
    polarizations at the signal time its point sees: the sample's time, less the delay from the
    geocentre to that point, plus the solar-system delay at the geocentre at that moment.
    """
    input_start, right_ascension, declination, d_right_ascension, d_declination = sky_line
    hp_vals = np.asarray(hp.data, dtype=float)
    hc_vals = np.asarray(hc.data, dtype=float)
    n_samples = len(hp_vals)
    dt = hp.dt
    offset = 0 if timing is None else timing.offset_samples

    if earth_rotation:
        # What the kernel reads, prepared once for every detector.
        pad, factor, hp_read, hc_read = _kernel_input(
            hp_vals, hc_vals, edge_padding(hp.sample_rate, sinc_taps, kaiser_beta), tapered=tapered
        )
        # Per sample, not per segment: see the anchor-and-rate comment in `_project`.
        precession_offsets = time_array - input_start
        sky = (
            _gmst_accurate_array(time_array),
            right_ascension + d_right_ascension * precession_offsets,
            declination + d_declination * precession_offsets,
        )
        # The whole sample and the delay are passed apart: summed in float64, a position near sample
        # 10**7 keeps only ~1e-9 of a sample of the delay, an error that grows with the buffer and peaks
        # where the strain is steepest. Apart, the delay keeps its own precision. On an oversampled series
        # both scale by the factor, exactly.
        index = factor * (pad + np.arange(len(time_array), dtype=np.int64) - offset)
        if timing is not None:
            # Each sample's time since the delay model's epoch, from offsets rather than GPS times.
            since_epoch = (float(time_array[0]) - timing.model.epoch) + np.arange(len(time_array)) * dt
    else:
        # Precomputed once for the exact frequency-domain shift.
        spectra = (np.fft.rfft(hp_vals), np.fft.rfft(hc_vals))
        frequencies = np.fft.rfftfreq(n_samples, d=dt)
        sky = (_gmst_accurate(reference_time), right_ascension, declination)

    strains: dict[str, TimeSeries] = {}
    for name, prefix in detectors:
        response = 0.0
        for term in _response_terms(prefix, None if arm_lengths is None else arm_lengths[name], hp.sample_rate):
            f_plus, f_cross = antenna_pattern(term.tensor, *sky, polarization_angle, xp=np)
            # Antenna pattern at the detector-time sample, i.e. the same time coordinate the output series
            # is labelled with. Evaluating it at t + tau would mix the detector and geocenter time
            # coordinates; LALSuite and the bilby-x-g frequency-domain implementation both use a single
            # consistent coordinate.
            site = time_delay_from_geocenter(term.location, *sky, xp=np)
            if earth_rotation:
                shift = -site / dt - lattice_shift
                if timing is not None:
                    shift = shift + timing.delay(since_epoch - site) / dt
                # Resample at the signal time with the shared band-limited kernel, gathered from a
                # zero-padded copy for the same reason as the device path -- taps reaching past either end
                # must read zero rather than clamping to and repeating the endpoint -- and with the same
                # padding, or the two paths disagree at the edges by the difference in padding alone.
                hp_shifted, hc_shifted = (
                    resample_uniform_sinc(series, index, fraction=factor * shift, taps=sinc_taps, beta=kaiser_beta)
                    for series in (hp_read, hc_read)
                )
                part = f_plus * hp_shifted + f_cross * hc_shifted
                if term.arm is not None:
                    # mu = cos(arm, source): the delay of a point one metre along the arm, in units of 1 / c.
                    alignment = -SPEED_OF_LIGHT_M_S * time_delay_from_geocenter(term.arm, *sky, xp=np)
                    part = finite_arm.apply_residual(
                        part,
                        alignment,
                        arm_length_m=term.length,
                        sample_rate=hp.sample_rate,
                        tapered=tapered,
                        taps=sinc_taps,
                        beta=kaiser_beta,
                    )
            else:
                delay = site + lattice_shift * dt
                if timing is not None:
                    # Held at the reference time, as the pattern and the site delay are on this branch.
                    delay = delay - float(timing.delay(np.array([reference_time - timing.model.epoch]))[0])
                # Exact FD phase shift: h(t-τ) <-> H(f)·exp(-2πifτ). Circular, which is negligible for
                # tapered polarizations (tested in test suite).
                transfer = np.exp(-2j * np.pi * frequencies * delay)
                if term.arm is not None:
                    alignment = -SPEED_OF_LIGHT_M_S * float(time_delay_from_geocenter(term.arm, *sky, xp=np))
                    transfer = transfer * finite_arm_residual(
                        frequencies * (term.length / SPEED_OF_LIGHT_M_S), alignment, xp=np
                    )
                hp_shifted, hc_shifted = (np.fft.irfft(spectrum * transfer, n=n_samples) for spectrum in spectra)
                part = f_plus * hp_shifted + f_cross * hc_shifted
            response = response + term.sign * part

        strains[name] = TimeSeries(
            response,
            t0=float(time_array[0]),
            sample_rate=hp.sample_rate,
            name=name,
        )

    return strains
