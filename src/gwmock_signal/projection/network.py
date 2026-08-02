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

import logging
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from astropy import constants
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock_signal.projection.geometry import DetectorSpec, reconstructed_geometry, resolve_detectors
from gwmock_signal.projection.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    edge_padding,
    require_terrestrial_location,
    resample_uniform_sinc,
)
from gwmock_signal.projection.sidereal import gmst_rad_astropy

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


def _validate_polarizations(polarizations: Mapping[str, GWpyTimeSeries]) -> tuple[GWpyTimeSeries, GWpyTimeSeries]:
    """Validate ``plus``/``cross`` GWpy series share one time grid."""
    if not isinstance(polarizations, Mapping):
        raise TypeError("polarizations must be a mapping with 'plus' and 'cross' keys.")
    if "plus" not in polarizations or "cross" not in polarizations:
        raise ValueError("polarizations must contain both 'plus' and 'cross' keys.")
    hp = polarizations["plus"]
    hc = polarizations["cross"]
    if not isinstance(hp, GWpyTimeSeries) or not isinstance(hc, GWpyTimeSeries):
        raise TypeError("polarizations['plus'] and polarizations['cross'] must be gwpy.timeseries.TimeSeries.")
    if len(hp) != len(hc):
        raise ValueError(f"plus and cross must have the same number of samples; got {len(hp)} and {len(hc)}.")
    if hp.sample_rate != hc.sample_rate:
        raise ValueError("plus and cross must have the same sample rate.")
    hp_times = np.asarray(hp.times.value, dtype=float)
    hc_times = np.asarray(hc.times.value, dtype=float)
    dt = float(hp.dt.value)
    if not np.allclose(hp_times, hc_times, rtol=0.0, atol=max(np.finfo(float).eps, 0.5 * dt)):
        raise ValueError("plus and cross must share the same time samples (same t0, dt, and length).")
    return hp, hc


def _gmst_accurate(t_gps: float) -> float:
    """Return Greenwich mean sidereal time in radians using Astropy."""
    return float(gmst_rad_astropy(float(t_gps)))


def _gmst_accurate_array(t_gps: np.ndarray) -> np.ndarray:
    """Return GMST in radians for an array of GPS times in one vectorized call."""
    return gmst_rad_astropy(t_gps)


def _time_delay_from_earth_center_lal(
    prefix: str,
    *,
    right_ascension: float,
    declination: float,
    t_gps: float,
) -> float:
    """Return the geocenter time delay for one reconstructed detector geometry."""
    _, location = reconstructed_geometry(prefix)
    gha = _gmst_accurate(t_gps) - right_ascension
    cosdec = np.cos(declination)
    propagation_direction = np.array(
        [
            cosdec * np.cos(gha),
            -cosdec * np.sin(gha),
            np.sin(declination),
        ],
        dtype=float,
    )
    earth_center = np.array([0, 0, 0])
    baseline = earth_center - location
    return float(np.dot(baseline, propagation_direction) / constants.c.value)


def _antenna_pattern_lal(
    prefix: str,
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    t_gps: float,
) -> tuple[float, float]:
    """Return tensor antenna-pattern factors for one reconstructed detector geometry."""
    response, _ = reconstructed_geometry(prefix)
    gha = _gmst_accurate(t_gps) - right_ascension
    cosgha = np.cos(gha)
    singha = np.sin(gha)
    cosdec = np.cos(declination)
    sindec = np.sin(declination)
    cospsi = np.cos(polarization_angle)
    sinpsi = np.sin(polarization_angle)

    x = np.array(
        [
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec,
        ],
        dtype=float,
    )
    dx = response.dot(x)

    y = np.array(
        [
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec,
        ],
        dtype=float,
    )
    dy = response.dot(y)

    return float(np.sum(x * dx - y * dy)), float(np.sum(x * dy + y * dx))


def _make_detectors(detector_specs: Sequence[DetectorSpec]) -> list[tuple[str, str]]:
    """Resolve detector names to one LAL lookup key per output channel.

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


def _project_rotating_on_device(  # noqa: PLR0913
    hp: GWpyTimeSeries,
    hc: GWpyTimeSeries,
    detectors: Sequence[tuple[str, str]],
    *,
    time_array: np.ndarray,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    sinc_taps: int,
    kaiser_beta: float,
) -> dict[str, GWpyTimeSeries]:
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

    from gwmock_signal.projection.jax_projection import (  # noqa: PLC0415
        project_polarizations_td_rotating,
    )
    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate  # noqa: PLC0415

    # Refused rather than allowed to degrade. Without x64 every `jnp.asarray(..., float64)` in
    # the device path truncates to float32 and the projection still returns a plausible-looking
    # series -- measured 2.7e-03 of peak over 256 s and 1.8e-02 over 1024 s against this
    # function's own host path, growing with the span because GPS times lose precision first.
    # A warning would not do: nothing downstream can tell such a series from a correct one.
    # Importing `ripplegw` happens to enable x64, which is why the compact-binary and
    # continuous-wave paths have never hit this; that is an accident of an unrelated import.
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "backend='jax' requires JAX in 64-bit mode; call "
            'jax.config.update("jax_enable_x64", True) before projecting. In 32-bit mode the '
            "delays and sidereal angles lose the precision this projection depends on and the "
            "result is wrong by of order a percent of peak while still looking like strain."
        )

    hp_vals = np.asarray(hp.value, dtype=float)
    hc_vals = np.asarray(hc.value, dtype=float)
    sampling_frequency = float(hp.sample_rate.value)
    n_samples = len(hp_vals)
    anchors, rate = gmst_anchor_and_rate(float(time_array[0]))

    strains: dict[str, GWpyTimeSeries] = {}
    for name, prefix in detectors:
        response, location = reconstructed_geometry(prefix)
        # Same precondition the host path asserts, and for the same reason: the padding width
        # assumes a terrestrial delay bound. Checked here too, because on the device `location`
        # is a traced argument and the check could not run there.
        require_terrestrial_location(location, name=f"location of {prefix}")
        projected = project_polarizations_td_rotating(
            hp_vals,
            hc_vals,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
            gmst_start=float(np.atleast_1d(anchors)[0]),
            gmst_rate=float(rate),
            sinc_taps=sinc_taps,
            kaiser_beta=kaiser_beta,
        )
        strains[name] = GWpyTimeSeries(
            np.asarray(projected, dtype=float),
            t0=float(time_array[0]),
            sample_rate=hp.sample_rate,
            name=name,
        )
    return strains


# PLR0915: this function was already at the 50-statement limit before the one-line call to
# `_warn_if_constant_pattern_is_stretched` below. Splitting it further would mean reshaping
# the two projection branches that the compact-binary path depends on, which is not worth
# the risk for a diagnostic; the warning body itself already lives in its own function.
def project_polarizations_to_network(  # noqa: PLR0913, PLR0915
    polarizations: Mapping[str, GWpyTimeSeries],
    detector_names: Sequence[DetectorSpec],
    *,
    right_ascension: float,
    declination: float,
    polarization_angle: float,
    earth_rotation: bool = True,
    backend: str = "numpy",
    sinc_taps: int = DEFAULT_SINC_TAPS,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
) -> dict[str, GWpyTimeSeries]:
    """Project tensor plus/cross strains onto detectors using detector geometry.

    Built-in and custom detector codes are resolved through the LAL cached
    detector registry. For ``earth_rotation=False``, the constant
    geocenter->detector delay is applied via an exact frequency-domain phase
    shift (``h(t-tau) <-> H(f)*exp(-2*pi*i*f*tau)``), which is lossless at all
    frequencies. For ``earth_rotation=True``, the polarizations are resampled at the
    time-dependent delayed times with the Kaiser-windowed sinc kernel in
    :mod:`gwmock_signal.projection.resampling`, gathered from a zero-padded copy so kernel
    taps reaching past either end read zero rather than repeating the endpoint.

    Args:
        polarizations: Mapping containing ``plus`` and ``cross`` GWpy time series
            on a common grid.
        detector_names: Sequence of IFO codes (e.g. ``H1``, ``L1``, ``V1``) or
            :class:`~gwmock_signal.detector.CustomDetector` instances, or a mix
            of both.
        right_ascension: Source right ascension in radians.
        declination: Source declination in radians.
        polarization_angle: Polarization angle psi in radians (tensor modes).
        earth_rotation: If ``True``, evaluate antenna patterns at time-dependent
            GPS times (recommended for longer signals). If ``False``, use a single
            reference time at the segment midpoint for patterns and delays.
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

    Returns:
            Mapping from each detector name to the projected strain as a GWpy time
            series (same length and sample rate as the inputs).

    Raises:
        TypeError: If ``polarizations`` is not a mapping of GWpy series as required.
        ValueError: If keys are missing, time grids disagree, a detector name is not
            recognized, ``backend`` is unknown, or ``backend="jax"`` is combined with
            ``earth_rotation=False``.
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}.")
    if backend == "jax" and not earth_rotation:
        # The constant-pattern branch is a frequency-domain phase shift, and no device
        # counterpart exists. Refused rather than silently served from the host path, which
        # would report a backend that did not run.
        raise ValueError(
            "backend='jax' is only available with earth_rotation=True. The constant-pattern "
            "branch applies one frequency-domain phase shift for the whole span and has no "
            "device implementation; it is also cheap, being the branch that skips the resampler."
        )
    hp, hc = _validate_polarizations(polarizations)
    normalized_names = [d if isinstance(d, str) else d.name for d in detector_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("detector_names must not contain duplicates.")
    detectors = _make_detectors(list(detector_names))

    time_array = cast(np.ndarray, hp.times.to_value())
    reference_time = float(0.5 * (time_array[0] + time_array[-1]))

    _warn_if_constant_pattern_is_stretched(time_array, earth_rotation=earth_rotation)
    hp_vals = hp.to_value()
    hc_vals = hc.to_value()

    # Precomputed once for the exact FD phase-shift (earth_rotation=False path).
    n_samples = len(hp_vals)
    dt = float(hp.dt.value)
    rfft_hp = np.fft.rfft(hp_vals)
    rfft_hc = np.fft.rfft(hc_vals)
    freqs_fd = np.fft.rfftfreq(n_samples, d=dt)

    strains: dict[str, GWpyTimeSeries] = {}

    cosdec = np.cos(declination)
    sindec = np.sin(declination)
    cospsi = np.cos(polarization_angle)
    sinpsi = np.sin(polarization_angle)
    gmst_array = _gmst_accurate_array(time_array)
    gha_array = gmst_array - right_ascension
    cosgha = np.cos(gha_array)
    singha = np.sin(gha_array)

    if backend == "jax":
        return _project_rotating_on_device(
            hp,
            hc,
            detectors,
            time_array=time_array,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
            sinc_taps=sinc_taps,
            kaiser_beta=kaiser_beta,
        )

    for name, prefix in detectors:
        if earth_rotation:
            response, location = reconstructed_geometry(prefix)

            # Vectorized time delay: time_delay = -location · prop_dir / c
            prop_dir = np.stack([cosdec * cosgha, -cosdec * singha, np.full(len(time_array), sindec)], axis=-1)
            time_delays = -np.dot(prop_dir, location) / constants.c.value

            # Antenna pattern at the detector-time sample, i.e. the same time coordinate
            # the output series is labelled with. Evaluating it at t + tau would mix the
            # detector and geocenter time coordinates; LALSuite and the bilby-x-g
            # frequency-domain implementation both use a single consistent coordinate.
            gha_a = gmst_array - right_ascension
            cosgha_a = np.cos(gha_a)
            singha_a = np.sin(gha_a)

            # Shape (N, 3) — polarization basis vectors
            x_vec = np.stack(
                [
                    -cospsi * singha_a - sinpsi * cosgha_a * sindec,
                    -cospsi * cosgha_a + sinpsi * singha_a * sindec,
                    np.full(len(time_array), sinpsi * cosdec),
                ],
                axis=-1,
            )
            y_vec = np.stack(
                [
                    sinpsi * singha_a - cospsi * cosgha_a * sindec,
                    sinpsi * cosgha_a + cospsi * singha_a * sindec,
                    np.full(len(time_array), cospsi * cosdec),
                ],
                axis=-1,
            )

            # dx[n] = response @ x_vec[n], using row-vector form: x_vec @ response.T
            dx = x_vec @ response.T
            dy = y_vec @ response.T
            fp_vals = np.sum(x_vec * dx - y_vec * dy, axis=-1)
            fc_vals = np.sum(x_vec * dy + y_vec * dx, axis=-1)

            # Resample at t - tau(t) with the shared band-limited kernel. Expressed as a
            # fractional sample index because the input grid is uniform. Gathered from a
            # zero-padded copy for the same reason as the device path -- taps reaching past
            # either end must read zero rather than clamping to and repeating the endpoint --
            # and with the same padding, or the two paths disagree at the edges by the
            # difference in padding alone.
            require_terrestrial_location(location, name=f"location of {prefix}")
            pad = edge_padding(float(hp.sample_rate.value), sinc_taps, kaiser_beta)
            index = pad + np.arange(len(time_array), dtype=float) - time_delays / dt
            hp_shifted = resample_uniform_sinc(np.pad(hp_vals, (pad, pad)), index, taps=sinc_taps, beta=kaiser_beta)
            hc_shifted = resample_uniform_sinc(np.pad(hc_vals, (pad, pad)), index, taps=sinc_taps, beta=kaiser_beta)
        else:
            time_delay = _time_delay_from_earth_center_lal(
                prefix,
                right_ascension=right_ascension,
                declination=declination,
                t_gps=reference_time,
            )
            fp_vals, fc_vals = _antenna_pattern_lal(
                prefix,
                right_ascension=right_ascension,
                declination=declination,
                polarization_angle=polarization_angle,
                t_gps=reference_time,
            )
            # Exact FD phase shift: h(t-τ) <-> H(f)·exp(-2πifτ)
            # Circular-wrap is negligible for tapered polarizations (tested in test suite).
            phase = np.exp(-2j * np.pi * freqs_fd * time_delay)
            hp_shifted = np.fft.irfft(rfft_hp * phase, n=n_samples)
            hc_shifted = np.fft.irfft(rfft_hc * phase, n=n_samples)

        response = fp_vals * hp_shifted + fc_vals * hc_shifted

        strains[name] = GWpyTimeSeries(
            response,
            t0=float(time_array[0]),
            sample_rate=hp.sample_rate,
            name=name,
        )

    return strains
