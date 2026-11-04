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
"""Simulator for continuous waves from isolated spinning neutron stars.

A continuous wave is unlike every other source this package generates. It is not a transient
placed at a coalescence time but a signal that is *on* for the whole observation, which is
typically months. Callers therefore ask for it one analysis segment at a time, and the segments
have to join up: a coherent search integrates over the entire run, so a phase discontinuity at a
segment boundary destroys the signal it is looking for.

That constraint drives the design of this class:

* **The reference epoch is fixed for the run, not derived per call.** A reference taken from the
  SSB time of each segment's first sample, as ripple's generator defaults to, silently restarts
  the phase at every boundary. Measured with three ten-minute segments, that reproduces the first
  segment exactly and then diverges by of order the signal amplitude itself; with the reference
  fixed, segmented and continuous generation agree to 1e-9 of peak. Because the failure is invisible -- each segment on its own is a perfectly good
  CW signal -- :class:`ContinuousWaveSimulator` requires ``reference_time_ssb`` rather than
  defaulting it.
* **Polarizations are generated in SSB time, and every delay is applied by the projection.** The
  isolated-pulsar phase is evaluated on an even grid of solar-system-barycentre times; the
  projection maps each detector sample to its SSB time -- the site delay, then the solar-system
  delay of :mod:`~gwmock_signal.projection.solar_system` -- through the same code every source
  uses, with the same :class:`~gwmock_signal.projection.effects.ProjectionEffects`. One
  implementation of each delay, so none can drift from another.
* **The ephemeris is an explicit input.** A standard LALPulsar table name is downloaded and cached
  when not found locally, which makes the physics depend on a file nobody chose deliberately.
  Paths are required here so a run records what it used.

.. note::
   **Validated against LALPulsar; one residual is known.** The solar-system delay equals
   ``lalpulsar.Barycenter``'s ``deltaT`` for a detector at the geocentre to 1e-13 s, and the projected
   signal equals the pulsar phase evaluated at the site delay plus that ``deltaT`` to within the
   resampling kernel's error. The residual is in the site term: this package's geocentre-to-detector
   delay differs from LAL's ``erot`` by **8.7e-07 s** worst case over detectors, sky positions and
   epochs. What remains is nutation, the short-period part of a motion whose secular part
   :func:`~gwmock_signal.projection.sidereal.precess_to_epoch` applies and LAL applies in full. The
   size fits: nutation's amplitude is ~17 arcseconds, which over an Earth radius is ~5e-07 s. It is a
   near-constant timing offset, so most of it is absorbed into the initial phase; the part that is
   not scales with frequency, reaching about 5e-03 rad at 1 kHz.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_signal.injection.core import _add_strain
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.projection import solar_system
from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.network import _project, _resolve_effects, validate_projection_backend
from gwmock_signal.projection.resampling import DEFAULT_KAISER_BETA, DEFAULT_SINC_TAPS, edge_padding
from gwmock_signal.projection.solar_system import SolarSystemEphemeris
from gwmock_signal.simulator import GWSimulator
from gwmock_signal.timeseries import TimeSeries, as_timeseries

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gwpy.timeseries

    from gwmock_signal.stochastic.overlap import DetectorSpec

#: Parameters every continuous-wave source must supply.
#:
#: ``h0``/``cos_iota``/``psi`` are deliberately *not* here: the two polarization amplitudes are
#: taken directly, so this class does not encode a second opinion about how ``h0`` and the
#: inclination combine. A caller wanting that convention converts before calling.
_REQUIRED_PARAMETERS: frozenset[str] = frozenset(
    {
        "right_ascension",
        "declination",
        "frequency",
        "initial_phase",
        "amplitude_plus",
        "amplitude_cross",
    }
)


class ContinuousWaveSimulator(GWSimulator):
    """Generate detector strain for continuous waves from isolated pulsars.

    Args:
        earth_ephemeris: Path to a LALPulsar ``earth*`` ephemeris file.
        sun_ephemeris: Path to the matching ``sun*`` ephemeris file.
        reference_time_ssb: SSB reference epoch for the spin parameters, in seconds. **Required,
            and constant for a whole run.** See the module docstring: deriving it per segment is
            what silently breaks phase coherence, so there is no default to fall into.
        spindowns: Spindown terms ``(f1, f2, ...)`` in Hz/s, Hz/s^2, ... applied at
            ``reference_time_ssb``.

    Raises:
        ValueError: If ``reference_time_ssb`` is not finite.
    """

    def __init__(
        self,
        *,
        earth_ephemeris: str,
        sun_ephemeris: str,
        reference_time_ssb: float,
        spindowns: Sequence[float] = (),
        projection_backend: str = "jax",
    ) -> None:
        """Initialize the continuous-wave simulator.

        Args:
            earth_ephemeris: Path to the Earth ephemeris table.
            sun_ephemeris: Path to the Sun ephemeris table.
            reference_time_ssb: Epoch the source parameters refer to, at the solar-system
                barycentre. Required; see the class docstring for why it has no default.
            spindowns: Spindown terms ``f1, f2, ...`` in Hz/s, Hz/s^2, ...
            projection_backend: Which projection implementation to use, ``"jax"`` (the default)
                or ``"numpy"``. The device path is the default because the projection is ~99% of
                a segment's cost here. It is an argument rather than a constant because "JAX imports" and
                "JAX runs" are different claims: device memory can be exhausted, a driver can be
                misconfigured, or a backend can have a defect, and in any of those the host path
                still works and the caller needs to be able to reach it.

                One limit comes with the default. The device path extrapolates sidereal time
                linearly across the span it is given and refuses a span over 86400 s, which is a
                *single* segment and needs ``"numpy"`` to go beyond. The span it measures is the
                padded one, not the nominal segment. ``edge_padding`` adds samples at each end,
                about 0.05 s to 4.13 s per end depending on sample rate, so the usable segment --
                and therefore the longest nominal segment this backend accepts -- is that much
                shorter than 86400 s. A run of any length made of ordinary segments is unaffected,
                since each re-anchors.

        Raises:
            ValueError: If ``reference_time_ssb`` or any spindown term is not finite, or
                ``projection_backend`` is not one of the two names.
        """
        self.projection_backend = validate_projection_backend(projection_backend, parameter="projection_backend")
        if projection_backend == "jax":
            # Enabled here rather than relied upon: the device projection refuses to run without x64.
            # Idempotent, so nothing changes for callers who already have it on.
            import jax  # noqa: PLC0415 — optional [jax] dep, and only needed for this backend

            jax.config.update("jax_enable_x64", True)
        if not np.isfinite(reference_time_ssb):
            raise ValueError("reference_time_ssb must be a finite GPS-scale time in seconds.")
        spindown_terms = tuple(float(term) for term in spindowns)
        if not all(np.isfinite(term) for term in spindown_terms):
            raise ValueError(f"spindowns must all be finite, got {spindown_terms}.")
        self.earth_ephemeris = str(earth_ephemeris)
        self.sun_ephemeris = str(sun_ephemeris)
        self.ephemeris = SolarSystemEphemeris(self.earth_ephemeris, self.sun_ephemeris)
        self.reference_time_ssb = float(reference_time_ssb)
        self.spindowns = spindown_terms

    @property
    def required_params(self) -> frozenset[str]:
        """Return the parameter keys a continuous-wave source must supply."""
        return _REQUIRED_PARAMETERS

    def _ssb_polarizations(
        self,
        params: Mapping[str, Any],
        *,
        start_ssb: float,
        n_samples: int,
        sampling_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return plus and cross polarizations on an even grid of SSB times from ``start_ssb``.

        The phase is ``phi0 + 2 pi sum_k f_k tau^(k+1) / (k+1)!`` with ``tau`` the SSB time since
        :attr:`reference_time_ssb`, and the polarizations ``A+ cos(phase)`` and ``Ax sin(phase)``: the
        isolated-pulsar model of LALPulsar's ``XLALGeneratePulsarSignal`` before its barycentring, which
        projection now applies with every other source's (:mod:`~gwmock_signal.projection.solar_system`).

        Raises:
            RuntimeError: If the polarizations are not all finite, which an extreme spindown can cause
                far from the reference epoch: its term grows as a power of the time from it.
        """
        tau = (start_ssb - self.reference_time_ssb) + np.arange(n_samples, dtype=float) / sampling_frequency
        coefficients = (float(params["frequency"]), *self.spindowns)
        phase = np.zeros(n_samples)
        power, factorial = np.ones(n_samples), 1.0
        for order, coefficient in enumerate(coefficients, start=1):
            power = power * tau
            factorial *= order
            phase = phase + coefficient * power / factorial
        phase = float(params["initial_phase"]) + 2.0 * np.pi * phase
        plus = float(params["amplitude_plus"]) * np.cos(phase)
        cross = float(params["amplitude_cross"]) * np.sin(phase)
        if not (np.all(np.isfinite(plus)) and np.all(np.isfinite(cross))):
            raise RuntimeError(
                "The continuous-wave polarizations are not all finite, so nothing was written. Check the source "
                "parameters: they are required finite but not bounded, and a spindown term grows as a power of "
                "the time from reference_time_ssb, so an extreme spindown far from it overflows the phase."
            )
        return plus, cross

    def _barycentric_offset(
        self,
        effects: ProjectionEffects,
        params: Mapping[str, Any],
        start: float,
        n_samples: int,
        sampling_frequency: float,
    ) -> tuple[int, int]:
        """Return how many samples the SSB grid runs ahead of the detector grid, and how many more it needs.

        The offset is the smallest solar-system delay across the span, rounded down, less one sample; the
        extra length covers the delay's range, plus one sample. Both are zero without the delays.
        """
        if not effects.solar_system:
            return 0, 0
        end = start + (n_samples - 1) / sampling_frequency
        times = np.concatenate([np.arange(start, end, 60.0), [end]])
        delay, _ = solar_system.delay(
            times,
            float(params["right_ascension"]),
            float(params["declination"]),
            self.ephemeris,
            roemer=effects.roemer_delay,
            einstein=effects.einstein_delay,
            shapiro=effects.shapiro_delay,
        )
        offset = math.floor(float(np.min(delay)) * sampling_frequency) - 1
        return offset, math.ceil(float(np.max(delay)) * sampling_frequency) - offset + 1

    def _validate_source(self, params: Mapping[str, Any], sampling_frequency: float) -> None:
        """Reject source values that would produce a silently wrong or all-NaN signal.

        Args:
            params: Source parameters, already checked for presence by the base class.
            sampling_frequency: Sample rate in Hz, needed for the Nyquist check.

        Raises:
            ValueError: If any required value is not finite, or the frequency is non-positive or
                at/above Nyquist.
        """
        for key in sorted(_REQUIRED_PARAMETERS):
            value = float(params[key])
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite, got {value!r}.")
        frequency = float(params["frequency"])
        if frequency <= 0.0:
            raise ValueError(f"frequency must be positive, got {frequency!r} Hz.")
        if frequency >= 0.5 * sampling_frequency:
            raise ValueError(
                f"frequency {frequency!r} Hz is at or above the Nyquist frequency "
                f"{0.5 * sampling_frequency!r} Hz for this sample rate, so it would alias."
            )

    @staticmethod
    def _as_series(background: Mapping[str, Any], names: Sequence[str]) -> dict[str, TimeSeries]:
        """Return the background channels for ``names`` as the package's time series.

        Raises:
            TypeError: If a background value is not a time series.
        """
        channels = {}
        for name in names:
            try:
                channels[name] = as_timeseries(background[name])
            except TypeError as exc:
                raise TypeError(
                    f"background values must be gwmock_signal.TimeSeries or GWpy TimeSeries; {name!r} is a "
                    f"{type(background[name]).__name__}."
                ) from exc
        return channels

    @staticmethod
    def _resolve_segment(
        background: Mapping[str, TimeSeries], names: Sequence[str], sampling_frequency: float
    ) -> tuple[float, int]:
        """Return the (epoch, sample count) the background defines, checking every channel agrees.

        Args:
            background: Existing strain, one channel per detector.
            names: Detector names, in output order.
            sampling_frequency: The rate the caller asked to generate at.

        Returns:
            The segment's epoch in GPS seconds and its length in samples.

        Raises:
            TypeError: If a background value is not a time series.
            ValueError: If the channels disagree, or the caller's rate differs from theirs.
        """
        reference = background[names[0]]
        epoch = reference.t0
        n_samples = reference.n_samples
        rate = reference.sample_rate

        # The argument drives generation while the background defines the grid the result is added
        # to, so a disagreement produces a signal sampled at one rate and labelled at another --
        # added elementwise, with no complaint, and time-stretched by the ratio.
        if rate != sampling_frequency:
            raise ValueError(
                f"sampling_frequency is {sampling_frequency!r} Hz but the background is at "
                f"{rate!r} Hz. The signal would be generated on one time grid and added to "
                f"another, stretching it by a factor of {sampling_frequency / rate:.6g}."
            )

        # Every channel must describe the same stretch of time. The polarizations are generated
        # once, for one epoch and length, and added to all of them -- so a channel that disagreed
        # would silently receive a signal from the wrong interval rather than failing.
        for name in names[1:]:
            channel = background[name]
            if channel.n_samples != n_samples:
                raise ValueError(
                    f"background channels must share a length; {name!r} has {channel.n_samples} samples "
                    f"against {n_samples} for {names[0]!r}."
                )
            if channel.t0 != epoch:
                raise ValueError(
                    f"background channels must share an epoch; {name!r} starts at "
                    f"{channel.t0!r} against {epoch!r} for {names[0]!r}."
                )
            if channel.sample_rate != rate:
                raise ValueError(
                    f"background channels must share a sample rate; {name!r} is at "
                    f"{channel.sample_rate!r} Hz against {rate!r} for {names[0]!r}."
                )
        return epoch, n_samples

    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[DetectorSpec],
        background: Mapping[str, TimeSeries | gwpy.timeseries.TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        effects: ProjectionEffects | None = None,
        interpolate_if_offset: bool = True,
        earth_rotation: bool | None = None,
    ) -> DetectorStrainStack:
        """Generate continuous-wave strain for one source over the background's span.

        Args:
            params: Source parameters; see :data:`_REQUIRED_PARAMETERS`.
            detector_names: IFO codes or ``CustomDetector`` instances.
            background: Existing strain to add the signal to. **Required**, because a continuous
                wave has no duration of its own: the segment being generated is defined by the
                background's epoch and length.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Unused. A continuous wave is monochromatic apart from its spindown
                and Doppler modulation, so there is no low-frequency cutoff to apply; accepted to
                keep the source-agnostic signature.
            effects: Which physical effects the projection includes; see
                :class:`~gwmock_signal.projection.effects.ProjectionEffects`. ``None``, the default,
                means all of them. The source's parameters are defined at the solar-system barycentre, so
                the solar-system delays enter in full; without them the signal is generated as if the
                geocentre were the barycentre. ``earth_rotation`` must stay on.
            interpolate_if_offset: Unused; the polarizations are generated on the segment's own
                grid, so there is never an offset to interpolate away.
            earth_rotation: Deprecated; ``effects=ProjectionEffects(earth_rotation=...)``.

        Returns:
            ``DetectorStrainStack`` with the signal added to *background*.

        Raises:
            ValueError: If ``background`` is absent or empty, or ``sampling_frequency`` is not
                positive.
            KeyError: If a detector has no background channel.
            RuntimeError: If the generated polarizations are not all finite -- an out-of-range
                source parameter. Refused rather than written; see ``_ssb_polarizations``.
        """
        del minimum_frequency, interpolate_if_offset
        effects = _resolve_effects(effects, earth_rotation=earth_rotation, caller="ContinuousWaveSimulator.simulate")
        self._validate_params(params)
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")

        if not effects.earth_rotation:
            raise ValueError(
                "earth_rotation=False is not available for continuous waves. That branch holds the "
                "antenna pattern fixed at the midpoint of the span, which is an approximation for "
                "signals short enough that Earth barely turns across them -- a continuous wave runs "
                "for months. It is refused rather than warned about because the output would look "
                "entirely normal while being wrong, and because the error would depend on how the "
                "run happened to be split into segments."
            )
        self._validate_source(params, sampling_frequency)

        if not background:
            raise ValueError(
                "ContinuousWaveSimulator requires a background: a continuous wave has no duration "
                "of its own, so the segment's epoch and length come from the data it is added to."
            )

        names = [d if isinstance(d, str) else d.name for d in detector_names]
        if not names:
            raise ValueError("detector_names must not be empty.")
        for name in names:
            if name not in background:
                raise KeyError(f"Missing background for detector {name!r}.")

        # Every channel must describe the same stretch of time. The polarizations are generated
        # once, for one epoch and length, and added to all of them -- so a channel that disagreed
        # would silently receive a signal from the wrong interval rather than failing.
        background = self._as_series(background, names)
        epoch, n_samples = self._resolve_segment(background, names, sampling_frequency)

        # Generated with a margin at both ends and trimmed after projecting. The rotating
        # projection resamples at the time-dependent detector delay with a windowed sinc kernel,
        # and taps reaching past the end of the array read zero. For a transient that is right --
        # there is nothing beyond the buffer. For a continuous wave there is: the signal carries on
        # into the neighbouring segments, so zeros at the seam would make each segment disagree
        # with the same stretch generated as part of a longer run, by far more than rounding.
        margin = edge_padding(sampling_frequency)
        padded_epoch = epoch - margin / sampling_frequency
        n_padded = n_samples + 2 * margin
        # The polarizations are generated in SSB time, which runs ahead of the detector's by the
        # solar-system delay D: up to 500 s, changing by up to 1e-4 s per second. They are labelled on the
        # detector grid and offset from it by a whole number of samples, the smallest D across the segment,
        # so the projection's positions stay small; and lengthened by D's range, so every sample the kept
        # span reads exists.
        offset, extra = self._barycentric_offset(effects, params, padded_epoch, n_padded, sampling_frequency)
        plus, cross = self._ssb_polarizations(
            params,
            start_ssb=padded_epoch + offset / sampling_frequency,
            n_samples=n_padded + extra,
            sampling_frequency=sampling_frequency,
        )
        polarizations = {
            "plus": TimeSeries(plus, t0=padded_epoch, sample_rate=sampling_frequency, unit="strain"),
            "cross": TimeSeries(cross, t0=padded_epoch, sample_rate=sampling_frequency, unit="strain"),
        }
        projected = _project(
            polarizations,
            detector_names,
            right_ascension=float(params["right_ascension"]),
            declination=float(params["declination"]),
            polarization_angle=float(params.get("polarization_angle", 0.0)),
            effects=effects,
            # The device implementation of the same algorithm. Measured against the host path at
            # this class's own segment sizes, three detectors, 512 Hz: 2.0x over 256 s and 3.4x
            # over 1024 s, agreeing to 1.4e-11 of peak. Not a choice about accuracy -- the two
            # differ only by floating-point reassociation -- but the projection is 99% of a
            # continuous-wave segment's cost, so it is the only part worth moving.
            #
            # Defaults to the device path; see ``projection_backend`` for why it can be changed.
            backend=self.projection_backend,
            sinc_taps=DEFAULT_SINC_TAPS,
            kaiser_beta=DEFAULT_KAISER_BETA,
            # The margin above already gives the delay somewhere to read from, and the trim below
            # assumes a series exactly as long as the polarizations. A transient's projection is
            # padded so delayed signal is not lost past the buffer; a continuous wave's signal does
            # not end at the buffer, so here that padding would only be cut off again.
            keep_delayed_samples=False,
            ephemeris=self.ephemeris,
            barycentric_offset_samples=offset,
        )

        # The projection returns dimensionless series; the trimmed signal is labelled ``strain``, as
        # it is, and added to a background that must be strain too.
        strains = {
            name: _add_strain(
                background[name], np.asarray(projected[name].data, dtype=float)[margin : margin + n_samples]
            )
            for name in names
        }
        return DetectorStrainStack.from_mapping(names, strains)
