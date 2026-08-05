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

* **The reference epoch is fixed for the run, not derived per call.** Ripple's
  ``ref_time_ssb`` defaults to the SSB time of the first sample it is given. Called once per
  segment, that silently restarts the phase at every boundary. Measured with three ten-minute
  segments, the default reproduces the first segment exactly and then diverges by of order the
  signal amplitude itself; with the reference fixed, segmented and continuous generation agree
  *bit for bit*. Because the failure is invisible -- each segment on its own is a perfectly good
  CW signal -- :class:`ContinuousWaveSimulator` requires ``reference_time_ssb`` rather than
  defaulting it.
* **Polarizations are generated at the geocentre.** The detector delay and antenna pattern are
  then applied by :func:`~gwmock_signal.projection.network.project_polarizations_to_network`, the
  same route the compact-binary path uses. Generating per detector instead would mean applying
  ripple's own detector delay *and* this package's, double-counting it -- and would leave two
  implementations of the same geometry to drift apart.
* **The ephemeris is an explicit input.** ``read_ephemeris_file`` will download a named LALPulsar
  table and cache it, which makes the physics depend on a file nobody chose deliberately. Paths
  are required here so a run records what it used.

.. note::
   **The geocentre composition is validated against LAL; one residual is known.** Checked with
   ``lalpulsar.Barycenter`` (DE405, alpha=1.1, delta=0.3, GPS 1577491218), whose ``EmissionTime``
   separates the site term from the barycentric part:

   * ``erot`` at the geocentre is exactly ``0``, so generating there gives the pure SSB-to-Earth-
     centre delay and nothing of the detector's position leaks in.
   * ``deltaT(H1) - deltaT(geocentre)`` equals LAL's own ``erot(H1)`` to 5.5e-15 s, confirming that
     the site term is the *only* difference between the two -- which is exactly the split this
     module relies on. Nothing is double-counted or dropped at the seam.

   The residual: this package's geocentre-to-detector delay differs from LAL's ``erot`` by
   **8.7e-07 s** worst case over detectors, sky positions and epochs. What remains is nutation, the
   short-period part of a motion whose secular part
   :func:`~gwmock_signal.projection.sidereal.precess_to_epoch` now applies and LAL applies in full.
   The size fits: nutation's amplitude is ~17 arcseconds, which over an Earth radius is ~5e-07 s.

   It was **1.8e-04 s** before that rotation existed, from combining plain GMST -- which measures
   the Earth's rotation from the mean equinox *of date* -- with a J2000 source direction. Like the
   residual it replaced, this applies to every source type, not only continuous waves, but matters
   most here because a coherent search integrates for months. It is a near-constant timing offset,
   so most of it is absorbed into the initial phase; the part that is not scales with frequency,
   which at the current 8.7e-07 s reaches about 5e-03 rad at 1 kHz.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.projection.resampling import edge_padding
from gwmock_signal.simulator import GWSimulator

if TYPE_CHECKING:  # pragma: no cover - typing only
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
                a segment's cost here and this class already requires ripple, so JAX is present
                either way. It is an argument rather than a constant because "JAX imports" and
                "JAX runs" are different claims: device memory can be exhausted, a driver can be
                misconfigured, or a backend can have a defect, and in any of those the host path
                still works and the caller needs to be able to reach it.

                One limit comes with the default. The device path extrapolates sidereal time
                linearly across the span it is given and accepts up to 86400 s; a *single*
                segment longer than that is refused and needs ``"numpy"``. The span it measures
                is the padded one -- ``edge_padding`` adds samples at both ends, 0.11 s to 0.56 s
                depending on sample rate -- so the usable background is that much shorter. A run
                of any length made of ordinary segments is unaffected, since each re-anchors.

        Raises:
            ValueError: If ``reference_time_ssb`` or any spindown term is not finite, or
                ``projection_backend`` is not one of the two names.
        """
        if projection_backend not in {"numpy", "jax"}:
            raise ValueError(f"projection_backend must be 'numpy' or 'jax', got {projection_backend!r}.")
        self.projection_backend = projection_backend
        if projection_backend == "jax":
            # Enabled here rather than relied upon. The device projection refuses to run without
            # x64, and it has always happened to be on because importing ``ripplegw`` sets it --
            # which this class does, for the polarizations. Depending on an unrelated package's
            # import side effect for a correctness precondition is a latent break: if ripple
            # stopped doing it, every default-configured call would raise instead of quietly
            # degrading, but it would still be a needless failure. Idempotent, and the same value
            # ripple sets, so nothing changes for callers who already have it on.
            import jax  # noqa: PLC0415 — optional [jax] dep, and only needed for this backend

            jax.config.update("jax_enable_x64", True)
        if not np.isfinite(reference_time_ssb):
            raise ValueError("reference_time_ssb must be a finite GPS-scale time in seconds.")
        spindown_terms = tuple(float(term) for term in spindowns)
        if not all(np.isfinite(term) for term in spindown_terms):
            raise ValueError(f"spindowns must all be finite, got {spindown_terms}.")
        self.earth_ephemeris = str(earth_ephemeris)
        self.sun_ephemeris = str(sun_ephemeris)
        self.reference_time_ssb = float(reference_time_ssb)
        self.spindowns = spindown_terms
        self._ephemeris_tables: tuple[Any, Any] | None = None

    @property
    def required_params(self) -> frozenset[str]:
        """Return the parameter keys a continuous-wave source must supply."""
        return _REQUIRED_PARAMETERS

    def _load_ephemeris(self) -> tuple[Any, Any]:
        """Return the (earth, sun) ephemeris tables, reading them at most once per instance."""
        if self._ephemeris_tables is None:
            from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file  # noqa: PLC0415

            self._ephemeris_tables = (
                read_ephemeris_file(self.earth_ephemeris),
                read_ephemeris_file(self.sun_ephemeris),
            )
        return self._ephemeris_tables

    def _geocentre_polarizations(
        self,
        params: Mapping[str, Any],
        *,
        epoch: float,
        n_samples: int,
        sampling_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return plus and cross polarizations at the geocentre for one source.

        The epoch is split into an integer GPS second and a remainder folded into the sample
        offsets, because ripple takes ``start_gps`` as an integer and the phase is accumulated
        from ``dt_rel``. Rounding the fraction away would shift the whole segment; carrying it in
        the offsets keeps the split exact.
        """
        from ripplegw.waveforms.cw.PulsarSignal import (  # noqa: PLC0415
            generate_pulsar_polarizations,
        )

        earth, sun = self._load_ephemeris()
        start_gps = int(np.floor(epoch))
        offset = epoch - start_gps
        sample_times = offset + np.arange(n_samples, dtype=float) / sampling_frequency

        plus, cross = generate_pulsar_polarizations(
            dt_rel=sample_times,
            start_gps=start_gps,
            alpha=float(params["right_ascension"]),
            delta=float(params["declination"]),
            f0=float(params["frequency"]),
            phi0=float(params["initial_phase"]),
            aplus=float(params["amplitude_plus"]),
            across=float(params["amplitude_cross"]),
            # The geocentre. Ripple applies the barycentring delay for whatever location it is
            # given; asking for the Earth centre leaves the geocentre-to-detector leg to the
            # projection below, so that geometry lives in exactly one place.
            det_location_m=(0.0, 0.0, 0.0),
            earth_gps0=earth.gps0,
            earth_dt=earth.dt,
            earth_pos=earth.pos,
            earth_vel=earth.vel,
            earth_acc=earth.acc,
            sun_gps0=sun.gps0,
            sun_dt=sun.dt,
            sun_pos=sun.pos,
            sun_vel=sun.vel,
            sun_acc=sun.acc,
            fkdot=self.spindowns,
            ref_time_ssb=self.reference_time_ssb,
        )
        plus_values = np.asarray(plus, dtype=float)
        cross_values = np.asarray(cross, dtype=float)

        # Refuse a non-finite signal rather than writing it. The motivating case is released
        # ripplegw before 0.3.1, which could not generate at the geocentre -- `barycenter.py`
        # computed the detector latitude as `arccos(lz / rd)`, which is 0/0 when the location is the
        # Earth centre, exactly what this class asks for so the projection can own the
        # geocentre-to-detector leg. Fixed in GW-JAX-Team/ripple#141, released in 0.3.1, and the
        # `jax` extra now floors ripplegw there -- so this is no longer reachable without a
        # deliberate downgrade.
        #
        # The check stays regardless. The floor closes one cause; the other has nothing to do with
        # ripple and is still live -- an out-of-range spindown overflowing the phase. Reachable
        # only when the reference epoch is also far from the data, because the term grows as the
        # square of the time from it: measured, `f1 = 1e300` stays finite with the reference beside
        # the data and produces NaN with it at GPS 0. Extreme, but demonstrable rather than
        # asserted, and `_validate_source` does not bound magnitudes.
        #
        # The condition is the general one, not a test for that bug, so the message must not
        # assert the cause. A finite but extreme spindown overflows the phase and produces NaN on
        # a *fixed* ripple too -- `_validate_source` checks that inputs are finite, not that they
        # are in range -- and telling that caller to go and install a fix they already have would
        # send them the wrong way entirely.
        #
        # Scope: this covers the polarizations only. Anything the projection, the antenna pattern
        # or the background introduces downstream is not checked here and nothing else checks it
        # either.
        if not np.all(np.isfinite(plus_values)) or not np.all(np.isfinite(cross_values)):
            import ripplegw  # noqa: PLC0415

            raise RuntimeError(
                f"the continuous-wave polarizations are not all finite, so nothing was written. "
                f"Two causes are worth checking, in order. First, the source parameters: "
                f"`_validate_source` requires them finite but not bounded, so an extreme spindown "
                f"or amplitude, combined with a reference epoch far from the data, can overflow "
                f"the phase and yield NaN from a perfectly good library -- the spindown term grows "
                f"as the square of the time from `reference_time_ssb`. Second, an old ripplegw: before 0.3.1 it returned NaN for *every* "
                f"sample when generating at the geocentre, which is what this simulator asks for, "
                f"so a wholly-NaN array points there. The `jax` extra floors ripplegw at 0.3.1, so "
                f"reaching that needs a deliberate downgrade. "
                f"Installed ripplegw: {getattr(ripplegw, '__version__', 'unknown')}."
            )
        return plus_values, cross_values

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
            TypeError: If a background value is not a GWpy time series.
            ValueError: If the channels disagree, or the caller's rate differs from theirs.
        """
        reference = background[names[0]]
        if not hasattr(reference, "t0") or not hasattr(reference, "sample_rate"):
            raise TypeError(f"background values must be gwpy TimeSeries; {names[0]!r} is a {type(reference).__name__}.")
        epoch = float(reference.t0.value)
        n_samples = len(reference)
        rate = float(reference.sample_rate.value)

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
            if len(channel) != n_samples:
                raise ValueError(
                    f"background channels must share a length; {name!r} has {len(channel)} samples "
                    f"against {n_samples} for {names[0]!r}."
                )
            if float(channel.t0.value) != epoch:
                raise ValueError(
                    f"background channels must share an epoch; {name!r} starts at "
                    f"{float(channel.t0.value)!r} against {epoch!r} for {names[0]!r}."
                )
            if float(channel.sample_rate.value) != rate:
                raise ValueError(
                    f"background channels must share a sample rate; {name!r} is at "
                    f"{float(channel.sample_rate.value)!r} Hz against {rate!r} for {names[0]!r}."
                )
        return epoch, n_samples

    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[DetectorSpec],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
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
            earth_rotation: Passed through to the projection. ``True`` evaluates the antenna
                pattern across the segment, which is what a signal of this length needs.
            interpolate_if_offset: Unused; the polarizations are generated on the segment's own
                grid, so there is never an offset to interpolate away.

        Returns:
            ``DetectorStrainStack`` with the signal added to *background*.

        Raises:
            ValueError: If ``background`` is absent or empty, or ``sampling_frequency`` is not
                positive.
            KeyError: If a detector has no background channel.
            RuntimeError: If the generated polarizations are not all finite -- an out-of-range
                source parameter, or a ripplegw without GW-JAX-Team/ripple#141, which returns NaN
                at the geocentre. Refused rather than written; see ``_geocentre_polarizations``.
        """
        del minimum_frequency, interpolate_if_offset
        self._validate_params(params)
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")

        if not earth_rotation:
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
        epoch, n_samples = self._resolve_segment(background, names, sampling_frequency)

        # Generated with a margin at both ends and trimmed after projecting. The rotating
        # projection resamples at the time-dependent detector delay with a windowed sinc kernel,
        # and taps reaching past the end of the array read zero. For a transient that is right --
        # there is nothing beyond the buffer. For a continuous wave there is: the signal carries on
        # into the neighbouring segments, so zeros at the seam would make each segment disagree
        # with the same stretch generated as part of a longer run, by far more than rounding.
        margin = edge_padding(sampling_frequency)
        plus, cross = self._geocentre_polarizations(
            params,
            epoch=epoch - margin / sampling_frequency,
            n_samples=n_samples + 2 * margin,
            sampling_frequency=sampling_frequency,
        )
        padded_epoch = epoch - margin / sampling_frequency
        polarizations = {
            "plus": TimeSeries(plus, t0=padded_epoch, sample_rate=sampling_frequency, unit="strain"),
            "cross": TimeSeries(cross, t0=padded_epoch, sample_rate=sampling_frequency, unit="strain"),
        }
        projected = project_polarizations_to_network(
            polarizations,
            detector_names,
            right_ascension=float(params["right_ascension"]),
            declination=float(params["declination"]),
            polarization_angle=float(params.get("polarization_angle", 0.0)),
            earth_rotation=earth_rotation,
            # The device implementation of the same algorithm. Measured against the host path at
            # this class's own segment sizes, three detectors, 512 Hz: 2.0x over 256 s and 3.4x
            # over 1024 s, agreeing to 1.4e-11 of peak. Not a choice about accuracy -- the two
            # differ only by floating-point reassociation -- but the projection is 99% of a
            # continuous-wave segment's cost, so it is the only part worth moving.
            #
            # Defaults to the device path; see ``projection_backend`` for why it can be changed.
            backend=self.projection_backend,
        )

        # Rebuilt rather than added directly: the projection returns dimensionless series, and
        # adding those to a ``strain`` background raises a unit-conversion error. Same approach as
        # the stochastic simulator, which constructs its output series explicitly.
        strains = {
            name: background[name]
            + TimeSeries(
                np.asarray(projected[name], dtype=float)[margin : margin + n_samples],
                t0=epoch,
                sample_rate=sampling_frequency,
                unit="strain",
            )
            for name in names
        }
        return DetectorStrainStack.from_mapping(names, strains)
