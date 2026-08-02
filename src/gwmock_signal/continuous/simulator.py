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

.. warning::
   **The geocentre composition has not been validated against an external reference.** Ripple
   applies the barycentring chain from the SSB to the location it is given, and the projection then
   adds the geocentre-to-detector leg; review agreed the split is correct in principle and avoids
   double-counting. But every test here compares this implementation against itself, so a
   convention error shared by both halves would pass all of them and produce a plausible signal
   with a systematically wrong timing model.

   Settling it means comparing against LALPulsar's ``SimulateExactPulsarSignal`` for a real
   detector. That is not currently reachable from Python: the signature names a
   ``PulsarSignalParams`` struct that the SWIG binding does not expose as a constructible type.
   Until that comparison exists, treat the absolute timing as unverified -- the *relative* property
   the tests do establish is that segments join up coherently.
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
    ) -> None:
        """Initialize the continuous-wave simulator."""
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
        return np.asarray(plus, dtype=float), np.asarray(cross, dtype=float)

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
        reference_channel = background[names[0]]
        epoch = float(reference_channel.t0.value)
        n_samples = len(reference_channel)
        rate = float(reference_channel.sample_rate.value)
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
