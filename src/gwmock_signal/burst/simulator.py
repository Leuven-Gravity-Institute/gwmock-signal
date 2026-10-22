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
"""Simulator for unmodelled gravitational-wave bursts."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gwmock_data.resampling import resample_uniform_sinc

from gwmock_signal.burst.waveforms import PARAMETRIC_MODELS, read_waveform_file
from gwmock_signal.simulator import TransientSimulator, _projected_duration
from gwmock_signal.timeseries import TimeSeries

#: Keys every burst event carries, whatever its model. The model-specific ones are checked when the
#: model is generated, because they differ between families.
_REQUIRED: frozenset[str] = frozenset(
    {"burst_model", "peak_time", "right_ascension", "declination", "polarization_angle"}
)

#: How many events' buffer extents a simulator remembers. A caller placing an event asks for its
#: start and its end, often more than once, before generating it; remembering the last few
#: answers makes every repeat free without holding on to a catalogue's worth of entries.
_EXTENT_CACHE_SIZE = 64


def _finite_time(params: Mapping[str, Any], key: str) -> float:
    """Return ``params[key]`` as a float, refusing a missing, non-numeric or non-finite value."""
    value = params.get(key)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number, got {value!r}.") from error
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number, got {value!r}.")
    return number


def _placement_time(params: Mapping[str, Any]) -> float:
    """Return the instant a segmenting caller places the event by: ``coa_time``, else ``peak_time``.

    ``peak_time`` is required either way: it alone positions the waveform, and a ``coa_time`` may
    lie anywhere inside the buffer, so it cannot stand in for the peak. ``None`` counts as absent,
    as it does for a population that leaves unused columns empty.

    Raises:
        ValueError: If ``peak_time`` is missing or not a finite number, or ``coa_time`` is given
            and is not a finite number.
    """
    peak_time = _finite_time(params, "peak_time")
    return peak_time if params.get("coa_time") is None else _finite_time(params, "coa_time")


def _span_key(params: Mapping[str, Any], sampling_frequency: float) -> Any:
    """Return the cache key of an event's polarization span, or ``None`` when it cannot be hashed.

    ``coa_time`` is left out: generation never reads it, so every placement of one event shares
    one entry.
    """
    try:
        key = (float(sampling_frequency), tuple(sorted((k, v) for k, v in params.items() if k != "coa_time")))
        hash(key)
    except (TypeError, ValueError):
        # An unhashable or unorderable parameter: measure without remembering.
        return None
    return key


#: Float64 steps, of the largest time involved, by which a placement may round outside the output
#: and still count as on its edge; see :func:`_placement_extent`.
_ROUNDING_STEPS = 16

#: Fraction of a sample interval the edge slack never exceeds, so a placement a whole sample outside
#: is refused at any rate; see :func:`_placement_extent`.
_EDGE_SLACK_SAMPLES = 0.5

#: Fewest float64 steps of the placement's GPS time a sample interval must span for an edge placement
#: to be judged at all: below it, rounding alone can move an exact edge by half a sample.
_MIN_STEPS_PER_SAMPLE = 4


def _placement_extent(params: Mapping[str, Any], span: tuple[float, float, float]) -> tuple[float, float]:
    """Return ``(pre, post)`` of the projected output around the placement time, or refuse it.

    The single check behind the duration queries, :meth:`BurstSimulator.generate_polarizations`
    and so :meth:`BurstSimulator.simulate`: they refuse exactly the same placements, with the same
    message.

    Args:
        params: One burst event.
        span: ``(start, end, sample_rate)`` of its polarization buffer.

    Raises:
        ValueError: If ``peak_time`` or a given ``coa_time`` is not a finite number, or the placement
            time lies before the projected output's first sample or after its end.
    """
    placement = _placement_time(params)
    start, end, rate = span
    pre = _projected_duration(placement - start, rate)
    post = _projected_duration(end - placement, rate)
    # A placement on the output's own first sample, or one sample past its last, is valid, but a
    # caller computes those instants with its own float arithmetic, and so does this check: at GPS
    # times of 1e9 to 2e9 s one float64 step is 1.2e-7 to 2.4e-7 s, and an exact edge comes out up to
    # about 0.5 step outside (measured at 4000, 6000 and 16000 Hz). Differences of that size are
    # rounding, not a placement outside the output, and are clamped to zero. The slack is
    # _ROUNDING_STEPS steps of the largest time involved, so it stays a few float steps however large
    # the times are, but never more than _EDGE_SLACK_SAMPLES of a sample: at a high rate 16 steps would
    # otherwise span several samples (3.8e-6 s at 2e9 s is almost four samples at 1 MHz), and a
    # placement that far outside would be accepted. With the cap, anything a whole sample outside is
    # refused at every rate. The two bounds only conflict when a sample spans so few float steps that
    # rounding alone could move an exact edge by half a sample; such a rate cannot place by GPS time to
    # sub-sample precision, and is refused below -- under _MIN_STEPS_PER_SAMPLE steps per sample, which
    # at 2e9 s is any rate above about 1.05 MHz, and at 1e9 s above about 2.1 MHz.
    step = math.ulp(max(abs(placement), abs(start), abs(end)))
    if 1.0 / rate < _MIN_STEPS_PER_SAMPLE * step:
        raise ValueError(
            f"At {rate:g} Hz one sample ({1.0 / rate:.3g} s) spans fewer than {_MIN_STEPS_PER_SAMPLE} float64 "
            f"steps ({step:.3g} s each) of the GPS time {placement!r}, so a placement cannot be resolved to "
            "within a sample; use a lower sampling rate or times measured from a nearer epoch."
        )
    slack = min(_ROUNDING_STEPS * step, _EDGE_SLACK_SAMPLES / rate)
    if -slack <= pre < 0:
        pre = 0.0
    if -slack <= post < 0:
        post = 0.0
    if pre < 0 or post < 0:
        key = "coa_time" if params.get("coa_time") is not None else "peak_time"
        raise ValueError(
            f"{key}={placement!r} lies outside this burst's output, which runs from {placement - pre!r} "
            f"to {placement + post!r} s (peak_time={params.get('peak_time')!r}); no segment placed by it "
            "could hold the burst. Set coa_time inside the output -- coa_time = peak_time is the simple "
            "choice -- or omit it."
        )
    return pre, post


InclinationDependence = Literal["none", "axisymmetric", "rotating"]


#: Largest fraction of a waveform file's power allowed above the new Nyquist frequency when it is
#: downsampled, since the shared kernel does not low-pass.
_ALIAS_TOLERANCE = 1e-12


def _resample(values: np.ndarray, rate: float, new_rate: float) -> np.ndarray:
    """Resample a waveform file's samples with the shared band-limited kernel.

    ``resample_uniform_sinc`` interpolates; it does not low-pass. A file being
    downsampled must therefore already be band-limited below the new Nyquist
    frequency, and one that is not is refused rather than aliased. The check
    uses a Hann-windowed spectrum, whose skirt can also refuse a strong tone
    within a few frequency bins below the new Nyquist frequency.

    Args:
        values: Samples at ``rate``.
        rate: Sample rate of ``values`` in Hz.
        new_rate: Sample rate to return, in Hz.

    Returns:
        The samples at ``new_rate`` over the same span.

    Raises:
        ValueError: If downsampling and more than ``_ALIAS_TOLERANCE`` of the
            waveform's power lies above the new Nyquist frequency.
    """
    if new_rate < rate:
        # A Hann window keeps the periodogram's leakage (about 1e-5 for an untapered tone) far below
        # the tolerance, so only content that is really above the new Nyquist frequency is counted.
        power = np.abs(np.fft.rfft(values * np.hanning(values.shape[0]))) ** 2
        above = power[np.fft.rfftfreq(values.shape[0], 1.0 / rate) > new_rate / 2.0].sum()
        if above > _ALIAS_TOLERANCE * power.sum():
            raise ValueError(
                f"The waveform file carries {above / power.sum():.2e} of its power above {new_rate / 2.0:g} Hz, "
                f"the Nyquist frequency of {new_rate:g} Hz; low-pass it before resampling."
            )
    n_new = max(1, int(np.floor((values.shape[0] - 1) * new_rate / rate)) + 1)
    return resample_uniform_sinc(values, np.arange(n_new) * (rate / new_rate))


@dataclass(frozen=True)
class WaveformFile:
    """A numerical burst waveform on disk and how to place it.

    Attributes:
        path: Text file of ``t h_+ h_x`` columns (see
            :func:`~gwmock_signal.burst.waveforms.read_waveform_file`).
        reference_distance: Distance, in Mpc, at which the file's strains are given.
        inclination_dependence: How the strain varies with the event's ``inclination``:

            * ``"none"`` -- used as written, whatever the inclination;
            * ``"axisymmetric"`` -- an axisymmetric (``m = 0``) quadrupole, given edge-on:
              ``h_+`` is scaled by ``sin^2(iota)`` and ``h_x`` is zero;
            * ``"rotating"`` -- a rotating (``m = +-2``) quadrupole, given face-on: ``h_+`` is
              scaled by ``(1 + cos^2 iota) / 2`` and ``h_x`` by ``cos(iota)``.

    Raises:
        ValueError: If ``reference_distance`` is not finite and positive, or
            ``inclination_dependence`` is not one of the three above.
    """

    path: str | Path
    reference_distance: float
    inclination_dependence: InclinationDependence = "none"

    def __post_init__(self) -> None:
        """Refuse a distance or an inclination dependence the simulator could not apply."""
        if not (math.isfinite(self.reference_distance) and self.reference_distance > 0):
            raise ValueError(f"reference_distance must be finite and positive (Mpc), got {self.reference_distance!r}.")
        if self.inclination_dependence not in _INCLINATION_FACTORS:
            raise ValueError(
                f"inclination_dependence must be one of {sorted(_INCLINATION_FACTORS)}, "
                f"got {self.inclination_dependence!r}."
            )


#: ``(h_+ factor, h_x factor)`` as functions of the inclination, for each inclination dependence.
_INCLINATION_FACTORS = {
    "none": lambda iota: (1.0, 1.0),
    "axisymmetric": lambda iota: (math.sin(iota) ** 2, 0.0),
    "rotating": lambda iota: ((1.0 + math.cos(iota) ** 2) / 2.0, math.cos(iota)),
}


class BurstSimulator(TransientSimulator):
    """Burst injections: the ad hoc LVK families and numerical waveforms read from file.

    Each event names its morphology in ``burst_model``, so one simulator serves a mixed injection
    set. The event keys follow the per-event record of established burst injection pipelines --
    peak time, distance, sky position, polarization angle, inclination, model -- under this
    package's canonical names:

    ========================  ====================================================================
    ``burst_model``           ``"sine_gaussian"``, ``"gaussian"``, ``"white_noise_burst"``, or a
                              name given in ``waveform_files``
    ``peak_time``             GPS time of the waveform's centre at the geocentre, s
    ``coa_time``              optional placement time for segmented output, s; see below
    ``right_ascension``       rad
    ``declination``           rad
    ``polarization_angle``    rad
    ``hrss``                  ad hoc families: ``sqrt(int (h_+^2 + h_x^2) dt)`` at the Earth
    ``distance``              file waveforms: source distance, Mpc
    ``inclination``           rad; ad hoc families that take it, and file waveforms whose
                              ``inclination_dependence`` is not ``"none"``
    ========================  ====================================================================

    plus the family's own shape parameters, listed on each function in
    :mod:`gwmock_signal.burst.waveforms`. Keys a model does not use are ignored, so an event may
    carry labels such as which injection set it belongs to.

    A file waveform is scaled by its ``reference_distance / distance`` and by the inclination
    factors of its :class:`WaveformFile`.

    **Two times, two meanings.** ``peak_time`` is the burst's physical reference, and it alone
    positions the waveform. ``coa_time`` is the key a caller writing segmented data places every
    transient by; it carries no physics for a burst and never moves or changes the waveform.
    :meth:`pre_coalescence_duration` and :meth:`post_coalescence_duration` measure from it -- from
    ``peak_time`` when it is absent -- so ``coa_time - pre`` and ``coa_time + post`` are where the
    output starts and ends. ``coa_time = peak_time`` is the simple choice; the start of the
    polarization buffer, or any other instant inside the output, is equally valid and gives the
    same strain. A ``coa_time`` that is not a finite number, or that lies outside the output -- before
    its first sample or after its end, padding included -- cannot place the event, and is refused
    with the same ``ValueError`` by the two queries, :meth:`generate_polarizations` and
    :meth:`simulate`, so a caller that ignores a failed query still cannot inject a cropped burst.

    Args:
        waveform_files: Model name -> :class:`WaveformFile`. Each file is read once, on first use.

    Raises:
        ValueError: If a file model reuses an ad hoc family's name, or a value is not a
            :class:`WaveformFile`.
    """

    def __init__(self, *, waveform_files: Mapping[str, WaveformFile] | None = None) -> None:
        """Initialise with the optional numerical waveform files."""
        files = dict(waveform_files or {})
        shadowed = sorted(set(files) & set(PARAMETRIC_MODELS))
        if shadowed:
            raise ValueError(f"waveform_files may not reuse the ad hoc model names {shadowed}.")
        wrong = sorted(name for name, entry in files.items() if not isinstance(entry, WaveformFile))
        if wrong:
            raise ValueError(f"waveform_files values must be WaveformFile instances; {wrong} are not.")
        self._waveform_files: dict[str, WaveformFile] = files
        self._file_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._extent_cache: OrderedDict[Any, tuple[float, float, float]] = OrderedDict()

    @property
    def required_params(self) -> frozenset[str]:
        """Return the keys every burst event must carry."""
        return _REQUIRED

    def pre_coalescence_duration(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> float:
        """Return how long before the placement time this simulator's output starts, in seconds.

        The burst counterpart of :meth:`CBCSimulator.pre_coalescence_duration
        <gwmock_signal.simulator.CBCSimulator.pre_coalescence_duration>`, with the same call, so a
        caller placing transients into segmented data can ask either simulator alike. For a burst
        "coalescence" is the placement time: ``coa_time`` when the event gives one, else
        ``peak_time``. A burst's buffer is centred on its peak, so a peak just past a segment
        boundary has content belonging to the segment before -- up to 5.6 s of it for a
        ``Q = 100`` sine-Gaussian at 30 Hz -- and the caller needs this answer to claim it there.

        **Measured, not modelled.** The answer is read off the polarization buffer
        :meth:`generate_polarizations` returns for this very event and sample rate, so it is exact
        by construction for every family, including the numerical ones, and cannot drift from the
        generator: no copy of LALSimulation's buffer-sizing rules is kept here. The cost is one
        generation of the polarizations -- no projection -- per distinct event. At 8192 Hz that was
        measured at about 7 ms for a draft white-noise burst, 2.8 ms for the longest buffer (a
        91,265-sample ``Q = 100`` sine-Gaussian at 30 Hz) and under 0.3 ms for every other draft
        family, small beside the projection :meth:`simulate` then pays. The extent is remembered
        for the last few events, so the start and end queries and repeats for the same event share
        one generation.

        Args:
            params: One burst event, as :meth:`simulate` takes it.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Unused, as in :meth:`generate_polarizations`; accepted so the call
                matches the compact-binary simulator's.

        Returns:
            Seconds from the first sample :meth:`simulate` returns to the placement time,
            including the padding the projection adds before the polarizations (see
            :func:`~gwmock_signal.simulator._projected_duration`). Never ``None``: a burst's
            buffer is always known, and never negative.

        Raises:
            ValueError: As :meth:`generate_polarizations`, for an event it would refuse; if
                ``coa_time`` is given and is not a finite number; or if the placement time lies
                outside the output (before its first sample or past its end), where no segment
                claimed from it could hold the event as placed.
        """
        del minimum_frequency
        lead, _ = self._buffer_extent(params, sampling_frequency)
        return lead

    def post_coalescence_duration(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> float:
        """Return how long after the placement time this simulator's output runs, in seconds.

        The complement of :meth:`pre_coalescence_duration`, measured the same way from the same
        buffer and remembered with it.

        Args:
            params: One burst event, as :meth:`simulate` takes it.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Unused; accepted so the call matches the compact-binary simulator's.

        Returns:
            Seconds from the placement time to one sample past the last sample :meth:`simulate`
            returns, including the projection's padding after the polarizations; never negative.

        Raises:
            ValueError: As :meth:`pre_coalescence_duration`.
        """
        del minimum_frequency
        _, tail = self._buffer_extent(params, sampling_frequency)
        return tail

    def _buffer_extent(self, params: Mapping[str, Any], sampling_frequency: float) -> tuple[float, float]:
        """Return ``(lead, tail)`` of the projected output around the placement time, in seconds."""
        _placement_time(params)  # Refuse unusable times before paying for a generation.
        key = _span_key(params, sampling_frequency)
        span = self._extent_cache.get(key) if key is not None else None
        if span is None:
            span = self._remember(key, *self._generate(params, sampling_frequency))
        elif key is not None:
            self._extent_cache.move_to_end(key)
        return _placement_extent(params, span)

    def _remember(self, key: Any, hp: TimeSeries, hc: TimeSeries) -> tuple[float, float, float]:
        """Return the polarization buffer's ``(start, end, sample_rate)``, remembering it under *key*.

        Remembered per event without ``coa_time``, which generation never reads, so an event asked
        about under several placement times, and then generated, is generated once per query run.
        """
        span = (
            min(hp.t0, hc.t0),
            max(hp.t0 + hp.n_samples / hp.sample_rate, hc.t0 + hc.n_samples / hc.sample_rate),
            hp.sample_rate,
        )
        if key is not None:
            self._extent_cache[key] = span
            self._extent_cache.move_to_end(key)
            if len(self._extent_cache) > _EXTENT_CACHE_SIZE:
                self._extent_cache.popitem(last=False)
        return span

    @property
    def burst_models(self) -> tuple[str, ...]:
        """Return every ``burst_model`` this instance can generate."""
        return (*PARAMETRIC_MODELS, *self._waveform_files)

    def generate_polarizations(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> tuple[TimeSeries, TimeSeries]:
        """Return ``(h_plus, h_cross)`` for one burst event, centred on its ``peak_time``.

        Args:
            params: One burst event; see the class docstring.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Unused. A burst's band is set by its own parameters, and cutting
                it here would change the hrss the event records.

        Returns:
            Tuple of ``(hp, hc)`` series.

        Raises:
            ValueError: If ``burst_model`` is unknown or the event lacks one of its parameters; or
                if the event gives a ``coa_time`` the duration queries refuse -- not a finite
                number, or outside the projected output. The check is the queries' own, made on the
                buffer just generated, so it costs no second generation. Without ``coa_time`` the
                event is generated as before.
        """
        placed = params.get("coa_time") is not None
        if placed:
            _placement_time(params)  # Refuse an unusable time before paying for a generation.
        hp, hc = self._generate(params, sampling_frequency)
        if placed:
            _placement_extent(params, self._remember(_span_key(params, sampling_frequency), hp, hc))
        return hp, hc

    def _generate(self, params: Mapping[str, Any], sampling_frequency: float) -> tuple[TimeSeries, TimeSeries]:
        """Return the polarizations for *params*, centred on ``peak_time``; ``coa_time`` is not read."""
        model = params["burst_model"]
        if model in PARAMETRIC_MODELS:
            return PARAMETRIC_MODELS[model](params, sampling_frequency)
        if model in self._waveform_files:
            return self._file_polarizations(model, params, sampling_frequency)
        raise ValueError(f"Unknown burst_model {model!r}; this simulator knows {list(self.burst_models)}.")

    def _file_polarizations(
        self, model: str, params: Mapping[str, Any], sampling_frequency: float
    ) -> tuple[TimeSeries, TimeSeries]:
        """Return a file waveform placed at ``peak_time``, scaled to ``distance``, at the target rate."""
        entry = self._waveform_files[model]
        needed = ["distance"] + ([] if entry.inclination_dependence == "none" else ["inclination"])
        missing = [key for key in needed if key not in params]
        if missing:
            raise ValueError(f"burst_model {model!r} needs parameters {missing}.")
        distance = float(params["distance"])
        if not (math.isfinite(distance) and distance > 0):
            raise ValueError(f"distance must be finite and positive, got {params['distance']!r}.")
        inclination = float(params.get("inclination", 0.0))
        if not math.isfinite(inclination):
            raise ValueError(f"inclination must be finite, got {params['inclination']!r}.")
        if model not in self._file_cache:
            self._file_cache[model] = read_waveform_file(entry.path)
        times, plus, cross = self._file_cache[model]

        scale = entry.reference_distance / distance
        plus_factor, cross_factor = _INCLINATION_FACTORS[entry.inclination_dependence](inclination)
        t0 = float(params["peak_time"]) + times[0]
        file_rate = 1.0 / (times[1] - times[0])
        resample = not math.isclose(file_rate, sampling_frequency, rel_tol=1e-9)
        series = []
        for strain, factor, name in ((plus, plus_factor, "plus"), (cross, cross_factor, "cross")):
            values = scale * factor * np.asarray(strain, dtype=float)
            if resample:
                values = _resample(values, file_rate, sampling_frequency)
            rate = sampling_frequency if resample else file_rate
            series.append(TimeSeries(values, t0=t0, sample_rate=rate, name=name))
        return series[0], series[1]
