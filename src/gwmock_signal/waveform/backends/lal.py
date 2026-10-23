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
"""LALSimulation-backed waveform generation, conditioned to the time domain.

The waveform is evaluated in the *frequency domain* -- ``SimInspiralChooseFDWaveform``
for FD-native approximants, ``SimInspiralFD`` (which conditions a time-domain
approximant and transforms it) otherwise -- so that coalescence sits at the
frequency-domain phase reference. This is the same convention used by the ripple
backend and by ``bilby``: the same source lands at the same sample in either
backend and ``FFT(TD) == FD`` holds, which frequency-domain inference on the
injected data relies on.

It deliberately does *not* use ``SimInspiralChooseTDWaveform``, whose returned
epoch is re-pinned to the (2,2) amplitude peak -- a mass-dependent ~12 M offset
from the phase reference that would place ``tc`` at a physically different point
than the ripple backend does. See ``bilby.gw.source`` for the reference
implementation of the ``dt = 1/deltaF + epoch`` conditioning correction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from gwmock_signal import _lalsuite
from gwmock_signal.constants import MPC_SI, MSUN_SI, MTSUN_SI
from gwmock_signal.timeseries import TimeSeries
from gwmock_signal.waveform.backends import conditioning
from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

MSUN = MSUN_SI
MPC = MPC_SI

logger = logging.getLogger("gwmock_signal")

#: Extra chirp time, as a fraction of the chirp time from the lower of the cutoff and the lowest
#: Kerr ISCO frequency, that the waveform is generated for below that frequency. The raised-cosine
#: taper rises across the band this opens. LALSimulation's frequency-domain conditioning uses the
#: same fraction (``extra_time_fraction = 0.1``).
EXTRA_TIME_FRACTION = 0.1

#: Cycles, at the lower of the cutoff and the lowest Kerr ISCO frequency, added at both ends of the
#: buffer for a frequency that sweeps rapidly near merger. LALSimulation's ``extra_cycles = 3``.
EXTRA_CYCLES = 3.0

#: Approximants without ``HM`` in their name whose default output still carries harmonics above
#: the quadrupole.
# typos:off -- approximant names (numerical-relativity surrogates), not misspellings
_HIGHER_MODE_APPROXIMANTS = frozenset({"IMRPhenomXO4a", "NRSur7dq4", "NRHybSur3dq8", "NRHybSur3dq8Tidal"})
# typos:on

#: The largest ``|m|`` assumed for a model that carries higher harmonics by default but whose mode
#: content LALSimulation cannot report (``SimInspiralChooseFDModes`` / ``SimInspiralChooseTDModes``
#: refuse it, or its data files are missing). It is the largest ``|m|`` any LALSimulation model
#: emits by default among those whose modes it reports -- the ``(5, 5)`` harmonic of SEOBNRv4PHM
#: and IMRPhenomTPHM -- so the fallback over-sizes such a model rather than under-sizing it.
_UNREPORTED_HIGHER_MODE_M = 5

#: Time reserved at both ends of the buffer for the taper's own time response, in units of
#: ``1 / (minimum_frequency - start_frequency)``, the width of the taper band.
#:
#: An estimate checked by measurement, not a bound. The window's transition is a raised cosine of
#: width ``W``; its time kernel spreads content that does not chirp through the transition band --
#: the merger of a heavy, precessing binary, whose band starts there -- for several times ``1 / W``
#: on both sides. Heavy precessing binaries (200 to 360 solar masses at a 5 Hz cutoff, ``W`` 0.7 to
#: 0.8 Hz) ring at 4.5 to 5 Hz for about ten seconds after coalescence. Measured against references
#: holding at least 1024 s on each side of coalescence, on IMRPhenomXPHM with multibanding off over a
#: test set of 259 binaries from 19 to 2612 solar masses, as the largest content outside the buffer
#: relative to each polarization's peak: one width left up to 5.1e-3, three 1.6e-4, four 1.0e-4 and
#: five 7.3e-5, the smallest number meeting 1e-4 everywhere. Before coalescence one width already left
#: under 1.5e-5; the reservation is kept symmetric because the window's response is. Cost over that
#: set: 3.4% (intermediate-mass) to 4.8% (stellar-mass) more generated strain than one width.
_TAPER_RESPONSE_WIDTHS = 5.0

#: Largest lighter-component mass, in solar masses, that LAL accepts for an approximant.
#:
#: LAL checks the mass it is given, but a model's bound is on the physical body: an injection
#: passes detector-frame masses ``(1 + z) m``, so ``IMRPhenomNSBH`` refuses a 1.4 solar-mass
#: neutron star from ``z = 1.15`` on (LALSimIMRPhenomNSBH.c: "NS mass must be less than or equal
#: to 3 solar masses"). Such a binary is generated by :func:`_mass_scale` rescaling instead.
#: Only frequency-domain approximants evaluated in closed form belong here: the rescaling is exact
#: for them (IMRPhenomNSBH and IMRPhenomD agree with direct generation to 1e-13 of the peak), but a
#: time-domain model integrated numerically is not scale-invariant to its integrator's tolerance
#: (SEOBNRv4 differs by 2e-3).
_LIGHTER_MASS_GUARDS: dict[str, float] = {"IMRPhenomNSBH": 3.0}


def _mass_scale(approximant: str, mass1: float, mass2: float) -> float:
    """Return the factor the masses are divided by before LAL is called.

    A frequency-domain waveform in general relativity depends on the masses only through
    ``M f`` and on the distance only as ``1 / d``, with amplitude ``M^2``, so
    ``h(f; M, d) = s * h(s f; M / s, d / s)`` holds exactly for any ``s``: the masses and the
    distance are divided by ``s``, every frequency of the grid is multiplied by ``s``, and the
    polarizations (and any epoch shift) are multiplied by ``s``. Tidal deformabilities and spins are dimensionless and do not change.

    Args:
        approximant: LAL approximant name.
        mass1: First component mass, in solar masses.
        mass2: Second component mass, in solar masses.

    Returns:
        ``1.0`` when the lighter mass is inside the approximant's guard, or the approximant has
        none; otherwise the factor that puts the lighter mass at half the guard, well inside it,
        so rounding cannot push it back over.
    """
    guard = _LIGHTER_MASS_GUARDS.get(approximant)
    lighter = min(mass1, mass2)
    if guard is None or lighter <= guard:
        return 1.0
    return lighter / (0.5 * guard)


@dataclass(frozen=True)
class _FrequencyGrid:
    """Frequency-domain evaluation grid shared by the FD generators."""

    delta_f: float
    minimum_frequency: float
    f_max: float
    f_ref: float


@dataclass(frozen=True)
class _ResolvedParameters:
    """Validated, backend-native CBC parameters for the LAL FD generators."""

    mass1: float
    mass2: float
    distance: float
    spin_1x: float
    spin_1y: float
    spin_1z: float
    spin_2x: float
    spin_2y: float
    spin_2z: float
    inclination: float
    coa_phase: float
    lambda_1: float
    lambda_2: float
    waveform_arguments: dict[str, object]


@dataclass(frozen=True)
class _ConditioningPlan:
    """Where generation starts, and the buffer the conditioned waveform is placed in."""

    start_frequency: float
    n_samples: int
    merger_index: int


def _sizing_harmonic(approximant: str, waveform_arguments: dict[str, object]) -> int:
    """Return the largest azimuthal number ``|m|`` the waveform carries, which the buffer is sized for.

    A harmonic ``m`` passes a given frequency when the quadrupole is at ``2 / m`` of it, so the
    largest ``|m|`` is the one that enters the generated band first. It is the largest ``|m|`` of an
    explicit ``ModeArray`` when one is given, and otherwise the largest ``|m|`` of the modes
    LALSimulation itself reports the model emits by default (:func:`_default_largest_m`). Never
    below 2, the quadrupole.
    """
    modes = waveform_arguments.get("ModeArray")
    if modes is not None:
        try:
            orders = [abs(int(m)) for _, m in modes]  # type: ignore[attr-defined]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ModeArray must be an iterable of (l, m) pairs, got {modes!r}") from exc
        return max([2, *orders])
    options = tuple(sorted((key, repr(value)) for key, value in waveform_arguments.items()))
    return _default_largest_m(approximant, options, waveform_arguments)


_DEFAULT_LARGEST_M: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}


def _default_largest_m(
    approximant: str, options: tuple[tuple[str, str], ...], waveform_arguments: dict[str, object]
) -> int:
    """Return the largest ``|m|`` of the modes LALSimulation reports *approximant* emits by default.

    Asked of LALSimulation's mode interface -- ``SimInspiralChooseFDModes`` for a frequency-domain
    model, ``SimInspiralChooseTDModes`` for a time-domain one -- with the caller's own extra
    options, for a short, cheap binary; the mode content is a property of the model and its
    options, not of the source, so the answer is remembered per approximant and options. The
    largest ``|m|`` of a precessing model's inertial-frame modes is its largest ``l``, the largest
    co-precessing ``|m|`` as well.

    A model whose modes LALSimulation cannot report is a quadrupole-only model, which has no mode
    interface (``m = 2``), unless it is known to carry higher harmonics -- ``HM`` in its name or
    listed in :data:`_HIGHER_MODE_APPROXIMANTS` -- in which case :data:`_UNREPORTED_HIGHER_MODE_M`
    is assumed.
    """
    key = (approximant, options)
    if key in _DEFAULT_LARGEST_M:
        return _DEFAULT_LARGEST_M[key]
    largest = _reported_largest_m(approximant, waveform_arguments)
    if largest is None:
        known = "HM" in approximant or approximant in _HIGHER_MODE_APPROXIMANTS
        largest = _UNREPORTED_HIGHER_MODE_M if known else 2
    _DEFAULT_LARGEST_M[key] = max(2, largest)
    return _DEFAULT_LARGEST_M[key]


def _reported_largest_m(approximant: str, waveform_arguments: dict[str, object]) -> int | None:
    """Return the largest ``|m|`` LALSimulation's mode interface lists for *approximant*, or ``None``."""
    lal = _lalsuite.require("lal", LALSimulationBackend.LAL_FEATURE)
    lalsimulation = _lalsuite.require("lalsimulation", LALSimulationBackend.LAL_FEATURE)
    try:
        approx_enum = lalsimulation.GetApproximantFromString(approximant)
    except RuntimeError:
        return None
    lal_params = lal.CreateDict()
    _apply_waveform_arguments(lal_params, waveform_arguments)
    # A 30 + 20 solar-mass binary from 20 Hz: short and inside every model's calibration.
    mass1, mass2, f_low = 30.0 * MSUN, 20.0 * MSUN, 20.0
    try:
        if lalsimulation.SimInspiralImplementedFDApproximants(approx_enum):
            modes = lalsimulation.SimInspiralChooseFDModes(
                mass1,
                mass2,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
                0.2,
                0.25,
                f_low,
                1024.0,
                f_low,
                0.0,
                MPC,
                0.0,
                lal_params,
                approx_enum,
            )
        else:
            modes = lalsimulation.SimInspiralChooseTDModes(
                0.0,
                1.0 / 2048.0,
                mass1,
                mass2,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
                0.2,
                f_low,
                f_low,
                MPC,
                lal_params,
                8,
                approx_enum,
            )
    except RuntimeError:
        return None
    largest = 0
    while modes is not None:
        largest = max(largest, abs(int(modes.m)))
        modes = modes.next
    return largest or None


def _to_onesided(data: object, n_freq: int) -> np.ndarray:
    """Coerce a LAL frequency-series buffer to a one-sided array of length ``n_freq``.

    LAL may return fewer bins (zero-padded here) or, when it internally rounds the
    segment up to a power of two, more (truncated here) -- mirroring bilby's length
    reconciliation against its own frequency grid.
    """
    out = np.zeros(n_freq, dtype=complex)
    arr = np.asarray(data)
    k = min(len(arr), n_freq)
    out[:k] = arr[:k]
    return out


def _apply_waveform_arguments(lal_params: object, waveform_arguments: dict[str, object]) -> None:
    """Insert extra waveform options into a LAL dictionary.

    Keys use LALSimulation's ``SimInspiralWaveformParamsInsert<Key>`` naming
    (e.g. ``PhenomXPrecVersion``, ``dQuadMon1``). ``ModeArray`` is
    special-cased and takes an iterable of ``(l, m)`` pairs. Value types must
    match the LAL setter (int, float, or str); mismatches raise from SWIG.
    """
    lalsimulation = _lalsuite.require("lalsimulation", LALSimulationBackend.LAL_FEATURE)
    for key, value in waveform_arguments.items():
        if key == "ModeArray":
            mode_array = lalsimulation.SimInspiralCreateModeArray()
            try:
                pairs = [(int(ell), int(m)) for ell, m in value]  # type: ignore[attr-defined]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ModeArray must be an iterable of (l, m) pairs, got {value!r}") from exc
            for ell, m in pairs:
                lalsimulation.SimInspiralModeArrayActivateMode(mode_array, ell, m)
            lalsimulation.SimInspiralWaveformParamsInsertModeArray(lal_params, mode_array)
            continue
        setter = getattr(lalsimulation, f"SimInspiralWaveformParamsInsert{key}", None)
        if setter is None:
            raise ValueError(
                f"Unknown waveform argument {key!r}: no lalsimulation.SimInspiralWaveformParamsInsert{key}"
            )
        setter(lal_params, value)


class LALSimulationBackend(WaveformBackend):
    """Time-domain waveform backend implemented with LALSimulation.

    Args:
        f_ref: Reference frequency in Hz. Defaults to ``minimum_frequency`` of each
            call when ``None``.
        ringdown_fraction: Smallest fraction of the analysis segment reserved after
            coalescence, in ``(0, 1)``. ``None`` (default) reserves what the remnant needs:
            LALSimulation's merger and ringdown time bounds for the final black hole, plus the
            taper's time response. See :meth:`generate_td_waveform`.
        segment_duration: Optional fixed analysis-segment length in seconds. When
            ``None`` (default) the length is estimated from the chirp time below the
            cutoff, the ringdown and the taper, so the inspiral, merger and ringdown fit
            without wraparound. Content a model places beyond them -- the grid-dependent content of
            IMRPhenomXPHM's multibanded precession angles, for one -- is not held; see the user
            guide's *Content the buffer does not hold*.

    Raises:
        ImportError: On construction, if ``lalsuite`` is not installed.
    """

    #: What a missing ``lalsuite`` stops, as named in the error.
    LAL_FEATURE = "The LAL waveform backend"

    def __init__(
        self,
        *,
        f_ref: float | None = None,
        ringdown_fraction: float | None = None,
        segment_duration: float | None = None,
    ) -> None:
        """Validate the placement configuration."""
        if ringdown_fraction is not None and not 0.0 < ringdown_fraction < 1.0:
            raise ValueError("ringdown_fraction must be in (0, 1)")
        if segment_duration is not None and segment_duration <= 0:
            raise ValueError("segment_duration must be > 0")
        # After the argument checks, so an invalid argument is reported as such whether or not
        # lalsuite is installed; before the first waveform request, so a missing lalsuite stops
        # construction with an actionable message.
        _lalsuite.require("lalsimulation", self.LAL_FEATURE)
        self._f_ref = f_ref
        self._ringdown_fraction = ringdown_fraction
        self._segment_duration = segment_duration

    def available_approximants(self) -> list[str]:
        """Return every LAL approximant this backend can generate.

        ``generate_td_waveform`` produces FD-native approximants via
        ``SimInspiralChooseFDWaveform`` and time-domain approximants via
        ``SimInspiralFD``, so the advertised set is the union of both. Iterating
        over approximant indices yields each name once, in a stable order.
        """
        lalsimulation = _lalsuite.require("lalsimulation", self.LAL_FEATURE)
        return [
            lalsimulation.GetStringFromApproximant(i)
            for i in range(lalsimulation.NumApproximants)
            if lalsimulation.SimInspiralImplementedTDApproximants(i)
            or lalsimulation.SimInspiralImplementedFDApproximants(i)
        ]

    @staticmethod
    def _resolve_waveform_arguments(value: object) -> dict[str, object]:
        """Validate the optional extra-argument mapping for the LAL dictionary."""
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("waveform_arguments must be a dict with string keys")
        for reserved in ("TidalLambda1", "TidalLambda2"):
            if reserved in value:
                raise ValueError(
                    f"Pass tidal deformabilities as lambda_1/lambda_2, not waveform_arguments[{reserved!r}]"
                )
        resolved = dict(value)
        if resolved.get("ModeArray") is not None:
            # Read once, here, into the copy both buffer sizing and generation use: the pairs may come
            # as a one-shot iterable, which a second reader would find empty -- sizing read it first,
            # and generation then activated no modes at all.
            modes = resolved["ModeArray"]
            try:
                resolved["ModeArray"] = tuple((int(ell), int(m)) for ell, m in modes)  # type: ignore[attr-defined]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ModeArray must be an iterable of (l, m) pairs, got {modes!r}") from exc
        return resolved

    @staticmethod
    def _resolve_parameters(
        sampling_frequency: float, minimum_frequency: float, **params: object
    ) -> _ResolvedParameters:
        """Validate inputs and translate canonical parameters to backend-native ones."""
        remaining = dict(params)
        waveform_arguments = LALSimulationBackend._resolve_waveform_arguments(
            _pop_alias(remaining, "waveform_arguments", default={})
        )
        resolved = _ResolvedParameters(
            waveform_arguments=waveform_arguments,
            mass1=float(_pop_alias(remaining, "detector_frame_mass_1", "mass1")),
            mass2=float(_pop_alias(remaining, "detector_frame_mass_2", "mass2")),
            distance=float(_pop_alias(remaining, "luminosity_distance", "distance")),
            spin_1x=float(_pop_alias(remaining, "spin_1x", "spin1x", default=0.0)),
            spin_1y=float(_pop_alias(remaining, "spin_1y", "spin1y", default=0.0)),
            spin_1z=float(_pop_alias(remaining, "spin_1z", "spin1z", default=0.0)),
            spin_2x=float(_pop_alias(remaining, "spin_2x", "spin2x", default=0.0)),
            spin_2y=float(_pop_alias(remaining, "spin_2y", "spin2y", default=0.0)),
            spin_2z=float(_pop_alias(remaining, "spin_2z", "spin2z", default=0.0)),
            inclination=float(_pop_alias(remaining, "inclination", default=0.0)),
            coa_phase=float(_pop_alias(remaining, "coa_phase", default=0.0)),
            lambda_1=float(_pop_alias(remaining, "lambda_1", "tidal_1", default=0.0)),
            lambda_2=float(_pop_alias(remaining, "lambda_2", "tidal_2", default=0.0)),
        )
        if remaining:
            extras = ", ".join(sorted(remaining))
            raise ValueError(f"Unsupported LAL waveform parameters: {extras}")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be > 0")
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be > 0")
        if resolved.lambda_1 < 0:
            raise ValueError("lambda_1 must be >= 0")
        if resolved.lambda_2 < 0:
            raise ValueError("lambda_2 must be >= 0")
        return resolved

    def _evaluate_fd_rescaled(
        self, approximant: str, p: _ResolvedParameters, grid: _FrequencyGrid
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Call :meth:`_evaluate_fd`, rescaling an approximant whose LAL implementation bounds a mass.

        The rescaling (:func:`_mass_scale`) wraps the generator hook rather than living
        inside it, so a subclass that overrides :meth:`_evaluate_fd` gets it too.
        """
        scale = _mass_scale(approximant, p.mass1, p.mass2)
        if scale == 1.0:
            return self._evaluate_fd(approximant, p, grid)
        scaled_p = replace(p, mass1=p.mass1 / scale, mass2=p.mass2 / scale, distance=p.distance / scale)
        scaled_grid = _FrequencyGrid(
            delta_f=grid.delta_f * scale,
            minimum_frequency=grid.minimum_frequency * scale,
            f_max=grid.f_max * scale,
            f_ref=grid.f_ref * scale,
        )
        hp, hc, epoch_shift = self._evaluate_fd(approximant, scaled_p, scaled_grid)
        # The scaled waveform runs on a clock 1 / scale times as fast, and its amplitude is
        # 1 / scale of the binary's at the same frequency index.
        return scale * hp, scale * hc, scale * epoch_shift

    def _evaluate_fd(
        self, approximant: str, p: _ResolvedParameters, grid: _FrequencyGrid
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the one-sided FD polarizations plus the epoch correction.

        Returns the raw plus/cross frequency-series buffers and the time shift
        (in seconds) that re-references them to the frequency-domain phase
        convention. Subclasses may override this to source the same quantities
        from a different generator while sharing the mass rescaling, segment
        sizing and conditioning performed by :meth:`generate_td_waveform`.
        """
        lal = _lalsuite.require("lal", self.LAL_FEATURE)
        lalsimulation = _lalsuite.require("lalsimulation", self.LAL_FEATURE)
        approx_enum = lalsimulation.GetApproximantFromString(approximant)
        lal_params = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, p.lambda_1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, p.lambda_2)
        _apply_waveform_arguments(lal_params, p.waveform_arguments)

        wf_args = (
            p.mass1 * MSUN,
            p.mass2 * MSUN,
            p.spin_1x,
            p.spin_1y,
            p.spin_1z,
            p.spin_2x,
            p.spin_2y,
            p.spin_2z,
            p.distance * MPC,
            p.inclination,
            p.coa_phase,
            0.0,  # longitude of ascending nodes
            0.0,  # eccentricity
            0.0,  # mean periastron anomaly
            grid.delta_f,
            grid.minimum_frequency,
            grid.f_max,
            grid.f_ref,
            lal_params,
            approx_enum,
        )
        # Follow bilby: FD-native approximants come back already referenced to the FD
        # phase; a TD approximant routed through SimInspiralFD carries an epoch from
        # the internal time-domain conditioning, undone by a dt = T + epoch shift.
        if lalsimulation.SimInspiralImplementedFDApproximants(approx_enum):
            hp, hc = lalsimulation.SimInspiralChooseFDWaveform(*wf_args)
            epoch_shift = 0.0
        else:
            hp, hc = lalsimulation.SimInspiralFD(*wf_args)
            epoch_shift = 1.0 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        return np.asarray(hp.data.data), np.asarray(hc.data.data), epoch_shift

    def pre_coalescence_duration(
        self,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> float | None:
        """Return the seconds before ``tc`` this backend's buffer starts.

        Computed from the same plan ``generate_td_waveform`` uses (:meth:`_plan`), so the answer
        cannot drift from what generation actually does. Reproducing the arithmetic here instead
        would be a second implementation of it.
        """
        plan = self._plan(approximant, sampling_frequency, minimum_frequency, **params)
        return plan.merger_index / sampling_frequency

    def post_coalescence_duration(
        self,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> float | None:
        """Return the seconds after ``tc`` this backend's buffer runs.

        The complement of :meth:`pre_coalescence_duration`, from the same plan, so the two cannot
        describe different waveforms: the buffer is ``n_samples`` long and coalescence sits at
        ``merger_index``, so what remains after it is the rest.
        """
        plan = self._plan(approximant, sampling_frequency, minimum_frequency, **params)
        return (plan.n_samples - plan.merger_index) / sampling_frequency

    def _plan(
        self, approximant: str, sampling_frequency: float, minimum_frequency: float, **params: object
    ) -> _ConditioningPlan:
        """Return the conditioning plan for one source: where generation starts, and the buffer.

        Shared by generation and both duration queries deliberately. They are views of one buffer,
        and computing them separately is how a change to one silently stops matching the other.
        """
        p = self._resolve_parameters(sampling_frequency, minimum_frequency, **params)
        return self._plan_resolved(approximant, p, sampling_frequency, minimum_frequency)

    def _plan_resolved(
        self, approximant: str, p: _ResolvedParameters, sampling_frequency: float, minimum_frequency: float
    ) -> _ConditioningPlan:
        """Plan the conditioning for resolved parameters; see :meth:`generate_td_waveform`.

        Follows LALSimulation's own frequency-domain conditioning
        (``XLALSimInspiralTDFromFD`` and its Python port in ``lalsimulation.gwsignal``) for where
        generation starts and how much merger and ringdown to keep, and departs from it where that
        would change the requested band; the user guide's waveform page lists both.
        """
        lalsimulation = _lalsuite.require("lalsimulation", self.LAL_FEATURE)
        mass1, mass2 = p.mass1 * MSUN, p.mass2 * MSUN
        total_mass = p.mass1 + p.mass2
        # The lowest gravitational-wave frequency of the innermost stable orbit of any Kerr black
        # hole of this mass (the retrograde extremal one). A cutoff above it starts in the merger,
        # where no chirp maps frequency to time, so the chirp quantities are taken from here.
        lowest_isco = 1.0 / (9.0**1.5 * math.pi * total_mass * MTSUN_SI)
        chirp_frequency = min(minimum_frequency, lowest_isco)
        chirp_seconds = lalsimulation.SimInspiralChirpTimeBound(chirp_frequency, mass1, mass2, p.spin_1z, p.spin_2z)
        start_frequency = lalsimulation.SimInspiralChirpStartFrequencyBound(
            (1.0 + EXTRA_TIME_FRACTION) * chirp_seconds, mass1, mass2
        )
        # The taper's own time response: a raised cosine of width W in frequency is a pulse whose
        # main lobe is ~1 / W long in time. It spreads whatever does not chirp through the taper
        # band -- the merger of a binary whose cutoff lies above its ISCO -- to both sides. An
        # estimate; see _TAPER_RESPONSE_WIDTHS.
        taper_seconds = _TAPER_RESPONSE_WIDTHS / (minimum_frequency - start_frequency)
        extra_seconds = EXTRA_CYCLES / chirp_frequency
        # Inspiral: the quadrupole's chirp time from where the largest harmonic the model emits
        # enters the generated band, the larger of LALSimulation's bound and this package's own
        # estimate.
        harmonic = _sizing_harmonic(approximant, p.waveform_arguments)
        onset = 2.0 * start_frequency / harmonic
        eta = p.mass1 * p.mass2 / total_mass**2
        chirp_mass = total_mass * eta**0.6
        inspiral = max(
            float(conditioning.inspiral_requirement_seconds(chirp_mass, eta, onset)),
            lalsimulation.SimInspiralChirpTimeBound(onset, mass1, mass2, p.spin_1z, p.spin_2z),
        )
        pre = inspiral + extra_seconds + taper_seconds
        # Merger and ringdown: LALSimulation's bounds for the plunge-merger and for the ringdown of
        # a remnant with the largest spin these components can produce (11 e-folds of its
        # fundamental quasi-normal mode).
        final_spin = lalsimulation.SimInspiralFinalBlackHoleSpinBound(p.spin_1z, p.spin_2z)
        post = (
            lalsimulation.SimInspiralMergeTimeBound(mass1, mass2)
            + lalsimulation.SimInspiralRingdownTimeBound(mass1 + mass2, final_spin)
            + extra_seconds
            + taper_seconds
        )
        fraction = self._ringdown_fraction
        if self._segment_duration is not None:
            seconds = self._segment_duration
        else:
            seconds = max(pre + post, pre / (1.0 - fraction) if fraction else 0.0, conditioning.MIN_SEGMENT_SECONDS)
        n_samples = conditioning.segment_sample_count(
            chirp_mass, minimum_frequency, sampling_frequency, eta=eta, segment_duration=seconds
        )
        post_samples = math.ceil(post * sampling_frequency)
        if fraction is not None:
            post_samples = max(post_samples, round(fraction * n_samples))
        if self._segment_duration is not None:
            # A pinned length is the caller's choice; it keeps at least half of it for the inspiral.
            post_samples = min(post_samples, n_samples // 2)
        return _ConditioningPlan(
            start_frequency=start_frequency, n_samples=n_samples, merger_index=n_samples - post_samples
        )

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations, conditioned from frequency to time domain.

        The waveform is generated from a start frequency below ``minimum_frequency`` and multiplied
        by a raised-cosine taper that rises from zero there to one at ``minimum_frequency``. Every
        bin from ``minimum_frequency`` up is the model's own value, unchanged; the content below it
        is tapered rather than cut, which is what suppresses the ringing a hard cut leaves across
        the buffer (on a 2302 + 248 solar-mass binary at a 5 Hz cutoff and 8192 Hz, the hard cut put
        its 4 s buffer up to 0.34 of peak away from the same waveform on a buffer eight times longer,
        with 0.26 on the first sample; tapered, 1.4e-6 and 2.6e-7). The buffer is sized from the physics of the source rather than as a fixed fraction:

        * **start frequency** -- as LALSimulation's frequency-domain conditioning chooses it, the
          frequency with 10% more chirp time than ``minimum_frequency``, or than the lowest Kerr
          ISCO frequency when the cutoff lies above it (heavy binaries);
        * **before coalescence** -- the quadrupole's chirp time from where the largest harmonic the
          model emits (the largest ``|m|`` LALSimulation lists for it, or of an explicit
          ``ModeArray``) enters the generated band, plus three cycles and the taper's time
          response;
        * **after coalescence** -- LALSimulation's merger and ringdown time bounds for the remnant,
          plus three cycles and the taper's time response; at least ``ringdown_fraction`` of the
          buffer when one is set.

        The buffer holds the inspiral, merger and ringdown so sized, not content a model places
        beyond them. IMRPhenomXPHM's default multibanding of its precession angles leaves its
        spectrum with content at every delay, on both sides of coalescence and changing with the
        frequency grid -- up to 1.6e-2 of peak at 250 + 5 solar masses with spins of 0.99 aligned
        -- which no buffer holds; ``waveform_arguments={"PhenomXPHMThresholdMband": 0.0}`` reduces
        it to below 1e-6 there. See the user guide's *Content the buffer does not hold*.

        If LALSimulation refuses to start below ``minimum_frequency`` for an approximant, the
        waveform is generated from ``minimum_frequency`` and the taper is placed *above* it instead,
        over the same width, which does reduce the lowest part of the band; a warning says so.
        """
        p = self._resolve_parameters(sampling_frequency, minimum_frequency, **params)
        plan = self._plan_resolved(approximant, p, sampling_frequency, minimum_frequency)
        n_samples = plan.n_samples
        delta_f = sampling_frequency / n_samples
        f_max = sampling_frequency / 2.0
        f_ref = self._f_ref if self._f_ref is not None else minimum_frequency

        taper_band = (plan.start_frequency, minimum_frequency)
        grid = _FrequencyGrid(delta_f=delta_f, minimum_frequency=plan.start_frequency, f_max=f_max, f_ref=f_ref)
        try:
            hp_raw, hc_raw, epoch_shift = self._evaluate_fd_rescaled(approximant, p, grid)
        except RuntimeError as below:
            try:
                hp_raw, hc_raw, epoch_shift = self._evaluate_fd_rescaled(
                    approximant, p, replace(grid, minimum_frequency=minimum_frequency)
                )
            except RuntimeError:
                raise below from None
            width = minimum_frequency - plan.start_frequency
            taper_band = (minimum_frequency, minimum_frequency + width)
            logger.warning(
                "%s could not be generated from %.3f Hz, below the %.3f Hz cutoff (%s). It is generated "
                "from the cutoff instead and tapered over [%.3f, %.3f] Hz, which reduces the band just "
                "above the cutoff rather than below it.",
                approximant,
                plan.start_frequency,
                minimum_frequency,
                below,
                *taper_band,
            )

        n_freq = n_samples // 2 + 1
        freqs = np.arange(n_freq) * delta_f
        hp_f = _to_onesided(hp_raw, n_freq)
        hc_f = _to_onesided(hc_raw, n_freq)
        if epoch_shift:
            time_shift = np.exp(-2j * np.pi * freqs * epoch_shift)
            hp_f = hp_f * time_shift
            hc_f = hc_f * time_shift
        window = conditioning.raised_cosine_taper(freqs, *taper_band)
        hp_f = np.nan_to_num(hp_f) * window
        hc_f = np.nan_to_num(hc_f) * window

        hp_t, hc_t, epoch = conditioning.condition_fd_to_td(
            hp_f, hc_f, n_samples, sampling_frequency, merger_index=plan.merger_index
        )
        t0 = epoch + tc
        return {
            "plus": TimeSeries(hp_t, t0=t0, sample_rate=sampling_frequency),
            "cross": TimeSeries(hc_t, t0=t0, sample_rate=sampling_frequency),
        }
