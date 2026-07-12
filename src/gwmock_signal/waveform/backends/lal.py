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

from dataclasses import dataclass

import lal
import lalsimulation
import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import conditioning
from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

MSUN = lal.MSUN_SI
MPC = lal.PC_SI * 1e6


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
    for key, value in waveform_arguments.items():
        if key == "ModeArray":
            mode_array = lalsimulation.SimInspiralCreateModeArray()
            for ell, m in value:  # type: ignore[attr-defined]
                lalsimulation.SimInspiralModeArrayActivateMode(mode_array, int(ell), int(m))
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
        ringdown_fraction: Fraction of the analysis segment reserved after
            coalescence. Must be in ``(0, 1)``.
        segment_duration: Optional fixed analysis-segment length in seconds. When
            ``None`` (default) the length is estimated from the post-Newtonian chirp
            time so the full inspiral fits without wraparound.
    """

    def __init__(
        self,
        *,
        f_ref: float | None = None,
        ringdown_fraction: float = conditioning.DEFAULT_RINGDOWN_FRACTION,
        segment_duration: float | None = None,
    ) -> None:
        """Validate the placement configuration shared with the ripple backend."""
        if not 0.0 < ringdown_fraction < 1.0:
            raise ValueError("ringdown_fraction must be in (0, 1)")
        if segment_duration is not None and segment_duration <= 0:
            raise ValueError("segment_duration must be > 0")
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
        return dict(value)

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

    def _evaluate_fd(
        self, approximant: str, p: _ResolvedParameters, grid: _FrequencyGrid
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the one-sided FD polarizations plus the epoch correction.

        Returns the raw plus/cross frequency-series buffers and the time shift
        (in seconds) that re-references them to the frequency-domain phase
        convention. Subclasses may override this to source the same quantities
        from a different generator while sharing the segment sizing and
        conditioning performed by :meth:`generate_td_waveform`.
        """
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

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations, conditioned from frequency to time domain."""
        p = self._resolve_parameters(sampling_frequency, minimum_frequency, **params)

        chirp_mass = (p.mass1 * p.mass2) ** 0.6 / (p.mass1 + p.mass2) ** 0.2
        n_samples = conditioning.segment_sample_count(
            chirp_mass,
            minimum_frequency,
            sampling_frequency,
            ringdown_fraction=self._ringdown_fraction,
            segment_duration=self._segment_duration,
        )
        delta_f = sampling_frequency / n_samples
        f_max = sampling_frequency / 2.0
        f_ref = self._f_ref if self._f_ref is not None else minimum_frequency

        grid = _FrequencyGrid(delta_f=delta_f, minimum_frequency=minimum_frequency, f_max=f_max, f_ref=f_ref)
        hp_raw, hc_raw, epoch_shift = self._evaluate_fd(approximant, p, grid)

        n_freq = n_samples // 2 + 1
        freqs = np.arange(n_freq) * delta_f
        hp_f = _to_onesided(hp_raw, n_freq)
        hc_f = _to_onesided(hc_raw, n_freq)
        if epoch_shift:
            time_shift = np.exp(-2j * np.pi * freqs * epoch_shift)
            hp_f = hp_f * time_shift
            hc_f = hc_f * time_shift
        in_band = freqs >= minimum_frequency
        hp_f = np.nan_to_num(np.where(in_band, hp_f, 0.0))
        hc_f = np.nan_to_num(np.where(in_band, hc_f, 0.0))

        hp_t, hc_t, epoch = conditioning.condition_fd_to_td(
            hp_f, hc_f, n_samples, sampling_frequency, self._ringdown_fraction
        )
        dt = 1.0 / sampling_frequency
        t0 = epoch + tc
        return {
            "plus": TimeSeries(hp_t, t0=t0, dt=dt),
            "cross": TimeSeries(hc_t, t0=t0, dt=dt),
        }
