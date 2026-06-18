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
"""ripple (JAX) waveform backend, conditioned from frequency to time domain.

[ripple](https://github.com/GW-JAX-Team/ripple) generates *frequency-domain*
polarizations as JAX arrays. This backend conditions them into the time-domain
GWpy ``plus``/``cross`` series required by :class:`WaveformBackend`, so ripple can
be used wherever the LAL/PyCBC backends are. The conversion runs on host (NumPy);
an on-device JAX pipeline is a separate, later effort (see ``PLAN.md``).

Supported so far: aligned-spin point-particle models (``IMRPhenomD``,
``IMRPhenomHM``, ``IMRPhenomXAS``, ``IMRPhenomXHM``) and their NRTidal variants
(``IMRPhenomD_NRTidalv2``, ``IMRPhenomXAS_NRTidalv3``). Precessing models are
added in a later PR.
"""

from __future__ import annotations

import importlib

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

_RIPPLE_IMPORT_ERROR = "ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'"

#: Aligned-spin, point-particle (non-tidal) models.
#: Each takes ripple params ``M_c, eta, s1_z, s2_z, d_L, phase_c, iota``.
_ALIGNED_SPIN_MODELS = ("IMRPhenomD", "IMRPhenomHM", "IMRPhenomXAS", "IMRPhenomXHM")

#: Aligned-spin models with an NRTidal sector; they additionally take
#: ``lambda_1, lambda_2``. Precessing models are added in a later PR.
_TIDAL_MODELS = ("IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3")

#: All approximants this backend can generate.
_SUPPORTED_APPROXIMANTS = _ALIGNED_SPIN_MODELS + _TIDAL_MODELS

#: Fraction of the analysis segment reserved *after* coalescence (ringdown + pad).
_DEFAULT_RINGDOWN_FRACTION = 0.1
#: Extra inspiral headroom (seconds) beyond the leading-order post-Newtonian chirp-time estimate.
_SEGMENT_BUFFER_SECONDS = 2.0
#: Floor on the segment length (seconds) for very short signals.
_MIN_SEGMENT_SECONDS = 1.0


class RippleBackend(WaveformBackend):
    """Time-domain waveform backend implemented with ripple (JAX).

    Args:
        f_ref: Reference frequency in Hz. Defaults to ``minimum_frequency`` of each
            call when ``None``.
        ringdown_fraction: Fraction of the analysis segment reserved after
            coalescence. Must be in ``(0, 1)``.
        segment_duration: Optional fixed analysis-segment length in seconds. When
            ``None`` (default) the length is estimated from the post-Newtonian
            chirp time so the full inspiral fits without wraparound.
    """

    def __init__(
        self,
        *,
        f_ref: float | None = None,
        ringdown_fraction: float = _DEFAULT_RINGDOWN_FRACTION,
        segment_duration: float | None = None,
    ) -> None:
        """Require ripple/JAX only when this backend is instantiated."""
        try:
            self._jax = importlib.import_module("jax")
            self._jnp = importlib.import_module("jax.numpy")
            self._ripplegw = importlib.import_module("ripplegw")
            self._conversions = importlib.import_module("ripplegw.conversions")
            self._constants = importlib.import_module("ripplegw.constants")
        except ImportError as exc:
            raise ImportError(_RIPPLE_IMPORT_ERROR) from exc
        # ripple needs double precision for waveform phase accuracy over long
        # inspirals. Importing ripplegw already enables this globally; set it
        # explicitly so correctness does not depend on import order.
        self._jax.config.update("jax_enable_x64", True)
        if not 0.0 < ringdown_fraction < 1.0:
            raise ValueError("ringdown_fraction must be in (0, 1)")
        if segment_duration is not None and segment_duration <= 0:
            raise ValueError("segment_duration must be > 0")
        self._f_ref = f_ref
        self._ringdown_fraction = ringdown_fraction
        self._segment_duration = segment_duration

    def available_approximants(self) -> list[str]:
        """Return the ripple approximants supported by this backend."""
        return list(_SUPPORTED_APPROXIMANTS)

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations from ripple, conditioned to time domain."""
        if approximant not in _SUPPORTED_APPROXIMANTS:
            raise ValueError(
                f"RippleBackend does not support approximant {approximant!r}. "
                f"Available: {list(_SUPPORTED_APPROXIMANTS)}."
            )
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be > 0")
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be > 0")

        remaining = dict(params)
        mass1 = float(_pop_alias(remaining, "detector_frame_mass_1", "mass1"))
        mass2 = float(_pop_alias(remaining, "detector_frame_mass_2", "mass2"))
        distance = float(_pop_alias(remaining, "luminosity_distance", "distance"))
        chi1 = float(_pop_alias(remaining, "spin_1z", "spin1z", default=0.0))
        chi2 = float(_pop_alias(remaining, "spin_2z", "spin2z", default=0.0))
        inclination = float(_pop_alias(remaining, "inclination", default=0.0))
        coa_phase = float(_pop_alias(remaining, "coa_phase", default=0.0))

        # Every supported approximant is non-precessing: in-plane spins must be zero.
        in_plane_spins = {
            "spin_1x": _pop_alias(remaining, "spin_1x", "spin1x", default=0.0),
            "spin_1y": _pop_alias(remaining, "spin_1y", "spin1y", default=0.0),
            "spin_2x": _pop_alias(remaining, "spin_2x", "spin2x", default=0.0),
            "spin_2y": _pop_alias(remaining, "spin_2y", "spin2y", default=0.0),
        }
        nonzero_in_plane = sorted(name for name, value in in_plane_spins.items() if float(value) != 0.0)
        if nonzero_in_plane:
            raise ValueError(
                f"{approximant} is an aligned-spin model; in-plane spins must be zero: {', '.join(nonzero_in_plane)}"
            )
        lambda_1 = float(_pop_alias(remaining, "lambda_1", "tidal_1", default=0.0))
        lambda_2 = float(_pop_alias(remaining, "lambda_2", "tidal_2", default=0.0))
        is_tidal = approximant in _TIDAL_MODELS
        if not is_tidal and (lambda_1 or lambda_2):
            raise ValueError(f"{approximant} does not support tidal parameters; use an NRTidal approximant.")
        if lambda_1 < 0:
            raise ValueError("lambda_1 must be >= 0")
        if lambda_2 < 0:
            raise ValueError("lambda_2 must be >= 0")
        if remaining:
            extras = ", ".join(sorted(remaining))
            raise ValueError(f"Unsupported ripple waveform parameters: {extras}")

        f_ref = self._f_ref if self._f_ref is not None else minimum_frequency
        hp_t, hc_t, epoch = self._condition_to_time_domain(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            is_tidal=is_tidal,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            f_ref=f_ref,
        )
        t0 = epoch + tc
        dt = 1.0 / sampling_frequency
        return {
            "plus": TimeSeries(hp_t, t0=t0, dt=dt),
            "cross": TimeSeries(hc_t, t0=t0, dt=dt),
        }

    def _segment_samples(self, chirp_mass_solar: float, minimum_frequency: float, sampling_frequency: float) -> int:
        """Return an even sample count whose duration contains the full inspiral.

        The duration is rounded up to a power of two seconds so the inspiral
        (estimated from the leading-order post-Newtonian chirp time) fits in the pre-coalescence
        portion of the buffer without cyclic wraparound.
        """
        if self._segment_duration is not None:
            seconds = self._segment_duration
        else:
            mc_seconds = chirp_mass_solar * self._constants.MTSUN
            # Leading-order (Newtonian) chirp time from minimum_frequency to merger.
            tau0 = (5.0 / 256.0) * (np.pi * minimum_frequency) ** (-8.0 / 3.0) * mc_seconds ** (-5.0 / 3.0)
            inspiral_room = 1.0 - self._ringdown_fraction
            seconds = max((tau0 + _SEGMENT_BUFFER_SECONDS) / inspiral_room, _MIN_SEGMENT_SECONDS)
        seconds_pow2 = float(2.0 ** np.ceil(np.log2(seconds)))
        n_samples = round(seconds_pow2 * sampling_frequency)
        if n_samples % 2:
            n_samples += 1
        return n_samples

    def _condition_to_time_domain(  # noqa: PLR0913
        self,
        *,
        approximant: str,
        mass1: float,
        mass2: float,
        chi1: float,
        chi2: float,
        distance: float,
        inclination: float,
        coa_phase: float,
        lambda_1: float,
        lambda_2: float,
        is_tidal: bool,
        sampling_frequency: float,
        minimum_frequency: float,
        f_ref: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate ripple in the frequency domain and inverse-FFT to time domain.

        Returns ``(hp, hc, epoch)`` where ``epoch`` is the time of the first sample
        relative to coalescence (negative), so the caller places coalescence at
        ``epoch + tc``.
        """
        jnp = self._jnp
        chirp_mass, eta = self._conversions.ms_to_Mc_eta(jnp.array([mass1, mass2]))

        n_samples = self._segment_samples(float(chirp_mass), minimum_frequency, sampling_frequency)
        dt = 1.0 / sampling_frequency
        delta_f = sampling_frequency / n_samples
        freqs = np.arange(n_samples // 2 + 1) * delta_f

        # ripple's class interface fixes its internal tc=0; coalescence is placed
        # in the time grid below via the roll.
        ripple_params = {
            "M_c": chirp_mass,
            "eta": eta,
            "s1_z": chi1,
            "s2_z": chi2,
            "d_L": distance,
            "phase_c": coa_phase,
            "iota": inclination,
        }
        if is_tidal:
            ripple_params["lambda_1"] = lambda_1
            ripple_params["lambda_2"] = lambda_2
        waveform = self._ripplegw.waveform_preset[approximant](f_ref=f_ref)
        polarizations = waveform(jnp.asarray(freqs), ripple_params)
        hp_f = polarizations["p"]
        hc_f = polarizations["c"]

        # Zero out-of-band bins (including DC, where the amplitude diverges) and
        # guard against any non-finite values before the inverse FFT.
        in_band = freqs >= minimum_frequency
        hp_f = np.nan_to_num(np.where(in_band, np.asarray(hp_f), 0.0))
        hc_f = np.nan_to_num(np.where(in_band, np.asarray(hc_f), 0.0))

        # Inverse real FFT: h(t) = irfft(h(f)) / dt (continuous-transform normalization).
        hp_t = np.fft.irfft(hp_f, n=n_samples) / dt
        hc_t = np.fft.irfft(hc_f, n=n_samples) / dt

        # With tc=0 coalescence lands at sample 0 and the inspiral wraps to the tail.
        # Roll it forward so coalescence sits near the segment end, leaving the
        # inspiral contiguous before it and a small ringdown pad after.
        merger_index = round((1.0 - self._ringdown_fraction) * n_samples)
        hp_t = np.roll(hp_t, merger_index)
        hc_t = np.roll(hc_t, merger_index)
        epoch = -merger_index * dt
        return hp_t, hc_t, epoch
