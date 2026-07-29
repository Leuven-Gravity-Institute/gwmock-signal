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

Supported: aligned-spin point-particle models (``IMRPhenomD``, ``IMRPhenomHM``,
``IMRPhenomXAS``, ``IMRPhenomXHM``), the tidal-capable ``TaylorF2`` and NRTidal
variants (``IMRPhenomD_NRTidalv2``, ``IMRPhenomXAS_NRTidalv3``), and the
precessing models (``IMRPhenomPv2``, ``IMRPhenomXP``, ``IMRPhenomXPHM``).

Extra ripple options travel through the ``waveform_arguments`` mapping, as with
the other backends, but here they are *constructor* kwargs of the ripple preset
rather than call-time options. The whitelist is intentionally narrow because
ripple's options surface is thin and pre-1.0: only ``no_taper`` (the NRTidal
variants) is forwarded. ``f_ref`` is owned by the backend, and
``use_lambda_tildes`` is refused because it would switch ripple to a
``lambda_tilde``/``delta_lambda_tilde`` parameterisation this backend does not
feed. The batch path takes the same options as a batch-wide keyword argument
(the preset is built once, so they are constructor-level, not per-event). See
``_ALLOWED_WAVEFORM_ARGUMENTS`` / ``_RESERVED_WAVEFORM_ARGUMENTS``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends.base import WaveformBackend, _pop_alias

if TYPE_CHECKING:
    from jax import Array

_RIPPLE_IMPORT_ERROR = "ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'"

#: Extra ripple *constructor* options this backend forwards, keyed by approximant.
#: ripple options are constructor-time (``waveform_preset[name](f_ref=..., **extras)``),
#: not call-time. ripple is pre-1.0, so this whitelist is deliberately narrow and is
#: hardened by ``test_waveform_arguments`` against ripple's real constructor signatures,
#: which will trip CI on a version bump that adds, renames, or removes an option.
_ALLOWED_WAVEFORM_ARGUMENTS: Final[dict[str, frozenset[str]]] = {
    "IMRPhenomD_NRTidalv2": frozenset({"no_taper"}),
    "IMRPhenomXAS_NRTidalv3": frozenset({"no_taper"}),
}

#: ripple constructor options this backend refuses through ``waveform_arguments``,
#: with the reason. ``f_ref`` is owned by the backend; ``use_lambda_tildes`` would
#: switch ripple to expect ``lambda_tilde``/``delta_lambda_tilde`` while this backend
#: always supplies ``lambda_1``/``lambda_2``, so it cannot be honoured here.
_RESERVED_WAVEFORM_ARGUMENTS: Final[dict[str, str]] = {
    "f_ref": "f_ref is configured on the backend, not through waveform_arguments",
    "use_lambda_tildes": (
        "use_lambda_tildes is not supported: this backend supplies lambda_1/lambda_2, "
        "not lambda_tilde/delta_lambda_tilde"
    ),
}

#: Aligned-spin, point-particle (non-tidal) models.
#: Each takes ripple params ``M_c, eta, s1_z, s2_z, d_L, phase_c, iota``.
_ALIGNED_SPIN_MODELS = ("IMRPhenomD", "IMRPhenomHM", "IMRPhenomXAS", "IMRPhenomXHM")

#: Aligned-spin models that additionally take tidal deformabilities
#: ``lambda_1, lambda_2`` (the NRTidal variants and the post-Newtonian inspiral TaylorF2).
_TIDAL_MODELS = ("TaylorF2", "IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3")

#: Precessing models taking the full six spin components
#: ``s1_x, s1_y, s1_z, s2_x, s2_y, s2_z``.
_PRECESSING_MODELS = ("IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM")

#: All approximants this backend can generate.
_SUPPORTED_APPROXIMANTS = _ALIGNED_SPIN_MODELS + _TIDAL_MODELS + _PRECESSING_MODELS

#: Fraction of the analysis segment reserved *after* coalescence (ringdown + pad).
_DEFAULT_RINGDOWN_FRACTION = 0.1
#: Extra inspiral headroom (seconds) beyond the leading-order post-Newtonian chirp-time estimate.
_SEGMENT_BUFFER_SECONDS = 2.0
#: Floor on the segment length (seconds) for very short signals.
_MIN_SEGMENT_SECONDS = 1.0


#: Distinct batched-kernel configurations kept compiled. Each entry holds one XLA
#: executable, so the cache is bounded; a handful covers any realistic run, which varies
#: catalogue size (a traced shape, handled by JAX's own cache) far more often than it
#: varies approximant, grid length or waveform options.
_KERNEL_CACHE_SIZE = 32


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _batched_polarization_kernel(  # noqa: PLR0913, PLR0917 - one key component per dependency
    approximant: str,
    n_samples: int,
    sampling_frequency: float,
    minimum_frequency: float,
    f_ref: float,
    waveform_arguments: tuple[tuple[str, object], ...],
) -> Callable[[dict], tuple]:
    """Return a cached, jitted, vmapped ripple evaluation for one kernel configuration.

    Keyed on every value the kernel closes over, so two calls that would build an identical
    kernel share one compiled executable instead of recompiling. Catalogue size is
    deliberately *not* part of the key: it changes the input shape, which JAX's own cache
    handles, and including it here would defeat the purpose whenever a chunked run ends with
    a shorter final chunk.

    Args:
        approximant: A supported ripple approximant name.
        n_samples: Length of the real time series the grid corresponds to.
        sampling_frequency: Sample rate in Hz.
        minimum_frequency: Low-frequency cutoff in Hz; bins below it are zeroed.
        f_ref: Reference frequency handed to the ripple preset.
        waveform_arguments: Resolved extra preset options, as sorted items so the key hashes.

    Returns:
        A callable taking the batched ripple parameter dict and returning ``(plus, cross)``.
    """
    import jax  # noqa: PLC0415 — optional [jax] dep, kept out of module import
    import jax.numpy as jnp  # noqa: PLC0415

    ripplegw = importlib.import_module("ripplegw")
    delta_f = sampling_frequency / n_samples
    freqs = jnp.arange(n_samples // 2 + 1) * delta_f
    in_band = freqs >= minimum_frequency
    waveform = ripplegw.waveform_preset[approximant](f_ref=f_ref, **dict(waveform_arguments))

    def _one(event: dict) -> tuple:
        polarizations = waveform(freqs, event)
        return (
            jnp.nan_to_num(jnp.where(in_band, polarizations["p"], 0.0)),
            jnp.nan_to_num(jnp.where(in_band, polarizations["c"], 0.0)),
        )

    return jax.jit(jax.vmap(_one))


@dataclass(frozen=True)
class FrequencyDomainPolarizations:
    """Frequency-domain plus/cross polarizations on a uniform one-sided grid.

    The polarizations are evaluated with coalescence at ``t = 0`` (ripple's internal
    ``tc = 0``) and out-of-band bins zeroed; ``frequencies``, ``plus`` and ``cross``
    are JAX arrays (on-device). ``n_samples`` and ``sampling_frequency`` describe the
    real time series an inverse real FFT would produce (``len(frequencies) ==
    n_samples // 2 + 1``). This is the on-device hand-off for the projection kernel;
    the time-domain backend conditions it into GWpy series.
    """

    frequencies: Array
    plus: Array
    cross: Array
    sampling_frequency: float
    n_samples: int


@dataclass(frozen=True)
class _ResolvedParameters:
    """Validated, backend-native CBC parameters shared by the FD and TD entry points."""

    mass1: float
    mass2: float
    spins: dict[str, float]
    distance: float
    inclination: float
    coa_phase: float
    lambda_1: float
    lambda_2: float
    is_tidal: bool
    is_precessing: bool
    f_ref: float
    waveform_arguments: dict[str, object]


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

    @property
    def segment_duration(self) -> float | None:
        """The fixed analysis-segment length in seconds, or ``None`` if auto-sized."""
        return self._segment_duration

    def with_segment_duration(self, segment_duration: float) -> RippleBackend:
        """Return a copy of this backend pinned to a fixed ``segment_duration``.

        Same ``f_ref`` and ``ringdown_fraction``; useful for forcing one shared grid
        across several batched calls (e.g. count-chunked catalogue generation).
        """
        return RippleBackend(
            f_ref=self._f_ref,
            ringdown_fraction=self._ringdown_fraction,
            segment_duration=segment_duration,
        )

    def segment_duration_for(
        self, chirp_mass_solar: float, minimum_frequency: float, sampling_frequency: float
    ) -> float:
        """Worst-case segment duration (seconds) the batch path uses for this chirp mass."""
        return self._segment_samples(chirp_mass_solar, minimum_frequency, sampling_frequency) / sampling_frequency

    def generate_td_waveform(
        self,
        approximant: str,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> dict[str, TimeSeries]:
        """Generate plus/cross polarizations from ripple, conditioned to time domain."""
        fd = self.generate_fd_polarizations(
            approximant,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            **params,
        )
        hp_t, hc_t, epoch = self._to_time_domain(fd)
        t0 = epoch + tc
        dt = 1.0 / sampling_frequency
        return {
            "plus": TimeSeries(hp_t, t0=t0, dt=dt),
            "cross": TimeSeries(hc_t, t0=t0, dt=dt),
        }

    def generate_fd_polarizations(
        self,
        approximant: str,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> FrequencyDomainPolarizations:
        """Generate ripple's frequency-domain plus/cross polarizations (on-device).

        This is the building block the on-device (GPU) projection path consumes: the
        polarizations stay as JAX arrays and are not conditioned to the time domain.
        ``generate_td_waveform`` calls this and then inverse-FFTs the result.

        Args:
            approximant: A supported ripple approximant name.
            sampling_frequency: Sample rate in Hz; sets the Nyquist frequency.
            minimum_frequency: Low-frequency cutoff in Hz; bins below it are zeroed.
            **params: CBC source parameters (gwmock-pop canonical names or aliases).

        Returns:
            A :class:`FrequencyDomainPolarizations` with coalescence at ``t = 0``.
        """
        resolved = self._resolve_parameters(approximant, sampling_frequency, minimum_frequency, **params)
        return self._evaluate_fd(approximant, resolved, sampling_frequency, minimum_frequency)

    def generate_fd_polarizations_batch(
        self,
        approximant: str,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        parameters: Mapping[str, object],
        waveform_arguments: Mapping[str, object] | None = None,
    ) -> FrequencyDomainPolarizations:
        """Generate ripple FD polarizations for a batch of events on one shared grid.

        Evaluates ripple under ``jax.vmap`` over the catalogue, so all events share a
        single frequency grid. Because ``vmap`` needs a fixed shape, the grid is sized
        (worst case) for the longest inspiral in the batch — the smallest chirp mass,
        via the same post-Newtonian estimate as the per-event path — unless a fixed
        ``segment_duration`` was set on the backend. This is the on-device entry point
        for catalogue-scale (GPU) generation.

        Args:
            approximant: A supported ripple approximant name.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz; bins below it are zeroed.
            parameters: Mapping of **canonical** gwmock-pop parameter names (no aliases)
                to 1-D arrays of equal length ``n_events`` (e.g. ``detector_frame_mass_1``,
                ``spin_1z``, ``inclination``). Omitted optional parameters default to zero.
            waveform_arguments: Optional extra ripple constructor options applied to the
                whole batch (the preset is built once). Same whitelist as the per-event
                path — e.g. ``{"no_taper": True}`` for the NRTidal variants. These are
                constructor-level, not per-event, so they take scalars, not arrays.

        Returns:
            A :class:`FrequencyDomainPolarizations` whose ``plus`` and ``cross`` are
            ``(n_events, n_samples // 2 + 1)`` JAX arrays (coalescence at ``t = 0``).
        """
        resolved_arguments = self._resolve_waveform_arguments(
            approximant, {} if waveform_arguments is None else dict(waveform_arguments)
        )
        ripple_params, n_samples = self._resolve_batch(approximant, sampling_frequency, minimum_frequency, parameters)
        jnp = self._jnp
        delta_f = sampling_frequency / n_samples
        freqs = jnp.arange(n_samples // 2 + 1) * delta_f
        f_ref = self._f_ref if self._f_ref is not None else minimum_frequency

        # Fetched from a cache keyed on everything the kernel depends on, so repeated calls
        # reuse one compiled executable. Building jax.jit around a closure defined here
        # would hand XLA a new callable every call and re-pay tracing, lowering and
        # compilation each time -- about 121 s per call for IMRPhenomXPHM on an A100, which
        # made the batched path slower than the per-event LAL loop it replaces.
        kernel = _batched_polarization_kernel(
            approximant,
            n_samples,
            sampling_frequency,
            minimum_frequency,
            f_ref,
            tuple(sorted(resolved_arguments.items())),
        )
        plus, cross = kernel(ripple_params)
        return FrequencyDomainPolarizations(
            frequencies=freqs,
            plus=plus,
            cross=cross,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
        )

    def _resolve_batch(
        self,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        parameters: Mapping[str, object],
    ) -> tuple[dict, int]:
        """Validate a batch of canonical parameters and build ripple-native arrays.

        Returns ``(ripple_params, n_samples)`` where ``ripple_params`` is a dict of
        equal-length JAX arrays ready for ``vmap`` and ``n_samples`` is the shared,
        worst-case segment length.
        """
        if approximant not in _SUPPORTED_APPROXIMANTS:
            raise ValueError(
                f"RippleBackend does not support approximant {approximant!r}. "
                f"Available: {list(_SUPPORTED_APPROXIMANTS)}."
            )
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be > 0")
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be > 0")
        if "waveform_arguments" in parameters:
            raise ValueError(
                "Pass waveform_arguments as its own keyword argument to "
                "generate_fd_polarizations_batch, not inside parameters"
            )

        jnp = self._jnp
        mass1 = self._batch_array(parameters, "detector_frame_mass_1")
        n_events = mass1.shape[0]
        mass2 = self._batch_array(parameters, "detector_frame_mass_2", n_events)
        distance = self._batch_array(parameters, "luminosity_distance", n_events)
        inclination = self._batch_array(parameters, "inclination", n_events, default=0.0)
        coa_phase = self._batch_array(parameters, "coa_phase", n_events, default=0.0)
        spins = {
            name: self._batch_array(parameters, name, n_events, default=0.0)
            for name in ("spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z")
        }
        lambda_1 = self._batch_array(parameters, "lambda_1", n_events, default=0.0)
        lambda_2 = self._batch_array(parameters, "lambda_2", n_events, default=0.0)

        is_precessing = approximant in _PRECESSING_MODELS
        if not is_precessing:
            for name in ("spin_1x", "spin_1y", "spin_2x", "spin_2y"):
                if bool(jnp.any(spins[name] != 0.0)):
                    raise ValueError(f"{approximant} is an aligned-spin model; {name} must be zero for all events.")
        is_tidal = approximant in _TIDAL_MODELS
        if not is_tidal and (bool(jnp.any(lambda_1 != 0.0)) or bool(jnp.any(lambda_2 != 0.0))):
            raise ValueError(f"{approximant} does not support tidal parameters; use an NRTidal approximant.")
        if bool(jnp.any(lambda_1 < 0.0)) or bool(jnp.any(lambda_2 < 0.0)):
            raise ValueError("lambda_1 and lambda_2 must be >= 0")

        chirp_mass, eta = self._jax.vmap(self._conversions.ms_to_Mc_eta)(jnp.stack([mass1, mass2], axis=-1))
        n_samples = self._segment_samples(float(jnp.min(chirp_mass)), minimum_frequency, sampling_frequency)

        ripple_params = {
            "M_c": chirp_mass,
            "eta": eta,
            "s1_z": spins["spin_1z"],
            "s2_z": spins["spin_2z"],
            "d_L": distance,
            "phase_c": coa_phase,
            "iota": inclination,
        }
        if is_precessing:
            ripple_params["s1_x"] = spins["spin_1x"]
            ripple_params["s1_y"] = spins["spin_1y"]
            ripple_params["s2_x"] = spins["spin_2x"]
            ripple_params["s2_y"] = spins["spin_2y"]
        if is_tidal:
            ripple_params["lambda_1"] = lambda_1
            ripple_params["lambda_2"] = lambda_2
        return ripple_params, n_samples

    def _batch_array(
        self,
        parameters: Mapping[str, object],
        name: str,
        n_events: int | None = None,
        *,
        default: float | None = None,
    ) -> Array:
        """Return one parameter as a 1-D float64 JAX array, validating its length."""
        jnp = self._jnp
        if name not in parameters:
            if default is None:
                raise ValueError(f"Missing required batch parameter: {name!r}")
            return jnp.full(n_events, default, dtype=jnp.float64)
        values = jnp.asarray(parameters[name], dtype=jnp.float64)
        if values.ndim != 1:
            raise ValueError(f"Batch parameter {name!r} must be 1-D; got shape {values.shape}.")
        if n_events is not None and values.shape[0] != n_events:
            raise ValueError(f"Batch parameter {name!r} has length {values.shape[0]}, expected {n_events}.")
        return values

    @staticmethod
    def _resolve_waveform_arguments(approximant: str, value: object) -> dict[str, object]:
        """Validate the optional extra ripple constructor options.

        Only the keys whitelisted in :data:`_ALLOWED_WAVEFORM_ARGUMENTS` for this
        approximant are accepted; backend-owned or contract-breaking keys
        (:data:`_RESERVED_WAVEFORM_ARGUMENTS`) are rejected with a specific reason,
        and any other key fails early rather than reaching ripple as an opaque
        ``TypeError``.
        """
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("waveform_arguments must be a dict with string keys")
        for key in value:
            if key in _RESERVED_WAVEFORM_ARGUMENTS:
                raise ValueError(_RESERVED_WAVEFORM_ARGUMENTS[key])
        allowed = _ALLOWED_WAVEFORM_ARGUMENTS.get(approximant, frozenset())
        unknown = sorted(key for key in value if key not in allowed)
        if unknown:
            joined = ", ".join(unknown)
            allowed_str = ", ".join(sorted(allowed)) if allowed else "(none)"
            raise ValueError(
                f"{approximant} does not accept waveform_arguments: {joined}. "
                f"Allowed for this approximant: {allowed_str}."
            )
        return dict(value)

    def _resolve_parameters(
        self,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> _ResolvedParameters:
        """Validate inputs and translate canonical parameters to backend-native ones."""
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
        waveform_arguments = self._resolve_waveform_arguments(
            approximant, _pop_alias(remaining, "waveform_arguments", default={})
        )
        mass1 = float(_pop_alias(remaining, "detector_frame_mass_1", "mass1"))
        mass2 = float(_pop_alias(remaining, "detector_frame_mass_2", "mass2"))
        distance = float(_pop_alias(remaining, "luminosity_distance", "distance"))
        spins = {
            "spin_1x": float(_pop_alias(remaining, "spin_1x", "spin1x", default=0.0)),
            "spin_1y": float(_pop_alias(remaining, "spin_1y", "spin1y", default=0.0)),
            "spin_1z": float(_pop_alias(remaining, "spin_1z", "spin1z", default=0.0)),
            "spin_2x": float(_pop_alias(remaining, "spin_2x", "spin2x", default=0.0)),
            "spin_2y": float(_pop_alias(remaining, "spin_2y", "spin2y", default=0.0)),
            "spin_2z": float(_pop_alias(remaining, "spin_2z", "spin2z", default=0.0)),
        }
        inclination = float(_pop_alias(remaining, "inclination", default=0.0))
        coa_phase = float(_pop_alias(remaining, "coa_phase", default=0.0))

        is_precessing = approximant in _PRECESSING_MODELS
        if not is_precessing:
            in_plane = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")
            nonzero_in_plane = sorted(name for name in in_plane if spins[name] != 0.0)
            if nonzero_in_plane:
                raise ValueError(
                    f"{approximant} is an aligned-spin model; "
                    f"in-plane spins must be zero: {', '.join(nonzero_in_plane)}"
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

        return _ResolvedParameters(
            mass1=mass1,
            mass2=mass2,
            spins=spins,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            is_tidal=is_tidal,
            is_precessing=is_precessing,
            f_ref=self._f_ref if self._f_ref is not None else minimum_frequency,
            waveform_arguments=waveform_arguments,
        )

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

    def _evaluate_fd(
        self,
        approximant: str,
        resolved: _ResolvedParameters,
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> FrequencyDomainPolarizations:
        """Evaluate ripple on the analysis frequency grid (coalescence at t=0)."""
        jnp = self._jnp
        spins = resolved.spins
        chirp_mass, eta = self._conversions.ms_to_Mc_eta(jnp.array([resolved.mass1, resolved.mass2]))

        n_samples = self._segment_samples(float(chirp_mass), minimum_frequency, sampling_frequency)
        delta_f = sampling_frequency / n_samples
        freqs = jnp.arange(n_samples // 2 + 1) * delta_f

        # ripple's class interface fixes its internal tc=0; coalescence is placed
        # in the time grid by _to_time_domain.
        ripple_params = {
            "M_c": chirp_mass,
            "eta": eta,
            "s1_z": spins["spin_1z"],
            "s2_z": spins["spin_2z"],
            "d_L": resolved.distance,
            "phase_c": resolved.coa_phase,
            "iota": resolved.inclination,
        }
        if resolved.is_precessing:
            ripple_params["s1_x"] = spins["spin_1x"]
            ripple_params["s1_y"] = spins["spin_1y"]
            ripple_params["s2_x"] = spins["spin_2x"]
            ripple_params["s2_y"] = spins["spin_2y"]
        if resolved.is_tidal:
            ripple_params["lambda_1"] = resolved.lambda_1
            ripple_params["lambda_2"] = resolved.lambda_2
        waveform = self._ripplegw.waveform_preset[approximant](f_ref=resolved.f_ref, **resolved.waveform_arguments)
        polarizations = waveform(freqs, ripple_params)

        # Zero out-of-band bins (including DC, where the amplitude diverges) and
        # guard against any non-finite values, keeping everything on device.
        in_band = freqs >= minimum_frequency
        hp_f = jnp.nan_to_num(jnp.where(in_band, polarizations["p"], 0.0))
        hc_f = jnp.nan_to_num(jnp.where(in_band, polarizations["c"], 0.0))
        return FrequencyDomainPolarizations(
            frequencies=freqs,
            plus=hp_f,
            cross=hc_f,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
        )

    def coalescence_placement(self, n_samples: int, sampling_frequency: float) -> tuple[int, float]:
        """Return ``(merger_index, epoch)`` for placing coalescence in a segment.

        ``merger_index`` is the sample at which coalescence sits after the
        time-domain roll (near the segment end, leaving a small ringdown pad), and
        ``epoch`` is the time of the first sample relative to coalescence (negative),
        so a caller places coalescence at ``epoch + tc``. Shared by the time-domain
        backend and the batched device path so both use the same convention.
        """
        merger_index = round((1.0 - self._ringdown_fraction) * n_samples)
        return merger_index, -merger_index / sampling_frequency

    def _to_time_domain(self, fd: FrequencyDomainPolarizations) -> tuple[np.ndarray, np.ndarray, float]:
        """Inverse-FFT frequency-domain polarizations and place coalescence in the segment.

        Returns ``(hp, hc, epoch)`` where ``epoch`` is the time of the first sample
        relative to coalescence (negative), so the caller places coalescence at
        ``epoch + tc``.
        """
        dt = 1.0 / fd.sampling_frequency
        # Inverse real FFT: h(t) = irfft(h(f)) / dt (continuous-transform normalization).
        hp_t = np.fft.irfft(np.asarray(fd.plus), n=fd.n_samples) / dt
        hc_t = np.fft.irfft(np.asarray(fd.cross), n=fd.n_samples) / dt

        # With tc=0 coalescence lands at sample 0 and the inspiral wraps to the tail.
        # Roll it forward so coalescence sits near the segment end, leaving the
        # inspiral contiguous before it and a small ringdown pad after.
        merger_index, epoch = self.coalescence_placement(fd.n_samples, fd.sampling_frequency)
        hp_t = np.roll(hp_t, merger_index)
        hc_t = np.roll(hc_t, merger_index)
        return hp_t, hc_t, epoch
