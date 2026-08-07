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
"""Abstract base class and CBC concrete implementation for GW signal simulation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gwpy.timeseries import TimeSeries

from gwmock_signal.injection import inject_strain
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.waveform import WaveformBackend, WaveformFactory

if TYPE_CHECKING:
    from gwmock_signal.detector import CustomDetector

logger = logging.getLogger("gwmock_signal.simulator")

_POLARIZATION_COUNT = 2
RegisteredWaveformOutput = dict[str, TimeSeries] | tuple[TimeSeries, TimeSeries]
RegisteredWaveformFactory = Callable[..., RegisteredWaveformOutput] | WaveformBackend


def _json_default(obj: Any) -> Any:
    """Convert NumPy types to Python natives for JSON serialization."""
    if hasattr(obj, "tolist"):  # NumPy array
        return obj.tolist()
    if hasattr(obj, "item"):  # NumPy scalar
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _normalize_registered_waveform_output(result: RegisteredWaveformOutput) -> dict[str, TimeSeries]:
    """Return one ``{"plus": ..., "cross": ...}`` mapping for a registered model."""
    if isinstance(result, tuple):
        if len(result) != _POLARIZATION_COUNT:
            raise TypeError("Registered waveform callables must return exactly two TimeSeries objects.")
        plus, cross = result
    elif isinstance(result, Mapping):
        try:
            plus = result["plus"]
            cross = result["cross"]
        except KeyError as exc:
            raise ValueError("Registered waveform callables must return 'plus' and 'cross' entries.") from exc
    else:
        raise TypeError("Registered waveform callables must return a ('plus', 'cross') tuple or mapping.")

    if not isinstance(plus, TimeSeries) or not isinstance(cross, TimeSeries):
        raise TypeError("Registered waveform callables must return GWpy TimeSeries objects for both polarizations.")
    return {"plus": plus, "cross": cross}


def _wrap_registered_waveform_callable(
    factory: Callable[..., RegisteredWaveformOutput],
) -> Callable[..., dict[str, TimeSeries]]:
    """Adapt one registered callable to the factory's ``{"plus", "cross"}`` contract."""

    def _call_registered_waveform(**kwargs: Any) -> dict[str, TimeSeries]:
        return _normalize_registered_waveform_output(factory(**kwargs))

    return _call_registered_waveform


def _wrap_registered_waveform_backend(
    name: str,
    backend: WaveformBackend,
) -> Callable[..., dict[str, TimeSeries]]:
    """Expose one backend approximant through ``WaveformFactory.register_model``."""

    def _call_backend(
        *,
        waveform_model: str | None = None,
        approximant: str | None = None,
        tc: float,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: Any,
    ) -> dict[str, TimeSeries]:
        for supplied_name in (waveform_model, approximant):
            if supplied_name is not None and supplied_name != name:
                raise ValueError(
                    f"Registered model {name!r} cannot be called with conflicting approximant {supplied_name!r}."
                )
        return backend.generate_td_waveform(
            approximant=name,
            tc=tc,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            **params,
        )

    return _call_backend


class GWSimulator(ABC):
    """Abstract base class for gravitational-wave signal simulators.

    Defines the stable, source-agnostic contract used across packages:
    subclasses must implement ``required_params`` and ``simulate`` and must
    return a ``DetectorStrainStack``. ``_validate_params`` is provided as a
    concrete helper.

    See ``docs/api/simulator/index.md`` for the API reference.
    """

    @property
    @abstractmethod
    def required_params(self) -> frozenset[str]:
        """Parameter keys that must be present in the params dict passed to ``simulate``."""

    def _validate_params(self, params: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` naming any key in ``required_params`` missing from ``params``.

        Args:
            params: Source parameters to validate.

        Raises:
            ValueError: If any required key is absent, naming each missing key.
        """
        missing = self.required_params - set(params)
        if missing:
            raise ValueError(f"Missing required parameters: {sorted(missing)}")

    @abstractmethod
    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[str],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Return detector strain for one source realization as a ``DetectorStrainStack``.

        Args:
            params: Source parameters.
            detector_names: IFO codes (e.g. ``'H1'``, ``'L1'``, ``'V1'``).
            background: Optional mapping of detector name to existing background
                strain. Non-transient subclasses may use this to inject into
                pre-existing data; transient subclasses may ignore it and return
                signal-only strain directly.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            earth_rotation: Optional backend-specific flag controlling whether
                detector response should vary across the signal duration.
            interpolate_if_offset: Optional backend-specific flag controlling
                how off-grid injections should be handled.

        Returns:
            ``DetectorStrainStack`` containing one aligned strain channel per
            detector in ``detector_names`` order.
        """


class TransientSimulator(GWSimulator):
    """Intermediate base class for transient GW source simulators.

    Provides the concrete ``simulate`` method that orchestrates the full
    transient injection pipeline: validation → polarizations → detector
    projection → strain injection → stacking. It also exposes
    ``register_waveform_model`` so each simulator instance can add custom
    waveform generators without poking private ``WaveformFactory`` attributes.
    Subclasses must implement ``generate_polarizations`` and ``required_params``.
    """

    @abstractmethod
    def generate_polarizations(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> tuple[TimeSeries, TimeSeries]:
        """Generate plus and cross polarization time series.

        Args:
            params: Source parameters.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.

        Returns:
            Tuple of ``(hp, hc)`` GWpy ``TimeSeries`` objects.
        """

    def register_waveform_model(
        self,
        name: str,
        factory: RegisteredWaveformFactory,
    ) -> None:
        """Register a one-shot waveform generator under ``name`` for this instance.

        Re-registering the same name with an equal factory is a no-op. Reusing
        the name for a different factory raises ``ValueError``.

        Args:
            name: Waveform model name used later as ``waveform_model``.
            factory: Callable returning either ``(plus, cross)`` or a
                ``{"plus": ..., "cross": ...}`` mapping, or a ``WaveformBackend``
                whose ``generate_td_waveform`` method should serve that name.
        """
        if not hasattr(self, "_waveform_factory"):
            raise AttributeError(
                "TransientSimulator subclasses must initialize _waveform_factory before registering waveform models."
            )

        registered_factories = self.__dict__.setdefault("_registered_waveform_models", {})
        if name in registered_factories:
            if registered_factories[name] == factory:
                return
            raise ValueError(f"Waveform model {name!r} is already registered for this simulator instance.")

        try:
            self._waveform_factory.get_model(name)
        except ValueError:
            pass
        else:
            raise ValueError(f"Waveform model {name!r} is already registered for this simulator instance.")

        if not isinstance(factory, WaveformBackend) and not callable(factory):
            raise TypeError(f"Waveform model {name!r} must be registered with a callable or WaveformBackend.")

        wrapped_factory = (
            _wrap_registered_waveform_backend(name, factory)
            if isinstance(factory, WaveformBackend)
            else _wrap_registered_waveform_callable(factory)
        )
        self._waveform_factory.register_model(name, wrapped_factory)
        registered_factories[name] = factory

    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[str | CustomDetector],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Run the full injection pipeline and return a ``DetectorStrainStack``.

        Calls ``_validate_params``, ``generate_polarizations``,
        ``project_polarizations_to_network``, and ``inject_strain`` in order,
        then assembles the result into a ``DetectorStrainStack``.

        Args:
            params: Source parameters; must contain all ``required_params`` keys
                plus ``right_ascension``, ``declination``, and ``polarization_angle``
                for antenna-response projection.
            detector_names: IFO codes (e.g. ``'H1'``, ``'L1'``, ``'V1'``).
            background: Optional mapping of detector name to background
                ``TimeSeries`` (e.g. noise or zeros). If omitted, the projected
                detector response is returned directly on the waveform time grid.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            earth_rotation: If ``True``, evaluate antenna patterns at
                time-dependent GPS times (recommended for longer signals).
                If ``False``, use a single reference time.
            interpolate_if_offset: If ``True``, interpolate the injection
                when its start is not on a target sample boundary.

        Returns:
            ``DetectorStrainStack`` containing the simulated strain for each
            detector in ``detector_names`` order.
        """
        self._validate_params(params)

        # Validate projection keys before direct params[...] access.
        # _validate_params only checks subclass-defined required_params. Non-CBC subclasses can pass validation and then fail mid-pipeline with KeyError.
        projection_keys = {"right_ascension", "declination", "polarization_angle"}
        missing_projection = projection_keys - set(params)
        if missing_projection:
            raise ValueError(f"Missing required parameters: {sorted(missing_projection)}")

        # Normalise detector entries so dict lookups use plain string keys.
        str_names: list[str] = [d.name if hasattr(d, "name") and not isinstance(d, str) else d for d in detector_names]

        hp, hc = self.generate_polarizations(params, sampling_frequency, minimum_frequency)
        projected = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            detector_names,
            right_ascension=params["right_ascension"],
            declination=params["declination"],
            polarization_angle=params["polarization_angle"],
            earth_rotation=earth_rotation,
        )
        if background is None:
            return DetectorStrainStack.from_mapping(str_names, projected)
        injected = {
            name: inject_strain(
                background[name],
                projected[name],
                interpolate_if_offset=interpolate_if_offset,
            )
            for name in str_names
        }
        return DetectorStrainStack.from_mapping(str_names, injected)


#: Parameters the *projection* consumes, which the waveform backend must never be handed.
#:
#: One definition, because two code paths need the same answer: generation, and
#: :meth:`CBCSimulator.pre_coalescence_duration`, which must be callable with exactly the mapping
#: generation takes. They were filtered in one place and not the other, so a caller passing a
#: complete CBC mapping got a `ValueError` about unsupported LAL parameters from the duration query
#: while generation accepted the same input -- the query being unusable by its only intended caller.
_PROJECTION_ONLY_PARAMETERS = frozenset({"right_ascension", "declination", "polarization_angle", "coa_time"})


def _waveform_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return *params* without the keys the projection owns, which the backend rejects."""
    return {key: value for key, value in params.items() if key not in _PROJECTION_ONLY_PARAMETERS}


class CBCSimulator(TransientSimulator):
    """Compact binary coalescence simulator backed by ``WaveformFactory``.

    Generates time-domain polarizations via a pluggable waveform backend,
    projects them onto the requested detectors, and injects them into background
    strain. ``waveform_model`` is a CBC concern and is supplied at construction
    time, keeping the base-class ``simulate`` interface source-agnostic.

    The public interface accepts **gwmock-pop canonical parameter names**
    (e.g. ``detector_frame_mass_1``, ``coa_time``, ``polarization_angle``).
    Backend-specific translation is handled internally and is invisible to callers.

    Args:
        waveform_model: Time-domain approximant name or any model
            registered with ``WaveformFactory`` (e.g. ``'IMRPhenomD'``).
        waveform_backend: Optional waveform backend instance. Defaults to
            ``LALSimulationBackend`` through ``WaveformFactory``.
    """

    #: Minimum parameter keys required by the CBC pipeline (gwmock-pop canonical names).
    _REQUIRED: frozenset[str] = frozenset(
        {
            "detector_frame_mass_1",
            "detector_frame_mass_2",
            "coa_time",
            "distance",
            "inclination",
            "right_ascension",
            "declination",
            "polarization_angle",
        }
    )

    def __init__(self, waveform_model: str, waveform_backend: WaveformBackend | None = None) -> None:
        """Initialise with the waveform model name.

        Args:
            waveform_model: Time-domain approximant name or any model
                registered with ``WaveformFactory``.
            waveform_backend: Optional waveform backend instance.
        """
        self._waveform_model = waveform_model
        self._waveform_factory = WaveformFactory(backend=waveform_backend)

    @property
    def waveform_model(self) -> str:
        """PyCBC approximant or custom model name."""
        return self._waveform_model

    @property
    def required_params(self) -> frozenset[str]:
        """Return the fixed set of required CBC parameter keys."""
        return self._REQUIRED

    def pre_coalescence_duration(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> float | None:
        """Return how long before ``coa_time`` this simulator's output starts, in seconds.

        Answered before generating, which is the point: a caller placing signals into segmented
        data needs to know that a compact binary's buffer begins well before its coalescence, so a
        ``coa_time`` just past a segment boundary belongs to an *earlier* segment. Choosing the
        claiming segment from ``coa_time`` alone crops the start away, and the loss is not marginal.

        **The size of that loss is not a single number** -- it moves by orders of magnitude with the
        low-frequency cutoff, the offset past the boundary, the backend, the approximant and the
        component spins, so no bare percentage is quoted here. The measured tables are in the user
        guide, under *How much signal a segment boundary costs*, together with what they are a proxy
        for and what they are not anchored against.

        Exposed here rather than left to callers to reach the backend through
        ``_waveform_factory``. Two private attributes across a package boundary would break
        silently on any internal rename, and a caller does not reliably own the backend instance --
        this class constructs one when none is supplied.

        Mirrors :meth:`generate_polarizations`, so the parameters, sample rate and cutoff are the
        same ones generation will be given and the answer describes the buffer it will produce.

        Args:
            params: Source parameters, as :meth:`generate_polarizations` takes them.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.

        Returns:
            Seconds between the first sample and coalescence, always positive; or ``None`` when the
            backend cannot say. Treat ``None`` as *unknown*, never as zero -- see
            :meth:`~gwmock_signal.waveform.backends.base.WaveformBackend.pre_coalescence_duration`.
            Note the value is where the buffer starts, not where audible signal begins, which makes
            it safe for placement: a segment chosen from it never crops real signal.
        """
        return self._waveform_factory.pre_coalescence_duration(
            self._waveform_model,
            sampling_frequency,
            minimum_frequency,
            **_waveform_parameters(params),
        )

    def post_coalescence_duration(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> float | None:
        """Return how long after ``coa_time`` this simulator's output runs, in seconds.

        The complement of :meth:`pre_coalescence_duration`, and the half that answers a different
        question: the pre side tells a caller an event *begins* before a segment, this one tells
        it an event has *finished* before one. Without it, a run whose population starts earlier
        than the run itself cannot distinguish an event whose content is entirely in the past from
        one whose tail lands in the segment being written, and must generate both to find out.

        **This is buffer padding, not physical ringdown.** It is a fraction of the buffer, so it
        scales with it: a fraction of a second for a stellar-mass binary, tens of seconds for a
        binary neutron star at the same cutoff, and longer as the cutoff falls. A caller
        substituting a constant is not approximating it.

        Exposed here for the same reason as the pre side: reaching the backend through
        ``_waveform_factory`` means two private attributes across a package boundary, and a caller
        does not reliably own the backend instance.

        Args:
            params: Source parameters, as :meth:`generate_polarizations` takes them.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.

        Returns:
            Seconds from coalescence to one sample past the buffer's end, always positive; or
            ``None`` when the backend cannot say. Treat ``None`` as *unknown*, never as zero --
            zero asserts the content stops at coalescence, which would discard exactly the events
            whose tail reaches into the segment being written.
        """
        return self._waveform_factory.post_coalescence_duration(
            self._waveform_model,
            sampling_frequency,
            minimum_frequency,
            **_waveform_parameters(params),
        )

    def generate_polarizations(
        self,
        params: Mapping[str, Any],
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> tuple[TimeSeries, TimeSeries]:
        """Delegate waveform generation to ``WaveformFactory``.

        Projection-specific keys are excluded because they are consumed by
        ``TransientSimulator.simulate`` rather than by waveform generation.

        Args:
            params: CBC source parameters using gwmock-pop canonical names
                (e.g. ``detector_frame_mass_1``, ``coa_time``).
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.

        Returns:
            Tuple of ``(hp, hc)`` GWpy ``TimeSeries`` objects.
        """
        waveform_params = _waveform_parameters(params)

        result = self._waveform_factory.generate(
            self._waveform_model,
            waveform_params,
            tc=params["coa_time"],
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
        )
        return result["plus"], result["cross"]

    def write(  # noqa: PLR0913
        self,
        path: str | Path,
        params: Mapping[str, Any],
        detector_names: Sequence[str | CustomDetector],
        background: Mapping[str, TimeSeries],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        format: Literal["gwf", "hdf5", "npy", "txt"] = "hdf5",  # noqa: A002
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Simulate a CBC injection, write the result, and save the parameter dict.

        Calls :meth:`simulate`, writes the returned ``DetectorStrainStack`` to
        *path* via :meth:`DetectorStrainStack.write`, and saves *params* as a
        JSON sidecar at ``<stem>_params.json`` next to *path*.

        Args:
            path: Output file path.
            params: CBC source parameters (same as :meth:`simulate`).
            detector_names: IFO codes for the target network.
            background: Mapping of detector name to background ``TimeSeries``.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            format: Output format for the strain data — one of ``'gwf'``,
                ``'hdf5'``, ``'npy'``, or ``'txt'``.  Defaults to ``'hdf5'``.
            earth_rotation: Passed through to :meth:`simulate`.
            interpolate_if_offset: Passed through to :meth:`simulate`.

        Returns:
            The ``DetectorStrainStack`` that was written to disk.
        """
        result = self.simulate(
            params,
            detector_names,
            background,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            earth_rotation=earth_rotation,
            interpolate_if_offset=interpolate_if_offset,
        )
        result.write(path, format=format)

        params_path = Path(path).with_name(Path(path).stem + "_params.json")
        params_path.write_text(json.dumps(dict(params), default=_json_default, indent=4))

        return result
