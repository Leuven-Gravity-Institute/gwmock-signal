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
"""Registry of waveform generators keyed by approximant or custom name."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import LALSimulationBackend, WaveformBackend

logger = logging.getLogger("gwmock_signal.waveform")


class WaveformFactory:
    """Registry and dispatcher for time-domain waveform generators.

    On construction, every name returned by the configured backend is registered
    and mapped to that backend's ``generate_td_waveform`` implementation.
    You may register additional names pointing at custom callables. See package docs for examples.
    """

    def __init__(self, backend: WaveformBackend | None = None) -> None:
        """Build the registry of built-in backend approximants.

        Note:
            Enumerating approximants can be slow; reuse one factory
            instance in tight loops instead of creating many factories.
        """
        self._backend = backend or LALSimulationBackend()
        self._models: dict[str, Callable[..., dict[str, TimeSeries]]] = {
            name: self._wrap_backend_call(name) for name in self._backend.available_approximants()
        }
        # Kept so :meth:`pre_coalescence_duration` can tell a backend approximant from a custom
        # registration. Compared by identity rather than by name, because
        # ``register_waveform_model`` may *shadow* a backend name -- and then the backend's answer
        # would describe a waveform nobody is generating.
        self._backend_models = dict(self._models)

    def _wrap_backend_call(self, default_approximant: str) -> Callable[..., dict[str, TimeSeries]]:
        """Adapt the backend interface to the factory's callable registry contract."""

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
                if supplied_name is not None and supplied_name != default_approximant:
                    raise ValueError(
                        f"Registered model {default_approximant!r} cannot be called with conflicting "
                        f"approximant {supplied_name!r}."
                    )
            model_name = default_approximant
            return self._backend.generate_td_waveform(
                approximant=model_name,
                tc=tc,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                **params,
            )

        return _call_backend

    def register_model(self, name: str, factory_func: Callable[..., Any] | str) -> None:
        """Register or overwrite a waveform model under ``name``.

        Args:
            name: Key used with ``WaveformFactory.generate`` and ``WaveformFactory.get_model``.
            factory_func: Callable that accepts merged waveform kwargs (including
                ``waveform_model``, ``tc``, ``sampling_frequency``, ``minimum_frequency``)
                and returns a dict of GWpy ``plus``/``cross`` series, **or** an import
                string: either ``module.path:callable`` (colon before the name) or
                ``package.module.callable`` (split on the last ``.`` for attribute lookup).

        Raises:
            ImportError: If a string path does not refer to an importable module.
            AttributeError: If the imported module has no such callable attribute.
            ValueError: If factory_func string is neither 'module.path:callable' nor 'package.module.callable'.
            TypeError: Registered model is not callable.
        """
        if isinstance(factory_func, str):
            if ":" in factory_func:
                module_path, func_name = factory_func.split(":", 1)
            else:
                if "." not in factory_func:
                    raise ValueError("factory_func string must be 'module.path:callable' or 'package.module.callable'")
                module_path, func_name = factory_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            factory_func = getattr(module, func_name)

        if not callable(factory_func):
            raise TypeError(f"Registered model '{name}' is not callable")

        self._models[name] = factory_func
        logger.info("Registered waveform model: %s", name)

    def get_model(self, name: str) -> Callable[..., dict[str, TimeSeries]]:
        """Look up the generator function registered for ``name``.

        Args:
            name: Registered model name (built-in approximant or custom).

        Returns:
            The callable registered for this name.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        if name in self._models:
            return self._models[name]
        raise ValueError(f"Waveform model '{name}' not found. Available: {list(self._models.keys())}.")

    def pre_coalescence_duration(
        self,
        name: str,
        sampling_frequency: float,
        minimum_frequency: float,
        **params: object,
    ) -> float | None:
        """Return how long before ``tc`` the buffer for ``name`` starts, or ``None`` if unknown.

        Delegates to the backend, which computes it from the same sizing its own generation uses.
        See :meth:`~gwmock_signal.waveform.backends.base.WaveformBackend.pre_coalescence_duration`
        for what the number means -- in particular that it is where the *buffer* starts, not where
        audible signal begins, and that ``None`` means unknown rather than zero.

        Returns ``None`` for a custom registered model. Those are arbitrary callables that never
        reach the backend, so the backend's sizing would not describe what they produce. The check
        is by identity against the wrappers built at construction, so a registration that shadows a
        backend approximant is also excluded.

        Args:
            name: Waveform model name, as passed to :meth:`get_model`.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            **params: Source parameters, as generation would receive them.

        Raises:
            ValueError: If ``name`` is not registered at all, matching :meth:`get_model`.
        """
        if name not in self._models:
            raise ValueError(f"Waveform model '{name}' not found. Available: {list(self._models.keys())}.")
        if self._models[name] is not self._backend_models.get(name):
            return None
        return self._backend.pre_coalescence_duration(name, sampling_frequency, minimum_frequency, **params)

    def list_models(self) -> list[str]:
        """Return every registered waveform model name, in dict iteration order.

        Returns:
            List of keys (backend approximants plus any custom registrations).
        """
        return list(self._models.keys())

    def generate(
        self,
        waveform_model: str,
        parameters: dict[str, Any],
        **extra_params: Any,
    ) -> dict[str, TimeSeries]:
        """Generate polarizations by calling the registered model with merged parameters.

        The callable is invoked with ``waveform_model``, then entries from ``parameters``,
        then ``extra_params`` (later keys override earlier ones).

        Args:
            waveform_model: Name of the registered model to run.
            parameters: Injection parameters (e.g. ``tc``, masses, spins) merged first.
            **extra_params: Additional fixed settings (e.g. ``sampling_frequency``,
                ``minimum_frequency``) merged after ``parameters``; later keys override.

        Returns:
            Dict whose keys are the strings ``plus`` and ``cross``, each mapping to a
            GWpy [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).

        Raises:
            ValueError: If ``waveform_model`` is not registered.
            TypeError: If the underlying generator is called with invalid arguments.
        """
        waveform_func = self.get_model(waveform_model)
        if "waveform_model" in parameters or "waveform_model" in extra_params:
            raise ValueError("Do not pass 'waveform_model' in parameters/extra_params.")
        all_params: dict[str, Any] = {**parameters, **extra_params, "waveform_model": waveform_model}
        return waveform_func(**all_params)
