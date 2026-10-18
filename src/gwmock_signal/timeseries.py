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
"""A uniformly sampled time series whose samples are a NumPy or a JAX array.

:class:`TimeSeries` holds an array and the few numbers that place it in time: the GPS time of the
first sample and the sample rate. Time runs along the **last** axis; leading axes are allowed and
share that timing, so the strain of several detectors over one segment can be a single series of
shape ``(n_detectors, n_samples)``.

The array is kept as given. A NumPy array stays on the host, and a JAX array stays where it is,
on a GPU included, through every method here: each works through the array's own namespace
(``data.__array_namespace__()``), never by converting to NumPy. :meth:`TimeSeries.to_numpy` and
:meth:`TimeSeries.to_jax` move data deliberately.

The class is deliberately small. It has no arithmetic, spectral or plotting methods: code that
needs them works on :attr:`TimeSeries.data`, and a caller who wants GWpy's API converts with
:meth:`TimeSeries.to_gwpy`. Conversions to and from GWpy and PyCBC import those packages only when
called.

Under JAX, :func:`register_pytree` makes a series a pytree, so it passes through ``jax.jit`` and
``jax.vmap``. The samples and ``t0`` are leaves; the sample rate, name and unit are static. The
modules that use JAX register it; importing this module never imports JAX. JAX must be in 64-bit mode
(``jax.config.update("jax_enable_x64", True)``), as the package's device projection already requires:
GPS times need float64, and in 32-bit mode the methods that would lose it raise instead.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import math
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_signal.sampling_grid import SamplingGrid

if TYPE_CHECKING:
    from types import ModuleType

    import gwpy.timeseries
    import pycbc.types

__all__ = ["TimeSeries", "as_timeseries", "register_pytree"]


def _namespace(data: Any) -> ModuleType:
    """Return the array namespace of ``data``: ``numpy``, ``jax.numpy``, or another array-API library."""
    return data.__array_namespace__()


def _import(module: str, extra: str, feature: str) -> ModuleType:
    """Import an optional package, naming the extra that provides it when it is missing."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] != module.split(".", maxsplit=1)[0]:
            raise
        raise ModuleNotFoundError(
            f"{feature} needs {module.split('.', maxsplit=1)[0]}, which is not installed. "
            f"Install it with: pip install 'gwmock-signal[{extra}]'",
            name=exc.name,
        ) from exc


@dataclass(frozen=True, eq=False)
class TimeSeries:
    """A uniformly sampled series: sample ``k`` along the last axis is at ``t0 + k / sample_rate``.

    Args:
        data: The samples, an array with at least one axis; time is the last. A NumPy or a JAX
            array is kept as given. Anything else is converted with :func:`numpy.asarray`.
        t0: GPS time of the first sample, in seconds.
        sample_rate: Samples per second, in Hz.
        name: Optional label, such as a detector or channel name. It labels the whole series.
        unit: Optional unit of the samples, such as ``"strain"``.

    Raises:
        ValueError: If ``data`` has no axis, or ``t0`` or ``sample_rate`` is not finite, or
            ``sample_rate`` is not positive.
        TypeError: If ``t0`` is not a single number.

    Two series are equal only if they are the same object: comparing arrays elementwise has no
    single truth value, and a JAX array cannot be compared without a device round trip.
    """

    data: Any
    t0: float
    sample_rate: float
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the timing and normalise the samples to an array."""
        data = self.data
        if not hasattr(data, "__array_namespace__"):
            data = np.asarray(data)
            object.__setattr__(self, "data", data)
        if len(data.shape) == 0:
            raise ValueError("data must have at least one axis; time is the last.")
        sample_rate = float(self.sample_rate)
        if not (math.isfinite(sample_rate) and sample_rate > 0.0):
            raise ValueError(f"sample_rate must be positive and finite; got {self.sample_rate!r}.")
        object.__setattr__(self, "sample_rate", sample_rate)
        try:
            t0 = float(self.t0)
        except TypeError as exc:
            # A traced value inside a JAX transformation has no number yet to check; it was checked
            # when the series entered the transformation. Matched by name so JAX is not imported.
            if type(exc).__name__ == "ConcretizationTypeError":
                return
            raise TypeError(f"t0 must be a single number; got {self.t0!r}.") from exc
        if not math.isfinite(t0):
            raise ValueError(f"t0 must be finite; got {self.t0!r}.")
        object.__setattr__(self, "t0", t0)

    # ----------------------------------------------------------------------------------------
    # Derived quantities

    @property
    def value(self) -> Any:
        """The samples; an alias of :attr:`data`, the name GWpy uses."""
        return self.data

    @property
    def xp(self) -> ModuleType:
        """The array namespace of :attr:`data`, such as ``numpy`` or ``jax.numpy``."""
        return _namespace(self.data)

    @property
    def n_samples(self) -> int:
        """Number of samples along the time axis."""
        return int(self.data.shape[-1])

    @property
    def dt(self) -> float:
        """Seconds between samples."""
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        """Seconds spanned by the samples: ``n_samples * dt``."""
        return self.n_samples * self.dt

    @property
    def end(self) -> float:
        """GPS time of the sample after the last, on the series' lattice: ``t0 + n_samples * dt``."""
        return self._time_of(self.n_samples)

    @property
    def grid(self) -> SamplingGrid:
        """The sample lattice of this series."""
        return SamplingGrid(epoch=self.t0, sampling_frequency=self.sample_rate)

    @property
    def times(self) -> Any:
        """GPS time of each sample, in the namespace of :attr:`data`, as float64.

        Raises:
            RuntimeError: If the namespace cannot hold float64, as JAX cannot outside its 64-bit
                mode. In single precision a GPS time near 1.4e9 s resolves only to 128 s, so every
                sample of a typical series would get the same time.
        """
        xp = self.xp
        times = self.t0 + xp.arange(self.n_samples, dtype=xp.float64) * self.dt
        if times.dtype != xp.float64:
            raise RuntimeError(_NO_FLOAT64)
        return times

    # ----------------------------------------------------------------------------------------
    # New series from this one

    def with_data(self, data: Any) -> TimeSeries:
        """Return a series with the same timing, name and unit, holding ``data``."""
        return dataclasses.replace(self, data=data)

    def copy(self) -> TimeSeries:
        """Return a series whose samples are a copy of these."""
        return self.with_data(self.xp.asarray(self.data, copy=True))

    def crop(self, start: float | None = None, end: float | None = None) -> TimeSeries:
        """Return the samples at GPS times in ``[start, end)``.

        The bounds decide the result's length, so ``crop`` needs a concrete ``t0``: it cannot run
        inside a JAX transformation over the series.

        The result starts at the first sample at or after ``start`` and stops before the first at
        or after ``end``. A bound that falls on a sample, to within the float64 resolution of GPS
        times (:meth:`~gwmock_signal.sampling_grid.SamplingGrid.lattice_tolerance_samples`), counts
        as on it, so cropping at a sample's own timestamp keeps that sample however large the GPS
        time. Bounds outside the series are clipped to it.

        GWpy differs for a bound between samples: it rounds down, keeping the sample before
        ``start``. Here every kept sample is at or after ``start``.

        Args:
            start: First GPS time to keep; ``None`` keeps from the first sample.
            end: GPS time to stop before; ``None`` keeps to the last sample.

        Returns:
            The cropped series, possibly with no samples. Its samples are a view where the array
            library makes one.

        Raises:
            ValueError: If a bound is not finite, or ``end`` is before ``start``.
        """
        for bound_name, bound in (("start", start), ("end", end)):
            if bound is not None and not math.isfinite(bound):
                raise ValueError(f"{bound_name} must be finite; got {bound!r}.")
        if start is not None and end is not None and end < start:
            raise ValueError(f"end ({end}) is before start ({start}).")
        first = 0 if start is None else self._first_index_at_or_after(start)
        stop = self.n_samples if end is None else self._first_index_at_or_after(end)
        first = min(max(first, 0), self.n_samples)
        stop = min(max(stop, first), self.n_samples)
        return dataclasses.replace(self, data=self.data[..., first:stop], t0=self._time_of(first))

    def pad(self, before: int = 0, after: int = 0) -> TimeSeries:
        """Return the series with ``before`` zeros prepended and ``after`` appended, in samples.

        ``t0`` moves back by ``before`` samples, so the existing samples keep their times. The
        counts decide the result's length, so inside a JAX transformation they must be Python
        integers, not traced values.

        Raises:
            ValueError: If either count is negative.
        """
        if before < 0 or after < 0:
            raise ValueError(f"pad counts must not be negative; got before={before}, after={after}.")
        xp = self.xp
        lead = self.data.shape[:-1]
        parts = [
            xp.zeros((*lead, before), dtype=self.data.dtype),
            self.data,
            xp.zeros((*lead, after), dtype=self.data.dtype),
        ]
        return dataclasses.replace(self, data=xp.concatenate(parts, axis=-1), t0=self._time_of(-before))

    def is_compatible(self, other: TimeSeries) -> bool:
        """Return whether ``other`` has the same sample rate and unit, so the two can be combined."""
        return self.sample_rate == other.sample_rate and self.unit == other.unit

    def _time_of(self, index: int) -> Any:
        # The arithmetic of SamplingGrid.time_of, written out so it also works on a traced t0.
        return self.t0 + index * self.dt

    def _first_index_at_or_after(self, gps_time: float) -> int:
        grid = self.grid
        exact = float(grid.index_of(gps_time))
        return math.ceil(exact - grid.lattice_tolerance_samples(gps_time))

    # ----------------------------------------------------------------------------------------
    # Moving data and converting

    def to_numpy(self) -> TimeSeries:
        """Return the series with its samples as a NumPy array on the host."""
        return self.with_data(np.asarray(self.data))

    def to_jax(self, device: Any = None) -> TimeSeries:
        """Return the series with its samples as a JAX array, on ``device`` if given.

        Raises:
            ModuleNotFoundError: If JAX is not installed.
            RuntimeError: If JAX is not in 64-bit mode, in which it would round double-precision
                samples to single precision.
        """
        jax = _import("jax", "jax", "TimeSeries.to_jax")
        _require_x64(jax)
        return self.with_data(jax.device_put(jax.numpy.asarray(self.data), device))

    def to_gwpy(self) -> gwpy.timeseries.TimeSeries:
        """Return a GWpy ``TimeSeries`` with the same samples, timing, name and unit.

        NumPy samples are shared, not copied, as :meth:`from_gwpy` shares GWpy's, so a write through
        either object is seen by the other. JAX samples are brought to the host with
        :func:`numpy.asarray`; for a CPU array that is a read-only view of JAX's buffer.

        Raises:
            ValueError: If the series has more than one axis; GWpy's is one-dimensional.
            ModuleNotFoundError: If GWpy is not installed.
        """
        if len(self.data.shape) != 1:
            raise ValueError(f"a GWpy TimeSeries is one-dimensional; this series has shape {tuple(self.data.shape)}.")
        timeseries = _import("gwpy.timeseries", "gwpy", "TimeSeries.to_gwpy")
        return timeseries.TimeSeries(
            np.asarray(self.data),
            t0=self.t0,
            sample_rate=self.sample_rate,
            name=self.name,
            unit=self.unit,
            copy=False,
        )

    @classmethod
    def from_gwpy(cls, series: gwpy.timeseries.TimeSeries) -> TimeSeries:
        """Return a series with the samples, timing, name and unit of a GWpy ``TimeSeries``.

        The samples are GWpy's array itself, not a copy. A dimensionless GWpy series gets
        ``unit=None``.

        Raises:
            ValueError: If the GWpy series is irregularly sampled.
        """
        try:
            sample_rate = series.sample_rate.to_value("Hz")
        except AttributeError as exc:
            # GWpy has no sample rate for a series built from an explicit, irregular times array.
            raise ValueError(
                "a TimeSeries is uniformly sampled; this GWpy series has an irregular time index."
            ) from exc
        unit = series.unit.to_string() if series.unit is not None else ""
        return cls(
            series.value,
            t0=series.t0.to_value("s"),
            sample_rate=sample_rate,
            name=None if series.name is None else str(series.name),
            unit=unit or None,
        )

    def to_pycbc(self) -> pycbc.types.TimeSeries:
        """Return a PyCBC ``TimeSeries`` with the same samples and timing.

        PyCBC series carry neither a name nor a unit.

        Raises:
            ValueError: If the series has more than one axis; PyCBC's is one-dimensional.
            ModuleNotFoundError: If PyCBC is not installed.
        """
        if len(self.data.shape) != 1:
            raise ValueError(f"a PyCBC TimeSeries is one-dimensional; this series has shape {tuple(self.data.shape)}.")
        types = _import("pycbc.types", "pycbc", "TimeSeries.to_pycbc")
        return types.TimeSeries(np.asarray(self.data), delta_t=self.dt, epoch=self.t0)

    @classmethod
    def from_pycbc(
        cls, series: pycbc.types.TimeSeries, *, name: str | None = None, unit: str | None = None
    ) -> TimeSeries:
        """Return a series with the samples and timing of a PyCBC ``TimeSeries``."""
        return cls(
            np.asarray(series.numpy()),
            t0=float(series.start_time),
            sample_rate=float(series.sample_rate),
            name=name,
            unit=unit,
        )


def as_timeseries(series: TimeSeries | gwpy.timeseries.TimeSeries) -> TimeSeries:
    """Return ``series`` as a :class:`TimeSeries`: itself, or converted from a GWpy ``TimeSeries``.

    Functions that take a time series call this on their input, so they accept either kind.

    Raises:
        TypeError: If ``series`` is neither.
    """
    if isinstance(series, TimeSeries):
        return series
    # A GWpy series can only exist if GWpy is imported, so it is not imported here to find out.
    gwpy_timeseries = sys.modules.get("gwpy.timeseries")
    if gwpy_timeseries is not None and isinstance(series, gwpy_timeseries.TimeSeries):
        return TimeSeries.from_gwpy(series)
    raise TypeError(
        f"expected a gwmock_signal.TimeSeries or a gwpy.timeseries.TimeSeries; got {type(series).__name__}."
    )


def _public_output(series: TimeSeries) -> gwpy.timeseries.TimeSeries:
    """Return ``series`` in the form the public API returns: a GWpy ``TimeSeries``, for now.

    The single place where results leave the package as GWpy series. Making the public API return
    :class:`TimeSeries` is a change here, not at every function.
    """
    return series.to_gwpy()


_NO_FLOAT64 = (
    "TimeSeries needs float64 timing, which this array library is not providing; with JAX, call "
    'jax.config.update("jax_enable_x64", True) first. '
    "In single precision a GPS time near 1.4e9 s resolves only to 128 s, so every sample of "
    "a typical series would get the same time."
)


def _require_x64(jax: ModuleType) -> None:
    """Refuse, rather than degrade, when JAX is in 32-bit mode, as the device projection does."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(_NO_FLOAT64)


#: Serialises the first registrations: the cache alone lets concurrent first calls both run.
_REGISTRATION_LOCK = threading.Lock()


def register_pytree() -> None:
    """Register :class:`TimeSeries` as a JAX pytree; safe to call more than once, from any thread.

    The samples and ``t0`` are leaves, so a batch of series differing in start time can be mapped
    over without recompiling. The sample rate, name and unit are static: they decide shapes and
    meaning, not values.

    Unflattening bypasses validation, because inside a transformation the leaves are tracers,
    which cannot be checked as numbers. A series rebuilt with its start time as a JAX array that is
    not float64 -- what ``jit``, ``vmap``, ``grad`` and ``device_put`` produce in JAX's 32-bit mode --
    raises ``RuntimeError``; structural uses that leave it a Python float, such as ``jax.tree.map``
    or ``jax.device_get``, work in either mode.

    Raises:
        ModuleNotFoundError: If JAX is not installed.
    """
    with _REGISTRATION_LOCK:
        _register_pytree_once()


# Cached, so registration, which JAX keeps per process, happens once.
@functools.cache
def _register_pytree_once() -> None:
    jax = _import("jax", "jax", "Registering TimeSeries with JAX")
    tree_util = jax.tree_util

    def flatten(series: TimeSeries) -> tuple[tuple[Any, Any], tuple[float, str | None, str | None]]:
        return (series.data, series.t0), (series.sample_rate, series.name, series.unit)

    def unflatten(static: tuple[float, str | None, str | None], leaves: tuple[Any, Any]) -> TimeSeries:
        # Checked where it can go wrong: JAX has turned t0 into one of its arrays -- a tracer under
        # jit, vmap or grad, or a device array after device_put -- and, in 32-bit mode, a
        # single-precision one, which cannot tell samples apart. A tree map or device_get that
        # leaves t0 a Python float passes in either mode.
        t0 = leaves[1]
        if isinstance(t0, jax.Array) and t0.dtype != jax.numpy.float64:
            raise RuntimeError(_NO_FLOAT64)
        series = object.__new__(TimeSeries)
        for field_name, value in zip(("data", "t0"), leaves, strict=True):
            object.__setattr__(series, field_name, value)
        for field_name, value in zip(("sample_rate", "name", "unit"), static, strict=True):
            object.__setattr__(series, field_name, value)
        return series

    tree_util.register_pytree_node(TimeSeries, flatten, unflatten)
