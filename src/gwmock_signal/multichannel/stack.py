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
"""Stack per-detector GWpy strains with a fixed channel order."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries, TimeSeriesDict

# Top-level HDF5 attribute storing the detector/channel name order.
#
# We serialize as a JSON list string to keep compatibility with simple
# h5py attribute types and avoid separate datasets.
_HDF5_STACK_ORDER_ATTR = "gwmock_signal_detector_strain_stack_order"


def _same_time_samples(series: TimeSeries, reference: TimeSeries) -> bool:
    """Return whether two equal-length series sample the same times.

    Compared through ``t0`` and ``dx`` rather than by materialising both time indices. GWpy
    builds a regular index as ``t0 + arange(n) * dx``, so for equal lengths those two scalars
    matching is not an approximation of index equality -- it is exactly equivalent, and it
    avoids allocating and comparing an array per channel. Assembly builds one stack per segment,
    where that allocation showed up as ~10% of the host cost.

    A series constructed with an explicit irregular ``times=`` array has no ``dx``; those fall
    back to comparing the indices directly.
    """
    try:
        return bool(series.t0 == reference.t0 and series.dx == reference.dx)
    except AttributeError:
        return bool(np.array_equal(series.times.value, reference.times.value))


def _validate_aligned_channels(channels: Sequence[TimeSeries]) -> None:
    """Require identical length, sample rate, and time samples across channels."""
    if len(channels) == 0:
        raise ValueError("At least one channel is required.")
    ref = channels[0]
    for i, s in enumerate(channels[1:], start=1):
        # GWpy's is_compatible() returns True or raises -- it never returns False -- so testing
        # its return value left this branch unreachable and reported gwpy's message without
        # saying which channel was at fault. Its own exception is re-raised with that index.
        try:
            s.check_compatible(ref)
        except ValueError as error:
            raise ValueError(
                f"Channel {i} is not compatible with reference channel 0 (mismatched unit or sample rate): {error}"
            ) from error
        if len(s) != len(ref):
            raise ValueError(f"Channel {i} length {len(s)} does not match reference length {len(ref)}.")
        if not _same_time_samples(s, ref):
            raise ValueError(
                f"Channel {i} time grid does not match reference (channel 0); "
                "all detectors must share identical sample times."
            )


class DetectorStrainStack:
    """Aligned strains for a fixed list of detectors (one GWpy series per row).

    See ``docs/user_guide/multi-channel-strains.md`` (examples) and
    ``docs/api/multichannel/index.md`` (API reference).
    """

    def __init__(self, detector_names: tuple[str, ...], channels: tuple[TimeSeries, ...]) -> None:
        """Prefer the ``from_mapping`` classmethod for public construction.

        Args:
            detector_names: Channel order; row ``i`` of ``data`` corresponds to
                ``detector_names[i]``.
            channels: Tuple of GWpy time series objects, one per detector.

        Raises:
            ValueError: If ``detector_names`` and ``channels`` have different lengths.
            ValueError: If ``detector_names`` is not unique.
            ValueError: If ``channels`` is empty.
            TypeError: If a channel is not a GWpy [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).
            ValueError: If the channels are not aligned on the same grid.
            ValueError: If the channels are not compatible (mismatched unit or sample rate).

        """
        if len(detector_names) != len(channels):
            raise ValueError("detector_names and channels must have the same length.")
        if len(set(detector_names)) != len(detector_names):
            raise ValueError("detector_names must be unique.")
        for i, s in enumerate(channels):
            if not isinstance(s, TimeSeries):
                raise TypeError(f"channels[{i}] must be gwpy.timeseries.TimeSeries, got {type(s)}")
        _validate_aligned_channels(channels)
        self._names = detector_names
        self._channels = channels

    @classmethod
    def from_mapping(
        cls,
        detector_names: Sequence[str],
        strains: Mapping[str, TimeSeries],
    ) -> DetectorStrainStack:
        """Build a stack in the given detector order; validate a single shared time grid.

        Args:
            detector_names: Channel order; row ``i`` of ``data`` corresponds to
                ``detector_names[i]``.
            strains: Mapping containing every listed name. Extra keys are ignored.

        Returns:
            New ``DetectorStrainStack`` instance.

        Raises:
            KeyError: If any ``detector_names`` entry is missing from ``strains``.
            ValueError: If channels are not aligned on the same grid.
            TypeError: If a strain is not a GWpy [`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/).
        """
        names = tuple(str(n) for n in detector_names)
        if not names:
            raise ValueError("detector_names must be non-empty.")
        missing = [n for n in names if n not in strains]
        if missing:
            raise KeyError(f"Missing strain keys for detectors: {missing}")
        channels = []
        for n in names:
            s = strains[n]
            if not isinstance(s, TimeSeries):
                raise TypeError(f"strains[{n!r}] must be gwpy.timeseries.TimeSeries, got {type(s)}")
            channels.append(s)
        ch_tuple = tuple(channels)
        _validate_aligned_channels(ch_tuple)
        return cls(names, ch_tuple)

    @property
    def detector_names(self) -> tuple[str, ...]:
        """IFO order used for stacking (immutable)."""
        return self._names

    @property
    def t0(self):
        """GPS start of the first sample (same for every channel)."""
        return self._channels[0].t0

    @property
    def sample_rate(self) -> Quantity:
        """Sample rate (GWpy quantity), identical for every channel.

        Returns:
            Sample rate of the first channel.
        """
        return self._channels[0].sample_rate

    @property
    def data(self) -> np.ndarray:
        """Strain samples shaped ``(n_detectors, n_samples)`` (C-contiguous copy).

        Returns:
            Strain samples shaped ``(n_detectors, n_samples)`` (C-contiguous copy).
        """
        rows = [np.asarray(s.value, dtype=float) for s in self._channels]
        return np.stack(rows, axis=0).copy()

    def __getitem__(self, key: int | str) -> TimeSeries:
        """Return one channel by index or detector name (same object as stored).

        Args:
            key: Index or detector name.

        Returns:
            GWpy TimeSeries object for the specified channel.

        Raises:
            KeyError: If the key is not found.
            TypeError: If the key is not an integer or string.
            TypeError: If the key is a boolean.
        """
        if isinstance(key, str):
            try:
                idx = self._names.index(key)
            except ValueError as exc:
                raise KeyError(key) from exc
            return self._channels[idx]
        if isinstance(key, bool):
            raise TypeError("index must be int or str, got bool")
        if isinstance(key, int):
            return self._channels[key]
        raise TypeError(f"index must be int or str, got {type(key)}")

    def __len__(self) -> int:
        """Number of detectors (channels).

        Returns:
            Number of detectors (channels).
        """
        return len(self._channels)

    def write(
        self,
        path: str | Path,
        format: Literal["gwf", "hdf5", "npy", "txt"] = "hdf5",  # noqa: A002
    ) -> None:
        """Write the stack to a file.

        Args:
            path: Output file path.
            format: Output format — one of ``'gwf'``, ``'hdf5'``, ``'npy'``,
                or ``'txt'``. Defaults to ``'hdf5'``.

        Raises:
            ValueError: If ``format`` is not recognised.

        Note:
            GWF writing requires an optional frame library
            (``python-ldas-tools-framecpp`` or ``framel``).  Install via your
            system package manager or conda.
        """
        path = Path(path)
        t0_val = float(self.t0.value)
        dt_val = 1.0 / float(self.sample_rate.value)
        channel_names = list(self._names)

        if format == "gwf":
            tsd = TimeSeriesDict()
            for name, ts in zip(self._names, self._channels, strict=True):
                ts_named = ts.copy()
                ts_named.name = name
                tsd[name] = ts_named
            tsd.write(str(path), format="gwf")

        elif format == "hdf5":
            with h5py.File(path, "w") as fh:
                # Preserve the detector/channel ordering explicitly since HDF5
                # group key iteration order is not guaranteed across environments.
                fh.attrs[_HDF5_STACK_ORDER_ATTR] = json.dumps(channel_names)
                for name, ts in zip(self._names, self._channels, strict=True):
                    ds = fh.create_dataset(name, data=np.asarray(ts.value, dtype=np.float64))
                    ds.attrs["t0"] = t0_val
                    ds.attrs["dt"] = dt_val
                    ds.attrs["unit"] = "strain"

        elif format == "npy":
            arr = np.stack([np.asarray(ts.value, dtype=np.float64) for ts in self._channels], axis=1)
            np.save(path, arr)
            sidecar = Path(path).with_suffix(".json")
            sidecar.write_text(json.dumps({"t0": t0_val, "dt": dt_val, "channels": channel_names}))

        elif format == "txt":
            arr = np.stack([np.asarray(ts.value, dtype=np.float64) for ts in self._channels], axis=1)
            header = f"t0={t0_val} dt={dt_val} channels={','.join(channel_names)}"
            np.savetxt(path, arr, header=header)

        else:
            raise ValueError(f"Unknown format {format!r}. Choose from 'gwf', 'hdf5', 'npy', 'txt'.")

    @classmethod
    def read(
        cls,
        path: str | Path,
        format: Literal["hdf5", "npy"],  # noqa: A002
    ) -> DetectorStrainStack:
        """Reconstruct a ``DetectorStrainStack`` from an HDF5 or npy file.

        Args:
            path: Input file path.  For ``'npy'`` format a JSON sidecar at
                ``<stem>.json`` must also be present.
            format: Input format — ``'hdf5'`` or ``'npy'``.

        Returns:
            Reconstructed ``DetectorStrainStack``.

        Raises:
            NotImplementedError: If ``format`` is ``'gwf'`` or ``'txt'``
                (write-only formats).
            ValueError: If ``format`` is not recognised.
        """
        path = Path(path)

        if format in ("gwf", "txt"):
            raise NotImplementedError(f"Reading format {format!r} is not yet supported.")

        if format == "hdf5":
            with h5py.File(path, "r") as fh:
                names = list(fh.keys())
                ordered_names = names

                order_raw = fh.attrs.get(_HDF5_STACK_ORDER_ATTR)
                if order_raw is not None:
                    if isinstance(order_raw, bytes):
                        order_raw = order_raw.decode("utf-8")
                    try:
                        candidate = json.loads(order_raw)
                    except (TypeError, ValueError):
                        candidate = None

                    if (
                        isinstance(candidate, list)
                        and all(isinstance(n, str) for n in candidate)
                        and len(set(candidate)) == len(candidate)
                        and set(candidate) == set(names)
                    ):
                        ordered_names = candidate

                channels = []
                for name in ordered_names:
                    ds = fh[name]
                    data = ds[...]
                    t0 = float(ds.attrs["t0"])
                    dt = float(ds.attrs["dt"])
                    channels.append(TimeSeries(data, t0=t0, dt=dt, unit="strain", name=name))
            return cls(tuple(ordered_names), tuple(channels))

        if format == "npy":
            arr = np.load(path)
            sidecar = path.with_suffix(".json")
            meta = json.loads(sidecar.read_text())
            t0 = float(meta["t0"])
            dt = float(meta["dt"])
            channel_names = meta["channels"]
            channels = [
                TimeSeries(arr[:, i], t0=t0, dt=dt, unit="strain", name=name) for i, name in enumerate(channel_names)
            ]
            return cls(tuple(channel_names), tuple(channels))

        raise ValueError(f"Unknown format {format!r}. Choose from 'hdf5', 'npy'.")

    def to_dict(self) -> dict[str, TimeSeries]:
        """Map detector name to GWpy series (same objects as ``__getitem__``).

        Returns:
            Mapping of detector name to GWpy series.
        """
        return dict(zip(self._names, self._channels, strict=True))
