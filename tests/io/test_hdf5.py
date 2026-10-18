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
"""HDF5 for a whole detector stack, and the extra each format names when its package is missing.

The HDF5 layout itself, ``write_hdf5`` and ``read_hdf5``, are gwmock-data's and are tested there; this
module covers what this package adds on top: ``DetectorStrainStack.write`` and ``read``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.timeseries import TimeSeries

_T0 = 1_400_000_000.25


def _series(rate: float = 4096.0, **kwargs: object) -> TimeSeries:
    return TimeSeries(np.arange(16.0), t0=_T0, sample_rate=rate, **kwargs)


def test_a_stack_round_trips_through_hdf5(tmp_path):
    """The stack's samples and timing survive a write and a read."""
    pytest.importorskip("h5py")
    stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _series(), "L1": _series()})
    stack.write(tmp_path / "stack.h5")

    back = DetectorStrainStack.read(tmp_path / "stack.h5", format="hdf5")

    assert list(back.detector_names) == ["H1", "L1"]
    assert back.t0 == _T0
    assert back.sample_rate == 4096.0
    np.testing.assert_array_equal(back.data, stack.data)


@pytest.mark.parametrize(
    ("call", "package", "extra"),
    [
        (lambda path: DetectorStrainStack.from_mapping(["H1"], {"H1": _series()}).write(path), "h5py", "hdf5"),
        (lambda path: DetectorStrainStack.read(path, format="hdf5"), "h5py", "hdf5"),
        (
            lambda path: DetectorStrainStack.from_mapping(["H1"], {"H1": _series()}).write(path, format="gwf"),
            "gwpy",
            "gwpy",
        ),
    ],
)
def test_a_missing_package_names_its_extra(monkeypatch, tmp_path, call, package, extra):
    """HDF5 needs only h5py and frames only GWpy, each from its own extra."""
    real_import = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == package or name.startswith(f"{package}."):
            raise ModuleNotFoundError(f"No module named {package!r}", name=package)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match=rf"pip install 'gwmock-signal\[{extra}\]'"):
        call(tmp_path / "out")
