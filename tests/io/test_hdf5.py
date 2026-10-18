"""HDF5 in GWpy's ``TimeSeriesDict`` layout, written and read with h5py alone.

A file written here must load with GWpy exactly, and a file GWpy wrote must load here exactly: the
point of the layout is that the two are interchangeable.
"""

from __future__ import annotations

import importlib

import h5py
import numpy as np
import pytest

from gwmock_signal.io import read_hdf5, write_hdf5
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.timeseries import TimeSeries

_T0 = 1_400_000_000.25


def _series(rate: float = 4096.0, **kwargs: object) -> TimeSeries:
    return TimeSeries(np.arange(16.0), t0=_T0, sample_rate=rate, **kwargs)


@pytest.mark.parametrize("rate", [4096.0, 3000.0, 7.5])
def test_a_round_trip_keeps_samples_timing_name_and_unit(tmp_path, rate):
    """Including rates that are not powers of two."""
    written = {"H1": _series(rate, name="H1:STRAIN", unit="strain"), "L1": _series(rate)}

    write_hdf5(written, tmp_path / "out.h5")
    back = read_hdf5(tmp_path / "out.h5")

    assert list(back) == ["H1", "L1"]
    for key, series in written.items():
        assert np.array_equal(back[key].data, series.data)
        assert (back[key].t0, back[key].sample_rate) == (_T0, rate)
    assert (back["H1"].name, back["H1"].unit) == ("H1:STRAIN", "strain")
    assert (back["L1"].name, back["L1"].unit) == (None, None)


def test_a_nameless_series_round_trips_with_gwpy_as_nameless(tmp_path):
    """GWpy writes no name attribute for a nameless series, and reads one as nameless; so do these."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    write_hdf5({"L1": _series()}, tmp_path / "ours.h5")
    timeseries.TimeSeriesDict({"L1": timeseries.TimeSeries(np.zeros(4), t0=0.0, sample_rate=1.0)}).write(
        tmp_path / "gwpy.h5", format="hdf5"
    )

    with h5py.File(tmp_path / "ours.h5") as ours, h5py.File(tmp_path / "gwpy.h5") as theirs:
        assert set(ours["L1"].attrs) == set(theirs["L1"].attrs)
    assert timeseries.TimeSeriesDict.read(tmp_path / "ours.h5", format="hdf5")["L1"].name is None
    assert read_hdf5(tmp_path / "gwpy.h5")["L1"].name is None


def test_gwpy_reads_what_is_written(tmp_path):
    """The file is a GWpy ``TimeSeriesDict`` file."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    write_hdf5({"H1": _series(3000.0, unit="strain")}, tmp_path / "out.h5")

    back = timeseries.TimeSeriesDict.read(tmp_path / "out.h5", format="hdf5")

    assert np.array_equal(back["H1"].value, np.arange(16.0))
    assert back["H1"].t0.value == _T0
    assert back["H1"].sample_rate.value == 3000.0
    assert str(back["H1"].unit) == "strain"


def test_what_gwpy_writes_is_read(tmp_path):
    """And the other way round, at a rate that is not a power of two."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    original = timeseries.TimeSeriesDict(
        {"V1": timeseries.TimeSeries(np.arange(8.0), t0=_T0, sample_rate=3000.0, name="V1", unit="strain")}
    )
    original.write(tmp_path / "gwpy.h5", format="hdf5")

    back = read_hdf5(tmp_path / "gwpy.h5")

    assert np.array_equal(back["V1"].data, np.arange(8.0))
    assert (back["V1"].t0, back["V1"].sample_rate, back["V1"].name, back["V1"].unit) == (_T0, 3000.0, "V1", "strain")


def test_the_datasets_are_compressed(tmp_path):
    """As GWpy writes them."""
    write_hdf5({"H1": _series()}, tmp_path / "out.h5")

    with h5py.File(tmp_path / "out.h5") as handle:
        assert handle["H1"].compression == "gzip"
        assert handle["H1"].attrs["xunit"] == "s"


def test_a_multi_row_series_is_refused(tmp_path):
    """A dataset is one channel."""
    with pytest.raises(ValueError, match="must be one-dimensional"):
        write_hdf5({"H1": TimeSeries(np.zeros((2, 4)), t0=0.0, sample_rate=1.0)}, tmp_path / "out.h5")


def test_a_dataset_without_timing_is_refused(tmp_path):
    """A bare array is not a time series."""
    with h5py.File(tmp_path / "bare.h5", "w") as handle:
        handle.create_dataset("H1", data=np.zeros(4))

    with pytest.raises(ValueError, match="no x0 and dx attributes"):
        read_hdf5(tmp_path / "bare.h5")


def test_a_time_axis_in_other_units_is_refused(tmp_path):
    """Only seconds are understood."""
    with h5py.File(tmp_path / "ms.h5", "w") as handle:
        dataset = handle.create_dataset("H1", data=np.zeros(4))
        dataset.attrs.update({"x0": 0.0, "dx": 1.0, "xunit": "ms"})

    with pytest.raises(ValueError, match="only seconds"):
        read_hdf5(tmp_path / "ms.h5")


def test_a_broken_installation_is_not_reported_as_missing(monkeypatch, tmp_path):
    """A part of an installed package that fails to import is re-raised as it is."""
    real_import = importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "h5py":
            raise ModuleNotFoundError("No module named 'h5py._internal'", name="h5py._internal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", broken)

    with pytest.raises(ModuleNotFoundError, match=r"h5py\._internal") as raised:
        write_hdf5({"H1": _series()}, tmp_path / "out.h5")
    assert "gwmock-signal[" not in str(raised.value)


@pytest.mark.parametrize(
    ("call", "package", "extra"),
    [
        (lambda path: write_hdf5({"H1": _series()}, path), "h5py", "hdf5"),
        (read_hdf5, "h5py", "hdf5"),
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
