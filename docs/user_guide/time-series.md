---
title: Time series
description:
    The time series every function takes and returns, and how to move code
    written against GWpy series.
---

# Time series

Every function in `gwmock-signal` that returns strain returns a
[`gwmock_signal.TimeSeries`](../api/timeseries/): an array of samples placed in
time by the GPS time of its first sample and its sample rate. It is the time
series of
[gwmock-data](https://github.com/Leuven-Gravity-Institute/gwmock-data),
re-exported here, so every gwmock package passes the same class around.

```python
import numpy as np

from gwmock_signal import TimeSeries

series = TimeSeries(np.zeros(4096), t0=1_400_000_000.0, sample_rate=4096.0, name="H1", unit="strain")

series.t0  # 1400000000.0, GPS seconds (a float)
series.sample_rate  # 4096.0, Hz
series.dt  # 1 / 4096, seconds
series.duration  # 1.0, seconds
series.n_samples  # 4096
series.times  # GPS time of each sample
series.data  # the samples; `series.value` is the same array
```

- **Samples.** `data` is a NumPy or a JAX array, kept as given. Time runs along
  the **last** axis; leading axes share the timing, so the strain of several
  detectors over one segment can be one series of shape
  `(n_detectors, n_samples)`.
- **Timing.** `t0` and `sample_rate` are plain floats. Sample `k` is at
  `t0 + k * dt`, the same lattice [`SamplingGrid`](../api/waveform/) uses.
- **New series.** `crop(start, end)` (GPS seconds), `pad(before, after)`
  (samples), `copy()` and `with_data(data)` return new series; a series is never
  changed in place by them.
- **JAX.** A JAX-backed series stays on its device. JAX must be in 64-bit mode
  (`jax.config.update("jax_enable_x64", True)`): GPS times need double
  precision, and the methods that would lose it raise instead.

## Inputs

Every function that takes a time series accepts this package's or a
[GWpy](https://gwpy.github.io) `TimeSeries`, in any mix. A GWpy series is
converted without copying its samples.

## Converting

| To                 | Call                                           | Needs               |
| ------------------ | ---------------------------------------------- | ------------------- |
| GWpy `TimeSeries`  | `series.to_gwpy()`                             | GWpy                |
| from GWpy          | `TimeSeries.from_gwpy(gwpy_ts)`                | GWpy                |
| PyCBC `TimeSeries` | `series.to_pycbc()`                            | the `[pycbc]` extra |
| from PyCBC         | `TimeSeries.from_pycbc(pycbc_ts)`              | the `[pycbc]` extra |
| NumPy samples      | `series.to_numpy()` or `numpy.asarray(series)` |                     |
| JAX samples        | `series.to_jax()`                              | the `[jax]` extra   |

`to_gwpy()` and `from_gwpy()` share NumPy samples rather than copying them.

## Files

`gwmock_signal.io.write_hdf5` and `read_hdf5` write and read a mapping of series
as HDF5 in the layout GWpy's `TimeSeriesDict` uses, so either side can read what
the other wrote. They need only `h5py` (the `[hdf5]` extra), not GWpy.

```python
from gwmock_signal.io import read_hdf5, write_hdf5

write_hdf5({"H1": series}, "strain.h5")
again = read_hdf5("strain.h5")  # {"H1": TimeSeries(...)}
```

`DetectorStrainStack.write` and `read` cover a whole stack, in HDF5, NumPy or
text; frame (GWF) output needs the `[gwpy]` extra.

## Moving from GWpy series

Earlier versions returned GWpy `TimeSeries`. Code that read their attributes
changes as follows:

| GWpy                           | `gwmock_signal.TimeSeries`                 |
| ------------------------------ | ------------------------------------------ |
| `ts.value`                     | `ts.value` or `ts.data`                    |
| `ts.t0.value`                  | `ts.t0`                                    |
| `ts.sample_rate.value`         | `ts.sample_rate`                           |
| `ts.dt.value`                  | `ts.dt`                                    |
| `ts.duration.value`            | `ts.duration`                              |
| `ts.times.value`               | `ts.times`                                 |
| `len(ts)`                      | `ts.n_samples`                             |
| `ts.unit` (an astropy unit)    | `ts.unit` (a string, or `None`)            |
| `TimeSeries(x, t0=t0, dt=dt)`  | `TimeSeries(x, t0=t0, sample_rate=1 / dt)` |
| arithmetic, `fft`, `plot`, ... | `ts.to_gwpy()` first                       |

`DetectorStrainStack.t0` and `sample_rate` are floats too, and indexing a stack
returns this package's series. Code that needs a GWpy object can convert any
result with `to_gwpy()`.
