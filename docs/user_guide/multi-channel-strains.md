---
title: Multichannel strain examples
description:
    Examples for stacking per-detector GWpy strains into one aligned
    multi-channel segment.
---

# Multichannel strain examples

[`project_polarizations_to_network`](../api/projection/) returns a **mapping**
from detector name to one GWpy
[`TimeSeries`](https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/)
each. Many workflows (frame writers, array-oriented numerics, neural nets, or
legacy code that expects an \((N*\mathrm{det}, N*\mathrm{samples})\) array) need
the **same data in a fixed channel order** as a single object.

This page is **examples only**. **`DetectorStrainStack` fields, validation
rules, and method contracts** are documented only under
**[API → Multichannel](../api/multichannel/)**.

<!-- markdownlint-disable -->

!!! tip "API reference"

    Use **API → Multichannel** as the single source of truth for stacking
    semantics; the snippets below show typical call patterns.

<!-- markdownlint-enable -->

## Example 1 — From projection dict to NumPy array

```python
import numpy as np

from gwmock_signal import DetectorStrainStack
from gwmock_signal.waveform import WaveformFactory
from gwmock_signal.projection import project_polarizations_to_network

names = ["H1", "L1", "V1"]
factory = WaveformFactory()
pol = factory.generate(
    "IMRPhenomD",
    {
        "tc": 1_400_000_000.0,
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 24.0,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "distance": 400.0,
        "inclination": 0.0,
        "coa_phase": 0.0,
    },
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)
strains = project_polarizations_to_network(
    pol,
    names,
    right_ascension=1.1,
    declination=-0.2,
    polarization_angle=0.4,
    earth_rotation=False,
)

stack = DetectorStrainStack.from_mapping(names, strains)
arr = stack.data
assert arr.shape == (len(names), len(strains["H1"]))
print(np.max(np.abs(arr)))

# Inspect stack properties
print(stack.t0)  # GPS start time (same for all channels)
print(stack.sample_rate)  # Sample rate as GWpy Quantity
print(stack.detector_names)  # ('H1', 'L1', 'V1')
```

## Example 2 — Index by name, export one channel

```python
h1_series = stack["H1"]
assert h1_series is stack[0]  # same order as names
```

## Example 3 — Round-trip to dict

```python
again = stack.to_dict()
assert set(again) == set(names)
```

## Example 4 — Save and reload a stack

Write the stack to an HDF5 file (the default format) and read it back:

```python
# Save
stack.write("my_injection.h5", format="hdf5")

# Reload — detector order is preserved
reloaded = DetectorStrainStack.read("my_injection.h5", format="hdf5")
assert reloaded.detector_names == stack.detector_names
```

Available write formats: `"hdf5"` (default), `"npy"` (+ JSON sidecar), `"gwf"`,
`"txt"`. Read formats: `"hdf5"`, `"npy"`.

## Example 5 — Write with NPY format

```python
# Writes my_injection.npy and my_injection.json (sidecar with metadata)
stack.write("my_injection.npy", format="npy")

# Read back
reloaded = DetectorStrainStack.read("my_injection.npy", format="npy")
```

## Pitfalls

- **Aligned grid:** Every channel must share one compatible GWpy time grid; see
  the API page for what is validated and which errors are raised.
- **Order:** Do not assume iteration order of Python dicts; always pass an
  explicit `detector_names` sequence when stacking.
- **Copy vs. view:** `data` may be a **copy** for contiguity; do not rely on
  mutating `data` to change internal GWpy series unless documented.
- **Units:** All channels should use compatible strain units (typically
  dimensionless).

## Scientific notes

- Row-stacked strains match how many **multi-IFO** pipelines treat network data
  before optional whitening or PSD weighting (those steps stay downstream).

## See also

- [User guide overview](index.md)
- [Detector projection](detector-projection.md)
- [Strain injection](strain-injection.md)
- [Multichannel API](../api/multichannel/)
- [API overview](../api/index.md)
- [Documentation home](../index.md)
