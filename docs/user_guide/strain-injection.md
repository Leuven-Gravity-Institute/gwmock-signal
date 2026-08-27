---
title: Strain injection examples
description:
    Example workflows for adding simulated strain into a target GWpy time
    series.
---

# Strain injection examples

After producing detector strain (e.g. with [Waveforms](waveform.md) and
[Detector projection](detector-projection.md)), you often need to **embed that
strain into a longer segment** aligned to a science run—typically starting from
**zeros** or from a **noise** realization (noise generation can live in a
separate package). This step is central to **software injections**, **end-to-end
mock challenges**, and **pipeline validation**.

This page is **examples only**. **Signatures, defaults, and raised exceptions**
for `inject_strain` / `inject_strains_sequential` are documented only under
**[API → Injection](../api/injection/)**.

<!-- markdownlint-disable -->

!!! tip "API reference"

    Use **API → Injection** for the authoritative behavior description; examples
    below illustrate common patterns.

<!-- markdownlint-enable -->

## Example 1 — Inject one CBC into a zero-filled segment

```python
import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform import WaveformFactory
from gwmock_signal.projection import project_polarizations_to_network
from gwmock_signal.injection import inject_strain

fs = 4096.0
duration = 8.0
t0 = 1_400_000_000.0
n = int(duration * fs)

target = TimeSeries(np.zeros(n), t0=t0, sample_rate=fs)

factory = WaveformFactory()
pol = factory.generate(
    "IMRPhenomD",
    {
        "tc": t0 + 2.0,
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "distance": 400.0,
        "inclination": 0.0,
        "coa_phase": 0.0,
    },
    sampling_frequency=fs,
    minimum_frequency=20.0,
)
strains = project_polarizations_to_network(
    pol,
    ["H1"],
    right_ascension=1.0,
    declination=0.5,
    polarization_angle=0.2,
    earth_rotation=False,
)
h1 = strains["H1"]

out = inject_strain(target, h1)
assert out is not target  # immutability: new object (recommended contract)
```

## Example 2 — Multiple injections in time order

```python
from gwmock_signal.injection import inject_strains_sequential

# target: long segment; inj1, inj2: non-overlapping or overlapping GWpy series
out = inject_strains_sequential(target, [inj1, inj2], interpolate_if_offset=True)
```

## Example 3 — Disable interpolation (strict grid)

If you know injections are **exactly aligned** to the target grid and want to
avoid cubic resampling at boundaries:

```python
out = inject_strain(target, h1, interpolate_if_offset=False)
```

When `interpolate_if_offset=False` and the injection start is not aligned to a
target-sample boundary, the function returns `target.copy()` — i.e., the
background unchanged, with a debug-level log message. Use this when you need
strict grid alignment guarantees.

## Example 4 — One-call CBC injection with `inject_cbc_signal`

The convenience function [`inject_cbc_signal`](../api/pipeline/) orchestrates
waveform generation, projection, and injection in one call:

```python
import numpy as np
from gwpy.timeseries import TimeSeries
from gwmock_signal.pipeline import inject_cbc_signal

fs = 4096.0
duration = 8.0
n = int(duration * fs)
t0 = 1_400_000_000.0

params = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "coa_time": t0 + duration / 2,
    "distance": 410.0,
    "inclination": 0.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0,
}

background = {name: TimeSeries(np.zeros(n), t0=t0, sample_rate=fs) for name in ["H1", "L1"]}

result = inject_cbc_signal(
    "IMRPhenomD",
    params,
    ["H1", "L1"],
    background,
    sampling_frequency=fs,
    minimum_frequency=20.0,
    earth_rotation=False,
)

for name in result.detector_names:
    rms = float(np.sqrt(np.mean(result[name].value ** 2)))
    print(f"{name}: rms={rms:.4e}")
```

## Example 5 — Simulate and save to disk with `CBCSimulator.write`

The highest-level convenience is [`CBCSimulator.write`](../api/simulator/),
which runs the full pipeline and writes both the strain data and a JSON sidecar
with the injection parameters:

```python
from gwmock_signal import CBCSimulator
from gwpy.timeseries import TimeSeries
import numpy as np

sim = CBCSimulator("IMRPhenomD")

# The waveform_model property exposes the approximant name:
print(sim.waveform_model)  # IMRPhenomD

params = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "coa_time": 1_126_259_462.4,
    "distance": 410.0,
    "inclination": 0.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0,
}
fs = 4096.0
n = 8192
background = {name: TimeSeries(np.zeros(n), t0=params["coa_time"] - 1, sample_rate=fs) for name in ["H1", "L1"]}

result = sim.write(
    "my_injection.h5",
    params,
    ["H1", "L1"],
    background,
    sampling_frequency=fs,
    minimum_frequency=20.0,
    format="hdf5",
)
# Also creates my_injection_params.json with the injection parameters
```

Supported write formats: `"hdf5"` (default), `"gwf"`, `"npy"`, `"txt"`.

## Pitfalls

- **Units:** Target and injection should use compatible strain units (typically
  dimensionless); mixed units should raise a clear error.
- **No overlap:** If the injection lies entirely outside the target span, the
  API returns a **copy** of the target (same samples, new object).
- **Interpolation:** Cubic interpolation can ring at edges; prefer aligned
  waveforms from the same `sampling_frequency` and GPS grid when possible.
- **Performance:** Large segments are memory-bound; avoid unnecessary copies if
  the API documents mutability (default recommendation: return a **new**
  `TimeSeries`).

## Scientific notes

- Conventions follow common **GWpy** / **LIGO** injection practice: coherent
  addition of strain in the time domain on a fixed grid.
- For **colored noise** backgrounds, build `target` as noise first, then inject;
  stationary Gaussian noise can be added in a separate noise-focused package.

## See also

- [User guide overview](index.md)
- [Waveforms](waveform.md)
- [Detector projection](detector-projection.md)
- [Multichannel strains](multi-channel-strains.md)
- [Injection API](../api/injection/)
- [API overview](../api/index.md)
- [Documentation home](../index.md)
