---
title: Custom detectors
description:
    How to construct a CustomDetector for non-standard ground-based
    interferometers and use it in projection and injection pipelines.
icon: material/plus-circle-outline
---

# Custom detectors

When a detector is not available in LAL's built-in list (returned by
[`Network.list_lal_detectors`](../api/network/)), construct a
**`CustomDetector`** with explicit geodetic coordinates and arm orientations.
Custom detectors integrate seamlessly with
[`project_polarizations_to_network`](../api/projection/) and the simulator
pipeline.

For a file-based approach (YAML/JSON), see
[Network configuration files](network-config.md).

## Geometry parameters

A [`CustomDetector`](../api/network/) requires eight values to describe a
ground-based interferometer:

| Parameter          | Type    | Unit    | Range                | Description                                                       |
| ------------------ | ------- | ------- | -------------------- | ----------------------------------------------------------------- |
| `name`             | `str`   | —       | Non-empty            | Key used in strain output dicts. Must be unique within a network. |
| `latitude_rad`     | `float` | radians | `[-pi/2, pi/2]`      | Geodetic latitude of the vertex.                                  |
| `longitude_rad`    | `float` | radians | `[-pi, pi]`          | Geodetic longitude of the vertex.                                 |
| `elevation_m`      | `float` | metres  | `[-1e4, 1e5]`        | Vertex elevation above WGS-84 ellipsoid.                          |
| `xarm_azimuth_rad` | `float` | radians | finite               | X-arm azimuth, clockwise from geodetic North (LAL convention).    |
| `yarm_azimuth_rad` | `float` | radians | finite               | Y-arm azimuth, clockwise from geodetic North (LAL convention).    |
| `xarm_tilt_rad`    | `float` | radians | finite (default `0`) | X-arm altitude above local horizon.                               |
| `yarm_tilt_rad`    | `float` | radians | finite (default `0`) | Y-arm altitude above local horizon.                               |

The `prefix` parameter is optional — when omitted, a unique two-character LAL
prefix is generated automatically so the projection layer can look up the
detector.

!!! tip "Degrees vs. radians" The `CustomDetector` constructor uses radians.
Network YAML/JSON files support `_deg` variants as well. See
[Network configuration files](network-config.md#file-schema).

---

## Example 1 — Construct a custom detector in Python

```python
import math
from gwmock_signal.detector import CustomDetector

# A hypothetical site in Sardinia (ET-like)
cust = CustomDetector(
    name="MY_SITE",
    latitude_rad=math.radians(40.5),
    longitude_rad=math.radians(9.4),
    elevation_m=50.0,
    xarm_azimuth_rad=math.radians(70.0),
    yarm_azimuth_rad=math.radians(130.0),
    xarm_tilt_rad=0.0,
    yarm_tilt_rad=0.0,
)

print(cust.name, cust.latitude_rad)
```

---

## Example 2 — Use in a network and projection

Custom detectors are valid arguments anywhere detector names are expected:

```python
import math
import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.waveform import WaveformFactory
from gwmock_signal.projection import project_polarizations_to_network

# Define two custom detectors
cd1 = CustomDetector(
    name="CUST_A",
    latitude_rad=math.radians(40.5),
    longitude_rad=math.radians(9.4),
    elevation_m=50.0,
    xarm_azimuth_rad=math.radians(70.0),
    yarm_azimuth_rad=math.radians(130.0),
)
cd2 = CustomDetector(
    name="CUST_B",
    latitude_rad=math.radians(40.6),
    longitude_rad=math.radians(9.5),
    elevation_m=55.0,
    xarm_azimuth_rad=math.radians(190.0),
    yarm_azimuth_rad=math.radians(250.0),
)

# Build a network from custom detectors
net = Network.from_detectors([cd1, cd2], name="Custom Network")

# Generate polarizations
factory = WaveformFactory()
tc = 1_400_000_000.0
pol = factory.generate(
    "IMRPhenomD",
    {
        "tc": tc,
        "detector_frame_mass_1": 36.0,
        "detector_frame_mass_2": 29.0,
        "distance": 410.0,
        "inclination": 0.0,
    },
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)

# Project onto custom network
strains = project_polarizations_to_network(
    pol,
    net.detector_names,  # CustomDetector instances
    right_ascension=1.375,
    declination=-1.211,
    polarization_angle=0.0,
)

for name, strain in strains.items():
    rms = float(np.sqrt(np.mean(strain.value**2)))
    print(f"{name}: rms={rms:.4e}")
```

---

## Example 3 — Mix custom and built-in detectors

```python
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network

cust = CustomDetector(
    name="ET1",
    latitude_rad=0.7,
    longitude_rad=0.16,
    elevation_m=100.0,
    xarm_azimuth_rad=1.23,
    yarm_azimuth_rad=2.28,
)

# Mix: built-in H1 and L1 plus a custom ET1
net = Network.from_detectors(["H1", "L1", cust], name="HL + ET")
print(net.detector_names)
# ('H1', 'L1', CustomDetector(name='ET1', latitude_rad=0.7, longitude_rad=0.16, elevation_m=100.0, xarm_azimuth_rad=1.23, yarm_azimuth_rad=2.28, xarm_tilt_rad=0.0, yarm_tilt_rad=0.0, prefix=''))
```

When a `Network` contains `CustomDetector` instances, the projection layer
registers them with LAL's prefix cache automatically on first use. Subsequent
lookups within the same process are cached.

---

## Automatic LAL prefix generation

Each `CustomDetector` registers itself in `lal.cached_detector_by_prefix` under
a two-character prefix. If you don't supply a `prefix`, one is generated
automatically (two unused uppercase/digit characters). If you _do_ supply one,
it must be exactly two characters and not already registered in LAL.

```python
# Explicit prefix (must be unique and not in use)
cust = CustomDetector(
    name="EXPLICIT",
    prefix="ZZ",
    latitude_rad=0.5,
    longitude_rad=0.1,
    elevation_m=10.0,
    xarm_azimuth_rad=1.0,
    yarm_azimuth_rad=2.0,
)
```

---

## See also

- [Network configuration files](network-config.md) — YAML/JSON format with
  `_deg` / `_rad` variants
- [Detector projection examples](detector-projection.md)
- [Network API reference](../api/network/)
- [User guide overview](index.md)
