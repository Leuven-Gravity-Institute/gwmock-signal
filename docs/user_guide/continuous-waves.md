---
title: Continuous waves
description:
    ContinuousWaveSimulator, projection backends (JAX, Numpy), time-domain
    examples.
---

# Continuous waves

Typical gravitational-wave sources in this package are **transients**: signals
placed at a single event time generated once, projected to a detector and
injected into a segment. A continuous wave (_CW_) is a signal from a rotating,
non-axisymmetric isolated neutron star. It is **on for the whole observation**,
which is typically months, and it is quasi-monochromatic rather than chirping.

The simulator produces a **time-domain** CW strain signal $h$, that is already
projected onto a detector frame: $h = F_+ \cdot h_+ + F_\times \cdot h_\times$.

!!! tip "API reference"

    Use **[Continuous waves API](../api/continuous/)** for the
    signatures, types, and raised exceptions. This page contains copy-paste examples and the reasoning behind the constraints.

## Optional JAX dependency

`ContinuousWaveSimulator` needs the **`[jax]` extra** as the polarizations are
produced with `rippleGW`, and the default detector projection backend runs on
JAX (`pip install 'gwmock-signal[jax]'`).

See [Installation](installation.md) for details.

## Import

Unlike `CBCSimulator` or `GWSimulator`, which are exported from the top-level
package, the `ContinuousWaveSimulator` should be exported from the CW submodule:

```python
from gwmock_signal.continuous import ContinuousWaveSimulator
```

## Ephemeris files

Barycentring of the signal needs an Earth and a Sun ephemeris table.

These should be provided as an **explicit paths** to the files. If path does not
exist locally but is a standard `LALPulsar` name (e.g.
_'earth00-40-DE405.dat.gz'_), the tables will be downloaded and cached (Internet
access is required once). The run records what it used.

```python
sim = ContinuousWaveSimulator(
    earth_ephemeris="earth00-40-DE405.dat.gz", sun_ephemeris="sun00-40-DE405.dat.gz", reference_time_ssb=1470600018.0
)
```

## The reference epoch

`reference_time_ssb` is a solar-system barycentre (SSB) reference epoch for the
source parameters. Provided in seconds.

It is a **required parameter**. It is fixed for the whole run which is reused
across every `ContinuousWaveSimulator.simulate(...)` call to **keep the segments
coherent** with each other. `rippleGW`'s own default derives the SSB reference
epoch from the first sample it is given. As it is called once per segment, it
silently restarts the phase at every segment boundary. Each segment on its own
still looks like a perfectly good CW signal, only a multi-segment run would
disagree with itself.

## Projection backend

`"jax"` is the default projection backend, as the projection is roughly 99% of a
segment's cost and `rippleGW` already brings JAX in for the polarizations, so
the device path is essentially free to default to. However, it comes with one
limit. This backend linearly extrapolates sidereal time across the span it is
given and accepts up to 86400 s for a single segment.

If you need a single segment longer than 86400 s, pass `"numpy"` instead:

```python
sim = ContinuousWaveSimulator(
    earth_ephemeris="earth00-40-DE405.dat.gz",
    sun_ephemeris="sun00-40-DE405.dat.gz",
    reference_time_ssb=1470600018.0,
    projection_backend="numpy",
)
```

Note that the span a segment measures is padded (`edge_padding`). This results
in the addition of samples, the duration of which depends on the sampling rate
(0.11 s - 0.56 s) at both ends. So the usable background is that much shorter.

## Accepted parameters

When using `ContinuousWaveSimulator.simulate(...)` only the following parameter
keys are accepted as `params` argument. Extra keys raise `ValueError`.

| Canonical name    | Default      | Description                              |
| ----------------- | ------------ | ---------------------------------------- |
| `right_ascension` | **required** | Source's right ascension (rad)           |
| `declination`     | **required** | Source's declination (rad)               |
| `frequency`       | **required** | **Wave** frequency (Hz)                  |
| `initial_phase`   | **required** | Signal's initial phase (rad)             |
| `amplitude_plus`  | **required** | CW plus polarization amplitude (strain)  |
| `amplitude_cross` | **required** | CW cross polarization amplitude (strain) |

Note that `amplitude_plus`/`amplitude_cross` are taken directly, not derived
from `h0`/`cos_iota`. If your source parameters are in the constant
amplitude/inclination convention, convert to plus/cross amplitudes before
calling.

The information on CW source spindowns should be provided in
`ContinuousWaveSimulator` call.

## Example 1 — one segment in a detector network

```python
import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.continuous import ContinuousWaveSimulator

fs = 64.0
epoch = 1470600018.0
segment_seconds = 1800.0
n = int(segment_seconds * fs)

sim = ContinuousWaveSimulator(
    earth_ephemeris="earth00-40-DE405.dat.gz",
    sun_ephemeris="sun00-40-DE405.dat.gz",
    reference_time_ssb=epoch,
    spindowns=(-1.0e-10,),
)

params = {
    "right_ascension": 1.1,
    "declination": 0.3,
    "frequency": 20.0,
    "initial_phase": 0.4,
    "amplitude_plus": 1.0e-24,
    "amplitude_cross": 7.0e-25,
    "polarization_angle": 0.2,
}


detectors = ["H1", "L1", "V1"]
background = {name: TimeSeries(np.zeros(n), t0=epoch, sample_rate=fs, unit="strain") for name in detectors}

strains = sim.simulate(
    params=params,
    detector_names=detectors,
    background=background,
    sampling_frequency=fs,
    minimum_frequency=0.0,  # unused for CW; required by backend
)

for name in strains.detector_names:
    rms = float(np.sqrt(np.mean(strains[name].value ** 2)))
    print(f"{name}: rms={rms:.4e}")
```

Swap the `background` parameter strain with your own noise data to inject the CW
signal into it. See [Strain injection](strain-injection.md) for more
information.

## Example 2 — a coherent multi-segment run

Multi-segment run reuses the **same simulator instance** (that is same
`reference_time_ssb`, same ephemeris table) across consecutive segments, and the
concatenated output reproduces a coherent signal which a single long call would
have produced.

```python
detectors = ["V1"]
segments_count = 3
strains = []

for index in range(segments_count):
    seg_epoch = epoch + segment_seconds * index
    segment_background = {"V1": TimeSeries(np.zeros(n), t0=seg_epoch, sample_rate=fs, unit="strain")}

    seg_strain = sim.simulate(
        params,
        detectors,
        segment_background,
        sampling_frequency=fs,
        minimum_frequency=0.0,
        earth_rotation=True,
    )
    strains.append(seg_strain["V1"].value)

multisegment = np.concatenate(strains)
```

## Constraints and errors

| Situation                                                     | Behavior                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `background` missing or empty                                 | `ValueError`. A continuous wave has no duration of its own, so the segment's epoch and length come from the data it is added to.                                                                                                                                      |
| `earth_rotation=False`                                        | `ValueError`. Not available. Holding the antenna pattern fixed is only a valid approximation for signals short enough that Earth barely turns across them. A CW run spans months, so this is refused rather than silently producing a plausible-looking wrong answer. |
| `reference_time_ssb` not finite                               | `ValueError`. Must be a finite GPS-scale time in seconds.                                                                                                                                                                                                             |
| `reference_time_ssb` ≥ ephemeris table time validity          | Warning. This may affect precision at the arcsec level.                                                                                                                                                                                                               |
| `frequency` ≤ 0 or ≥ Nyquist                                  | `ValueError`. Frequency must be positive and below Nyquist frequency (1/2 of sampling rate). Otherwise it would alias.                                                                                                                                                |
| A detector in `detector_names` has no `background` channel    | `KeyError`.                                                                                                                                                                                                                                                           |
| Background channels disagree on epoch, length, or sample rate | `ValueError`. Background channels for all detectors must share these parameters.                                                                                                                                                                                      |
| Polarizations come back non-finite                            | `RuntimeError`. Usually an old `rippleGW` (<0.3.1 returns `NaN` at the geocentre for every sample), or an extreme spindown/reference-epoch combination overflowing the phase. The spindown term grows as the square of the time from `reference_time_ssb`.            |

## See also

- [User guide overview](index.md)
- [Continuous API](../api/continuous/)
- [Simulator API](../api/simulator/)
- [Waveforms](waveform.md)
- [Installation](installation.md)
- [Documentation home](../index.md)
