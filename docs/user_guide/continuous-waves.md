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
package, `ContinuousWaveSimulator` is reached through the CW submodule:

```python
from gwmock_signal.continuous import ContinuousWaveSimulator
```

## Ephemeris files

Barycentring of the signal needs an Earth and a Sun ephemeris table.

These should be provided as **explicit paths** to the files. If a path does not
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
source parameters, provided in seconds.

It is a **required parameter**, and it is fixed for the whole run: the same
value is reused across every `ContinuousWaveSimulator.simulate(...)` call, which
is what **keeps the segments coherent** with each other. `rippleGW`'s own
default derives the SSB reference epoch from the first sample it is given. As it
is called once per segment, that default silently restarts the phase at every
segment boundary. Each segment on its own still looks like a perfectly good CW
signal; only a multi-segment run would disagree with itself.

## Projection backend

`"jax"` is the default projection backend, as the projection is roughly 99% of a
segment's cost and `rippleGW` already brings JAX in for the polarizations, so
the device path is essentially free to default to. However, it comes with one
limit: this backend linearly extrapolates sidereal time across the span it is
given, and refuses a span longer than 86400 s.

That limit applies to the **padded** span, not to the nominal segment. The
projection is given a buffer padded at both ends (see below), so a segment whose
padded span exceeds 86400 s is refused even when its nominal length is a little
under. To stay on the device path, keep the nominal segment below
`86400 s - 2 x padding`. If you need a longer single segment, pass `"numpy"`
instead:

```python
sim = ContinuousWaveSimulator(
    earth_ephemeris="earth00-40-DE405.dat.gz",
    sun_ephemeris="sun00-40-DE405.dat.gz",
    reference_time_ssb=1470600018.0,
    projection_backend="numpy",
)
```

Note that the span a segment measures is padded (`edge_padding`): the resampling
kernel's taps reach past the end of the buffer, and for a continuous wave the
signal continues into the neighbouring segment rather than stopping there. So
each segment is generated with extra samples at both ends and trimmed back
afterwards, and the usable background is shorter than 86400 s by that padding.
`edge_padding` returns the padding in **samples**, so its duration in seconds
scales inversely with the sample rate. With the default resampling kernel:

| Sample rate | Padding per end | Usable segment shortened by |
| ----------- | --------------- | --------------------------- |
| 32 Hz       | 2.06 s          | 4.12 s                      |
| 64 Hz       | 1.05 s          | 2.09 s                      |
| 128 Hz      | 0.53 s          | 1.06 s                      |
| 256 Hz      | 0.28 s          | 0.55 s                      |
| 512 Hz      | 0.15 s          | 0.30 s                      |

These are the values at the sample rates the examples use. Read the padding for
your own rate from `edge_padding(sampling_frequency)`, and treat a nominal
segment of `86400 s - 2 x padding` as the longest one the JAX backend accepts.

## Accepted parameters

When using `ContinuousWaveSimulator.simulate(...)`, the following parameter keys
are accepted in the `params` argument. The six listed as **required** must be
present, or `ValueError` names the ones that are missing; extra keys are ignored
rather than rejected, so a typo in a key name passes silently.

| Canonical name       | Default      | Description                              |
| -------------------- | ------------ | ---------------------------------------- |
| `right_ascension`    | **required** | Source's right ascension (rad)           |
| `declination`        | **required** | Source's declination (rad)               |
| `frequency`          | **required** | **Wave** frequency (Hz)                  |
| `initial_phase`      | **required** | Signal's initial phase (rad)             |
| `amplitude_plus`     | **required** | CW plus polarization amplitude (strain)  |
| `amplitude_cross`    | **required** | CW cross polarization amplitude (strain) |
| `polarization_angle` | `0.0`        | Polarization angle (rad); optional       |

Note that `amplitude_plus`/`amplitude_cross` are taken directly, not derived
from `h0`/`cos_iota`. If your source parameters are in the constant
amplitude/inclination convention, convert to plus/cross amplitudes before
calling.

`polarization_angle` is applied when the geocentre polarizations are projected
onto a detector, which is the only place it enters: `rippleGW`'s pulsar
generator takes no polarization angle, because it fixes its own polarization
basis internally, and this simulator passes it the amplitudes directly.

Spindowns are **not** a `params` key. They are passed to
`ContinuousWaveSimulator` as the `spindowns` constructor argument, together with
the reference epoch and the ephemeris tables, so that they are fixed for the
whole run alongside them.

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
    "polarization_angle": 0.2,  # optional; defaults to 0.0
}

detectors = ["H1", "L1", "V1"]
background = {name: TimeSeries(np.zeros(n), t0=epoch, sample_rate=fs, unit="strain") for name in detectors}

strains = sim.simulate(
    params=params,
    detector_names=detectors,
    background=background,
    sampling_frequency=fs,
    minimum_frequency=0.0,  # required by the shared simulator signature; unused for CW
)

for name in strains.detector_names:
    rms = float(np.sqrt(np.mean(strains[name].value ** 2)))
    print(f"{name}: rms={rms:.4e}")
```

Swap the `background` parameter strain with your own noise data to inject the CW
signal into it. See [Strain injection](strain-injection.md) for more
information.

## Example 2 — a coherent multi-segment run

A multi-segment run reuses the **same simulator instance** (that is, the same
`reference_time_ssb` and the same ephemeris tables) across consecutive segments.
The concatenated output then reproduces the coherent signal that a single long
call would have produced.

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
| `reference_time_ssb` ≥ ephemeris table time validity          | Caution for callers. The package does not emit a warning, but this may affect precision at the arcsec level.                                                                                                                                                          |
| `frequency` ≤ 0 or ≥ Nyquist                                  | `ValueError`. Frequency must be positive and below Nyquist frequency (1/2 of sampling rate). Otherwise it would alias.                                                                                                                                                |
| Any required parameter is not finite                          | `ValueError`, naming the key.                                                                                                                                                                                                                                         |
| `sampling_frequency` ≤ 0                                      | `ValueError`. The sample rate must be positive.                                                                                                                                                                                                                       |
| `sampling_frequency` differs from the background's            | `ValueError`. Otherwise the signal is generated on one time grid and added to another, stretching it by the ratio.                                                                                                                                                    |
| `detector_names` is empty                                     | `ValueError`.                                                                                                                                                                                                                                                         |
| A detector in `detector_names` has no `background` channel    | `KeyError`.                                                                                                                                                                                                                                                           |
| A `background` value is not a GWpy `TimeSeries`               | `TypeError`.                                                                                                                                                                                                                                                          |
| Background channels disagree on epoch, length, or sample rate | `ValueError`. Background channels for all detectors must share these parameters.                                                                                                                                                                                      |
| The JAX backend is given a padded span over 86400 s           | `ValueError` naming the span; project in shorter segments or pass `projection_backend="numpy"`.                                                                                                                                                                       |
| Polarizations come back non-finite                            | `RuntimeError`. Usually an old `rippleGW` (<0.3.1 returns `NaN` at the geocentre for every sample), or an extreme spindown/reference-epoch combination overflowing the phase. The spindown term grows as the square of the time from `reference_time_ssb`.            |

## See also

- [User guide overview](index.md)
- [Continuous API](../api/continuous/)
- [Simulator API](../api/simulator/)
- [Waveforms](waveform.md)
- [Installation](installation.md)
- [Documentation home](../index.md)
