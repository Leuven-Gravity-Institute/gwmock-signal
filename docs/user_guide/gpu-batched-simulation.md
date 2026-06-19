---
title: Batched GPU simulation
description:
    Catalogue-scale, on-device (GPU) CBC simulation with the JAX/ripple backend
    — batched waveform generation, projection, and assembly into fixed-duration
    data segments.
---

# Batched GPU simulation

For large injection catalogues (e.g. a year of events) the
JAX/[ripple](waveform.md) backend can generate and project the whole catalogue
**on device** in batched, `vmap`-ped calls, then assemble the signals into
uniform-duration data-segment files. The single-event CPU path
([Strain injection](strain-injection.md), `CBCSimulator`) is unchanged and
remains the default; this page covers the batched device path.

!!! note "Requires the JAX extra"

    Install with **`pip install 'gwmock-signal[jax]'`** (see
    [Installation](installation.md)). The batched API lives in
    `gwmock_signal.jax_batch`.

!!! tip "See also"

    The speed of this batched/GPU path against the per-event CPU baseline is
    quantified in [Performance](../dev/performance.md), and the ripple backend's
    agreement with the LAL baseline in [Consistency](../dev/consistency.md).

## Two independent durations

A catalogue mixes two notions of "duration" that this API keeps separate:

- **Generation buffer** — how long each waveform must be to contain its inspiral
  from `minimum_frequency`. It is estimated from the post-Newtonian chirp time.
  A `vmap` needs one fixed shape, so the batch uses a **single worst-case
  buffer** sized for the longest (lowest-mass) event, unless you fix it with
  `RippleBackend(segment_duration=…)`.
- **Data-segment duration** — the fixed length of each output data file. It is
  independent of any signal's length: a long signal simply **spans several
  consecutive segments**.

## End to end in one call

```python
import numpy as np
from gwmock_signal.jax_batch import simulate_cbc_catalogue

# Canonical gwmock-pop parameter names as a struct-of-arrays (one entry per event).
catalogue = {
    "detector_frame_mass_1": np.array([40.0, 36.0]),
    "detector_frame_mass_2": np.array([31.0, 29.0]),
    "luminosity_distance": np.array([400.0, 410.0]),
    "spin_1z": np.array([0.3, 0.0]),
    "spin_2z": np.array([-0.2, 0.0]),
    "inclination": np.array([0.9, 0.4]),
    "coa_phase": np.array([0.3, 0.0]),
    "right_ascension": np.array([1.375, 0.5]),
    "declination": np.array([-1.211, 0.3]),
    "polarization_angle": np.array([2.659, 0.0]),
    "coa_time": np.array([1_126_259_500.0, 1_126_259_600.0]),
}

segments = simulate_cbc_catalogue(
    "IMRPhenomD",
    ["H1", "L1"],
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
    parameters=catalogue,
    segment_duration=64.0,
    start_time=1_126_259_456.0,
    end_time=1_126_259_968.0,
)
# segments: list[DetectorStrainStack], one per 64 s segment, zero-noise + injected signals.
for stack in segments:
    ...  # write each stack (see Multichannel strains)
```

`simulate_cbc_catalogue` is a convenience over the two building blocks below;
use them directly when you need non-zero backgrounds or your own segment tiling.

## Building blocks

**`simulate_cbc_batch`** — batched, on-device generation and projection. Returns
a `BatchedDetectorStrain`: a `(n_events, n_detectors, n_samples)` JAX array (on
device) plus per-event timing metadata (`coa_time`, `epoch`,
`sampling_frequency`). Each event/detector row is a signal with coalescence near
the segment end; its first sample is at `epoch + coa_time[event]`.

**`assemble_segments`** — host-side scatter of those signals onto a tiling of
fixed-duration segments. Each signal is injected into every segment it overlaps
via [`inject_strains_sequential`](strain-injection.md), which crops it to the
segment — so a signal longer than one segment contributes its overlapping part
to each segment it spans. Segments are **zero-noise by default**; pass
`backgrounds` (one mapping per segment) to inject into existing data, e.g.
coloured noise from `gwmock-noise`.

```python
from gwmock_signal.jax_batch import simulate_cbc_batch, assemble_segments

batch = simulate_cbc_batch(
    "IMRPhenomD", ["H1", "L1"],
    sampling_frequency=4096.0, minimum_frequency=20.0, parameters=catalogue,
)
segments = assemble_segments(
    batch,
    segment_duration=64.0,
    segment_start_times=[1_126_259_456.0, 1_126_259_520.0, ...],
    # backgrounds=[{"H1": ts, "L1": ts}, ...],  # optional, aligned with the starts
)
```

## Conventions and current limitations

- **Earth rotation.** The device path evaluates the antenna pattern and
  geocenter delay **once per event at the segment midpoint**
  (`earth_rotation=False`), matching
  [`project_polarizations_to_network`](detector-projection.md). The
  time-dependent (`earth_rotation=True`) response is **not** available on
  device.
- **Sidereal time.** Sidereal time uses an on-device IAU model with default leap
  seconds and `dut1 = 0`, accurate to ~1e-4 rad; this is the small difference
  from the Astropy-based CPU path.
- **Worst-case buffer.** A wide mass range makes the shared generation buffer as
  long as the lowest-mass event needs. For very heterogeneous catalogues,
  simulate in chirp-mass groups to bound memory.
- **Validation.** Every event/detector of the device path is cross-checked
  against the CPU pipeline (white overlap > 0.999); the small gap is the
  sidereal-time default above.
