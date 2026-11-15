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
    [Installation](installation.md)).

    The batched entry points are re-exported from the package root, so prefer
    `from gwmock_signal import simulate_cbc_catalogue` over the
    `gwmock_signal.jax_batch` submodule path. Both work; the root import is the
    advertised one and does not tie you to the module layout.

    Import resolution is lazy, so `import gwmock_signal` succeeds without the
    extra installed. Generating a waveform without it raises an `ImportError`
    naming the install command.

!!! tip "See also"

    The speed of this batched/GPU path against the per-event CPU baseline is
    quantified on the gwmock-benchmark site
    ([Performance](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/performance/)),
    and the ripple backend's agreement with the LAL baseline in
    [Consistency](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/consistency/).

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
from gwmock_signal import simulate_cbc_catalogue

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
the segment end; its first sample is at `epoch + coa_time[event]`. The rows are
the waveform segment padded at each end, as described under "Segment length"
below, and `epoch` describes the padded row.

**`assemble_segments`** — host-side scatter of those signals onto a tiling of
fixed-duration segments. Each signal is injected into every segment it overlaps
via [`inject_strains_sequential`](strain-injection.md), which crops it to the
segment — so a signal longer than one segment contributes its overlapping part
to each segment it spans. Segments are **zero-noise by default**; pass
`backgrounds` (one mapping per segment) to inject into existing data, e.g.
coloured noise from `gwmock-noise`.

```python
from gwmock_signal import assemble_segments, simulate_cbc_batch

batch = simulate_cbc_batch(
    "IMRPhenomD",
    ["H1", "L1"],
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
    parameters=catalogue,
)
segments = assemble_segments(
    batch,
    segment_duration=64.0,
    segment_start_times=[1_126_259_456.0, 1_126_259_520.0, ...],
    # backgrounds=[{"H1": ts, "L1": ts}, ...],  # optional, aligned with the starts
)
```

## Conventions and current limitations

- **Earth rotation.** Both branches of
  [`project_polarizations_to_network`](detector-projection.md) run on device. By
  default (`earth_rotation=True`) the antenna pattern and geocenter delay are
  evaluated **per sample**, and the polarizations are resampled at the delayed
  times. With `earth_rotation=False` they are evaluated **once per event at the
  segment midpoint** and the delay is applied as an exact frequency-domain phase
  shift, which is cheaper but valid only for signals short compared with an
  hour.
- **Segment length.** Each event's strain is longer than its waveform segment
  and starts earlier. As in
  [`project_polarizations_to_network`](detector-projection.md), the segment is
  zero-padded by `edge_padding(sampling_frequency)` samples at each end before
  it is projected, so the geocentre-to-detector delay (up to about 21 ms) cannot
  move signal past either end: without it the rotating branch dropped that
  signal and the phase-shift branch wrapped it round to the other end. Each row
  has `2 * edge_padding(sampling_frequency)` more samples than the waveform
  segment (240 at each end, or 29.3 ms, at 8192 Hz) and holds what the host
  projection returns for the same waveform, start time included. `epoch` and, on
  an aligned batch, `start_index` describe the padded row, so a consumer that
  places rows by either needs no change; one that assumed the waveform segment's
  length must read it from the strain's shape.
- **Sidereal time.** Both paths take sidereal time from Astropy, IERS UT1-UTC
  included. With `earth_rotation=False` the batched path evaluates it on the
  host, once per event at the segment midpoint. With `earth_rotation=True`
  Astropy's sidereal time is a straight line within each UTC day that bends at
  every UTC midnight, and the host describes it per event by a few numbers: its
  value at the event's first sample, the rate to the next UTC midnight, and the
  change of rate at that midnight and the next, each rate a chord between
  Astropy values (`gwmock_signal.projection.sidereal.gmst_piecewise`). The
  device evaluates the line, and adds the bends only for the events whose
  segment crosses a midnight, which are projected apart from the others.
  Measured against Astropy every 0.5 s over 4096 s segments from 2005 to 2026,
  one crossing each UTC midnight and four inside each UTC day, the largest error
  was 9.95e-14 rad (a geocentre delay of 2.1e-15 s). A segment starting in the
  last nine seconds before a leap second is the exception: up to 1.6e-12 rad
  measured there, 3.2e-12 rad bounded (see `gwmock_signal.projection.sidereal`).
  A batch that mixes events crossing a midnight with events crossing none runs
  each group separately: it compiles and launches a second kernel variant per
  detector, and each variant is its own entry in the bounded kernel cache (32
  executables), so a run whose batches mix the two can hold up to twice as many
  compiled kernels. While it runs, the full batch's inputs, the two gathered
  subsets, each group's output and the reassembled result are live at once. In
  one test on a Linux CPU (peak resident memory, 8 events of 1,048,718 samples
  on three detectors), the projection added 1.0 GB for a batch split half and
  half, 1.66 GB with one crossing event and seven clear, and 1.7 GB for an
  all-crossing or all-clear batch; the batch memory estimate, unchanged by this,
  gave 5.9 GB there. Peak memory on an accelerator was not measured, and this
  test does not validate the estimate on a GPU.
- **Worst-case buffer.** A wide mass range makes the shared generation buffer as
  long as the lowest-mass event needs. For very heterogeneous catalogues,
  simulate in chirp-mass groups to bound memory.
- **Validation.** The tests compare the batched path with
  `project_polarizations_to_network` on the same polarizations, for each event
  and detector of their catalogues and sample by sample over the padded span.
  Measured relative to each series' peak, the largest difference is about 1e-12
  on the rotating branch (2e-14 RMS) and below 1e-14 on the phase-shift branch
  (1e-16 RMS): rounding, from the two paths ordering the same arithmetic
  differently.
