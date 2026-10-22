---
title: Waveforms and backends
description:
    Waveform backends (LAL, gwsignal, PyCBC, ripple), WaveformFactory registry,
    supported approximant names, and time-domain examples.
---

# Waveforms and backends

## Overview

This page explains how **time-domain** gravitational-wave polarizations $h_{+}$
and $h_{\times}$ are produced and how they surface as
[`TimeSeries`](../api/timeseries/) values (`dict` keys `"plus"` and `"cross"`).
Typical uses include **injection studies**, **matched-filter sanity checks**,
and **mock data challenges**.

The helpers are **stateless at call time** for a single generation: one call
returns one waveform pair. [`WaveformFactory`](../api/waveform/) adds a
**registry**: every built-in name from the active **waveform backend** is
pre-registered, and you may add more with `WaveformFactory.register_model`.

!!! tip "API reference"

    Exact signatures, types, and exceptions live under **[Waveform API](../api/waveform/)**;
    the sections below are narrative and copy-paste examples.

## Backend architecture

Waveform generation is split into three layers:

1. **`WaveformBackend`** (abstract interface; see
   [Waveform API](../api/waveform/)) Implementations expose
   `available_approximants()` and
   `generate_td_waveform(approximant, tc, sampling_frequency, minimum_frequency, **params)`.

2. **Concrete backends**
    - **`LALSimulationBackend`** (the default where lalsuite is installed) —
      uses **LALSimulation**, evaluating in the frequency domain
      (`SimInspiralChooseFDWaveform`, or `SimInspiralFD` for time-domain-native
      approximants) and conditioning to the time domain so coalescence sits at
      the frequency-domain phase reference. Shipped with the core dependency
      **`lalsuite`**; no optional extra is required. On Python 3.15 and later,
      which lalsuite publishes no wheel for yet, it is unavailable and
      constructing it raises an `ImportError` saying so; the default then falls
      back to `RippleBackend` (see below). Parameter handling is **strict**:
      only documented mass, spin, distance, inclination, and coalescence-phase
      aliases are accepted; unknown keys raise `ValueError`. How the
      frequency-domain waveform becomes a time series is described under
      [Frequency-to-time conditioning](#frequency-to-time-conditioning-lal-and-gwsignal).

    - **`GWSignalBackend`** — routes **frequency-domain-native** approximants
      through the LVK's forward-looking
      [`lalsimulation.gwsignal`](https://docs.ligo.org/waveforms/reviews/newwaveforminterface/)
      interface, sharing the LAL backend's segment sizing, phase-reference
      handling, and FD-to-time conditioning; outputs are **bit-identical** to
      `LALSimulationBackend`. Time-domain-native approximants fall back to the
      LAL path (gwsignal's frequency-domain output discards the conditioning
      epoch those approximants need). Ships with the core **`lalsuite`**
      dependency, so it has the same Python 3.15 limitation; no optional extra
      is required. External Python models (e.g. SEOBNRv5 via `pyseobnr`) are not
      exposed yet.

    - **`PyCBCBackend`** (optional) — delegates to PyCBC’s time-domain waveform
      path via the internal `pycbc_waveform_wrapper`. Requires installing
      **`gwmock-signal[pycbc]`** (see [Installation](installation.md)). Extra
      parameters are forwarded to PyCBC like a direct `get_td_waveform` call.

    - **`RippleBackend`** (optional) — uses
      **[ripple](https://github.com/GW-JAX-Team/ripple)** (JAX). ripple is a
      **frequency-domain** generator; the backend conditions its output into the
      time-domain `plus`/`cross` series via an inverse FFT. Requires installing
      **`gwmock-signal[jax]`** (see [Installation](installation.md)). Supports a
      fixed set of approximants (see below). The conversion runs on **CPU host
      memory** — there is **no GPU acceleration yet**. Constructor options:
      `f_ref` (reference frequency; defaults to `minimum_frequency`),
      `segment_duration` (fixed analysis length in seconds; auto-estimated from
      the chirp time when omitted), and `ringdown_fraction` (fraction of the
      segment reserved after coalescence).

3. **`WaveformFactory`** — on construction, takes
   `backend: WaveformBackend | None` (default `default_waveform_backend()`:
   `LALSimulationBackend()` where lalsuite is installed, otherwise
   `RippleBackend()` with a logged warning when the `jax` extra is installed,
   otherwise an `ImportError` naming both remedies), registers **one callable
   per** name returned by `backend.available_approximants()`, and dispatches
   `generate(waveform_model, parameters, **extra)` to the right callable.

[`CBCSimulator`](../api/simulator/) and [`inject_cbc_signal`](../api/pipeline/)
accept an optional `waveform_backend` and build a `WaveformFactory(backend=…)`
internally; given none, they use `default_waveform_backend()` (LAL where
lalsuite is installed, ripple where it is not and the `jax` extra is, otherwise
an `ImportError` naming both). The CLI `inject cbc` exposes **`--backend lal`**,
**`--backend gwsignal`**, **`--backend pycbc`**, or **`--backend ripple`**, and
without `--backend` takes the same default.

## Supported waveform model names

Built-in model names are **not hard-coded in this documentation**: they depend
on your installed **`lalsuite`** / **PyCBC** versions.

| Backend                                       | Where names come from                                                                                           | Install                  |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **LAL** (default where lalsuite is installed) | Every approximant for which LAL marks a **time-domain** implementation (`SimInspiralImplementedTDApproximants`) | Core (`lalsuite`)        |
| **PyCBC**                                     | `pycbc.waveform.td_approximants()`                                                                              | Optional extra `[pycbc]` |
| **ripple**                                    | Fixed curated set (see below)                                                                                   | Optional extra `[jax]`   |

Unlike LAL/PyCBC, the **ripple** backend supports a **fixed** set of
approximants: `IMRPhenomD`, `IMRPhenomHM`, `IMRPhenomXAS`, `IMRPhenomXHM`,
`TaylorF2`, `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3`, `IMRPhenomPv2`,
`IMRPhenomXP`, `IMRPhenomXPHM`. List them with
`RippleBackend().available_approximants()`. Aligned-spin models reject in-plane
spins; only the NRTidal variants and `TaylorF2` accept `lambda_1`/`lambda_2`;
only the precessing models (`IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomXPHM`)
accept in-plane spin components.

**List every name in your environment**

```bash
python -c "from gwmock_signal.waveform import WaveformFactory; print('\n'.join(WaveformFactory().list_models()))"
```

With PyCBC installed:

```bash
python -c "from gwmock_signal.waveform import WaveformFactory, PyCBCBackend; print('\n'.join(WaveformFactory(backend=PyCBCBackend()).list_models()))"
```

The two lists **overlap in spirit** (many IMR models exist in both stacks) but
are **not identical** string-for-string. If a name fails, use the listing above
or check upstream tables (LALSimulation approximant documentation;
[PyCBC waveforms](https://pycbc.org/pycbc/latest/html/waveform.html)).

**Illustrative examples** (always confirm with `list_models()` on your machine):
`IMRPhenomD`, `IMRPhenomPv2`, `SEOBNRv4`, `SEOBNRv4PHM`, `TaylorF2`,
`SpinTaylorT4`, …

## Examples

### Example 1 — `WaveformFactory` with the LAL backend (BBH, IMRPhenomD)

```python
from gwmock_signal.waveform import LALSimulationBackend, WaveformFactory

factory = WaveformFactory(LALSimulationBackend())
pol = factory.generate(
    "IMRPhenomD",
    {
        "tc": 1_400_000_000.0,
        "detector_frame_mass_1": 36.0,
        "detector_frame_mass_2": 29.0,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "distance": 410.0,
        "inclination": 0.0,
        "coa_phase": 0.0,
    },
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)
hp, hx = pol["plus"], pol["cross"]
print(hp.t0, hp.duration)
```

### Example 2 — Explicit PyCBC backend (requires optional install)

```python
from gwmock_signal.waveform import PyCBCBackend, WaveformFactory

factory = WaveformFactory(backend=PyCBCBackend())
params = {
    "tc": 1_400_000_000.0,
    "mass1": 40.0,
    "mass2": 30.0,
    "spin1z": 0.5,
    "spin2z": -0.3,
    "distance": 400.0,
    "inclination": 0.0,
    "coa_phase": 0.0,
}
pol = factory.generate(
    "IMRPhenomD",
    params,
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)
```

### Example 3 — Direct PyCBC wrapper (lazy import)

`pycbc_waveform_wrapper` is still available for **direct** PyCBC calls without
going through `WaveformFactory`. Importing it triggers a PyCBC import; install
**`gwmock-signal[pycbc]`** first.

```python
from gwmock_signal.waveform import pycbc_waveform_wrapper

pol = pycbc_waveform_wrapper(
    tc=1_400_000_000.0,
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
    waveform_model="IMRPhenomD",
    mass1=36.0,
    mass2=29.0,
    spin1z=0.0,
    spin2z=0.0,
    distance=410.0,
    inclination=0.0,
    coa_phase=0.0,
)
hp, hx = pol["plus"], pol["cross"]
```

### Example 4 — Register your own waveform model

Use this when you have a custom generator (e.g. **numerical relativity** or a
**ROM**) and want the same `{"plus", "cross"}` contract as the built-in
backends. The callable does **not** have to call PyCBC or LAL.

```python
import numpy as np

from gwmock_signal import TimeSeries
from gwmock_signal.waveform import WaveformFactory


def toy_gaussian_sine_burst(
    *,
    waveform_model: str,
    tc: float,
    sampling_frequency: float,
    minimum_frequency: float,
    **kwargs,
) -> dict[str, TimeSeries]:
    width_s = float(kwargs.get("width", 0.1))
    f_hz = float(kwargs.get("f0", 150.0))
    dt = 1.0 / sampling_frequency
    n_half = max(8, int(width_s * sampling_frequency))
    t = (np.arange(-n_half, n_half + 1) * dt) + tc
    env = np.exp(-0.5 * ((t - tc) / (width_s / 3)) ** 2)
    phase = 2 * np.pi * f_hz * (t - tc)
    hp = env * np.cos(phase)
    hc = env * np.sin(phase)
    return {
        "plus": TimeSeries(hp, t0=float(t[0]), sample_rate=sampling_frequency),
        "cross": TimeSeries(hc, t0=float(t[0]), sample_rate=sampling_frequency),
    }


factory = WaveformFactory()
factory.register_model("toy_burst", toy_gaussian_sine_burst)
out = factory.generate(
    "toy_burst",
    {"tc": 1_400_000_000.0, "f0": 120.0, "width": 0.05},
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)
```

### Example 5 — Inspect registered names

```python
from gwmock_signal.waveform import WaveformFactory

factory = WaveformFactory()
names = factory.list_models()
print(len(names), names[:5])
```

You can look up a single registered generator with `get_model()`:

```python
generator = factory.get_model("IMRPhenomD")
# generator is the wrapped backend callable
```

### Example 6 — Register a model via import string

The `register_model` method also accepts a string path to a callable, so you can
register waveform generators without importing them at registration time:

```python
factory = WaveformFactory()

# Colon syntax: "module.path:callable"
factory.register_model("my_nr_model", "mypackage.waveforms:my_nr_generator")

# Dot syntax: "package.module.callable"
factory.register_model("another_model", "mypackage.waveforms.a_func")
```

The string is resolved via `importlib`; the resolved object must be callable.

## LAL backend — accepted parameters

When using `LALSimulationBackend` (the default where lalsuite is installed),
only the following parameter keys are accepted. Extra keys raise `ValueError`.

| Canonical name          | Aliases    | Default      | Description                   |
| ----------------------- | ---------- | ------------ | ----------------------------- |
| `detector_frame_mass_1` | `mass1`    | **required** | Primary mass (solar masses)   |
| `detector_frame_mass_2` | `mass2`    | **required** | Secondary mass (solar masses) |
| `luminosity_distance`   | `distance` | **required** | Luminosity distance (Mpc)     |
| `spin_1x`               | `spin1x`   | `0.0`        | Primary spin x-component      |
| `spin_1y`               | `spin1y`   | `0.0`        | Primary spin y-component      |
| `spin_1z`               | `spin1z`   | `0.0`        | Primary spin z-component      |
| `spin_2x`               | `spin2x`   | `0.0`        | Secondary spin x-component    |
| `spin_2y`               | `spin2y`   | `0.0`        | Secondary spin y-component    |
| `spin_2z`               | `spin2z`   | `0.0`        | Secondary spin z-component    |
| `inclination`           | —          | `0.0`        | Inclination angle (radians)   |
| `coa_phase`             | —          | `0.0`        | Coalescence phase (radians)   |

The **PyCBC backend** forwards all extra parameters to `get_td_waveform` instead
of rejecting unknown keys.

!!! note "`sampling_frequency` type" The Python API (`WaveformFactory.generate`,
`WaveformBackend.generate_td_waveform`) accepts `sampling_frequency` as `float`.
The CLI `--sample-rate` flag accepts an `int`. Both represent Hertz.

## Tips and pitfalls

- Unknown model names raise `ValueError`; use `list_models()` after install.
- **LAL** surfaces invalid physics or unsupported parameter combinations as
  LAL/LALSimulation errors; **PyCBC** behaves like PyCBC’s `get_td_waveform`.
- **Redshifted neutron stars.** LAL's `IMRPhenomNSBH` refuses a neutron star
  heavier than 3 M☉, and the backend passes detector-frame masses, so a 1.4 M☉
  neutron star is refused from `z ≈ 1.15` on. For that approximant the LAL and
  gwsignal backends evaluate the binary at masses and distance divided by a
  common factor, on a frequency grid multiplied by it, and multiply the result
  by it, which is exact for a frequency-domain model: the masses enter only
  through `M f`. The result agrees with direct generation to about 1e-13 of the
  peak. Errors on other limits (tidal deformability above 5000, for example)
  still surface from LAL.
- Series are **offset by `tc`** in the sense documented for each backend; they
  are **not** cropped to a science segment — windowing is downstream.
- **Reuse one `WaveformFactory`** in hot loops: construction calls
  `available_approximants()` once and can be slow on cold import.

## Frequency-to-time conditioning (LAL and gwsignal)

The LAL and gwsignal backends evaluate the model in the frequency domain, with
coalescence at the frequency-domain phase reference, and inverse-transform it
into a buffer. A periodic transform folds whatever extends past either end of
that buffer back into it, so the buffer is sized to hold the inspiral, merger
and ringdown, and the spectrum must not end in a step. Content a model places
outside those — described under
[content the buffer does not hold](#content-the-buffer-does-not-hold) — is not
held.

- **Tapered, not cut, below `minimum_frequency`.** The model is generated from a
  start frequency below the cutoff and multiplied by a raised cosine that rises
  from zero there to one at `minimum_frequency`. Every bin from
  `minimum_frequency` up is the model's own value, unchanged; the content below
  it is tapered and is part of the injected strain.
- **Start frequency.** As LALSimulation's own conditioning chooses it: the
  frequency with 10% more chirp time than `minimum_frequency`, or, when the
  cutoff lies above the lowest innermost-stable-orbit frequency of a Kerr black
  hole of the binary's mass (heavy binaries, whose band starts in the merger),
  than that frequency instead.
- **Before coalescence**, the buffer holds the quadrupole's chirp time from
  where the largest harmonic the model emits enters the generated band, the
  larger of LALSimulation's bound and this package's 1PN estimate, plus three
  cycles and the taper's time response (one over its width). A harmonic `m`
  reaches a frequency when the quadrupole is at `2 / m` of it, so the largest
  `|m|` enters first. It is the largest `|m|` of an explicit `ModeArray`, and
  otherwise of the modes LALSimulation's mode interface lists for the model by
  default: 4 for IMRPhenomXHM, XPHM and HM, 5 for IMRPhenomTPHM and SEOBNRv4PHM,
  2 for a quadrupole-only model such as IMRPhenomD, TaylorF2, NRTidalv2 or
  IMRPhenomNSBH. A model known to carry higher harmonics whose modes
  LALSimulation cannot list is sized for 5, the largest any of its models emits.
- **After coalescence**, LALSimulation's merger and ringdown time bounds for the
  remnant (eleven e-folds of the fundamental quasi-normal mode at the largest
  spin the components allow), plus three cycles and the taper's time response.
  `ringdown_fraction`, when set, is a lower bound on this as a fraction of the
  buffer; by default no fixed fraction is used.
- The total is rounded up to a power of two seconds; a pinned `segment_duration`
  replaces it and keeps at least half for the inspiral.

How this relates to LALSimulation's `SimInspiralTDFromFD`: the start frequency,
the raised-cosine window, the three extra cycles and the merger and ringdown
bounds are LALSimulation's. Two things differ, both to keep the requested band
exactly as the model gives it. LALSimulation finishes with an eighth-order
Butterworth high-pass at the cutoff and a time-domain taper at both ends, which
change the band just above the cutoff; here the frequency taper sits entirely
below it and the buffer is made long enough that its ends are already quiet. And
the buffer reaches back for higher harmonics, which enter the band before the
quadrupole does.

**What is bounded and what is estimated.** The chirp time from the onset of the
largest harmonic and the merger and ringdown times are LALSimulation's own
bounds (`SimInspiralChirpTimeBound`, `SimInspiralMergeTimeBound`,
`SimInspiralRingdownTimeBound` at `SimInspiralFinalBlackHoleSpinBound`). The
three extra cycles and the taper's time response are estimates, sized by
measurement. The taper's window spreads content that does not chirp through the
taper band — the merger of a heavy, precessing binary, whose band starts there —
for several times one over the band's width on both sides: binaries of 200 to
360 solar masses at a 5 Hz cutoff ring at 4.5 to 5 Hz for about ten seconds
after coalescence. Five widths are reserved at each end. Over a test set of 259
binaries from 19 to 2612 solar masses (IMRPhenomXPHM with multibanding off,
against references holding at least 1024 s on each side), one width left up to
5.1e-03 of a polarization's peak outside the buffer, four 1.0e-04 and five
7.3e-05, the smallest number meeting 1e-04 everywhere; the cost is 3% to 5% more
generated strain than one width. Before coalescence one width already sufficed
(under 1.5e-05); the reservation is symmetric because the window's response is.
None of these terms bounds content a model places outside the inspiral, merger
and ringdown; what lies past the buffer is measured below.

Measured with a 5 Hz cutoff against the same tapered waveform on a buffer four
times longer, at 512 to 4096 Hz (enough for each source's band; the rate does
not change the result, checked at 8192 Hz on the worst cases), as the largest
content outside the buffer as a fraction of peak strain, over equal and unequal
masses (to 250+5 solar masses), aligned and anti-aligned spins of 0.99, in-plane
spins of 0.9, and total masses to 2612 solar masses:

| Approximant                                  | Worst case outside the buffer                                                                                                                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IMRPhenomD                                   | 5e-06                                                                                                                                                                                          |
| IMRPhenomXHM                                 | 5.3e-05 (250+5)                                                                                                                                                                                |
| IMRPhenomXPHM                                | 1.7e-02 (250+5, spins 0.99 aligned); 5e-04 to 3e-03 at 60+30, 130+13 and 250+5 with spins of 0.99 or 0.9 in-plane; 2e-06 to 5e-06 at 2612 solar masses unless spins are 0.99 aligned (1.5e-03) |
| IMRPhenomTPHM, SEOBNRv4PHM, SEOBNRv4HM       | 6e-06                                                                                                                                                                                          |
| IMRPhenomNSBH                                | 4.3e-05                                                                                                                                                                                        |
| TaylorF2 (binary neutron star)               | 3.7e-05 (20 Hz cutoff, 8192 Hz)                                                                                                                                                                |
| IMRPhenomPv2_NRTidalv2 (binary neutron star) | 8e-09 (20 Hz cutoff, 8192 Hz)                                                                                                                                                                  |

The binary neutron stars were measured at a 20 Hz cutoff, where a reference at
8192 Hz is affordable; at a 5 Hz cutoff only a 1024 Hz reference was, and there
the 2.5e-04 left after coalescence is the spectrum cut at that rate's Nyquist
frequency (the same source at 20 Hz leaves 1.2e-03 at 1024 Hz and 8e-09 at 8192
Hz).

These were measured with one taper width at each end; five only lengthen the
buffer. The IMRPhenomXPHM row is the default model, whose multibanded precession
angles put content outside any buffer (below). With multibanding off, the same
sources and a test set of 259 binaries leave at most 7.3e-05 outside the buffer.
SEOBNRv5HM and SEOBNRv5PHM are not part of LALSimulation and were not checked.

### Content the buffer does not hold

- **IMRPhenomXPHM's multibanded precession angles.** By default IMRPhenomXPHM
  evaluates its precession angles on a coarse frequency grid and interpolates
  them (`PhenomXPHMThresholdMband`). The interpolation leaves the spectral phase
  slightly non-smooth, which in time puts content at every delay, on both sides
  of coalescence: at 250+5 solar masses with spins of 0.99 aligned and a 5 Hz
  cutoff, 1.6e-02 of peak strain 5 s after coalescence, still 1e-04 two to four
  thousand seconds away, and 3e-05 out to ±30000 s. It is an artefact of the
  interpolation, not of precession — it appears with zero in-plane spin — and it
  depends on the frequency grid: the same source generated into buffers 1, 2 and
  4 times as long differs inside the buffer by 2e-02 to 3e-02 of peak (mismatch
  3e-04 to 6e-04). Lengthening the buffer helps only slowly — 126 s more after
  coalescence still leaves 1.3e-03 outside — and does not touch the grid
  dependence. Setting `waveform_arguments={"PhenomXPHMThresholdMband": 0.0}`
  reduces both by four orders of magnitude: the same source leaves 8.8e-07
  outside the buffer and changes by 1.3e-06 with the grid (with
  `PhenomXHMThresholdMband` also set to 0, as measured here). The cost is about
  5 times the frequency-domain evaluation time, a few percent of the time per
  event once projection is counted. In band the two agree closely: over 50
  stellar- and intermediate-mass binaries the mismatch between the default and
  the multibanding-off waveform (the 10 km full-cryogenic ET design sensitivity,
  5 Hz cutoff) is 5e-08 at the median and at most 5e-06 maximised over time and
  phase, although they differ by up to 6e-03 of peak in time. LALSimulation's
  own `SimInspiralTD` drops the content after coalescence too — its buffer ends
  about 0.7 s after coalescence, with a taper.
- **A spectrum that ends in a step** rings with a tail falling as one over time.
  TaylorF2 stops at its innermost stable orbit (3.7e-05 of peak outside the
  buffer above).
- **IMRPhenomXHM's own multibanding** (`PhenomXHMThresholdMband`, also used
  inside IMRPhenomXPHM) leaves a much weaker component of the same kind: 1e-05
  to 5e-05 of peak outside the buffer, and a waveform that changes with the
  frequency grid by up to 3.5e-05 of peak (mismatch at most 2.4e-09) on a 250+5
  solar-mass binary with spins of 0.99 aligned and two heavier aligned-spin
  binaries. With `PhenomXHMThresholdMband = 0` the grid dependence falls to
  below 1.5e-05 of peak and to 1e-08 once the buffer is twice as long.

### Recommended: turn off IMRPhenomXPHM and IMRPhenomXHM multibanding for injection data

With its default multibanding, the waveform IMRPhenomXPHM returns depends on the
length of the buffer it is generated into — and so on the segment duration of a
simulation — and leaves content outside any buffer, as described above. For
injection data that must not depend on how it was segmented, set both
`PhenomXPHMThresholdMband` and `PhenomXHMThresholdMband` to 0. IMRPhenomXHM
alone is affected far less (up to 3.5e-05 of peak, above), and setting
`PhenomXHMThresholdMband` to 0 reduces its grid dependence too: to at most
1.5e-05 of peak at the planned buffer and 1e-08 once the buffer is twice as
long, on the same three binaries.

- **Cost.** About 4.7 times the frequency-domain evaluation time, measured over
  50 stellar- and intermediate-mass binaries at a 5 Hz cutoff: a few percent of
  the time per event once projection onto the detectors is counted.
- **Agreement in band.** Over the same 50 binaries the mismatch between the
  default and the multibanding-off waveform (h+, 10 km full-cryogenic ET design
  sensitivity, 5 Hz cutoff) is 5e-08 at the median and at most 5e-06 maximised
  over time and phase.

From the Python API, pass them as `waveform_arguments`, which reach the LAL
dictionary of every call:

```python
from gwmock_signal import CBCSimulator
from gwmock_signal.waveform.backends.lal import LALSimulationBackend

simulator = CBCSimulator("IMRPhenomXPHM", waveform_backend=LALSimulationBackend(f_ref=5.0))
params = {
    **event,  # the source parameters
    "waveform_arguments": {
        "PhenomXPHMThresholdMband": 0.0,
        "PhenomXHMThresholdMband": 0.0,
    },
}
strain = simulator.simulate(params, detectors, sampling_frequency=8192.0, minimum_frequency=5.0)
```

From a gwmock configuration, put them under `signal.waveform-options`, which
gwmock passes to gwmock-signal as `waveform_arguments`:

```yaml
signal:
    waveform-model: IMRPhenomXPHM
    waveform-backend: lal
    waveform-options:
        PhenomXPHMThresholdMband: 0.0
        PhenomXHMThresholdMband: 0.0
```

Not under `signal.waveform-arguments`: those are merged into each event's source
parameters, which the LAL backend rejects for these keys, so they are not
applied. The configuration above was checked end to end: the strain gwmock
writes equals, bit for bit, the strain `CBCSimulator.simulate` returns with the
same `waveform_arguments`, and differs from the default by 1.2e-03 of peak on a
precessing 111+111 solar-mass binary.

Sizing for the largest harmonic the model emits, rather than for `m = 3`,
lengthens the buffers of higher-harmonic models: on a 250+5 solar mass binary
the earlier sizing left 6.7e-03 of peak before the buffer, and 5.3e-05 is left
now. The previous conditioning — a hard cut at the cutoff and a buffer sized for
the inspiral above it with a fixed tenth after coalescence — left up to 0.34 of
peak at the start of a 4 s buffer for a 2302+248 solar mass binary.

Buffers are longer as a result, most for heavy binaries and for models with
higher harmonics: at a 5 Hz cutoff and 8192 Hz, a set of IMRPhenomXPHM binary
black holes spanning 14 to 1665 solar masses needs about 6.8 times the strain of
the previous conditioning: twice what `m = 3` needed, and 5% more for five taper
widths rather than one. With one taper width this was measured at 2.5 times the
generation and projection time per event. Binary-neutron-star and
neutron-star–black-hole buffers grew by 3% and 6% before the taper reservation,
which was not re-measured on them.

If LALSimulation refuses to start an approximant below the cutoff, the backend
generates it from the cutoff and places the same taper _above_ it instead, which
does reduce the lowest part of the band, and logs a warning. Every approximant
tried — the IMRPhenomX, D, NSBH and NRTidalv2 families, TaylorF2, IMRPhenomTPHM
and SEOBNRv4 — starts below the cutoff without complaint. The ripple backend has
its own taper, described in its constructor documentation.

## Scientific notes

- Default **LAL** path follows `SimInspiralChooseTDWaveform` conventions; see
  LIGO Algorithm Library / LALSimulation documentation for parameter meanings.
- **PyCBC** wrapper path follows PyCBC’s `get_td_waveform` keyword conventions.
- Outputs are this package's `TimeSeries`; `to_gwpy()` and `to_pycbc()` convert
  them for **GWpy**, **frame**, **PyCBC** and **Bilby** workflows (see
  [Time series](time-series.md)).

## See also

- [User guide overview](index.md)
- [Installation](installation.md) — core vs `[pycbc]` optional extra
- [Command-line interface](cli.md) — `inject cbc --backend` / `--approximant`
- [Detector projection](detector-projection.md)
- [Continuous waves](continuous-waves.md)
- [Waveform API reference](../api/waveform/)
- [Documentation home](../index.md)
