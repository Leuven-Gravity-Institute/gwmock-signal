---
title: Detector projection examples
description:
    Example workflows for projecting GW polarizations onto ground-based detector
    networks.
---

# Detector projection examples

After generating **plus** and **cross** polarizations (see
[Waveforms](waveform.md)), the next step in many pipelines is to compute the
**strain in each interferometer** using the detector **antenna patterns**
$F_{+}$, $F_{\times}$ and **geometric time delays** relative to the geocenter.
This is required for **injection into multi-detector data**, **end-to-end
simulations**, and cross-checks with **matched filtering** that use the same sky
location and polarization as the search.

This page is **examples only**. **Signatures, parameter semantics, return types,
and exceptions** for `project_polarizations_to_network` live exclusively in the
**[Projection API](../api/projection/)** (generated from docstrings).

<!-- markdownlint-disable -->

!!! tip "API reference"

    Use **API → Projection** for the full contract; the sections below are
    narrative + runnable snippets.

<!-- markdownlint-enable -->

## Example 1 — Waveform then H1 / L1 / V1 projection

```python
from gwmock_signal.waveform import WaveformFactory
from gwmock_signal.projection import project_polarizations_to_network

tc = 1_400_000_000.0
factory = WaveformFactory()
pol = factory.generate(
    "IMRPhenomD",
    {
        "tc": tc,
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

# Sky location and polarization (radians)
ra = 1.23
dec = -0.45
psi = 0.78

strains = project_polarizations_to_network(
    pol,
    ["H1", "L1", "V1"],
    right_ascension=ra,
    declination=dec,
    polarization_angle=psi,
    earth_rotation=True,
)
for name, h in strains.items():
    print(name, h.shape, h.t0)
```

## Example 2 — Short segment, fixed antenna pattern

For very short waveforms where Earth rotation over the segment is negligible:

```python
strains = project_polarizations_to_network(
    pol,
    ["H1", "L1"],
    right_ascension=0.5,
    declination=0.3,
    polarization_angle=1.0,
    earth_rotation=False,
)
```

## Example 3 — Custom detectors (advanced)

For observatories not in the built-in LAL cache, pass
`gwmock_signal.detector.CustomDetector` instances (or load a YAML/JSON network
with [`Network.from_file`](../api/network/) or a bundled ET preset such as
`Network.from_preset("ET-Triangle-Sardinia")`) and use the same
`project_polarizations_to_network` call pattern as for `H1` / `L1` strings. See
the **[Projection API](../api/projection/)** for the supported `detector_names`
types.

## Parameter and units checklist

| Quantity                                               | Unit       | Notes                                                   |
| ------------------------------------------------------ | ---------- | ------------------------------------------------------- |
| `right_ascension`, `declination`, `polarization_angle` | radians    | Same convention as PyCBC `Detector.antenna_pattern`.    |
| GW polarizations                                       | strain     | Dimensionless \(h\); output strains are dimensionless.  |
| Time bases                                             | GPS / GWpy | Input `plus`/`cross` must share a compatible time axis. |

## Edge cases and pitfalls

- **Misaligned arrays:** If `plus` and `cross` differ in length, sample rate, or
  epoch, the implementation should raise a clear error before projecting.
- **Unknown detector:** Invalid IFO names should fail with an actionable message
  listing supported or loaded detectors.
- **Unconfigured custom sites:** Custom interferometer configs must be loaded
  successfully; otherwise projection is undefined.
- **Numerics:** Cubic interpolation (if used) can ring at edges; for production
  injections, prefer waveforms that are windowed or long enough that edge
  effects are negligible.

## Choosing the projection implementation

The `earth_rotation=True` branch has two implementations of the same algorithm,
selected by `backend` on `project_polarizations_to_network`, or by
`projection_backend` on a simulator that projects for you:

```python
from gwmock_signal import CBCSimulator

simulator = CBCSimulator(waveform_model="IMRPhenomD", projection_backend="jax")
```

- `"numpy"` — the default on `CBCSimulator` and on the projection function. Runs
  on the host, needs no optional dependency, and asks Astropy for sidereal time
  at every sample.
- `"jax"` — the same algorithm compiled into one fused kernel. Requires JAX
  (`pip install 'gwmock-signal[jax]'`), and **runs on whatever JAX backend is
  installed**: on a CPU with the plain wheel, on a GPU with a CUDA build. The
  continuous-wave simulator already defaults to it.

**What it buys.** Projection is where a long transient spends its time: measured
at 1024 s and 8192 Hz across five ET detectors, it was 604 s of a 620 s
single-event run — 97% — and the device path did the same work in 225 s, **2.7x
faster on the same CPU**. A short segment will not show this, since the fixed
compilation cost is then a larger share.

**It is not a different answer.** The two agree to ~1e-10 of peak, and through
`CBCSimulator.simulate` at 32 s and 256 Hz the measured worst disagreement was
8.0e-13 of peak. The difference is floating-point reassociation.

Two constraints come with `"jax"`:

- It is only available with `earth_rotation=True`. The constant-pattern branch
  is a single frequency-domain phase shift with no device implementation, and it
  is already the cheap branch; asking for both is refused rather than silently
  served from the host.
- One call may span at most `MAX_LINEAR_SIDEREAL_SPAN_SECONDS` (86400 s). The
  device path extrapolates sidereal time linearly from a single Astropy anchor,
  which is validated to that span and refused beyond it. Consecutive segments
  are unaffected at any run length, because each one re-anchors.

It also requires JAX in 64-bit mode, which the simulators enable for you when
you select it. Calling the projection function directly does not: in 32-bit mode
it raises rather than returning a series that is wrong by ~1% of peak while
still looking like strain.

## Scientific notes

- Antenna patterns and delays follow **PyCBC/LAL** conventions, consistent with
  many **Bilby** and **PyCBC** analyses.
- For long signals or high precision, keep **`earth_rotation=True`** unless you
  have verified the approximation.

## See also

- [User guide overview](index.md)
- [Waveforms](waveform.md) — produce `plus` / `cross`
- [Strain injection](strain-injection.md) — embed projected strain in a segment
- [Projection API](../api/projection/)
- [API overview](../api/index.md)
- [Documentation home](../index.md)
