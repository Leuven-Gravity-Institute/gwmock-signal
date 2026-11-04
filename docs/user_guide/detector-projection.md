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
)
for name, h in strains.items():
    print(name, h.shape, h.t0)
```

Each returned series is longer than the polarizations and starts earlier. The
projection delays the signal by the light-travel time from the geocentre to the
detector, up to about 21 ms, so the polarizations are zero-padded at both ends
by `edge_padding(sample_rate, sinc_taps, kaiser_beta)` samples (from
`gwmock_signal.projection.resampling`) before projecting, and the output covers
the padded span on the input's sample lattice. The padding is the bound for any
ground-based detector plus the resampling kernel's half-width, so every
detector, sky position and backend gets the same start time and length. Inside
the input's span the samples are those of a same-length projection, up to
rounding; outside it they hold the delayed signal a same-length projection would
drop. This matters most for signals shorter than the delay: a 0.1 ms burst, 17
samples at 8192 Hz, would otherwise be lost entirely. Crop to the input's span
if you need it, at the cost of that content; injecting the series into data
places it by its own start time.

## Example 2 — Short segment, fixed antenna pattern

For very short waveforms where Earth rotation over the segment is negligible:

```python
from gwmock_signal.projection.effects import ProjectionEffects

strains = project_polarizations_to_network(
    pol,
    ["H1", "L1"],
    right_ascension=0.5,
    declination=0.3,
    polarization_angle=1.0,
    effects=ProjectionEffects(earth_rotation=False),
)
```

## Projection effects

Every source -- compact binaries, bursts and continuous waves -- is projected by
the same code, and `ProjectionEffects` says which physical effects it includes.
**All are on by default**, and each can be turned off on its own:

| Effect              | What it is                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `earth_rotation`    | The antenna pattern and delays follow the Earth's rotation sample by sample. Off, they are held at the span's midpoint, which suits only a signal of seconds. |
| `finite_arm_length` | Each arm responds to the strain averaged along the light's round trip; see [Finite-arm response](#finite-arm-response).                                       |
| `source_precession` | The J2000 source direction is precessed to the epoch of date before the Earth's orientation is applied, as LALPulsar's barycentring does.                     |
| `roemer_delay`      | The light-travel time between the geocentre and the solar-system barycentre, from the Earth's orbit.                                                          |
| `einstein_delay`    | The relativistic time dilation of the Earth's clock, TDB minus TT.                                                                                            |
| `shapiro_delay`     | The Sun's gravitational delay.                                                                                                                                |

```python
from gwmock_signal.projection.effects import ProjectionEffects

everything = ProjectionEffects()  # the default
no_solar_system = ProjectionEffects(roemer_delay=False, einstein_delay=False, shapiro_delay=False)
lal_convention = ProjectionEffects.long_wavelength()  # all off
lal_convention_rotating = ProjectionEffects.long_wavelength(earth_rotation=True)
```

`ProjectionEffects.long_wavelength()` is the convention of Bilby's and PyCBC's
default projection: a point detector at its vertex, a geocentre at rest and a
J2000 direction with sidereal time of date. An analysis using that convention
recovers a right ascension offset from an injection made with
`source_precession` on (about 0.43 degrees in 2030), and a slightly different
waveform for a long or loud signal; turn the effects off to match it exactly.

**The solar-system delays for a transient.** A compact binary's `coa_time` and a
burst's `peak_time` are geocentre times, and a catalogue's detector-frame masses
already include whatever constant Doppler shift the source has. So for a
transient the solar-system delay enters only through how it **changes** across
the signal: its value and its rate at the reference time are taken out. What
remains is the Earth's orbital acceleration -- for a binary neutron star in the
Einstein Telescope band from 5 Hz, about 4.5 hours long, up to 2.7 ms of timing
over the sky (2.5 ms at the sky position of the package's test) and 0.1 rad of
phase; for anything lasting minutes, microseconds or less. The simulators pass
the reference time themselves; `project_polarizations_to_network` takes
`reference_time`, and without it the time of the polarizations' peak power. A
continuous wave, whose parameters are defined at the barycentre, takes the
delays in full.

**The ephemeris.** The delays are LALPulsar's (`XLALBarycenter` in its original
time system, with the site term from the projection's own geometry), computed
from the JPL DE405 tables LALPulsar ships, `earth00-40-DE405.dat.gz` and
`sun00-40-DE405.dat.gz`. They agree with `lalpulsar.Barycenter` at the geocentre
to 1e-13 s. A table not found locally is downloaded once from the LALSuite
repository into `~/.cache/gwmock-signal/ephemeris` (or
`$GWMOCK_SIGNAL_CACHE_DIR/ephemeris`); on a machine without network access, copy
the files there or pass their paths
(`ephemeris=SolarSystemEphemeris(earth=..., sun=...)` to the projection, the
simulators' `ephemeris` argument). The tables cover 2000 to 2040; a span outside
them is refused while any solar-system delay is on.

## Example 3 — Custom detectors (advanced)

For observatories not in the built-in LAL cache, pass
`gwmock_signal.detector.CustomDetector` instances (or load a YAML/JSON network
with [`Network.from_file`](../api/network/) or a bundled ET preset such as
`Network.from_preset("ET-Triangle-Sardinia")`) and use the same
`project_polarizations_to_network` call pattern as for `H1` / `L1` strings. See
the **[Projection API](../api/projection/)** for the supported `detector_names`
types.

Custom detectors, the ET presets included, project in double precision by
default. Earlier versions projected them through LAL in single precision, so
their strain now differs at that rounding, about 1.5e-7 of peak; see
[Custom detectors](custom-detectors.md) for the `precision` option that
reproduces it.

## Parameter and units checklist

| Quantity                                               | Unit        | Notes                                                   |
| ------------------------------------------------------ | ----------- | ------------------------------------------------------- |
| `right_ascension`, `declination`, `polarization_angle` | radians     | Same convention as PyCBC `Detector.antenna_pattern`.    |
| GW polarizations                                       | strain      | Dimensionless \(h\); output strains are dimensionless.  |
| Time bases                                             | GPS seconds | Input `plus`/`cross` must share a compatible time axis. |

## Edge cases and pitfalls

- **Misaligned arrays:** If `plus` and `cross` differ in length, sample rate, or
  epoch, the implementation should raise a clear error before projecting.
- **Unknown detector:** Invalid IFO names should fail with an actionable message
  listing supported or loaded detectors.
- **Unconfigured custom sites:** Custom interferometer configs must be loaded
  successfully; otherwise projection is undefined.
- **Numerics:** Off-grid injection resamples with a windowed-sinc kernel, which
  can ring at the edges of an injection that does not taper to zero; for
  production injections, prefer waveforms that are windowed or long enough that
  edge effects are negligible.
- **Content near Nyquist:** With `earth_rotation=True` the polarizations are
  resampled at the delayed times with the same kernel, accurate to 0.8 x
  Nyquist. For a whole signal tapered to zero at both ends, pass `tapered=True`:
  the polarizations are oversampled exactly first, and the projection is
  accurate up to their Nyquist frequency. Not for a piece cut from a longer
  signal, such as a continuous-wave segment, whose cut ends the global
  interpolation would ring from. `earth_rotation=False` shifts exactly in the
  frequency domain and ignores the option.

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

## Finite-arm response

LAL, Bilby and PyCBC's default projection treat each detector as a point at its
vertex: the long-wavelength response. A real arm measures the strain averaged
along the light's round trip to its end mirror and back. For an arm of length
`L` the two differ at order `f L / c`: about 1e-2 at 100 Hz and 0.1 at 1 kHz for
a 10 km arm. The round-trip response is on by default (`finite_arm_length` in
`ProjectionEffects`):

```python
from gwmock_signal.network import Network
from gwmock_signal.projection import project_polarizations_to_network

triangle = Network.from_preset("ET-Triangle-EMR")
strain = project_polarizations_to_network(
    {"plus": hp, "cross": hc},
    triangle.detector_names,
    right_ascension=ra,
    declination=dec,
    polarization_angle=psi,
    tapered=True,
)
```

Each arm is weighted by the single-arm transfer function of Essick, Vitale &
Evans, "Frequency-dependent responses in third generation gravitational-wave
detectors",
[Phys. Rev. D 96, 084004 (2017)](https://doi.org/10.1103/PhysRevD.96.084004),
Eq. (4). PyCBC offers the same function as
`Detector.antenna_pattern(..., frequency=f)` and jim as
`finite_arm_length=True`; the formula is in the docstring of
`gwmock_signal.projection.kernels.finite_arm_residual`.

**Where it matters most: the triangle's null stream.** In a triangle each arm
line is read by two interferometers with opposite signs, so a signal cancels in
`E1 + E2 + E3` only if both read it as the same average. At their vertices they
read it 10 km apart. For a 100 Hz wave packet on the `ET-Triangle-EMR` preset,
at the sky position of `tests/projection/test_finite_arm.py`, the null stream
holds 1.9e-2 of a single detector's peak with the long-wavelength response and
7e-7 with the finite-arm one; at 700 Hz, 0.13 and 2.4e-4. What remains with
finite arms grows as `f^3`, from the difference between the two ends of each
round trip.

**How it follows Earth's rotation.** On the `earth_rotation=True` branch each
arm is resampled at its own midpoint delay, sample by sample, as the vertex is
for the long-wavelength response, and the rest of the round trip is applied as a
filter that follows the arm's changing angle to the source. Against the
definition -- the strain of an analytic wave averaged by quadrature along each
arm's light path, with Earth turning during the trip -- it agrees to 1e-11 of
peak at 100 Hz, 4e-10 at 700 Hz and 3e-9 at 1.8 kHz. The difference is bounded
by the Doppler shift of Earth's rotation (at most 1.5e-6) times the round trip's
own correction, since the filter is applied in detector time. With
`tapered=True` that filter is exact up to Nyquist; otherwise it is as accurate
as the resampling kernel, to 0.8 of Nyquist. The `earth_rotation=False` branch
applies the whole response exactly in the frequency domain, at the span's
midpoint.

**What it needs and costs.**

- An arm length for every detector. A built-in code has LAL's (twice the
  vertex-to-midpoint distance LAL records); the ET presets carry their design
  lengths, 10 km for the triangles and 15 km for the 2L; a `CustomDetector`
  takes `arm_length_m`. A detector without one is refused, as are LAL's resonant
  bars.
- Nothing of the backend: the device projection includes it too. The
  stochastic-background simulator projects nothing, and its detector response is
  the overlap reduction function it is given.
- About twice the time of the long-wavelength projection on the rotating branch,
  since each arm is resampled separately: 28.5 s against 14.0 s for 64 s of
  signal at 8192 Hz on a three-detector triangle.

## Scientific notes

- Antenna patterns and delays follow **PyCBC/LAL** conventions, consistent with
  many **Bilby** and **PyCBC** analyses, except for sidereal time (next note).
  With `ProjectionEffects.long_wavelength()`, that is; by default the response
  is the finite-arm one and the direction is precessed (see
  [Projection effects](#projection-effects)).
- **Sidereal time follows the Earth's measured rotation (UT1), not UTC.** The
  orientation of a detector on the rotating Earth is set by the Earth Rotation
  Angle, which IAU 2000 Resolution B1.8 defines as a linear function of UT1;
  Greenwich Mean Sidereal Time is that angle plus the accumulated precession
  (Kaplan 2005, USNO Circular 179, eq. 2.12 and Resolution B1.8,
  [arXiv:astro-ph/0602086](https://arxiv.org/abs/astro-ph/0602086)). UT1 is
  measured and published by the IERS; UTC is kept within 0.9 s of it by leap
  seconds. gwmock-signal takes sidereal time from Astropy with the IERS UT1-UTC,
  so the detectors rotate with the Earth's measured rotation. (Sidereal time is
  the mean one: nutation is not modelled.) LAL's `GreenwichMeanSiderealTime`,
  which the PyCBC and Bilby analyses built on it use, converts GPS time to UTC
  and evaluates its own sidereal-time series with UTC in place of UT1. It agrees
  with the IAU 2006 series at UT1 = UTC to 1e-7 rad, so the difference from
  gwmock's is that substitution: LAL's exceeds it by minus UT1-UTC times the
  Earth's rotation rate (+2.2e-5 rad at GPS 1e9, where UT1-UTC = -0.305 s; at
  most 6.6e-5 rad either way), plus the 1e-7 rad between the two series.
- **What that difference does to an analysis.** Sidereal time enters the
  projection only through the hour angle, sidereal time minus right ascension.
  At any one time, then, strain simulated here is the strain a UTC-convention
  pipeline computes for the right ascension shifted by the whole sidereal-time
  difference: minus (UT1-UTC) times 7.29e-5 rad/s, to within the 1e-7 rad
  between the two series. An analysis with LAL, PyCBC or Bilby recovers the
  right ascension offset by that much (+2.2e-5 rad, about 5 arcsec, at GPS 1e9),
  and the waveform and the SNR are unchanged. The offset is fixed only as far as
  UT1-UTC is: it drifts with it, by about 0.7 ms a day near GPS 1e9 (5e-8 rad a
  day, so 4e-9 rad across a 1.8-hour signal, and from 2.2e-5 to 2.5e-5 rad over
  the following month), and it steps by 7.3e-5 rad at a leap second, where
  UT1-UTC changes by 1 s. With `source_precession` on the offset applies to the
  right ascension of date that the projection uses, not to the J2000 value
  passed in. Compare recovered and injected right ascensions with this offset in
  mind when the comparison is at arcsecond level.
- For long signals or high precision, keep **`earth_rotation=True`** unless you
  have verified the approximation.

## See also

- [User guide overview](index.md)
- [Waveforms](waveform.md) — produce `plus` / `cross`
- [Strain injection](strain-injection.md) — embed projected strain in a segment
- [Projection API](../api/projection/)
- [API overview](../api/index.md)
- [Documentation home](../index.md)
