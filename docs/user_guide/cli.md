---
title: Command-line interface
description: The gwmock-signal Typer CLI (logging and inject commands).
icon: material/console-line
---

# Command-line interface

The distribution installs a console script **`gwmock-signal`** (defined in
`pyproject.toml`). Import name for the library remains **`gwmock_signal`**
(underscore).

## Global options

```bash
gwmock-signal --help
gwmock-signal -v DEBUG   # or: --verbose DEBUG
```

`-v` / `--verbose` sets the log level for the `gwmock_signal` logger (Rich
handler on stderr). Subcommands inherit this after the callback runs.

## `inject cbc` — CBC into zero-noise segments

End-to-end helper: load CBC parameters from JSON, build a zero-noise GWpy
background centred on `coa_time`, run the same pipeline as
[`inject_cbc_signal`](../api/pipeline/), then either write HDF5 or print a short
per-detector RMS summary.

```bash
gwmock-signal inject cbc --help
```

### Parameters file (`--params`)

The JSON object must include every **required** CBC key understood by
[`CBCSimulator`](../api/simulator/) (gwmock-pop canonical names). At minimum:

| Key                     | Meaning                                                                     |
| ----------------------- | --------------------------------------------------------------------------- |
| `detector_frame_mass_1` | Primary mass in the detector frame (solar masses)                           |
| `detector_frame_mass_2` | Secondary mass in the detector frame (solar masses)                         |
| `coa_time`              | Coalescence GPS time (seconds) — also used to centre the background segment |
| `distance`              | Luminosity distance (Megaparsec)                                            |
| `inclination`           | Inclination angle (radians)                                                 |
| `right_ascension`       | Right ascension (radians)                                                   |
| `declination`           | Declination (radians)                                                       |
| `polarization_angle`    | Polarization angle ψ (radians)                                              |

Additional keys (for example `spin_1z`, `spin_2z`, `coa_phase`) are passed to
the waveform backend. The default **`--backend lal`** implementation accepts
only the parameters documented for LAL time-domain generation (unknown keys
error). **`--backend pycbc`** forwards extras like PyCBC’s `get_td_waveform`.
**`--backend ripple`** (JAX; requires `gwmock-signal[jax]`) supports a fixed set
of approximants and converts ripple’s frequency-domain output to the time domain
on CPU — see [Waveforms](waveform.md).

Minimal example file `cbc.json`:

```json
{
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "coa_time": 1126259462.4,
    "distance": 410.0,
    "inclination": 0.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0
}
```

### Network (`--network`)

The value is resolved in order:

1. If it is an existing path, or ends with `.yaml`, `.yml`, or `.json`, it is
   passed to [`Network.from_file`](../api/network/) (detector network definition
   file).
2. Otherwise it is treated as a **named preset** (for example `H1L1V1`) via
   `Network.from_name`.
3. If that fails, it is split on commas and interpreted as PyCBC detector codes
   (for example `H1,L1,V1`).

Bundled named presets include:

| Preset                   | Detectors                                         |
| ------------------------ | ------------------------------------------------- |
| `H1L1`, `H1L1V1`, `HLVK` | Built-in LAL detector-code groups                 |
| `ET-triangle`, `ET-L`    | Built-in LAL detector-code groups                 |
| `ET-Triangle-Sardinia`   | `ET1_SARD`, `ET2_SARD`, `ET3_SARD`                |
| `ET-Sardinia` (alias)    | Same as `ET-Triangle-Sardinia`                    |
| `ET-Triangle-EMR`        | `ET1_EMR`, `ET2_EMR`, `ET3_EMR`                   |
| `ET-EMR` (alias)         | Same as `ET-Triangle-EMR`                         |
| `ET-2L-Aligned`          | `ET1_2L_ALIGNED_SARD`, `ET2_2L_ALIGNED_EMR`       |
| `ET-2L-Misaligned`       | `ET1_2L_MISALIGNED_SARD`, `ET2_2L_MISALIGNED_EMR` |

### Output (`--output`)

- If **`--output` is set**: strains are written as GWpy **`TimeSeriesDict`**
  HDF5 (one dataset per detector).
- If **omitted**: one line per detector is printed to stdout showing RMS strain
  and duration. Example output:

```text
H1  rms=1.2345e-21  duration=16.0s
L1  rms=1.1132e-21  duration=16.0s
V1  rms=8.7621e-22  duration=16.0s
```

### Other flags

| Flag                                                     |    Default | Role                                                                  |
| -------------------------------------------------------- | ---------: | --------------------------------------------------------------------- |
| `--sample-rate`                                          |       4096 | Sample rate (Hz) of the synthetic background (integer)                |
| `--f-min`                                                |         20 | Low-frequency cutoff (Hz) for waveform generation                     |
| `--duration`                                             |         16 | Length (seconds) of zero background; centred on `coa_time`            |
| `--approximant`                                          | IMRPhenomD | Time-domain approximant string for the active `--backend`             |
| `--backend`                                              |        lal | Waveform engine: `lal` (LALSimulation, default), `pycbc`, or `ripple` |
| `--seed`                                                 |   _(none)_ | Optional integer seed passed to `numpy.random.seed`                   |
| `--earth-rotation` / `--no-earth-rotation`               |       true | Enable time-dependent antenna patterns (disable for short waveforms)  |
| `--interpolate-if-offset` / `--no-interpolate-if-offset` |       true | Enable cubic resampling when injection is off-grid                    |

### Example

```bash
gwmock-signal inject cbc --params cbc.json --network H1L1V1 --output injected.h5
```

## Python vs CLI

Use the **CLI** for quick checks and scripted runs from the shell. Use the
**Python API** when you already have `TimeSeries` / `TimeSeriesDict` data,
custom backgrounds, or need `inject_cbc_signal` / `CBCSimulator` options such as
custom `earth_rotation` and `interpolate_if_offset` tuning beyond the boolean
CLI flags (see **[Pipeline API](../api/pipeline/)** and
**[Simulator API](../api/simulator/)**).

## See also

- [Installation](installation.md) — environment and `uv` usage
- [Strain injection examples](strain-injection.md) — library-first injection
  patterns
- [API overview](../api/index.md)
