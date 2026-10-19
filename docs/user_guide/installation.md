---
title: Installation
description:
    Install gwmock-signal with uv, dependency groups from source, and verify the
    CLI.
---

# Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock-signal` (PyPI distribution name, hyphen) and importing
**`gwmock_signal`** (underscore) in Python.

After install, continue with the [user guide overview](index.md),
[Quick Start](quick_start.md), or [Command-line interface](cli.md).

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- **Python:** 3.13, 3.14 or 3.15 (`requires-python` in `pyproject.toml` is
  `>=3.13`).
- **Operating system:** Linux or macOS (same range as the published wheels and
  CI).

<!-- prettier-ignore-start -->

!!! note "Python version pin"

    When creating a virtual environment with `uv`, pass an explicit interpreter
    (for example `uv venv --python 3.13`) so you do not pick an unsupported
    Python from your machine default.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install the library for downstream use is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-signal
```

The wheel includes **runtime dependencies** declared in `pyproject.toml`:
**`numpy`**, **`astropy`**, **`typer`**, **`pyyaml`**, **`lalsuite`** (on Python
3.14 and earlier), and their transitive installs. **GWpy**, **h5py**, **PyCBC**
and **ripple** are **not** core dependencies; install them with the optional
extras **`gwmock-signal[gwpy]`** (conversion to and from GWpy series, frame
output), **`gwmock-signal[hdf5]`** (faster HDF5), **`gwmock-signal[pycbc]`**
(PyCBC waveform backend or the `pycbc_waveform_wrapper` helper) and
**`gwmock-signal[jax]`** (the JAX/ripple waveform backend, which pulls in
`rippleGW` and JAX).

There is **no** `gwmock-signal[dev]` extra on PyPI; contributor tooling lives in
**uv dependency groups** in the repository (see **Install from source**).

```bash
# Optional waveform backend (PyCBC)
uv pip install 'gwmock-signal[pycbc]'

# Optional waveform backend (JAX/ripple); installs CPU JAX by default
uv pip install 'gwmock-signal[jax]'
```

## Install from source

For the latest `main` branch:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-signal.git
cd gwmock-signal
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Development dependencies

From the repository root, install the `dev` group (pytest, ruff, pre-commit, …):

```bash
uv sync --group dev
```

Documentation build tools (`zensical`, `mkdocstrings-python`):

```bash
uv sync --group docs
```

Multiple groups:

```bash
uv sync --group dev --group docs
```

### Development setup (full)

Typical contributor workflow:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-signal.git
cd gwmock-signal

uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev

npm ci
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Verify installation

```bash
gwmock-signal --help
gwmock-signal inject --help
```

```bash
python -c "import gwmock_signal; print(gwmock_signal.__version__)"
```

## Dependencies (direct)

Declared in `pyproject.toml` for the library:

- **numpy** — arrays throughout
- **astropy** — sidereal time for detector projection
- **typer** — CLI (`gwmock-signal` entry point)
- **lalsuite** — default time-domain CBC waveforms (LALSimulation) and related
  LAL types. Not installed on Python 3.15 and later, for which `lalsuite`
  publishes no wheel yet; there, the LAL waveform backends are unavailable
  (detector geometry does not need it), and with the `jax` extra installed the
  default waveform backend is the ripple one instead.
- **pyyaml** — configuration parsing where used
- **gwmock-data** — the time series, its sampling grid and HDF5 input and
  output, shared by the gwmock packages; `gwmock_signal.TimeSeries` is
  `gwmock_data.TimeSeries`

**Optional** (`pip install 'gwmock-signal[pycbc]'`):

- **pycbc** — alternate waveform backend (`PyCBCBackend`) and
  `pycbc_waveform_wrapper`

**Optional** (`pip install 'gwmock-signal[jax]'`):

- **rippleGW** (and JAX) — JAX/ripple waveform backend (`RippleBackend`); CPU
  JAX by default

**Optional** (`pip install 'gwmock-signal[hdf5]'`):

- **h5py** — faster HDF5 input and output, gzip-compressed. HDF5 files
  (`DetectorStrainStack.write`/`read`,
  `gwmock_signal.io.write_hdf5`/`read_hdf5`, the CLI's `--output`) also work
  without it, through gwmock-data's own HDF5 writer and the pure-Python pyfive
  reader.

**Optional** (`pip install 'gwmock-signal[gwpy]'`):

- **gwpy** — `TimeSeries.to_gwpy()`/`from_gwpy()` and frame (GWF) output

The time series and `gwmock_signal.io.write_hdf5`/`read_hdf5` come from
gwmock-data, so when one of them is missing an optional package the error names
gwmock-data's extra, for example `pip install 'gwmock-data[gwpy]'`. It installs
the same package as the `gwmock-signal` extra of the same name.

## Python 3.15

On Python 3.15, `lalsuite` and `h5py` publish no wheel yet, so a plain
`pip install gwmock-signal` installs without them. What needs neither works: the
time series, detector projection onto built-in and custom detectors, injection,
simulation with a waveform model you register, the CLI, and NumPy, text and HDF5
files. The LAL waveform backends need `lalsuite`, and report how to install it
when used. Install the `jax` extra (`pip install 'gwmock-signal[jax]'`) to
generate CBC waveforms there: with lalsuite absent, the default waveform backend
is then ripple's (IMRPhenomD, IMRPhenomHM, IMRPhenomXAS, IMRPhenomXHM,
IMRPhenomPv2, TaylorF2 and the tidal variants), with a warning that the
waveforms are ripple's rather than LAL's.

From Python 3.15 the package also imports some of its heavier dependencies only
when they are first used (PEP 810): `astropy.time`, needed only to compute
sidereal time, and the simulator behind the command line, which `--help` does
not need. Importing the command-line module takes about 25 ms there, against
about 155 ms without it. Earlier Python versions import them as before.

## Getting help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock-signal/issues)
3. Open a new issue with your OS, Python version, full traceback, and minimal steps to reproduce

<!-- prettier-ignore-end -->
