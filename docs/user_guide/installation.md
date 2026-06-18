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

- **Python:** 3.12 or 3.13 (`requires-python` in `pyproject.toml` is
  `>=3.12,<3.14`).
- **Operating system:** Linux or macOS (same range as the published wheels and
  CI).

<!-- prettier-ignore-start -->

!!! note "Python version pin"

    When creating a virtual environment with `uv`, pass an explicit interpreter
    (for example `uv venv --python 3.12`) so you do not pick an unsupported
    Python from your machine default.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install the library for downstream use is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-signal
```

The wheel includes **runtime dependencies** declared in `pyproject.toml`:
**`typer`**, **`gwpy`**, **`lalsuite`**, **`pyyaml`**, and their transitive
installs. **PyCBC** and **ripple** are **not** core dependencies; install them
with the optional extras **`gwmock-signal[pycbc]`** (PyCBC waveform backend or
the `pycbc_waveform_wrapper` helper) and **`gwmock-signal[jax]`** (the
JAX/ripple waveform backend, which pulls in `rippleGW` and JAX).

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
uv venv --python 3.12
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

uv venv --python 3.12
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

- **typer** — CLI (`gwmock-signal` entry point)
- **gwpy** — `TimeSeries` / GW I/O conventions
- **lalsuite** — default time-domain CBC waveforms (LALSimulation) and related
  LAL types
- **pyyaml** — configuration parsing where used

**Optional** (`pip install 'gwmock-signal[pycbc]'`):

- **pycbc** — alternate waveform backend (`PyCBCBackend`) and
  `pycbc_waveform_wrapper`

**Optional** (`pip install 'gwmock-signal[jax]'`):

- **rippleGW** (and JAX) — JAX/ripple waveform backend (`RippleBackend`); CPU
  JAX by default

Numerical arrays (`numpy`, etc.) are pulled in transitively (for example by
`gwpy` and `lalsuite`).

## Getting help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock-signal/issues)
3. Open a new issue with your OS, Python version, full traceback, and minimal steps to reproduce

<!-- prettier-ignore-end -->
