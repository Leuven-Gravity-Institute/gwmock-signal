# gwmock-signal

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-signal/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-signal/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-signal/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-signal/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-signal/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-signal/)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-signal/graph/badge.svg?token=COF8341N60)](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-signal)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-signal)](https://pypi.org/project/gwmock-signal/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-signal)](https://pypi.org/project/gwmock-signal/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1180014442.svg)](https://doi.org/10.5281/zenodo.19202406)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

A Python package for simulating gravitational wave signals.

## Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock-signal`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Requirements

- Python 3.12-3.14 (We adopt the SPEC 0 policy for the Python support version)
- Operating System: Linux, or macOS

**Note:** The package is built and tested against **Python 3.12-3.14**
(`requires-python` is `>=3.12`). When creating a virtual environment with `uv`,
pin a supported interpreter, for example `uv venv --python 3.12`, so you do not
accidentally pick an unsupported version.

### Install from PyPI

The recommended way to install `gwmock-signal` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-signal
```

#### PyPI install scope

The PyPI wheel includes **runtime dependencies only** (see `dependencies` in
`pyproject.toml`). The published optional extras are `gwmock-signal[pycbc]` (the
PyCBC waveform backend) and `gwmock-signal[jax]` (the JAX/ripple waveform
backend and batched GPU simulation); development and documentation dependencies
remain **uv dependency groups** used when you work from a **git clone** (see
**Development installation** below).

### Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-signal.git
cd gwmock-signal
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

#### Development installation

To set up for development (tests, linters, pre-commit):

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-signal.git
cd gwmock-signal

# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev

# Install pre-commit hooks
uv run prek install
```

To build or serve the documentation site locally:

```bash
uv sync --group docs
uv run zensical serve
# or: uv run zensical build  → static site under site/
```

### Verify Installation

Check that the CLI and import path work (console script **`gwmock-signal`**,
Python package **`gwmock_signal`**):

```bash
gwmock-signal --help
gwmock-signal inject --help
```

```bash
python -c "import gwmock_signal; print(gwmock_signal.__version__)"
```

## Backends

- `lal` (default): ships with the core install via `lalsuite` and supports the
  built-in CBC pipeline without installing `pycbc`.
- `pycbc` (optional): install with `pip install 'gwmock-signal[pycbc]'` to use
  `PyCBCBackend`.
- `ripple` (optional): install with `pip install 'gwmock-signal[jax]'` to use
  the JAX/ripple backend (`RippleBackend`). Beyond the CLI/CPU path, it enables
  **batched, on-device (GPU) catalogue simulation** that assembles signals into
  fixed-duration data segments — see
  [Batched GPU simulation](https://leuven-gravity-institute.github.io/gwmock-signal/user_guide/gpu-batched-simulation/).
  Its agreement with the LAL baseline and its speed across backends/hardware are
  tracked on the
  [gwmock-benchmark](https://leuven-gravity-institute.github.io/gwmock-benchmark/)
  site
  ([Consistency](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/consistency/),
  [Performance](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/performance/)).

Python API:

```python
from gwmock_signal import CBCSimulator, LALSimulationBackend

sim = CBCSimulator("IMRPhenomD", waveform_backend=LALSimulationBackend())

# For a complete end-to-end example, see the
# Quick Start in the user guide:
# https://leuven-gravity-institute.github.io/gwmock-signal/user_guide/quick_start/
```

CLI:

```bash
gwmock-signal inject cbc --params examples/gw150914_like.json --network H1L1V1 --backend lal

# Same injection with the JAX/ripple backend (install with gwmock-signal[jax])
gwmock-signal inject cbc --params examples/gw150914_like.json --network H1L1V1 --backend ripple
```

Ready-to-run parameter files live in [`examples/`](examples/).

## Documentation

- **Site:**
  [gwmock-signal documentation](https://leuven-gravity-institute.github.io/gwmock-signal/)
- **User guide:** overview, **Command-line interface** (`inject cbc`, JSON
  parameters, `--network`), **Custom backends**, and **Examples** (library
  walkthroughs)
- **API reference:** signatures, types, and exceptions (mkdocstrings)
- **Benchmarks:** performance and ripple-vs-LAL consistency live on the
  [gwmock-benchmark](https://leuven-gravity-institute.github.io/gwmock-benchmark/)
  site — see
  [Performance](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/performance/)
  and
  [Consistency](https://leuven-gravity-institute.github.io/gwmock-benchmark/signal/consistency/).
  The underlying records are also published as a static JSON
  [Data API](https://leuven-gravity-institute.github.io/gwmock-benchmark/data-api/)
  (`.../data/v1/records.json`) for programmatic retrieval

### Public source-type backend lookup

`gwmock-signal` exposes a small public registry so downstream packages can
resolve the signal backend from a gwmock-pop `source_type` string instead of
hardcoding a concrete simulator class:

```python
from gwmock_signal import resolve_simulator_backend

backend_cls = resolve_simulator_backend("bbh")
simulator = backend_cls(waveform_model="IMRPhenomD")
```

The built-in compact-binary backend is registered under `bbh`. Future source
families keep the same downstream lookup contract by registering a new
`GWSimulator` subclass inside `gwmock-signal` with
`register_simulator_backend(source_type, backend_cls)`.

The stable backend boundary is
`GWSimulator.simulate(...) -> DetectorStrainStack`. For a minimal non-transient
backend example, see
[User guide → Custom backends](https://leuven-gravity-institute.github.io/gwmock-signal/user_guide/custom-backends/).

Build locally: `uv sync --group docs && uv run zensical serve` (or
`uv run zensical build` for static output in `site/`).

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock-signal/releases).

## Testing

From a repository checkout with **dev** dependencies installed
(`uv sync --group dev`):

```bash
uv run pytest
```

## License

This project is licensed under **GPL-3.0-or-later**. See the [LICENSE](LICENSE)
file for the full license text.

> gwmock-signal is free software: you can redistribute it and/or modify it under
> the terms of the GNU General Public License as published by the Free Software
> Foundation, either version 3 of the License, or (at your option) any later
> version.
>
> gwmock-signal is distributed in the hope that it will be useful, but WITHOUT
> ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
> FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
>
> You should have received a copy of the GNU General Public License along with
> gwmock-signal. If not, see <https://www.gnu.org/licenses/>.

### Downstream dependency note

`gwmock-signal` should be treated as a **GPL-3.0-or-later** dependency by any
downstream package that distributes it, including `gwmock`. In practice, that
means `gwmock` may depend on `gwmock-signal` only if the resulting downstream
distribution is compatible with GPL-3.0-or-later.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock-signal/issues/new/choose)
or contact the maintainers.
