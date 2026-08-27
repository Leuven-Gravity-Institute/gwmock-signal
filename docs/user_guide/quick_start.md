---
title: Quick Start
description: Orient yourself and run a minimal import check for gwmock-signal.
icon: material/rocket-launch
---

# Quick Start

Welcome to the **gwmock-signal** documentation.

## Before you begin

1. Follow **[Installation](installation.md)** to create an environment and
   install the package.
2. Read the **[User guide overview](index.md)** for the recommended workflow
   (waveforms → projection → injection → multichannel).

## Verify the install

```bash
gwmock-signal --help
gwmock-signal inject --help
```

```bash
python -c "import gwmock_signal; print(gwmock_signal.__version__)"
python -c "from gwmock_signal import list_registered_source_types; print(list_registered_source_types())"
```

You can also run the package as a module to print version information:

```bash
python -m gwmock_signal
```

## Configuring logging

The package uses Python's standard `logging` module under the `"gwmock_signal"`
logger. In the CLI, use `--verbose DEBUG` to increase log level. From Python:

```python
from gwmock_signal.utils import setup_logger, get_version_information

# Set up console logging at INFO level
setup_logger(log_level="INFO", print_version=True)

# Or direct version query
print(get_version_information())
```

`setup_logger` accepts `outdir` and `label` arguments to write a log file
alongside console output.

## End-to-end demo

Once installed, run a quick CBC injection with the bundled example parameters.
This uses the GW150914-like parameters shipped with the package under
`examples/gw150914_like.json`:

```bash
gwmock-signal inject cbc \
    --params examples/gw150914_like.json \
    --network H1L1
```

This prints the per-detector RMS strain to stdout. Add `--output injected.h5` to
write the result to an HDF5 file instead.

The same workflow in Python:

```python
import json, numpy as np
from pathlib import Path
from gwpy.timeseries import TimeSeries
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.network import Network

# Load bundled example parameters
params = json.loads(Path("examples/gw150914_like.json").read_text())

# Resolve a two-detector network
net = Network.from_name("H1L1")

# Build a zero-noise background centred on coa_time
fs = 4096.0
duration = 8.0  # seconds
n = int(duration * fs)
t0 = params["coa_time"] - duration / 2
background = {name: TimeSeries(np.zeros(n), t0=t0, sample_rate=fs) for name in net.detector_names}

# Run the injection pipeline
result = inject_cbc_signal(
    "IMRPhenomD",
    params,
    net.detector_names,
    background,
    sampling_frequency=fs,
    minimum_frequency=20.0,
)

# Print RMS per detector
for name in result.detector_names:
    rms = float(np.sqrt(np.mean(result[name].value ** 2)))
    print(f"{name}: rms={rms:.4e}")
```

The repository also ships additional example files:

- `examples/gw150914_like.json` — GW150914-like CBC parameters
- `examples/networks/hlvk.yaml` — 4-detector LIGO-Virgo-KAGRA network
- `examples/networks/et_triangle.yaml` — Einstein Telescope triangle network

## Next steps

- Read **[Command-line interface](cli.md)** if you plan to use
  `gwmock-signal inject`.
- Work through **User guide → Examples** (start with [Waveforms](waveform.md)).
- Browse the **[API overview](../api/index.md)** for full function and class
  reference.
