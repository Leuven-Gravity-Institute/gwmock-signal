---
title: Custom backends
description:
    How to implement and register a custom GWSimulator backend that returns a
    DetectorStrainStack.
---

# Custom backends

Downstream packages should treat `GWSimulator.simulate(...)` as the stable
extension point:

```python
def simulate(
    self,
    params,
    detector_names,
    background=None,
    *,
    sampling_frequency,
    minimum_frequency,
    earth_rotation=False,
    interpolate_if_offset=True,
) -> DetectorStrainStack: ...
```

The important part of the contract is the **return type**:
[`DetectorStrainStack`](../api/multichannel/) is the stable multi-detector
container that downstream orchestration should consume.

## When to implement `GWSimulator` directly

Use a direct `GWSimulator` subclass when your source family does **not** fit the
transient "generate polarizations, then project" pipeline. Typical examples are
stochastic backgrounds, glitch models, burst populations, or any backend that
already produces detector-frame strain.

In that case:

- `background` is available when you need to inject into existing strain.
- `background=None` is valid when your backend can produce detector strain from
  scratch.
- You should return a `DetectorStrainStack` directly instead of depending on
  `TransientSimulator.generate_polarizations`.

## Minimal example

```python
from collections.abc import Mapping, Sequence

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal import DetectorStrainStack, GWSimulator


class ConstantBurstSimulator(GWSimulator):
    @property
    def required_params(self) -> frozenset[str]:
        return frozenset({"amplitude"})

    def simulate(
        self,
        params: Mapping[str, float],
        detector_names: Sequence[str],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = False,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        del minimum_frequency, earth_rotation, interpolate_if_offset

        self._validate_params(params)
        amplitude = float(params["amplitude"])

        if background is None:
            strains = {
                name: TimeSeries(np.full(8, amplitude), t0=0.0, sample_rate=sampling_frequency)
                for name in detector_names
            }
        else:
            strains = {
                name: TimeSeries(
                    np.asarray(background[name].value, dtype=float) + amplitude,
                    t0=float(background[name].t0.value),
                    sample_rate=float(background[name].sample_rate.value),
                )
                for name in detector_names
            }

        return DetectorStrainStack.from_mapping(detector_names, strains)
```

## When to use `TransientSimulator`

Use [`TransientSimulator`](../api/simulator/) only when your backend naturally
fits the existing waveform-to-projection pipeline:

1. generate plus/cross polarizations,
2. project them onto detectors,
3. optionally inject them into a background.

That helper remains convenient for CBC-like sources, but it is not the required
entry point for every source family.

### Registering custom waveform models on a simulator instance

`TransientSimulator` (and by inheritance, `CBCSimulator`) exposes a public
method `register_waveform_model` so each simulator instance can register custom
waveform generators (e.g., numerical relativity hybrids, ROM surrogates, or
analytic burst models) without touching internal `WaveformFactory` details:

```python
from gwpy.timeseries import TimeSeries
from gwmock_signal import CBCSimulator


def my_callback_burst(*, waveform_model, tc, sampling_frequency, **kw):
    """Return (plus, cross) GWpy TimeSeries tuple."""
    import numpy as np

    dt = 1.0 / sampling_frequency
    n = int(kw.get("width", 0.1) * sampling_frequency)
    t = np.arange(n) * dt + tc
    hp = TimeSeries(np.sin(2 * np.pi * 150 * t), t0=tc, dt=dt)
    hc = TimeSeries(np.cos(2 * np.pi * 150 * t), t0=tc, dt=dt)
    return hp, hc


sim = CBCSimulator("IMRPhenomD")
sim.register_waveform_model("my_burst", my_callback_burst)
# Now sim.generate_polarizations(params, ...) can resolve "my_burst"
```

The registered callable must return either a `(plus, cross)` tuple or a
`{"plus": ..., "cross": ...}` mapping. Re-registering the same name with a
different callable raises `ValueError`.

## Registration by `source_type`

The `source_type` registry lets downstream packages resolve a `GWSimulator`
subclass from a simple string key, decoupling orchestration from concrete
backend classes. This is the primary integration point for gwmock-pop and
similar multi-source simulation frameworks.

If a downstream package wants to resolve your backend from a gwmock-pop
`source_type`, register it once:

```python
from gwmock_signal import register_simulator_backend

register_simulator_backend("burst", ConstantBurstSimulator)
```

Later, orchestration can look it up without hard-coding the class:

```python
from gwmock_signal import resolve_simulator_backend

backend_cls = resolve_simulator_backend("burst")
simulator = backend_cls()
```

The built-in `bbh` entry is pre-registered to `CBCSimulator`. Downstream code
should prefer `resolve_simulator_backend(source_type)` over importing
`CBCSimulator` directly, so that the orchestration layer stays source-agnostic.

## Projection note

`gwmock_signal.projection` remains available as a module for internal
transient-style workflows, but custom non-transient backends should generally
return their own `DetectorStrainStack` instead of depending on projection
helpers from the top-level package contract.

!!! note "Canonical module paths" The imports
`from gwmock_signal import GWSimulator, TransientSimulator` work because the
top-level package re-exports them. The canonical source locations are
`gwmock_signal.simulator` for these classes and
`gwmock_signal.multichannel.stack` for `DetectorStrainStack`. Either import
style is supported.

## See also

- [Simulator API](../api/simulator/)
- [Registry API](../api/registry/)
- [Multichannel API](../api/multichannel/)
- [User guide overview](index.md)
- [Waveforms and backends](waveform.md)
- [Custom detectors](custom-detectors.md)
- [Network configuration files](network-config.md)
