---
title: API Reference
description: Auto-generated reference for gwmock-signal public Python APIs.
icon: material/api
---

# API reference

This section documents the **public Python API** of **gwmock-signal**. Pages are
generated with **mkdocstrings** from **Google-style docstrings** in the source
tree (`src/gwmock_signal/`).

**How to use this site**

- For **workflow overview and example snippets**, use
  **[User guide → Examples](../user_guide/index.md)** (sidebar).
- For **exact signatures, types, and raised exceptions**, use the API sections
  below.
- For **`gwmock-signal` CLI** flags and JSON formats, use
  **[User guide → Command-line interface](../user_guide/cli.md)**.

## Sections

| Section                           | Contents                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **[Waveform](waveform/)**         | `WaveformFactory`, `WaveformBackend`, `LALSimulationBackend`, `PyCBCBackend`, `pycbc_waveform_wrapper` |
| **[Projection](projection/)**     | `project_polarizations_to_network`                                                                     |
| **[Injection](injection/)**       | `inject_strain`, `inject_strains_sequential`                                                           |
| **[Pipeline](pipeline/)**         | `inject_cbc_signal` (CBC orchestration)                                                                |
| **[Stochastic](stochastic/)**     | `StochasticBackgroundSimulator`, power-law SGWB spectra, ORF helpers                                   |
| **[Multichannel](multichannel/)** | `DetectorStrainStack`                                                                                  |
| **[Network](network/)**           | `Network` (presets, `from_file`, detector lists)                                                       |
| **[Simulator](simulator/)**       | `GWSimulator`, `TransientSimulator`, `CBCSimulator`, stable `DetectorStrainStack` return contract      |
| **[Registry](registry/)**         | `resolve_simulator_backend`, `register_simulator_backend`, `list_registered_source_types`              |
| **[Utility](utils/)**             | Logging and other helpers                                                                              |

## Main entry points (quick links)

- **[Waveform](waveform/)** — Time-domain polarizations: `WaveformFactory` over
  `LALSimulationBackend` (default) or `PyCBCBackend` (optional); direct
  `pycbc_waveform_wrapper` for PyCBC-only workflows.
- **[Projection](projection/)** — Strain per detector
  (`project_polarizations_to_network`).
- **[Injection](injection/)** — Strain into segments (`inject_strain`,
  `inject_strains_sequential`).
- **[Pipeline](pipeline/)** — One-call CBC injection (`inject_cbc_signal`).
- **[Stochastic](stochastic/)** — SGWB signal generation as correlated
  multi-detector Gaussian strain.
- **[Multichannel](multichannel/)** — Stacked IFO strains
  (`DetectorStrainStack`).
- **[Network](network/)** — Named networks and YAML/JSON network configs.
- **[Simulator](simulator/)** — Simulator base class, CBC implementation, and
  the stable `DetectorStrainStack` return contract.
- **[Registry](registry/)** — Lookup `GWSimulator` subclasses by `source_type`
  string (for example `bbh`).
- **[Utility](utils/)** — Utility functions.

## Top-level package exports

`import gwmock_signal` re-exports the following symbols via
`gwmock_signal.__init__`. Prefer these imports in application code; submodule
paths are stable but longer.

Resolution is lazy: each symbol imports its own implementation module on first
access, so `import gwmock_signal` does not import JAX.

The **Batched** symbols come from
[`gwmock_signal.jax_batch`](../user_guide/gpu-batched-simulation.md) and need
the `[jax]` extra to _run_, as does `RippleBackend`; without it, generating a
waveform raises an `ImportError` naming the install command. `SamplingGrid` is
pure NumPy — it is the lattice contract the batched path aligns to, and it is
usable on its own without the extra.

| Symbol                          | Category     | Reference                         |
| ------------------------------- | ------------ | --------------------------------- |
| `CBCSimulator`                  | Simulator    | [Simulator API](simulator/)       |
| `StochasticBackgroundSimulator` | Simulator    | [Stochastic API](stochastic/)     |
| `StochasticBackgroundSpectrum`  | Spectrum     | [Stochastic API](stochastic/)     |
| `TransientSimulator`            | Simulator    | [Simulator API](simulator/)       |
| `GWSimulator`                   | Simulator    | [Simulator API](simulator/)       |
| `DetectorStrainStack`           | Multichannel | [Multichannel API](multichannel/) |
| `Network`                       | Network      | [Network API](network/)           |
| `CustomDetector`                | Detector     | [Network API](network/)           |
| `WaveformBackend`               | Waveform     | [Waveform API](waveform/)         |
| `LALSimulationBackend`          | Waveform     | [Waveform API](waveform/)         |
| `RippleBackend`                 | Waveform     | [Waveform API](waveform/)         |
| `simulate_cbc_catalogue`        | Batched      | [Waveform API](waveform/)         |
| `simulate_cbc_batch`            | Batched      | [Waveform API](waveform/)         |
| `assemble_segments`             | Batched      | [Waveform API](waveform/)         |
| `BatchedDetectorStrain`         | Batched      | [Waveform API](waveform/)         |
| `SamplingGrid`                  | Grid         | [Waveform API](waveform/)         |
| `resolve_simulator_backend`     | Registry     | [Registry API](registry/)         |
| `register_simulator_backend`    | Registry     | [Registry API](registry/)         |
| `list_registered_source_types`  | Registry     | [Registry API](registry/)         |
| `__version__`                   | Version      | _(string)_                        |

## See also

- [User guide overview](../user_guide/index.md)
- [Documentation home](../index.md)
