---
title: Performance
description:
    Benchmarks of gwmock-signal CBC catalogue generation across backends,
    methods, and hardware (wall time, core-hours, memory, output-data size).
---

# Performance

These benchmarks compare the cost of generating a CBC catalogue data product
across **backends** (`lal`, `pycbc`, `ripple`), **methods** (per-event vs the
batched on-device path, see
[Batched GPU simulation](../user_guide/gpu-batched-simulation.md)), and
**hardware** (CPU and GPU nodes). They are produced by the scripts in
[`benchmarks/`](https://github.com/Leuven-Gravity-Institute/gwmock-signal/tree/main/benchmarks)
and are reproducible.

## Metrics

Each run produces the catalogue **twice** — a **cold** run that pays one-time
JIT/XLA compilation and a **warm** steady-state run — and records, with full
provenance (the released `gwmock-signal` version, the CPU/GPU model, and library
versions):

- **Wall time (cold / warm)** — end-to-end seconds, before and after
  compilation.
- **Compile time** — `cold − warm`, the one-time cost a production-scale
  catalogue amortizes away. (The eager per-event path has no JIT, so cold ≈
  warm.)
- **CPU core-hours / GPU-hours (cold / warm)** — wall time × allocated CPU cores
  (or GPUs).
- **Peak / average memory** — peak resident set across both runs, average over
  the warm run (and peak GPU memory where present).
- **Output-data size** — bytes of the produced data segments.
- **Throughput (cold / warm)** — events per second.

The headline comparison is the **warm** throughput: at catalogue scale the
compile cost vanishes, so steady state is what a year-long run actually sees.
The cold bars are kept beside it because the GPU's compile is larger than the
CPU's, which can mask the device's advantage at small event counts.

## Reproducing

Run one configuration per invocation, then aggregate:

```bash
uv run --extra jax python benchmarks/run_performance_benchmark.py \
    --backend ripple --method batched --n-events 5000 \
    --output-json benchmarks/results/ripple_gpu.json --label "ripple batched (GPU)"

uv run python benchmarks/plot_performance.py --results-dir benchmarks/results
```

See
[`benchmarks/README.md`](https://github.com/Leuven-Gravity-Institute/gwmock-signal/tree/main/benchmarks)
for the full matrix and the cluster submission templates.

## Results

!!! note "Populated per release"

    The figures below are generated from runs on **released** versions (each figure
    is annotated with the `gwmock-signal` version and the CPU/GPU model that produced
    it). They are added/refreshed after a release benchmark campaign.

<!-- The release benchmark run adds the figures here, e.g.:
![Wall time](figures/performance_walltime.svg)
![CPU core-hours](figures/performance_cpu_core_hours.svg)
![GPU-hours](figures/performance_gpu_hours.svg)
![Peak memory](figures/performance_peak_memory.svg)
![Throughput](figures/performance_throughput.svg)
-->
