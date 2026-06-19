---
title: Consistency
description:
    Agreement between the ripple (JAX) waveform backend and the LAL baseline
    across supported approximants.
---

# Consistency

The [ripple](../user_guide/waveform.md) (JAX) backend is an alternative
implementation of the same waveform models LAL provides. This page tracks their
agreement so the JAX/GPU path can be trusted against the LAL baseline.

## What is measured

For every approximant that LAL implements in the time domain, the white,
time/phase-maximized **match** between the ripple and LAL backends is computed
for both polarizations across several parameter sets; the **worst-case** and
**median** match per approximant are recorded (with the released version that
produced them).

`TaylorF2` is covered separately by the test suite — LAL provides no time-domain
TaylorF2, so it is anchored against LAL's _frequency-domain_ TaylorF2.

## Reproducing

```bash
uv run --extra jax python benchmarks/run_consistency_benchmark.py \
    --output-json benchmarks/results/consistency.json

uv run python benchmarks/plot_consistency.py \
    --results-json benchmarks/results/consistency.json
```

## Results

!!! note "Populated per release"

    The figure and table below are generated from a run on a **released** version
    (annotated with that version) and refreshed per release. In development the
    backends agree to a worst-case match better than 0.999 across all approximants.

<!-- The release benchmark run adds the figure and table here, e.g.:
![ripple vs LAL match](figures/consistency_matches.svg)

{% include "dev/figures/consistency_table.md" %}
-->
