# gwmock-signal benchmarks

Reproducible scripts for the **performance** and **ripple-vs-LAL consistency**
results shown in the documentation
([Performance](https://leuven-gravity-institute.github.io/gwmock-signal/dev/performance/),
[Consistency](https://leuven-gravity-institute.github.io/gwmock-signal/dev/consistency/)).

Every result records its **provenance** — the released `gwmock-signal` version,
the CPU/GPU model, and the library versions — so figures stay attributable. Run
these on a released version (`uv pip install gwmock-signal[jax]`), not a dirty
checkout.

## Scripts

| Script                         | Purpose                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `run_performance_benchmark.py` | One backend/method/hardware run → JSON record (wall time, CPU/GPU core-hours, peak & average memory, output-data size, provenance). |
| `run_consistency_benchmark.py` | ripple-vs-LAL white match per approximant → JSON record.                                                                            |
| `plot_performance.py`          | Aggregate performance JSON records → figures in `docs/dev/figures/`.                                                                |
| `plot_consistency.py`          | Consistency JSON → figure + Markdown table in `docs/dev/figures/`.                                                                  |
| `benchutils.py`                | Shared provenance + resource-measurement helpers.                                                                                   |

`benchmarks/results/` (raw JSON) and `benchmarks/submit/` (cluster submission
scripts) are git-ignored; the generated figures under `docs/dev/figures/` are
committed with the documentation.

## The matrix

One `run_performance_benchmark.py` invocation = one cell. A typical matrix:

- **CPU node** — `lal`, `pycbc`, `ripple` per-event; `ripple` batched.
- **GPU node** — `ripple` batched (the headline), optionally with `--chunk-size`
  / `--n-chirp-mass-bins`.

`--method batched` is ripple-only. Pass `--n-cpu-cores` / `--n-gpus` to record
the _allocated_ resources for core-hours when the scheduler env does not expose
them.

## Run locally

```bash
uv run --extra jax python benchmarks/run_performance_benchmark.py \
    --backend ripple --method batched --n-events 2000 \
    --output-json benchmarks/results/ripple_batched_cpu.json --label "ripple batched (CPU)"

uv run --extra jax python benchmarks/run_consistency_benchmark.py \
    --output-json benchmarks/results/consistency.json

uv run python benchmarks/plot_performance.py --results-dir benchmarks/results
uv run python benchmarks/plot_consistency.py --results-json benchmarks/results/consistency.json
```

## Run on the clusters

Submission templates (SLURM for `kuleuven-fys`, HTCondor for `stadius-nc-5`)
live in `benchmarks/submit/` (git-ignored — copy and adapt the
account/partition/paths). After the jobs finish, copy the JSON records into
`benchmarks/results/`, run the plotting scripts, and commit the figures with the
version recorded in each record.
