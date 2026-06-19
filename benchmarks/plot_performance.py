#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""Render performance figures from the JSON records written by the benchmark runner.

Reads every ``*.json`` under ``--results-dir`` and produces one figure per metric
into ``--output-dir``. Timing metrics that have a cold and a warm variant (wall time,
core-hours, throughput) are drawn as grouped cold/warm bars so the one-time compile
cost is visible next to the steady state. Each bar is labelled with the run label,
device, and the gwmock-signal version that produced it, so figures stay
self-describing in the docs.

    uv run python benchmarks/plot_performance.py --results-dir results --output-dir docs/dev/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchutils import load_results

# Metrics with a cold/warm pair -> grouped bars: (cold_key, warm_key, axis label, filename).
_PAIRED_METRICS = (
    ("wall_seconds_cold", "wall_seconds_warm", "Wall time [s]", "performance_walltime.svg"),
    ("cpu_core_hours_cold", "cpu_core_hours_warm", "CPU core-hours", "performance_cpu_core_hours.svg"),
    ("gpu_hours_cold", "gpu_hours_warm", "GPU-hours", "performance_gpu_hours.svg"),
    ("events_per_second_cold", "events_per_second_warm", "Throughput [events/s]", "performance_throughput.svg"),
)
# Single-value metrics -> one bar each.
_SINGLE_METRICS = (
    ("compile_seconds", "One-time compile [s]", "performance_compile.svg"),
    ("peak_rss_bytes", "Peak memory [GB]", "performance_peak_memory.svg"),
    ("output_bytes", "Output data [GB]", "performance_output.svg"),
)


def _bar_label(record: dict) -> str:
    """Return a compact axis label: ``label`` plus the device it ran on.

    A GPU run is labelled by its GPU; a CPU run (``n_gpus == 0``) always by its CPU,
    regardless of any GPU that merely happens to be present on the node.
    """
    provenance = record["provenance"]
    gpu_models = provenance.get("gpu_models") or []
    device = gpu_models[0] if provenance.get("n_gpus") and gpu_models else provenance["cpu_model"]
    return f"{record['label']}\n{device}"


def _value(record: dict, metric: str) -> float:
    """Return one metric, converting bytes to GB, treating missing values as zero."""
    raw = record["metrics"].get(metric)
    if raw is None:
        return 0.0
    return raw / 1e9 if metric.endswith("_bytes") else float(raw)


def main() -> None:
    """Render one figure per metric from the benchmark records."""
    parser = argparse.ArgumentParser(description="Plot gwmock-signal performance benchmarks.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/dev/figures"))
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        raise SystemExit(f"No benchmark records found in {args.results_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    versions = sorted({record["provenance"]["gwmock_signal_version"] for record in records})
    labels = [_bar_label(record) for record in records]
    subtitle = f"gwmock-signal {', '.join(versions)}"
    positions = range(len(records))

    def _new_axes():
        figure, axes = plt.subplots(figsize=(max(6.0, 1.6 * len(records)), 4.5))
        axes.set_xticks(list(positions))
        axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axes.grid(axis="y", alpha=0.3)
        return figure, axes

    def _save(figure, filename):
        figure.tight_layout()
        figure.savefig(args.output_dir / filename)
        plt.close(figure)
        print(f"wrote {args.output_dir / filename}")

    for cold_key, warm_key, axis_label, filename in _PAIRED_METRICS:
        cold = [_value(record, cold_key) for record in records]
        warm = [_value(record, warm_key) for record in records]
        if not any(cold) and not any(warm):
            continue  # e.g. no GPU runs -> skip the GPU-hours figure
        figure, axes = _new_axes()
        width = 0.4
        axes.bar([p - width / 2 for p in positions], cold, width, label="cold (incl. compile)", color="#7fcdbb")
        axes.bar([p + width / 2 for p in positions], warm, width, label="warm (steady state)", color="#2c7fb8")
        axes.set_ylabel(axis_label)
        axes.set_title(f"{axis_label} — cold vs warm\n{subtitle}", fontsize=10)
        axes.legend(fontsize=8)
        _save(figure, filename)

    for metric, axis_label, filename in _SINGLE_METRICS:
        values = [_value(record, metric) for record in records]
        if not any(values):
            continue
        figure, axes = _new_axes()
        axes.bar(list(positions), values, color="#2c7fb8")
        axes.set_ylabel(axis_label)
        axes.set_title(f"{axis_label}\n{subtitle}", fontsize=10)
        _save(figure, filename)


if __name__ == "__main__":
    main()
