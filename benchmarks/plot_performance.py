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
(wall time, core-hours, peak memory, throughput) into ``--output-dir``. Each bar is
labelled with the run label, device, and the gwmock-signal version that produced it,
so figures stay self-describing in the docs.

    uv run python benchmarks/plot_performance.py --results-dir results --output-dir docs/dev/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchutils import load_results

_METRICS = (
    ("wall_seconds", "Wall time [s]", "performance_walltime.svg"),
    ("cpu_core_hours", "CPU core-hours", "performance_cpu_core_hours.svg"),
    ("gpu_hours", "GPU-hours", "performance_gpu_hours.svg"),
    ("peak_rss_bytes", "Peak memory [GB]", "performance_peak_memory.svg"),
    ("events_per_second", "Throughput [events/s]", "performance_throughput.svg"),
)


def _bar_label(record: dict) -> str:
    """Return a compact axis label: ``label`` plus the device it ran on."""
    provenance = record["provenance"]
    device = (provenance["gpu_models"] or [provenance["cpu_model"]])[0]
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

    for metric, axis_label, filename in _METRICS:
        values = [_value(record, metric) for record in records]
        if not any(values):
            continue  # e.g. no GPU runs -> skip the GPU-hours figure
        figure, axes = plt.subplots(figsize=(max(6.0, 1.6 * len(records)), 4.5))
        axes.bar(range(len(records)), values, color="#2c7fb8")
        axes.set_xticks(range(len(records)))
        axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axes.set_ylabel(axis_label)
        axes.set_title(f"{axis_label}\n{subtitle}", fontsize=10)
        axes.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output_dir / filename)
        plt.close(figure)
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
