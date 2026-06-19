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
"""Render the ripple-vs-LAL consistency figure and table from the consistency record.

uv run python benchmarks/plot_consistency.py --results-json results/consistency.json \
        --output-dir docs/dev/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THRESHOLD = 0.99


def main() -> None:
    """Render the consistency figure (worst match per approximant) and a Markdown table."""
    parser = argparse.ArgumentParser(description="Plot ripple-vs-LAL consistency.")
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/dev/figures"))
    args = parser.parse_args()

    record = json.loads(args.results_json.read_text())
    approximants = record["approximants"]
    version = record["provenance"]["gwmock_signal_version"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = list(approximants)
    min_matches = [approximants[name]["min_match"] for name in names]

    figure, axes = plt.subplots(figsize=(max(6.0, 1.1 * len(names)), 4.5))
    axes.bar(range(len(names)), min_matches, color="#41ab5d")
    axes.axhline(_THRESHOLD, color="#d7301f", linestyle="--", label=f"threshold {_THRESHOLD}")
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes.set_ylabel("Worst-case match vs LAL")
    axes.set_ylim(min(0.98, min(min_matches) - 0.005), 1.0005)
    axes.set_title(f"ripple vs LAL time-domain agreement\ngwmock-signal {version}", fontsize=10)
    axes.legend(fontsize=8)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "consistency_matches.svg")
    plt.close(figure)
    print(f"wrote {args.output_dir / 'consistency_matches.svg'}")

    table = [
        "| Approximant | f_min [Hz] | worst match | median match |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        entry = approximants[name]
        table.append(
            f"| `{name}` | {entry['minimum_frequency']:.0f} | {entry['min_match']:.5f} | {entry['median_match']:.5f} |"
        )
    table_path = args.output_dir / "consistency_table.md"
    table_path.write_text("\n".join(table) + "\n")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
