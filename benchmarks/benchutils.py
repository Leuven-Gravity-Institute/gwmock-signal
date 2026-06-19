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
"""Shared helpers for the gwmock-signal benchmark scripts.

Collects **run provenance** (release version, hardware, library versions) and
**resource usage** (wall time, peak/average resident memory, optional GPU memory)
and writes one JSON record per run, so every result stays attributable to the code
and machine that produced it. Used by ``run_performance_benchmark.py`` and
``run_consistency_benchmark.py``; the plotting scripts read the JSON records back.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

_PROVENANCE_LIBRARIES = ("gwmock-signal", "ripplegw", "jax", "jaxlib", "lalsuite", "pycbc", "gwpy", "numpy")


def _library_versions() -> dict[str, str]:
    """Return installed versions of the packages relevant to a benchmark run."""
    versions: dict[str, str] = {}
    for name in _PROVENANCE_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def _cpu_model() -> str:
    """Return the CPU model name (Linux ``/proc/cpuinfo``), or a best-effort fallback."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_models() -> list[str]:
    """Return the GPU model names reported by ``nvidia-smi`` (empty if none)."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def allocated_cpu_cores() -> int:
    """Return the number of CPU cores allocated to this job (scheduler env or all cores)."""
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "OMP_NUM_THREADS"):
        value = os.environ.get(variable, "")
        if value.isdigit():
            return int(value)
    return os.cpu_count() or 1


def provenance(*, n_cpu_cores: int | None = None, n_gpus: int | None = None) -> dict:
    """Return a record of the code version and hardware that produced a benchmark.

    Args:
        n_cpu_cores: Override the allocated CPU-core count (defaults to the scheduler
            allocation or the machine core count) — used for CPU core-hours.
        n_gpus: Override the GPU count (defaults to the number of GPUs ``nvidia-smi``
            reports) — used for GPU-hours.
    """
    gpu_models = _gpu_models()
    return {
        "gwmock_signal_version": _library_versions().get("gwmock-signal", "unknown"),
        "library_versions": _library_versions(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "gpu_models": gpu_models,
        "n_cpu_cores": n_cpu_cores if n_cpu_cores is not None else allocated_cpu_cores(),
        "n_gpus": n_gpus if n_gpus is not None else len(gpu_models),
    }


def _peak_rss_bytes() -> int:
    """Return peak resident set size in bytes (ru_maxrss is KiB on Linux, bytes on macOS)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss * 1024 if platform.system() == "Linux" else maxrss


def _current_rss_bytes() -> int:
    """Return the current resident set size in bytes (Linux ``/proc/self/statm``)."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return resident_pages * resource.getpagesize()


def _gpu_peak_bytes() -> int | None:
    """Return peak GPU bytes-in-use from JAX, or ``None`` if unavailable (e.g. on CPU)."""
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


@dataclass
class ResourceUsage:
    """Wall time and memory measured around a workload."""

    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    average_rss_bytes: float = 0.0
    gpu_peak_bytes: int | None = None
    output_bytes: int = field(default=0)


@contextmanager
def measure(sample_interval_seconds: float = 0.1) -> Iterator[ResourceUsage]:
    """Measure wall time and resident memory around a ``with`` block.

    Average memory is sampled in a background thread; peak memory comes from
    ``getrusage`` and (if a GPU is present) JAX. The yielded :class:`ResourceUsage`
    is filled in on exit; the caller may set ``output_bytes`` inside the block.
    """
    usage = ResourceUsage()
    samples: list[int] = [_current_rss_bytes()]
    stop = threading.Event()

    def _sample() -> None:
        while not stop.wait(sample_interval_seconds):
            samples.append(_current_rss_bytes())

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        yield usage
    finally:
        usage.wall_seconds = time.perf_counter() - start
        stop.set()
        sampler.join()
        usage.peak_rss_bytes = _peak_rss_bytes()
        usage.average_rss_bytes = sum(samples) / len(samples)
        usage.gpu_peak_bytes = _gpu_peak_bytes()


def build_catalogue(n_events: int, seed: int = 0, *, gps_start: float = 1_126_259_462.0, span: float = 3.0e7) -> dict:
    """Return a synthetic struct-of-arrays catalogue of ``n_events`` aligned-spin BBHs."""
    rng = np.random.default_rng(seed)
    return {
        "detector_frame_mass_1": rng.uniform(25.0, 50.0, n_events),
        "detector_frame_mass_2": rng.uniform(20.0, 45.0, n_events),
        "luminosity_distance": rng.uniform(200.0, 1500.0, n_events),
        "spin_1z": rng.uniform(-0.5, 0.5, n_events),
        "spin_2z": rng.uniform(-0.5, 0.5, n_events),
        "inclination": rng.uniform(0.0, np.pi, n_events),
        "coa_phase": rng.uniform(0.0, 2.0 * np.pi, n_events),
        "right_ascension": rng.uniform(0.0, 2.0 * np.pi, n_events),
        "declination": rng.uniform(-0.5 * np.pi, 0.5 * np.pi, n_events),
        "polarization_angle": rng.uniform(0.0, np.pi, n_events),
        "coa_time": gps_start + rng.uniform(0.0, span, n_events),
    }


def write_result(path: str | Path, record: dict) -> None:
    """Write one benchmark record as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))


def load_results(directory: str | Path) -> list[dict]:
    """Load every ``*.json`` benchmark record under ``directory`` (sorted by name)."""
    return [json.loads(path.read_text()) for path in sorted(Path(directory).glob("*.json"))]
