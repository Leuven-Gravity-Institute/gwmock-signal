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
"""Benchmark producing a CBC catalogue data product for one backend/method/hardware.

Each invocation runs **one** configuration (backend x method) on a synthetic
catalogue and writes a single JSON record. The workload is run twice: a **cold**
run that pays one-time JIT/XLA compilation and a **warm** steady-state run; the
record reports both wall times plus the ``compile_seconds`` difference (which a
production-scale catalogue amortizes away), alongside CPU/GPU core-hours,
peak/average memory, output-data size, and full run provenance (release version,
CPU/GPU model, library versions). A submission script launches one invocation per
point in the backend x method x hardware matrix; the plotting script aggregates the
records.

Examples::

    # CPU baseline, per-event LAL
    uv run --extra pycbc python benchmarks/run_performance_benchmark.py \
        --backend lal --method per-event --n-events 200 --output-json results/lal.json

    # GPU, batched ripple
    uv run --extra jax python benchmarks/run_performance_benchmark.py \
        --backend ripple --method batched --n-events 5000 --output-json results/ripple_gpu.json
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path

import numpy as np
from benchutils import build_catalogue, measure, provenance, write_result
from gwpy.timeseries import TimeSeries

_INTRINSIC_KEYS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
    "spin_1z",
    "spin_2z",
    "inclination",
    "coa_phase",
)
_BYTES_PER_SAMPLE = 8  # float64 strain


def _waveform_backend(name: str):
    """Return a fresh waveform backend for ``name`` ('lal', 'pycbc', or 'ripple')."""
    from gwmock_signal.waveform.backends import LALSimulationBackend, PyCBCBackend, RippleBackend

    return {"lal": LALSimulationBackend, "pycbc": PyCBCBackend, "ripple": RippleBackend}[name]()


def _segment_start_times(start_time: float, end_time: float, segment_duration: float) -> np.ndarray:
    """Return contiguous segment start times tiling ``[start_time, end_time)``."""
    n_segments = int(np.ceil((end_time - start_time) / segment_duration))
    return start_time + np.arange(n_segments) * segment_duration


def _per_event_catalogue(  # noqa: PLR0913
    backend,
    approximant,
    detector_names,
    catalogue,
    *,
    sampling_frequency,
    minimum_frequency,
    segment_duration,
    start_time,
    end_time,
):
    """Build the segmented data product one event at a time (the per-event CPU path)."""
    from gwmock_signal.injection import inject_strains_sequential
    from gwmock_signal.multichannel.stack import DetectorStrainStack
    from gwmock_signal.projection.network import project_polarizations_to_network

    starts = _segment_start_times(start_time, end_time, segment_duration)
    n_segment_samples = round(segment_duration * sampling_frequency)
    n_events = len(catalogue["coa_time"])

    # One zero background per (segment, detector); signals are injected on top.
    channels = [
        {
            name: TimeSeries(np.zeros(n_segment_samples), t0=float(start), sample_rate=sampling_frequency)
            for name in detector_names
        }
        for start in starts
    ]
    for i in range(n_events):
        polarizations = backend.generate_td_waveform(
            approximant,
            tc=float(catalogue["coa_time"][i]),
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            **{key: float(catalogue[key][i]) for key in _INTRINSIC_KEYS},
        )
        projected = project_polarizations_to_network(
            {"plus": polarizations["plus"], "cross": polarizations["cross"]},
            detector_names,
            right_ascension=float(catalogue["right_ascension"][i]),
            declination=float(catalogue["declination"][i]),
            polarization_angle=float(catalogue["polarization_angle"][i]),
            earth_rotation=False,
        )
        for name in detector_names:
            signal = projected[name]
            signal_start, signal_end = signal.t0.value, signal.t0.value + signal.duration.value
            for k, start in enumerate(starts):
                if signal_start < start + segment_duration and signal_end > start:
                    channels[k][name] = inject_strains_sequential(channels[k][name], [signal])
    return [DetectorStrainStack.from_mapping(detector_names, channel) for channel in channels]


def _batched_catalogue(  # noqa: PLR0913
    approximant,
    detector_names,
    catalogue,
    *,
    sampling_frequency,
    minimum_frequency,
    segment_duration,
    start_time,
    end_time,
    chunk_size,
    n_chirp_mass_bins,
):
    """Build the segmented data product with the on-device batched path."""
    from gwmock_signal.jax_batch import simulate_cbc_catalogue

    return simulate_cbc_catalogue(
        approximant,
        detector_names,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        parameters=catalogue,
        segment_duration=segment_duration,
        start_time=start_time,
        end_time=end_time,
        chunk_size=chunk_size,
        n_chirp_mass_bins=n_chirp_mass_bins,
    )


def _output_bytes(segments, *, write_dir: Path | None) -> int:
    """Return the data-product size: on-disk bytes if ``write_dir`` is set, else in-memory."""
    if write_dir is None:
        return sum(stack.data.size * _BYTES_PER_SAMPLE for stack in segments)
    total = 0
    for index, stack in enumerate(segments):
        path = write_dir / f"segment_{index:06d}.hdf5"
        stack.write(path, format="hdf5")
        total += path.stat().st_size
    return total


def main() -> None:  # noqa: PLR0915
    """Run one configuration and write its benchmark record."""
    parser = argparse.ArgumentParser(description="Benchmark CBC catalogue generation for one configuration.")
    parser.add_argument("--backend", choices=("lal", "pycbc", "ripple"), required=True)
    parser.add_argument("--method", choices=("per-event", "batched"), required=True)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"])
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--sampling-frequency", type=float, default=4096.0)
    parser.add_argument("--minimum-frequency", type=float, default=20.0)
    parser.add_argument("--segment-duration", type=float, default=64.0)
    parser.add_argument("--start-time", type=float, default=1_126_259_462.0)
    parser.add_argument(
        "--end-time",
        type=float,
        default=1_126_259_462.0 + 8192.0,
        help="Span [start, end) is tiled with fixed segments and held in memory; the "
        "default is ~128 x 64 s segments. A full year of segments is TBs — keep the "
        "span bounded (or raise --max-product-gb deliberately).",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Batched method only.")
    parser.add_argument("--n-chirp-mass-bins", type=int, default=1, help="Batched method only.")
    parser.add_argument("--n-cpu-cores", type=int, default=None, help="Override for CPU core-hours.")
    parser.add_argument("--n-gpus", type=int, default=None, help="Override for GPU-hours.")
    parser.add_argument("--label", default=None, help="Human-readable label for this run.")
    parser.add_argument("--write-data", action="store_true", help="Write segments to measure on-disk size.")
    parser.add_argument(
        "--max-product-gb",
        type=float,
        default=8.0,
        help="Refuse to run if the in-memory data product would exceed this (guards against OOM).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.method == "batched" and args.backend != "ripple":
        parser.error("the batched method is only available for the ripple backend")

    # The span is tiled with fixed-duration segments and the whole product is held in
    # memory; refuse absurd spans up front instead of OOMing the node mid-run.
    n_segments = int(np.ceil((args.end_time - args.start_time) / args.segment_duration))
    n_segment_samples = round(args.segment_duration * args.sampling_frequency)
    product_gb = n_segments * len(args.detectors) * n_segment_samples * _BYTES_PER_SAMPLE / 1e9
    if product_gb > args.max_product_gb:
        parser.error(
            f"data product is ~{product_gb:.1f} GB ({n_segments} segments x {len(args.detectors)} detectors), "
            f"over --max-product-gb={args.max_product_gb}. Shorten the span (--end-time) or raise the cap."
        )

    catalogue = build_catalogue(args.n_events, gps_start=args.start_time)
    catalogue["coa_time"] = np.clip(catalogue["coa_time"], args.start_time, args.end_time)
    run = {
        "sampling_frequency": args.sampling_frequency,
        "minimum_frequency": args.minimum_frequency,
        "segment_duration": args.segment_duration,
        "start_time": args.start_time,
        "end_time": args.end_time,
    }

    def workload():
        if args.method == "batched":
            return _batched_catalogue(
                args.approximant,
                args.detectors,
                catalogue,
                chunk_size=args.chunk_size,
                n_chirp_mass_bins=args.n_chirp_mass_bins,
                **run,
            )
        return _per_event_catalogue(_waveform_backend(args.backend), args.approximant, args.detectors, catalogue, **run)

    def run_once(write_dir):
        """Run the workload once under resource measurement and return the usage."""
        with measure() as usage:
            segments = workload()
            usage.output_bytes = _output_bytes(segments, write_dir=write_dir)
        return usage

    with tempfile.TemporaryDirectory() as tmp:
        write_dir = Path(tmp) if args.write_data else None
        # The cold run pays one-time JIT/XLA compilation (and OS/page-cache warm-up);
        # the warm run is the steady state a production-scale catalogue actually sees.
        # Their difference is the compile cost that amortizes away at scale. Both are
        # recorded so the docs can report cold *and* warm side by side.
        cold = run_once(write_dir)
        warm = run_once(write_dir)

    prov = provenance(n_cpu_cores=args.n_cpu_cores, n_gpus=args.n_gpus)

    def _per_second(wall: float) -> float | None:
        return args.n_events / wall if wall else None

    def _core_hours(wall: float, units: int) -> float:
        return wall / 3600.0 * units

    record = {
        "label": args.label or f"{args.backend}-{args.method}",
        "configuration": {
            "backend": args.backend,
            "method": args.method,
            "approximant": args.approximant,
            "detectors": args.detectors,
            "n_events": args.n_events,
            "chunk_size": args.chunk_size,
            "n_chirp_mass_bins": args.n_chirp_mass_bins,
            **run,
        },
        "metrics": {
            "wall_seconds_cold": cold.wall_seconds,
            "wall_seconds_warm": warm.wall_seconds,
            "compile_seconds": max(cold.wall_seconds - warm.wall_seconds, 0.0),
            "events_per_second_cold": _per_second(cold.wall_seconds),
            "events_per_second_warm": _per_second(warm.wall_seconds),
            "cpu_core_hours_cold": _core_hours(cold.wall_seconds, prov["n_cpu_cores"]),
            "cpu_core_hours_warm": _core_hours(warm.wall_seconds, prov["n_cpu_cores"]),
            "gpu_hours_cold": _core_hours(cold.wall_seconds, prov["n_gpus"]),
            "gpu_hours_warm": _core_hours(warm.wall_seconds, prov["n_gpus"]),
            "peak_rss_bytes": max(cold.peak_rss_bytes, warm.peak_rss_bytes),
            "average_rss_bytes": warm.average_rss_bytes,
            "gpu_peak_bytes": max(cold.gpu_peak_bytes or 0, warm.gpu_peak_bytes or 0),
            "output_bytes": cold.output_bytes,
        },
        "provenance": prov,
    }
    write_result(args.output_json, record)
    m = record["metrics"]
    print(
        f"{record['label']}: cold {m['wall_seconds_cold']:.2f} s / warm {m['wall_seconds_warm']:.2f} s "
        f"(compile {m['compile_seconds']:.2f} s), "
        f"warm {m['cpu_core_hours_warm']:.3f} CPU-core-h, {m['gpu_hours_warm']:.3f} GPU-h, "
        f"peak {m['peak_rss_bytes'] / 1e9:.2f} GB, output {m['output_bytes'] / 1e6:.1f} MB  "
        f"(gwmock-signal {prov['gwmock_signal_version']})"
    )
    with contextlib.suppress(KeyError):
        print(f"  CPU={prov['cpu_model']}  GPU={prov['gpu_models'] or 'none'}")


if __name__ == "__main__":
    main()
