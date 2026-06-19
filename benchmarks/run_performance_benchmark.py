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
catalogue and writes a single JSON record with the timing, CPU/GPU core-hours,
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


def main() -> None:
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
    parser.add_argument("--end-time", type=float, default=1_126_259_462.0 + 3.0e7)
    parser.add_argument("--chunk-size", type=int, default=None, help="Batched method only.")
    parser.add_argument("--n-chirp-mass-bins", type=int, default=1, help="Batched method only.")
    parser.add_argument("--n-cpu-cores", type=int, default=None, help="Override for CPU core-hours.")
    parser.add_argument("--n-gpus", type=int, default=None, help="Override for GPU-hours.")
    parser.add_argument("--label", default=None, help="Human-readable label for this run.")
    parser.add_argument("--write-data", action="store_true", help="Write segments to measure on-disk size.")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.method == "batched" and args.backend != "ripple":
        parser.error("the batched method is only available for the ripple backend")

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

    with tempfile.TemporaryDirectory() as tmp:
        write_dir = Path(tmp) if args.write_data else None
        with measure() as usage:
            segments = workload()
            usage.output_bytes = _output_bytes(segments, write_dir=write_dir)

    prov = provenance(n_cpu_cores=args.n_cpu_cores, n_gpus=args.n_gpus)
    wall_hours = usage.wall_seconds / 3600.0
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
            "wall_seconds": usage.wall_seconds,
            "cpu_core_hours": wall_hours * prov["n_cpu_cores"],
            "gpu_hours": wall_hours * prov["n_gpus"],
            "peak_rss_bytes": usage.peak_rss_bytes,
            "average_rss_bytes": usage.average_rss_bytes,
            "gpu_peak_bytes": usage.gpu_peak_bytes,
            "output_bytes": usage.output_bytes,
            "events_per_second": args.n_events / usage.wall_seconds if usage.wall_seconds else None,
        },
        "provenance": prov,
    }
    write_result(args.output_json, record)
    metrics = record["metrics"]
    print(
        f"{record['label']}: {metrics['wall_seconds']:.2f} s, {metrics['cpu_core_hours']:.3f} CPU-core-h, "
        f"{metrics['gpu_hours']:.3f} GPU-h, peak {metrics['peak_rss_bytes'] / 1e9:.2f} GB, "
        f"output {metrics['output_bytes'] / 1e6:.1f} MB  (gwmock-signal {prov['gwmock_signal_version']})"
    )
    with contextlib.suppress(KeyError):
        print(f"  CPU={prov['cpu_model']}  GPU={prov['gpu_models'] or 'none'}")


if __name__ == "__main__":
    main()
