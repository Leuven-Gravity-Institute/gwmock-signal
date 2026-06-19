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
"""Benchmark batched (GPU-ready) CBC catalogue simulation against the per-event path.

Requires the optional JAX extra. Run on CPU and again on a CUDA box to see the
device speedup, e.g.::

    uv run --extra jax python benchmarks/benchmark_simulate_cbc_batch.py --n-events 500

It times the on-device batched path (``simulate_cbc_batch``, vmap + jit) against the
existing per-event CPU pipeline (``generate_td_waveform`` + projection) — the
honest "one event at a time" baseline. JAX compilation is excluded from the timed
batched run by a warm-up call first.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import jax
import numpy as np

from gwmock_signal.jax_batch import simulate_cbc_batch
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.waveform.backends import RippleBackend

_INTRINSIC_KEYS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
    "spin_1z",
    "spin_2z",
    "inclination",
    "coa_phase",
)


def build_catalogue(n_events: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return a synthetic struct-of-arrays catalogue of ``n_events`` aligned-spin BBHs."""
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
        "coa_time": 1_126_259_462.0 + rng.uniform(0.0, 3.0e7, n_events),
    }


def _time(fn: Callable[[], object]) -> float:
    """Return the wall-clock seconds of one call to ``fn``."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    """Parse arguments, run both paths, and print a timing summary."""
    parser = argparse.ArgumentParser(description="Benchmark batched vs per-event CBC simulation.")
    parser.add_argument("--n-events", type=int, default=500)
    parser.add_argument("--sampling-frequency", type=float, default=2048.0)
    parser.add_argument("--minimum-frequency", type=float, default=20.0)
    parser.add_argument("--segment-duration", type=float, default=8.0)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    args = parser.parse_args()

    catalogue = build_catalogue(args.n_events, np.random.default_rng(0))
    backend = RippleBackend(segment_duration=args.segment_duration)
    shared = {
        "sampling_frequency": args.sampling_frequency,
        "minimum_frequency": args.minimum_frequency,
    }

    def batched() -> object:
        result = simulate_cbc_batch(args.approximant, args.detectors, parameters=catalogue, backend=backend, **shared)
        return jax.block_until_ready(result.strain)

    def per_event() -> object:
        outputs = []
        for i in range(args.n_events):
            polarizations = backend.generate_td_waveform(
                args.approximant,
                tc=float(catalogue["coa_time"][i]),
                **shared,
                **{key: float(catalogue[key][i]) for key in _INTRINSIC_KEYS},
            )
            outputs.append(
                project_polarizations_to_network(
                    {"plus": polarizations["plus"], "cross": polarizations["cross"]},
                    args.detectors,
                    right_ascension=float(catalogue["right_ascension"][i]),
                    declination=float(catalogue["declination"][i]),
                    polarization_angle=float(catalogue["polarization_angle"][i]),
                    earth_rotation=False,
                )
            )
        return outputs

    batched()  # warm up: exclude JAX compilation from the timed run
    batched_seconds = _time(batched)
    per_event_seconds = _time(per_event)

    device = jax.devices()[0].platform
    print(f"device={device}  n_events={args.n_events}  detectors={args.detectors}")
    print(f"batched (vmap+jit): {batched_seconds:.3f} s  ({args.n_events / batched_seconds:.0f} events/s)")
    print(f"per-event CPU path: {per_event_seconds:.3f} s  ({args.n_events / per_event_seconds:.0f} events/s)")
    print(f"speedup: {per_event_seconds / batched_seconds:.1f}x")


if __name__ == "__main__":
    main()
