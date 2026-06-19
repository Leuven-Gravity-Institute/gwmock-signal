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
"""Measure the agreement between the ripple (JAX) backend and the LAL baseline.

For each ripple approximant that LAL implements in the time domain, this computes
the white, time/phase-maximized match between the two backends across several
parameter sets and records the worst and median match per approximant, with full
run provenance (release version, library versions). The plotting script renders the
result as the consistency table/figure in the docs.

Run with::

    uv run --extra jax python benchmarks/run_consistency_benchmark.py --output-json results/consistency.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from benchutils import provenance, write_result

# (approximant, family) — TaylorF2 is omitted: LAL provides no time-domain TaylorF2.
_ALIGNED = ("IMRPhenomD", "IMRPhenomHM", "IMRPhenomXAS", "IMRPhenomXHM")
_TIDAL = ("IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3")
_PRECESSING = ("IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM")

# Per-family parameter sets (canonical names) spanning a few configurations each.
_ALIGNED_CONFIGS = [
    {"detector_frame_mass_1": 40.0, "detector_frame_mass_2": 31.0, "spin_1z": 0.5, "spin_2z": -0.2, "inclination": 0.9},
    {"detector_frame_mass_1": 36.0, "detector_frame_mass_2": 29.0, "spin_1z": 0.0, "spin_2z": 0.0, "inclination": 0.4},
    {"detector_frame_mass_1": 60.0, "detector_frame_mass_2": 55.0, "spin_1z": -0.3, "spin_2z": 0.2, "inclination": 1.2},
]
_TIDAL_CONFIGS = [
    {
        "detector_frame_mass_1": 1.6,
        "detector_frame_mass_2": 1.4,
        "spin_1z": 0.02,
        "spin_2z": -0.01,
        "inclination": 0.6,
        "lambda_1": 400.0,
        "lambda_2": 500.0,
    },
    {
        "detector_frame_mass_1": 2.0,
        "detector_frame_mass_2": 1.5,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "inclination": 0.3,
        "lambda_1": 300.0,
        "lambda_2": 600.0,
    },
]
_PRECESSING_CONFIGS = [
    {
        "detector_frame_mass_1": 40.0,
        "detector_frame_mass_2": 30.0,
        "spin_1x": 0.3,
        "spin_1y": 0.1,
        "spin_1z": 0.2,
        "spin_2x": -0.1,
        "spin_2y": 0.2,
        "spin_2z": 0.1,
        "inclination": 0.6,
    },
    {
        "detector_frame_mass_1": 50.0,
        "detector_frame_mass_2": 35.0,
        "spin_1x": -0.2,
        "spin_1y": 0.3,
        "spin_1z": 0.1,
        "spin_2x": 0.2,
        "spin_2y": -0.1,
        "spin_2z": 0.0,
        "inclination": 1.0,
    },
]
_FAMILIES = dict.fromkeys(_ALIGNED, _ALIGNED_CONFIGS)
_FAMILIES.update(dict.fromkeys(_TIDAL, _TIDAL_CONFIGS))
_FAMILIES.update(dict.fromkeys(_PRECESSING, _PRECESSING_CONFIGS))


def _white_match(a: np.ndarray, b: np.ndarray, sampling_frequency: float, minimum_frequency: float) -> float:
    """White, time/phase-maximized match between two real time series."""
    n = 1 << (int(np.ceil(np.log2(max(len(a), len(b))))) + 1)
    spectrum_a = np.fft.rfft(a, n=n)
    spectrum_b = np.fft.rfft(b, n=n)
    in_band = np.fft.rfftfreq(n, d=1.0 / sampling_frequency) >= minimum_frequency
    spectrum_a = np.where(in_band, spectrum_a, 0.0)
    spectrum_b = np.where(in_band, spectrum_b, 0.0)
    cross = spectrum_a * np.conj(spectrum_b)
    full = np.zeros(n, dtype=complex)
    full[: len(cross)] = cross
    correlation = np.fft.ifft(full) * n
    norm = np.sqrt(np.sum(np.abs(spectrum_a) ** 2) * np.sum(np.abs(spectrum_b) ** 2))
    return float(np.max(np.abs(correlation)) / norm)


def main() -> None:
    """Run the ripple-vs-LAL match for every supported approximant and write the record."""
    parser = argparse.ArgumentParser(description="Measure ripple-vs-LAL waveform agreement.")
    parser.add_argument("--sampling-frequency", type=float, default=2048.0)
    parser.add_argument("--minimum-frequency", type=float, default=20.0)
    parser.add_argument("--tidal-minimum-frequency", type=float, default=40.0, help="Used for BNS/NRTidal (long).")
    parser.add_argument("--distance", type=float, default=400.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    from gwmock_signal.waveform.backends import LALSimulationBackend, RippleBackend

    ripple_backend = RippleBackend()
    lal_backend = LALSimulationBackend()
    tc = 1_126_259_462.4

    approximants: dict[str, dict] = {}
    for approximant, configs in _FAMILIES.items():
        f_min = args.tidal_minimum_frequency if approximant in _TIDAL else args.minimum_frequency
        matches: list[float] = []
        for config in configs:
            common = {
                "tc": tc,
                "sampling_frequency": args.sampling_frequency,
                "minimum_frequency": f_min,
                "luminosity_distance": args.distance,
                **config,
            }
            ripple = ripple_backend.generate_td_waveform(approximant, **common)
            lal = lal_backend.generate_td_waveform(approximant, **common)
            for polarization in ("plus", "cross"):
                matches.append(
                    _white_match(ripple[polarization].value, lal[polarization].value, args.sampling_frequency, f_min)
                )
        approximants[approximant] = {
            "minimum_frequency": f_min,
            "n_matches": len(matches),
            "min_match": min(matches),
            "median_match": float(np.median(matches)),
        }
        print(
            f"{approximant}: min={approximants[approximant]['min_match']:.5f} "
            f"median={approximants[approximant]['median_match']:.5f}"
        )

    write_result(args.output_json, {"approximants": approximants, "provenance": provenance()})


if __name__ == "__main__":
    main()
