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
"""Overlap-reduction helpers for stochastic backgrounds."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from gwmock_signal.detector import CustomDetector
from gwmock_signal.projection.network import _make_detectors, _reconstructed_geometry

DETECTOR_PAIR_SIZE = 2
DetectorSpec = str | CustomDetector
OverlapReductionInput = (
    Mapping[tuple[str, str], NDArray[np.floating]]
    | Mapping[tuple[str, str], NDArray[np.complexfloating]]
    | Callable[[NDArray[np.float64], Sequence[str]], Mapping[tuple[str, str], NDArray[np.complex128]]]
)


def detector_names(detectors: Sequence[DetectorSpec]) -> list[str]:
    """Return public detector names from detector specs."""
    names = [detector if isinstance(detector, str) else detector.name for detector in detectors]
    if not names:
        raise ValueError("detector_names must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError("detector_names must be unique.")
    return names


def detector_tensors(detectors: Sequence[DetectorSpec]) -> dict[str, NDArray[np.float64]]:
    """Resolve detector response tensors through the package geometry layer."""
    return {name: _reconstructed_geometry(prefix)[0] for name, prefix in _make_detectors(detectors)}


def long_wavelength_overlap_reduction(
    detectors: Sequence[DetectorSpec],
    frequencies: NDArray[np.float64],
) -> dict[tuple[str, str], NDArray[np.float64]]:
    """Return frequency-independent tensor ORFs from detector response tensors.

    This is the long-wavelength, co-located limit ``gamma_ij = 2 D_i : D_j``.
    It is a useful default for ET-style low-frequency studies and tests. For
    separated detectors or paper-facing production datasets, pass explicit ORF
    arrays to :class:`~gwmock_signal.stochastic.StochasticBackgroundSimulator`.
    """
    names = detector_names(detectors)
    tensors = detector_tensors(detectors)
    n_frequencies = np.asarray(frequencies).shape[0]
    overlap_reduction: dict[tuple[str, str], NDArray[np.float64]] = {}
    for index_a, detector_a in enumerate(names):
        for detector_b in names[index_a + 1 :]:
            gamma = 2.0 * float(np.sum(tensors[detector_a] * tensors[detector_b]))
            overlap_reduction[(detector_a, detector_b)] = np.full(n_frequencies, gamma, dtype=float)
    return overlap_reduction


def normalize_overlap_reduction(
    overlap_reduction: OverlapReductionInput | None,
    *,
    detectors: Sequence[DetectorSpec],
    names: Sequence[str],
    frequencies: NDArray[np.float64],
) -> dict[tuple[str, str], NDArray[np.complex128]]:
    """Return pairwise ORF arrays keyed by normalized detector-name pairs."""
    if overlap_reduction is None:
        raw = long_wavelength_overlap_reduction(detectors, frequencies)
    elif callable(overlap_reduction):
        raw = overlap_reduction(frequencies, names)
    else:
        raw = overlap_reduction

    detector_set = set(names)
    normalized: dict[tuple[str, str], NDArray[np.complex128]] = {}
    expected_shape = np.asarray(frequencies).shape
    for pair, values in raw.items():
        if len(pair) != DETECTOR_PAIR_SIZE:
            raise ValueError("overlap_reduction pairs must contain exactly two detector names.")
        detector_a, detector_b = tuple(sorted(pair))
        if detector_a == detector_b:
            raise ValueError("overlap_reduction pairs must reference two distinct detectors.")
        if detector_a not in detector_set or detector_b not in detector_set:
            raise ValueError("overlap_reduction pairs must reference configured detectors.")
        arr = np.asarray(values, dtype=np.complex128)
        if arr.shape != expected_shape:
            raise ValueError("overlap_reduction arrays must have shape (n_frequencies,).")
        normalized[(detector_a, detector_b)] = arr
    return normalized
