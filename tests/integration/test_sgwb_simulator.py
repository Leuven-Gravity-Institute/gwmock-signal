"""Slow integration tests for SGWB signal generation."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.stochastic import StochasticBackgroundSimulator

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SAMPLING_FREQUENCY = 128.0
DURATION = 16.0
MINIMUM_FREQUENCY = 4.0
OMEGA_REF = 1.0e30


def _one_sided_correlation(data: np.ndarray) -> float:
    """Return the Pearson correlation between two simulated detector channels."""
    return float(np.corrcoef(data[0], data[1])[0, 1])


def test_sgwb_simulator_reproducible_and_correlated() -> None:
    """SGWB simulation is deterministic by seed and follows explicit ORF sign."""
    n_samples = round(DURATION * SAMPLING_FREQUENCY)
    masked_bins = np.count_nonzero(np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_FREQUENCY) >= MINIMUM_FREQUENCY)
    overlap_reduction = {("H1", "L1"): np.full(masked_bins, 0.75)}
    params = {"omega_ref": OMEGA_REF, "spectral_index": 0.0, "reference_frequency": 25.0}

    simulator_a = StochasticBackgroundSimulator(
        duration=DURATION,
        seed=12345,
        overlap_reduction=overlap_reduction,
    )
    simulator_b = StochasticBackgroundSimulator(
        duration=DURATION,
        seed=12345,
        overlap_reduction=overlap_reduction,
    )

    stack_a = simulator_a.simulate(
        params,
        ["H1", "L1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        minimum_frequency=MINIMUM_FREQUENCY,
    )
    stack_b = simulator_b.simulate(
        params,
        ["H1", "L1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        minimum_frequency=MINIMUM_FREQUENCY,
    )

    assert stack_a.detector_names == ("H1", "L1")
    assert stack_a.data.shape == (2, n_samples)
    assert np.all(np.isfinite(stack_a.data))
    assert np.std(stack_a.data[0]) > 0.0
    assert np.std(stack_a.data[1]) > 0.0
    np.testing.assert_allclose(stack_a.data, stack_b.data)
    assert _one_sided_correlation(stack_a.data) > 0.2
