"""Tests for stochastic-background signal simulation."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal import resolve_simulator_backend
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.stochastic import (
    StochasticBackgroundSimulator,
    StochasticBackgroundSpectrum,
    long_wavelength_overlap_reduction,
)

pytest.importorskip("gwmock_noise")


def test_power_law_spectrum_converts_omega_to_strain_psd() -> None:
    """The power-law spectrum is finite, positive, and zero at f=0."""
    spectrum = StochasticBackgroundSpectrum(omega_ref=1.0e-9, spectral_index=2.0, reference_frequency=25.0)
    frequencies = np.array([0.0, 25.0, 50.0])

    omega = spectrum.omega(frequencies)
    psd = spectrum.strain_psd(frequencies)

    assert omega[0] == 0.0
    assert omega[1] == pytest.approx(1.0e-9)
    assert omega[2] == pytest.approx(4.0e-9)
    assert psd[0] == 0.0
    assert np.all(psd[1:] > 0.0)


def test_stochastic_simulator_returns_detector_strain_stack() -> None:
    """SGWB simulator produces aligned signal-only detector strain."""
    frequencies = np.fft.rfftfreq(256, d=1.0 / 128.0)
    overlap_reduction = {("H1", "L1"): np.full(np.count_nonzero(frequencies >= 8.0), 0.25)}
    simulator = StochasticBackgroundSimulator(duration=2.0, seed=10, overlap_reduction=overlap_reduction)

    stack = simulator.simulate(
        {"omega_ref": 1.0e30},
        ["H1", "L1"],
        sampling_frequency=128.0,
        minimum_frequency=8.0,
    )

    assert isinstance(stack, DetectorStrainStack)
    assert stack.detector_names == ("H1", "L1")
    assert stack.data.shape == (2, 256)
    assert np.all(np.isfinite(stack.data))
    assert np.any(stack.data != 0.0)


def test_stochastic_simulator_is_reproducible_with_seed() -> None:
    """A fixed seed makes stochastic signal generation deterministic."""
    overlap_reduction = {("H1", "L1"): np.zeros(65)}
    params = {"omega_ref": 1.0e30}
    simulator_a = StochasticBackgroundSimulator(duration=1.0, seed=123, overlap_reduction=overlap_reduction)
    simulator_b = StochasticBackgroundSimulator(duration=1.0, seed=123, overlap_reduction=overlap_reduction)

    stack_a = simulator_a.simulate(params, ["H1", "L1"], sampling_frequency=128.0, minimum_frequency=0.0)
    stack_b = simulator_b.simulate(params, ["H1", "L1"], sampling_frequency=128.0, minimum_frequency=0.0)

    np.testing.assert_allclose(stack_a.data, stack_b.data)


def test_stochastic_simulator_adds_background() -> None:
    """SGWB signal can be added to an existing aligned background."""
    overlap_reduction = {("H1", "L1"): np.zeros(65)}
    background = {
        "H1": TimeSeries(np.ones(128), t0=100.0, sample_rate=128.0, unit="strain"),
        "L1": TimeSeries(np.ones(128) * 2.0, t0=100.0, sample_rate=128.0, unit="strain"),
    }
    simulator = StochasticBackgroundSimulator(duration=1.0, seed=4, overlap_reduction=overlap_reduction)

    injected = simulator.simulate(
        {"omega_ref": 1.0e30},
        ["H1", "L1"],
        background=background,
        sampling_frequency=128.0,
        minimum_frequency=0.0,
    )
    signal_only = simulator.simulate(
        {"omega_ref": 1.0e30},
        ["H1", "L1"],
        sampling_frequency=128.0,
        minimum_frequency=0.0,
    )

    np.testing.assert_allclose(injected["H1"].value, signal_only["H1"].value + 1.0)
    np.testing.assert_allclose(injected["L1"].value, signal_only["L1"].value + 2.0)
    assert float(injected.t0.value) == 100.0


def test_long_wavelength_overlap_reduction_has_pair_keys() -> None:
    """Default ORF helper returns one array per detector pair."""
    frequencies = np.array([10.0, 20.0, 30.0])

    overlap_reduction = long_wavelength_overlap_reduction(["H1", "L1"], frequencies)

    assert set(overlap_reduction) == {("H1", "L1")}
    assert overlap_reduction[("H1", "L1")].shape == frequencies.shape


def test_stochastic_exports_and_registry() -> None:
    """SGWB simulator is available from public package entry points."""
    import gwmock_signal

    assert gwmock_signal.StochasticBackgroundSimulator is StochasticBackgroundSimulator
    assert resolve_simulator_backend("sgwb") is StochasticBackgroundSimulator
