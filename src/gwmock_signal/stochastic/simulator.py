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
"""Simulator for isotropic stochastic gravitational-wave backgrounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gwpy.timeseries import TimeSeries

from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.simulator import GWSimulator
from gwmock_signal.stochastic.overlap import (
    DetectorSpec,
    OverlapReductionInput,
    normalize_overlap_reduction,
)
from gwmock_signal.stochastic.overlap import (
    detector_names as normalize_detector_names,
)
from gwmock_signal.stochastic.spectrum import H0_SI, StochasticBackgroundSpectrum


class StochasticBackgroundSimulator(GWSimulator):
    """Generate isotropic SGWB detector strain as a correlated Gaussian signal.

    The simulator samples Fourier coefficients from a one-sided detector
    covariance ``C_ij(f) = gamma_ij(f) S_h(f)``, where ``S_h(f)`` is derived
    from a power-law ``Omega_GW`` spectrum. It returns signal-only strain by
    default or adds the SGWB signal to an optional background mapping.

    Args:
        duration: Signal duration in seconds.
        seed: Optional random seed.
        overlap_reduction: Optional pairwise ORF arrays or callable. When
            omitted, the long-wavelength geometric ORF is used.
        regularization_epsilon: Positive diagonal regularization scale passed
            to the spectral Cholesky builder.
    """

    _REQUIRED: frozenset[str] = frozenset({"omega_ref"})

    def __init__(
        self,
        *,
        duration: float,
        seed: int | None = None,
        overlap_reduction: OverlapReductionInput | None = None,
        regularization_epsilon: float = 1.0e-12,
    ) -> None:
        """Initialize the SGWB simulator."""
        if duration <= 0.0:
            raise ValueError("duration must be positive.")
        if regularization_epsilon <= 0.0:
            raise ValueError("regularization_epsilon must be positive.")
        self.duration = duration
        self.seed = seed
        self.overlap_reduction = overlap_reduction
        self.regularization_epsilon = regularization_epsilon

    @property
    def required_params(self) -> frozenset[str]:
        """Return required SGWB spectrum parameter keys."""
        return self._REQUIRED

    def simulate(  # noqa: PLR0913
        self,
        params: Mapping[str, Any],
        detector_names: Sequence[DetectorSpec],
        background: Mapping[str, TimeSeries] | None = None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Generate SGWB strain on a fixed detector network."""
        del earth_rotation, interpolate_if_offset
        self._validate_params(params)
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")
        if minimum_frequency < 0.0:
            raise ValueError("minimum_frequency must be non-negative.")

        names = normalize_detector_names(detector_names)
        n_samples = round(self.duration * sampling_frequency)
        if n_samples <= 0:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        frequency_grid = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
        frequency_mask = frequency_grid >= minimum_frequency
        if not np.any(frequency_mask):
            raise ValueError("minimum_frequency leaves no FFT bins to simulate.")

        masked_frequencies = frequency_grid[frequency_mask]
        delta_frequency = sampling_frequency / n_samples
        spectrum = StochasticBackgroundSpectrum(
            omega_ref=float(params["omega_ref"]),
            spectral_index=float(params.get("spectral_index", 0.0)),
            reference_frequency=float(params.get("reference_frequency", 25.0)),
            hubble_constant_si=float(params.get("hubble_constant_si", H0_SI)),
        )
        strain_psd = spectrum.strain_psd(masked_frequencies)
        psd = dict.fromkeys(names, strain_psd)
        overlap_reduction = normalize_overlap_reduction(
            self.overlap_reduction,
            detectors=detector_names,
            names=names,
            frequencies=masked_frequencies,
        )
        csd = {pair: gamma * strain_psd for pair, gamma in overlap_reduction.items()}
        factors = self._spectral_factors(psd, csd, names=names, delta_frequency=delta_frequency)
        signal = self._sample_signal(
            np.random.default_rng(self.seed),
            factors,
            names=names,
            frequency_grid_size=frequency_grid.size,
            frequency_mask=frequency_mask,
            delta_frequency=delta_frequency,
            n_samples=n_samples,
        )
        return self._to_stack(signal, names, background=background, sampling_frequency=sampling_frequency)

    def _spectral_factors(
        self,
        psd: Mapping[str, np.ndarray],
        csd: Mapping[tuple[str, str], np.ndarray],
        *,
        names: Sequence[str],
        delta_frequency: float,
    ) -> np.ndarray:
        """Build spectral Cholesky factors using the optional gwmock-noise extra."""
        try:
            from gwmock_noise.spectral import (  # noqa: PLC0415
                assemble_hermitian_spectral_matrices,
                cholesky_factors_from_spectral_matrices,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install gwmock-signal[sgwb] to simulate stochastic backgrounds.") from exc

        matrices = assemble_hermitian_spectral_matrices(names, psd, csd)
        return cholesky_factors_from_spectral_matrices(
            matrices,
            delta_frequency=delta_frequency,
            regularization_epsilon=self.regularization_epsilon,
        )

    def _sample_signal(  # noqa: PLR0913
        self,
        rng: np.random.Generator,
        factors: np.ndarray,
        *,
        names: Sequence[str],
        frequency_grid_size: int,
        frequency_mask: np.ndarray,
        delta_frequency: float,
        n_samples: int,
    ) -> Mapping[str, np.ndarray]:
        """Sample real detector strain arrays using the optional gwmock-noise extra."""
        try:
            from gwmock_noise.spectral import simulate_spectral_covariance_chunk  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install gwmock-signal[sgwb] to simulate stochastic backgrounds.") from exc

        return simulate_spectral_covariance_chunk(
            rng,
            factors,
            detectors=names,
            frequency_grid_size=frequency_grid_size,
            frequency_mask=frequency_mask,
            delta_frequency=delta_frequency,
            window_size=n_samples,
        )

    def _to_stack(
        self,
        signal: Mapping[str, np.ndarray],
        names: Sequence[str],
        *,
        background: Mapping[str, TimeSeries] | None,
        sampling_frequency: float,
    ) -> DetectorStrainStack:
        """Convert sampled arrays to a detector stack, adding background if supplied."""
        t0 = 0.0 if background is None else float(next(iter(background.values())).t0.value)
        if background is not None:
            for detector in names:
                if detector not in background:
                    raise KeyError(f"Missing background for detector {detector!r}.")
                if len(background[detector]) != len(signal[detector]):
                    raise ValueError("background channels must match the SGWB sample count.")
            strains = {
                detector: background[detector]
                + TimeSeries(signal[detector], t0=t0, sample_rate=sampling_frequency, unit="strain")
                for detector in names
            }
        else:
            strains = {
                detector: TimeSeries(
                    signal[detector],
                    t0=t0,
                    sample_rate=sampling_frequency,
                    unit="strain",
                    name=detector,
                )
                for detector in names
            }
        return DetectorStrainStack.from_mapping(names, strains)
