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
"""Power-law SGWB spectral models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

H0_SI = 67.74 * 1000.0 / 3.0856775814913673e22


@dataclass(frozen=True, slots=True)
class StochasticBackgroundSpectrum:
    """Power-law isotropic SGWB spectrum.

    Args:
        omega_ref: Dimensionless energy-density amplitude at
            ``reference_frequency``.
        spectral_index: Power-law index for ``Omega_GW(f)``.
        reference_frequency: Reference frequency in Hz.
        hubble_constant_si: Hubble constant in inverse seconds.
    """

    omega_ref: float
    spectral_index: float = 0.0
    reference_frequency: float = 25.0
    hubble_constant_si: float = H0_SI

    def omega(self, frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate ``Omega_GW(f)`` on a frequency grid."""
        if self.omega_ref < 0.0:
            raise ValueError("omega_ref must be non-negative.")
        if self.reference_frequency <= 0.0:
            raise ValueError("reference_frequency must be positive.")
        frequencies = np.asarray(frequencies, dtype=float)
        omega = np.zeros_like(frequencies)
        positive = frequencies > 0.0
        omega[positive] = self.omega_ref * (frequencies[positive] / self.reference_frequency) ** self.spectral_index
        return omega

    def strain_psd(self, frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert ``Omega_GW(f)`` to one-sided strain PSD."""
        if self.hubble_constant_si <= 0.0:
            raise ValueError("hubble_constant_si must be positive.")
        frequencies = np.asarray(frequencies, dtype=float)
        psd = np.zeros_like(frequencies)
        positive = frequencies > 0.0
        coefficient = 3.0 * self.hubble_constant_si**2 / (10.0 * np.pi**2)
        psd[positive] = coefficient * self.omega(frequencies[positive]) / frequencies[positive] ** 3
        return psd
