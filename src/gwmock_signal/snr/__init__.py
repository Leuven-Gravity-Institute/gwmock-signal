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
"""Signal-to-noise ratio utilities for gravitational-wave signals."""

from __future__ import annotations

from gwmock_signal.snr.core import matched_filter_snr, noise_weighted_inner_product, optimal_snr
from gwmock_signal.snr.psd import evaluate_psd, from_numpy_psd, load_design_psd

__all__ = [
    "evaluate_psd",
    "from_numpy_psd",
    "load_design_psd",
    "matched_filter_snr",
    "noise_weighted_inner_product",
    "optimal_snr",
]
