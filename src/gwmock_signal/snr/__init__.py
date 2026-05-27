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
"""SNR computation functions for gravitational-wave signals.

The pycbc-backed functions (`matched_filter_snr`, `optimal_snr`) require pycbc as an optional dependency and raise
``ImportError`` with a helpful message when it is absent.
"""

from __future__ import annotations

from gwmock_signal.snr._network import network_optimal_snr
from gwmock_signal.snr._pycbc import matched_filter_snr, optimal_snr

__all__ = ["matched_filter_snr", "network_optimal_snr", "optimal_snr"]
