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
"""Built-in constants: LAL's values, written out so they need no ``lalsuite``.

Each literal is the shortest decimal that round-trips to LAL's double. Change one
only together with the ``lalsuite`` floor, after the comparison against LAL shows
a release has changed it.
"""

from __future__ import annotations

#: Nominal solar mass in kg (``lal.MSUN_SI``).
MSUN_SI = 1.9884098706980507e30
#: Solar mass in seconds, ``G M_sun / c^3`` (``lal.MTSUN_SI``).
MTSUN_SI = 4.925490947641267e-06
#: Parsec in metres (``lal.PC_SI``).
PC_SI = 3.085677581491367e16
#: Megaparsec in metres. Derived exactly as LAL-based codes derive it, ``PC_SI * 1e6``.
MPC_SI = PC_SI * 1e6
