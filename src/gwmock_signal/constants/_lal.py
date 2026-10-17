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
"""The constants as the installed ``lalsuite`` gives them.

``MSUN_SI``, ``MTSUN_SI`` and ``PC_SI`` are read from ``lal``; ``MPC_SI`` is derived from
``PC_SI`` exactly as the built-in provider derives it, since LAL defines no megaparsec.
Defines the same names as :mod:`gwmock_signal.constants._builtin` so the two can be
compared name by name. Importing it requires ``lalsuite``.
"""

from __future__ import annotations

import lal

MSUN_SI = lal.MSUN_SI
MTSUN_SI = lal.MTSUN_SI
PC_SI = lal.PC_SI
MPC_SI = PC_SI * 1e6
