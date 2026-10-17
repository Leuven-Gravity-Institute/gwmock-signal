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
"""Physical constants used to convert source parameters to SI units.

The values come from :mod:`gwmock_signal.constants._builtin`, written out in this
package, so converting masses and distances does not require ``lalsuite`` to be
installed. ``MSUN_SI``, ``MTSUN_SI`` and ``PC_SI`` equal LAL's constants of the
same names bit-for-bit, and ``MPC_SI`` is ``PC_SI * 1e6``. They must stay that way:
LALSimulation converts with its own constants internally, so a value that differed in
the last bit would carry a one-ulp error into every waveform generated through it. The
test suite checks them against ``lal`` whenever ``lalsuite`` is installed, and against
``ripplegw`` whenever it is.

:mod:`gwmock_signal.constants._lal` reads the same names from LAL. It exists for that
comparison; the package itself never converts with it.
"""

from __future__ import annotations

from gwmock_signal.constants._builtin import MPC_SI, MSUN_SI, MTSUN_SI, PC_SI

#: The names every constants provider defines, in the order they are reported.
CONSTANT_NAMES = ("MSUN_SI", "MTSUN_SI", "PC_SI", "MPC_SI")

__all__ = ["CONSTANT_NAMES", "MPC_SI", "MSUN_SI", "MTSUN_SI", "PC_SI"]
