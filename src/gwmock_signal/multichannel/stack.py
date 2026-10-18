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
"""Stack per-detector strains with a fixed channel order.

:class:`DetectorStrainStack` is :mod:`gwmock_data.multichannel`'s, shared by the gwmock packages and
re-exported here, so ``gwmock_signal.multichannel.stack.DetectorStrainStack`` keeps working. Its HDF5
files keep their layout, including the ``gwmock_signal_detector_strain_stack_order`` attribute, and are
written and read without h5py when it is not installed.
"""

from __future__ import annotations

from gwmock_data.multichannel import DetectorStrainStack

__all__ = ["DetectorStrainStack"]
