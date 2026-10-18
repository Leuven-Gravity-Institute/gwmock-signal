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
"""The time series this package passes around, which is :mod:`gwmock_data`'s.

:class:`TimeSeries`, :func:`as_timeseries` and :func:`register_pytree` are defined in
``gwmock_data.timeseries`` and re-exported here, so a series from either package is the same class: the
gwmock packages share one time series rather than each converting to the others'. See gwmock-data's
documentation for the class itself.
"""

from __future__ import annotations

from gwmock_data.timeseries import TimeSeries, as_timeseries, register_pytree

__all__ = ["TimeSeries", "as_timeseries", "register_pytree"]
