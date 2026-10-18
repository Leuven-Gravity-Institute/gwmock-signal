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
"""Time-domain strain injection into segments.

See ``docs/user_guide/strain-injection.md`` (examples) and ``docs/api/injection/index.md`` (API).
"""

from __future__ import annotations

from gwmock_signal.injection.core import inject_strain, inject_strains_sequential

__all__ = ["inject_strain", "inject_strains_sequential"]
