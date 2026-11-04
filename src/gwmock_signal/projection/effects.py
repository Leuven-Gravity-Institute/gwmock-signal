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
"""Which physical effects the projection includes, for every source alike."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class ProjectionEffects:
    """The physical effects projection includes. All are on by default; each can be turned off alone.

    Compact binaries, bursts and continuous waves are projected by the same code, so a given set of
    effects means the same thing for each. :meth:`long_wavelength` turns them all off: a point detector at
    its vertex, a geocentre at rest and a J2000 direction, the convention of Bilby's and PyCBC's default
    projection.

    Attributes:
        earth_rotation: The antenna pattern and the geocentre-to-detector delay follow the Earth's
            rotation sample by sample. Off, both are held at the midpoint of the span, which suits only a
            signal of seconds.
        finite_arm_length: Each arm responds to the strain averaged along the light's round trip to its
            end mirror and back rather than to the strain at the vertex; see
            :mod:`~gwmock_signal.projection.finite_arm`. Needs every detector's arm length.
        source_precession: The source direction is rotated from J2000 into the mean equator and equinox of
            date before the Earth's orientation is applied, as LALPulsar's barycentring does. Off, a J2000
            direction is used with sidereal time of date, as LAL, Bilby and PyCBC do for transients; the
            two differ by up to 1.8e-4 s of geocentre-to-detector delay and about 0.4 degrees of right
            ascension in 2030.
        roemer_delay: The travel time between the geocentre and the solar-system barycentre, from the
            Earth's orbit; see :mod:`~gwmock_signal.projection.solar_system`.
        einstein_delay: The relativistic time dilation of the Earth's clock, TDB minus TT.
        shapiro_delay: The Sun's gravitational delay.

    For a transient the three solar-system delays enter only through how they change across the signal:
    their value and rate at the signal's reference time (a compact binary's coalescence, a burst's peak)
    are absorbed, so that time keeps its meaning at the geocentre and the detector-frame masses theirs.
    For a continuous wave, whose parameters are defined at the barycentre, they enter in full.
    """

    earth_rotation: bool = True
    finite_arm_length: bool = True
    source_precession: bool = True
    roemer_delay: bool = True
    einstein_delay: bool = True
    shapiro_delay: bool = True

    @classmethod
    def long_wavelength(cls, *, earth_rotation: bool = False) -> ProjectionEffects:
        """Return every effect off: a point detector at its vertex, a geocentre at rest, J2000 directions.

        Args:
            earth_rotation: Keep the Earth's rotation on, for a signal long enough to need it.
        """
        return replace(
            cls(**dict.fromkeys((field.name for field in fields(cls)), False)), earth_rotation=earth_rotation
        )

    @property
    def solar_system(self) -> bool:
        """Whether any of the three solar-system delays is included."""
        return self.roemer_delay or self.einstein_delay or self.shapiro_delay
