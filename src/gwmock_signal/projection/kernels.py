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
"""Detector geometry and projection factors, written once for NumPy and JAX.

Every function takes the array namespace as ``xp`` -- :mod:`numpy` or :mod:`jax.numpy` -- and uses only
operations both provide, so the NumPy projection path and the on-device JAX path compute the same
quantities from the same code. Under JAX the functions are traceable, so detector geometry and the
projection factors can be jitted, vmapped and differentiated together with a JAX waveform model.

Inputs broadcast: the sidereal time, right ascension and declination may each be a scalar or an array
(per sample, when Earth rotation or precession is followed), and the results take their broadcast
shape. Under JAX, 64-bit mode must be enabled; see :mod:`gwmock_signal.projection.jax_projection`.
"""

from __future__ import annotations

import math
from typing import Any

from gwmock_signal.projection.resampling import SPEED_OF_LIGHT_M_S

#: WGS-84 semi-major axis in metres and flattening, as ERFA's ``eraEform`` defines them.
WGS84_EQUATORIAL_RADIUS_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563


def _rotation(xp: Any, angle: Any, axis: str) -> Any:
    """Return the 3x3 frame rotation by ``angle`` about ``"y"`` or ``"z"``.

    The sense is astropy's ``rotation_matrix``: counterclockwise looking down the positive axis, applied
    to the coordinate frame.
    """
    s, c = xp.sin(angle), xp.cos(angle)
    zero, one = xp.zeros_like(c), xp.ones_like(c)
    if axis == "z":
        rows = ([c, s, zero], [-s, c, zero], [zero, zero, one])
    elif axis == "y":
        rows = ([c, zero, -s], [zero, one, zero], [s, zero, c])
    else:
        raise ValueError(f"axis must be 'y' or 'z', got {axis!r}")
    return xp.stack([xp.stack(row) for row in rows])


def detector_geometry(fields: Any, *, xp: Any) -> tuple[Any, Any]:
    """Return ``(response, location)`` from a detector's seven defining fields.

    Args:
        fields: ``(vertex longitude, vertex latitude, vertex elevation, x-arm azimuth, y-arm azimuth,
            x-arm altitude, y-arm altitude)``, angles in radians and elevation in metres -- the order of
            :data:`gwmock_signal._detector_table.FIELDS`.
        xp: Array namespace, :mod:`numpy` or :mod:`jax.numpy`.

    Returns:
        The 3x3 response tensor ``(x x^T - y y^T) / 2`` of the two arm directions, and the Earth-fixed
        vertex position in metres on the WGS-84 ellipsoid, computed with the same operations as ERFA's
        ``eraGd2gce``.
    """
    longitude, latitude, elevation, x_azimuth, y_azimuth, x_altitude, y_altitude = (
        xp.asarray(value, dtype=xp.float64) for value in fields
    )
    arm_response = xp.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=xp.float64)
    rotation_longitude = _rotation(xp, -longitude, "z")
    rotation_latitude = _rotation(xp, -(math.pi / 2.0 - latitude), "y")

    responses = []
    for azimuth, altitude in ((y_azimuth, y_altitude), (x_azimuth, x_altitude)):
        rotation = rotation_longitude @ rotation_latitude @ _rotation(xp, azimuth, "z") @ _rotation(xp, -altitude, "y")
        responses.append(rotation @ arm_response @ rotation.T / 2.0)

    sin_lat, cos_lat = xp.sin(latitude), xp.cos(latitude)
    w = (1.0 - WGS84_FLATTENING) * (1.0 - WGS84_FLATTENING)
    d = cos_lat * cos_lat + w * sin_lat * sin_lat
    ac = WGS84_EQUATORIAL_RADIUS_M / xp.sqrt(d)
    radius = (ac + elevation) * cos_lat
    location = xp.stack([radius * xp.cos(longitude), radius * xp.sin(longitude), (w * ac + elevation) * sin_lat])
    return responses[0] - responses[1], location


def arm_directions(fields: Any, *, xp: Any) -> tuple[Any, Any]:
    """Return the Earth-fixed unit vectors along the x and y arms, pointing from the vertex outwards.

    Built from the local geodetic frame at the vertex as LAL's ``XLALCreateDetector`` builds it: an arm
    at azimuth ``a`` (clockwise from North) and altitude ``h`` points along
    ``cos h (cos a North + sin a East) + sin h Up``. Half the difference of their outer products is the
    response tensor :func:`detector_geometry` returns. The sense of each arm, which the tensor does not
    carry, is what the finite-arm response needs.

    Args:
        fields: The seven defining fields, in the order of :func:`detector_geometry`.
        xp: Array namespace, :mod:`numpy` or :mod:`jax.numpy`.

    Returns:
        ``(x_arm, y_arm)``, each a unit 3-vector.
    """
    longitude, latitude, _, x_azimuth, y_azimuth, x_altitude, y_altitude = (
        xp.asarray(value, dtype=xp.float64) for value in fields
    )
    sin_lat, cos_lat = xp.sin(latitude), xp.cos(latitude)
    sin_lon, cos_lon = xp.sin(longitude), xp.cos(longitude)
    north = xp.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    east = xp.stack([-sin_lon, cos_lon, xp.zeros_like(cos_lon)])
    up = xp.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    def direction(azimuth: Any, altitude: Any) -> Any:
        return xp.cos(altitude) * (xp.cos(azimuth) * north + xp.sin(azimuth) * east) + xp.sin(altitude) * up

    return direction(x_azimuth, x_altitude), direction(y_azimuth, y_altitude)


def finite_arm_residual(arm_light_crossings: Any, source_alignment: Any, *, xp: Any) -> Any:
    """Return the part of one arm's finite-length response beyond a delay to the arm's midpoint.

    An arm measures the gravitational-wave strain averaged along the light's round trip from the vertex
    to the end mirror and back, not the strain at the vertex. For a wave of frequency ``f`` from a source
    at cosine ``mu`` to an arm of length ``L``, the round trip multiplies the arm's long-wavelength
    response by the single-arm transfer function of R. Essick, S. Vitale and M. Evans,
    "Frequency-dependent responses in third generation gravitational-wave detectors", Phys. Rev. D 96,
    084004 (2017), Eq. (4), also Eq. (7) of P. Baral et al., "Localization of binary neutron star
    mergers with a single Cosmic Explorer", Phys. Rev. D 108, 043010 (2023). With ``x = f L / c`` and
    the response timed at the reflection, the midpoint in time of the round trip, that function is
    ``exp(i pi x mu)`` times

    ``R(x, mu) = (exp(-i pi x) sinc(x (1 + mu)) + exp(i pi x) sinc(x (1 - mu))) / 2``,

    with ``sinc(u) = sin(pi u) / (pi u)``. The first factor is a pure time shift: the strain taken at
    the arm's midpoint rather than at the vertex. This function returns ``R``, the rest: half for each
    leg of the trip, the strain averaged over the time the light spends on that leg. The outward leg
    is centred ``L / (2c)`` before the reflection and lasts ``(1 + mu) L / c``; the return leg is
    centred as much after it and lasts ``(1 - mu) L / c``. ``R`` is one at zero frequency and differs
    from one only at second order in ``x``, so the long-wavelength response is recovered for short
    arms or low frequencies.

    The exponents follow the Fourier convention of :func:`numpy.fft.rfft`, in which a delay ``tau``
    multiplies a spectrum by ``exp(-2 pi i f tau)``. PyCBC (``pycbc.detector.single_arm_frequency_response``)
    and jim (``GroundBased3G._arm_transfer_function``) implement the same transfer function, both with
    the midpoint shift included; PyCBC times the response when the light returns to the vertex, a
    further delay of ``L / c`` common to every arm.

    Args:
        arm_light_crossings: ``x = f L / c``, the arm length in gravitational wavelengths.
        source_alignment: ``mu``, the cosine of the angle between the arm (vertex to end mirror) and the
            direction to the source, in ``[-1, 1]``.
        xp: Array namespace, :mod:`numpy` or :mod:`jax.numpy`.

    Returns:
        ``R(x, mu)``, complex, of the broadcast shape of the inputs.
    """
    x, mu = arm_light_crossings, source_alignment
    outward = xp.exp(-1j * math.pi * x) * xp.sinc(x * (1.0 + mu))
    inward = xp.exp(1j * math.pi * x) * xp.sinc(x * (1.0 - mu))
    return 0.5 * (outward + inward)


def _polarization_basis(
    xp: Any, gmst: Any, right_ascension: Any, declination: Any, polarization_angle: Any
) -> tuple[Any, Any]:
    """Return the polarization basis vectors ``x`` and ``y``, each with a trailing axis of 3."""
    gha = gmst - right_ascension
    cosgha, singha = xp.cos(gha), xp.sin(gha)
    cosdec, sindec = xp.cos(declination), xp.sin(declination)
    cospsi, sinpsi = xp.cos(polarization_angle), xp.sin(polarization_angle)
    shape = xp.broadcast_shapes(xp.shape(cosgha), xp.shape(cosdec), xp.shape(cospsi))

    def stack(components: list[Any]) -> Any:
        return xp.stack([xp.broadcast_to(component, shape) for component in components], axis=-1)

    x = stack(
        [
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec,
        ]
    )
    y = stack(
        [
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec,
        ]
    )
    return x, y


def antenna_pattern(  # noqa: PLR0913 -- the five projection inputs plus the array namespace
    response: Any,
    gmst: Any,
    right_ascension: Any,
    declination: Any,
    polarization_angle: Any,
    *,
    xp: Any,
) -> tuple[Any, Any]:
    """Return the tensor antenna-pattern factors ``(F+, Fx)``.

    Args:
        response: 3x3 detector response tensor, from :func:`detector_geometry`.
        gmst: Greenwich Mean Sidereal Time in radians.
        right_ascension: Source right ascension in radians, in the mean equator and equinox of date.
        declination: Source declination in radians.
        polarization_angle: Polarization angle in radians.
        xp: Array namespace, :mod:`numpy` or :mod:`jax.numpy`.

    Returns:
        ``(f_plus, f_cross)``, each of the broadcast shape of the angles.
    """
    x, y = _polarization_basis(xp, gmst, right_ascension, declination, polarization_angle)
    dx = x @ response.T
    dy = y @ response.T
    return xp.sum(x * dx - y * dy, axis=-1), xp.sum(x * dy + y * dx, axis=-1)


def time_delay_from_geocenter(
    location: Any,
    gmst: Any,
    right_ascension: Any,
    declination: Any,
    *,
    xp: Any,
) -> Any:
    """Return the arrival-time delay at the detector relative to the geocentre, in seconds.

    Args:
        location: Earth-fixed detector position in metres, from :func:`detector_geometry`.
        gmst: Greenwich Mean Sidereal Time in radians.
        right_ascension: Source right ascension in radians, in the mean equator and equinox of date.
        declination: Source declination in radians.
        xp: Array namespace, :mod:`numpy` or :mod:`jax.numpy`.

    Returns:
        The delay, of the broadcast shape of the angles.
    """
    gha = gmst - right_ascension
    cosdec, sindec = xp.cos(declination), xp.sin(declination)
    cosgha, singha = xp.cos(gha), xp.sin(gha)
    shape = xp.broadcast_shapes(xp.shape(cosgha), xp.shape(cosdec))
    propagation_direction = xp.stack(
        [xp.broadcast_to(component, shape) for component in (cosdec * cosgha, -cosdec * singha, sindec)], axis=-1
    )
    return -(propagation_direction @ location) / SPEED_OF_LIGHT_M_S
