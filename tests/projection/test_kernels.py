"""The array-namespace-generic geometry and projection kernels.

One implementation serves the NumPy projection path and the JAX path, so these tests pin it against
references it does not share code with: ERFA's own geodetic-to-geocentric conversion, astropy's
rotation matrices, and a scalar-by-scalar evaluation of the broadcast forms.
"""

from __future__ import annotations

import math

import erfa
import numpy as np
import pytest
from astropy import coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix

from gwmock_signal._detector_table import builtin_detectors
from gwmock_signal.projection import kernels

CODES = sorted(builtin_detectors())


def _astropy_response(fields: tuple[float, ...]) -> np.ndarray:
    """The response tensor as the package built it with astropy's rotation matrices."""
    longitude, latitude, _, x_azimuth, y_azimuth, x_altitude, y_altitude = fields
    arm = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotation_longitude = rotation_matrix(-longitude * units.rad, "z")
    rotation_latitude = rotation_matrix(-(np.pi / 2.0 - latitude) * units.rad, "y")
    responses = []
    for azimuth, altitude in ((y_azimuth, y_altitude), (x_azimuth, x_altitude)):
        rotation = (
            rotation_longitude
            @ rotation_latitude
            @ rotation_matrix(azimuth * units.rad, "z")
            @ rotation_matrix(-altitude * units.rad, "y")
        )
        responses.append(rotation @ arm @ rotation.T / 2.0)
    return responses[0] - responses[1]


@pytest.mark.parametrize("code", CODES)
def test_the_location_is_erfa_gd2gc_exactly(code):
    """The vertex position is ERFA's WGS-84 conversion of the stored fields, to the last bit."""
    longitude, latitude, elevation = builtin_detectors()[code][:3]
    _, location = kernels.detector_geometry(builtin_detectors()[code], xp=np)

    np.testing.assert_array_equal(location, erfa.gd2gc(1, longitude, latitude, elevation))


@pytest.mark.parametrize("code", CODES)
def test_the_response_is_astropy_rotation_matrices_exactly(code):
    """The response tensor matches the astropy construction it replaces, to the last bit."""
    response, _ = kernels.detector_geometry(builtin_detectors()[code], xp=np)

    np.testing.assert_array_equal(response, _astropy_response(builtin_detectors()[code]))


@pytest.mark.parametrize("code", CODES)
def test_the_location_agrees_with_astropy_to_rounding(code):
    """The position agrees with astropy's ``EarthLocation`` to a few parts in 1e16.

    astropy converts the radians through degrees and wraps negative longitudes, which moves some
    inputs by one ulp before ERFA sees them.
    """
    longitude, latitude, elevation = builtin_detectors()[code][:3]
    reference = coordinates.EarthLocation.from_geodetic(
        longitude * units.rad, latitude * units.rad, height=elevation * units.meter
    )
    _, location = kernels.detector_geometry(builtin_detectors()[code], xp=np)

    expected = np.array([reference.x.value, reference.y.value, reference.z.value])
    assert np.max(np.abs(location - expected)) <= 1e-15 * np.linalg.norm(expected)


def test_broadcast_forms_equal_a_scalar_loop():
    """Arrays of sidereal times, right ascensions and declinations give the per-element scalar values.

    To the last place, not bit for bit: NumPy evaluates ``sin`` and ``cos`` of an array with SIMD
    routines that can differ from the scalar ones by one ulp.
    """
    response, location = kernels.detector_geometry(builtin_detectors()["H1"], xp=np)
    rng = np.random.default_rng(7)
    gmst = rng.uniform(0.0, 2.0 * math.pi, 64)
    right_ascension = rng.uniform(0.0, 2.0 * math.pi, 64)
    declination = rng.uniform(-1.5, 1.5, 64)
    psi = 0.4

    f_plus, f_cross = kernels.antenna_pattern(response, gmst, right_ascension, declination, psi, xp=np)
    delay = kernels.time_delay_from_geocenter(location, gmst, right_ascension, declination, xp=np)

    assert f_plus.shape == f_cross.shape == delay.shape == (64,)
    scalar = np.array(
        [
            [
                *kernels.antenna_pattern(response, gmst[i], right_ascension[i], declination[i], psi, xp=np),
                kernels.time_delay_from_geocenter(location, gmst[i], right_ascension[i], declination[i], xp=np),
            ]
            for i in range(64)
        ]
    )
    # A few ulps of an O(1) antenna factor, and of a delay of at most ~21 ms.
    np.testing.assert_allclose(f_plus, scalar[:, 0], rtol=0, atol=4e-16)
    np.testing.assert_allclose(f_cross, scalar[:, 1], rtol=0, atol=4e-16)
    np.testing.assert_allclose(delay, scalar[:, 2], rtol=0, atol=4 * np.spacing(0.025))


def test_scalar_inputs_give_scalar_results():
    """No stray axes when every angle is a scalar."""
    response, location = kernels.detector_geometry(builtin_detectors()["V1"], xp=np)

    f_plus, f_cross = kernels.antenna_pattern(response, 1.0, 0.5, 0.2, 0.3, xp=np)
    delay = kernels.time_delay_from_geocenter(location, 1.0, 0.5, 0.2, xp=np)

    assert np.shape(f_plus) == np.shape(f_cross) == np.shape(delay) == ()


def test_the_delay_is_bounded_by_the_earth_radius():
    """A physical anchor: the geocentre-to-detector delay never exceeds the vertex radius over c."""
    _, location = kernels.detector_geometry(builtin_detectors()["H1"], xp=np)
    rng = np.random.default_rng(3)
    delay = kernels.time_delay_from_geocenter(
        location, rng.uniform(0, 7, 512), rng.uniform(0, 7, 512), rng.uniform(-1.5, 1.5, 512), xp=np
    )

    bound = np.linalg.norm(location) / 299792458.0
    assert np.all(np.abs(delay) <= bound)
    assert np.max(np.abs(delay)) > 0.9 * bound  # the sweep reaches near-alignment


def test_an_unknown_rotation_axis_is_rejected():
    """Only the two axes the geometry uses are defined."""
    with pytest.raises(ValueError, match="axis"):
        kernels._rotation(np, 0.1, "x")


class TestUnderJax:
    """The same code under ``jax.numpy`` gives the NumPy results and is differentiable."""

    @pytest.fixture(autouse=True)
    def _jax(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = pytest.importorskip("jax.numpy")

    @pytest.mark.parametrize("code", CODES)
    def test_geometry_matches_numpy(self, code):
        """Response and location are bit-identical to the NumPy evaluation."""
        fields = builtin_detectors()[code]
        response, location = kernels.detector_geometry(fields, xp=self.jnp)
        np_response, np_location = kernels.detector_geometry(fields, xp=np)

        np.testing.assert_array_equal(np.asarray(response), np_response)
        np.testing.assert_array_equal(np.asarray(location), np_location)

    def test_projection_factors_match_numpy(self):
        """Antenna pattern and delay agree with NumPy to the last place or two.

        Not bit for bit: XLA and NumPy may evaluate the trigonometric functions and the reductions
        differently.
        """
        response, location = kernels.detector_geometry(builtin_detectors()["L1"], xp=np)
        rng = np.random.default_rng(11)
        gmst, ra, dec = rng.uniform(0, 7, 256), 1.1, -0.4

        np_fp, np_fc = kernels.antenna_pattern(response, gmst, ra, dec, 0.9, xp=np)
        np_delay = kernels.time_delay_from_geocenter(location, gmst, ra, dec, xp=np)
        jfp, jfc = kernels.antenna_pattern(
            self.jnp.asarray(response), self.jnp.asarray(gmst), ra, dec, 0.9, xp=self.jnp
        )
        jdelay = kernels.time_delay_from_geocenter(
            self.jnp.asarray(location), self.jnp.asarray(gmst), ra, dec, xp=self.jnp
        )

        np.testing.assert_allclose(np.asarray(jfp), np_fp, rtol=0, atol=4e-16)
        np.testing.assert_allclose(np.asarray(jfc), np_fc, rtol=0, atol=4e-16)
        np.testing.assert_allclose(np.asarray(jdelay), np_delay, rtol=0, atol=4 * np.spacing(0.025))

    def test_the_geometry_is_differentiable(self):
        """d(F+)/d(latitude) through the geometry agrees with a central finite difference."""
        fields = np.asarray(builtin_detectors()["H1"])

        def f_plus(latitude):
            values = self.jnp.asarray(fields).at[1].set(latitude)
            response, _ = kernels.detector_geometry(tuple(values), xp=self.jnp)
            return kernels.antenna_pattern(response, 1.3, 0.7, 0.2, 0.5, xp=self.jnp)[0]

        latitude = float(fields[1])
        step = 1e-6
        finite = (f_plus(latitude + step) - f_plus(latitude - step)) / (2 * step)

        assert abs(float(self.jax.grad(f_plus)(latitude)) - float(finite)) < 1e-8
