"""The geocentre-to-detector delay, measured against ``lalpulsar.Barycenter``.

The projection combines Greenwich Mean Sidereal Time -- which measures the Earth's rotation from the
mean equinox *of date* -- with a source right ascension. If that right ascension is referred to
J2000, the two quantities live in different frames and the mismatch grows as the equinox precesses:
0.43 degrees in right ascension by 2030, which reached 1.8e-04 s of timing error at worst.

So this is not a tolerance to be widened. It is a frame conversion, and the reference that decides
whether it is right is LAL, which the rest of this pipeline is validated against.

``lalpulsar`` and DE405 ephemeris tables are required, so these skip where either is absent. The
ephemeris files are the same ones the continuous-wave tests use.
"""

from __future__ import annotations

import functools
import itertools
import pathlib

import numpy as np
import pytest

from gwmock_signal.projection.geometry import reconstructed_geometry
from gwmock_signal.projection.sidereal import gmst_rad_astropy, lunisolar_precession_angles, precess_to_epoch

#: Worst-case agreement required with LAL's ``erot``, in seconds.
#:
#: Measured at 8.7e-07 s over the grid below. What remains is nutation, the short-period part of the
#: same motion, which LAL adds as a separate term and this rotation does not model: its amplitude is
#: ~17 arcseconds, which over an Earth radius is ~5e-07 s, so the residue is exactly the size the
#: omission predicts. The bound is set just above the measurement rather than at a round number, so
#: that losing the precession -- which would give 1.8e-04 s -- cannot slip under it.
_MAX_DELAY_DISAGREEMENT_SECONDS = 1.0e-06

_DETECTOR_KEYS = ("H1", "L1", "V1")
_RIGHT_ASCENSIONS = (0.2, 1.1, 3.0, 5.5)
_DECLINATIONS = (-1.2, -0.3, 0.3, 1.2)
#: Spread across the ephemeris span rather than clustered: the error is periodic in sidereal time and
#: grows with time from J2000, so a single epoch could sit anywhere in that structure.
_EPOCHS = (1.0e9, 1.26e9, 1.577491218e9, 1.75e9)


@functools.lru_cache(maxsize=1)
def _ephemeris():
    """Return the parsed DE405 tables, once.

    ``InitBarycenter`` reads and decompresses two files. Called per grid point it turned this module
    into minutes of table parsing; cached it is a few seconds.
    """
    pytest.importorskip("lal", reason="lalsuite is not installed")
    lalpulsar = pytest.importorskip("lalpulsar", reason="lalpulsar is not installed")

    try:
        from ripplegw.waveforms.cw.ephemeris import _cache_dir
    except ImportError:  # pragma: no cover - exercised only without the jax extra
        pytest.skip("ripplegw is needed to locate the DE405 ephemeris tables")

    cache = pathlib.Path(_cache_dir())
    earth_file = cache / "earth00-40-DE405.dat.gz"
    sun_file = cache / "sun00-40-DE405.dat.gz"
    if not earth_file.is_file() or not sun_file.is_file():
        pytest.skip(f"DE405 ephemeris tables are not in {cache}; fetch with ripplegw-fetch-ephemeris")
    return lalpulsar.InitBarycenter(str(earth_file), str(sun_file))


def _lal_erot(prefix: str, right_ascension: float, declination: float, t_gps: float) -> float:
    """Return LAL's geocentre-to-detector delay in seconds, or skip if LAL cannot be asked."""
    lal = pytest.importorskip("lal", reason="lalsuite is not installed")
    lalpulsar = pytest.importorskip("lalpulsar", reason="lalpulsar is not installed")
    ephemeris = _ephemeris()

    index = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
    }[prefix]

    barycenter_input = lalpulsar.BarycenterInput()
    barycenter_input.tgps = lal.LIGOTimeGPS(t_gps)
    barycenter_input.site = lal.CachedDetectors[index]
    # LAL wants the site location in seconds, not metres.
    for axis in range(3):
        barycenter_input.site.location[axis] /= lal.C_SI
    barycenter_input.alpha = right_ascension
    barycenter_input.delta = declination
    barycenter_input.dInv = 0.0

    earth = lalpulsar.EarthState()
    emission = lalpulsar.EmissionTime()
    lalpulsar.BarycenterEarth(earth, barycenter_input.tgps, ephemeris)
    lalpulsar.Barycenter(emission, barycenter_input, earth)
    return float(emission.erot)


def _our_erot(prefix: str, right_ascension: float, declination: float, t_gps: float, *, precess: bool) -> float:
    """Return this package's geocentre-to-detector delay, with the same sign convention as LAL.

    Reproduces what ``_time_delay_from_earth_center_lal`` computes rather than calling it, because
    that function takes the detector-to-geocentre sign and this comparison is clearer with LAL's.
    """
    _, location = reconstructed_geometry(prefix)
    from astropy import constants

    if precess:
        right_ascension, declination = precess_to_epoch(right_ascension, declination, t_gps)
    gha = float(gmst_rad_astropy(t_gps)) - right_ascension
    direction = np.array(
        [np.cos(declination) * np.cos(gha), -np.cos(declination) * np.sin(gha), np.sin(declination)],
        dtype=float,
    )
    return float(np.dot(location, direction) / constants.c.value)


_GRID = tuple(itertools.product(_DETECTOR_KEYS, _RIGHT_ASCENSIONS, _DECLINATIONS, _EPOCHS))


@pytest.fixture(scope="module")
def lal_delays() -> dict[tuple, float]:
    """LAL's answer for every grid point, computed once and shared by the tests below."""
    return {point: _lal_erot(*point) for point in _GRID}


def test_the_delay_agrees_with_lal_across_detectors_sky_and_epoch(lal_delays):
    """The whole point, over 192 combinations rather than one favourable position.

    A single sky position gave 8.6e-05 s before this rotation and 3.4e-08 s after, which reads as a
    2500x improvement. Over the grid the honest figures are 1.8e-04 s and 8.7e-07 s -- 204x. The
    single point was flattering in both directions, which is why the bound is set from the grid.
    """
    disagreements = [
        abs(_our_erot(prefix, ra, dec, gps, precess=True) - lal_delays[(prefix, ra, dec, gps)])
        for prefix, ra, dec, gps in _GRID
    ]

    worst = max(disagreements)
    assert worst < _MAX_DELAY_DISAGREEMENT_SECONDS, (
        f"worst disagreement with LAL is {worst:.3e} s over {len(_GRID)} combinations, above the "
        f"{_MAX_DELAY_DISAGREEMENT_SECONDS:.1e} s this is meant to hold; the residue should be "
        f"nutation-scale and nothing larger"
    )


def test_dropping_the_precession_is_two_orders_worse(lal_delays):
    """Pins the size of what the rotation buys, so a revert cannot pass quietly.

    Without this, the test above could be satisfied by a tolerance nobody had questioned. Measuring
    both sides makes the improvement itself the assertion.
    """
    with_precession = max(
        abs(_our_erot(prefix, ra, dec, gps, precess=True) - lal_delays[point])
        for point in _GRID
        for prefix, ra, dec, gps in (point,)
    )
    without_precession = max(
        abs(_our_erot(prefix, ra, dec, gps, precess=False) - lal_delays[point])
        for point in _GRID
        for prefix, ra, dec, gps in (point,)
    )

    assert without_precession / with_precession > 100.0, (
        f"precession improves worst-case agreement with LAL by only "
        f"{without_precession / with_precession:.0f}x ({without_precession:.3e} s to "
        f"{with_precession:.3e} s); it was 204x when measured, so something has changed"
    )


def test_the_precession_angles_are_lals_own_polynomials():
    """Checked against LAL's computed values, not against the formulas they came from.

    Copying three polynomials from the Explanatory Supplement and asserting they match the
    Supplement proves only that the copy is faithful. What matters is agreeing with the values LAL
    actually uses, so this reads them off ``EarthState``.
    """
    lal = pytest.importorskip("lal", reason="lalsuite is not installed")
    lalpulsar = pytest.importorskip("lalpulsar", reason="lalpulsar is not installed")

    ephemeris = _ephemeris()

    for epoch in _EPOCHS:
        earth = lalpulsar.EarthState()
        lalpulsar.BarycenterEarth(earth, lal.LIGOTimeGPS(epoch), ephemeris)
        zeta_a, z_a, theta_a = lunisolar_precession_angles(epoch)

        # 1e-9 rad is 2e-4 arcseconds, four orders below the nutation this model already omits, so a
        # tighter bound would be pinning floating-point noise in the GPS-to-centuries conversion.
        assert abs(zeta_a - earth.tzeA) < 1e-9, f"zeta_A at {epoch}: {zeta_a} against LAL {earth.tzeA}"
        assert abs(z_a - earth.zA) < 1e-9, f"z_A at {epoch}: {z_a} against LAL {earth.zA}"
        assert abs(theta_a - earth.thetaA) < 1e-9, f"theta_A at {epoch}: {theta_a} against LAL {earth.thetaA}"


def test_precession_at_j2000_is_the_identity():
    """The rotation must vanish where the frames coincide, which is the one case known a priori."""
    from gwmock_signal.projection.sidereal import _GPS_AT_J2000

    right_ascension, declination = precess_to_epoch(1.1, 0.3, _GPS_AT_J2000)

    assert right_ascension == pytest.approx(1.1, abs=1e-12)
    assert declination == pytest.approx(0.3, abs=1e-12)
