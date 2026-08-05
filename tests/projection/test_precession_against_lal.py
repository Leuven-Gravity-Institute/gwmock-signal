"""The geocentre-to-detector delay, measured against ``lalpulsar.Barycenter``.

The projection combines Greenwich Mean Sidereal Time -- which measures the Earth's rotation from the
mean equinox *of date* -- with a source right ascension. If that right ascension is referred to
J2000, the two live in different frames and the mismatch grows as the equinox precesses: 0.43 degrees
in right ascension by 2030, reaching 1.8e-04 s of timing error at worst.

**LAL does this two different ways, and these tests pin only one of them.** ``lalpulsar``'s
``XLALBarycenter`` applies lunisolar precession and nutation; ``lal``'s
``XLALTimeDelayFromEarthCenter`` and ``XLALComputeDetAMResponse`` do not, computing ``gha = gmst -
ra`` from the right ascension as given. The compact-binary world follows the second convention --
Bilby, PyCBC, LALInference and GstLAL all go through those two functions -- so a CBC injection that
precessed would be recovered 0.43 degrees away from where it was injected. The continuous-wave world
follows the first, because the SSB-to-geocentre phase it is added to comes from a barycentering
routine that precesses.

So ``precess_source_direction=True`` is not simply the more correct setting; it is the setting that
matches ``lalpulsar``, and this module measures how well. What pins the other convention is
``test_matches_pycbc_reference_on_gw150914_like_case`` in ``test_projection_network.py``, and the two
are supposed to disagree.

One test here deliberately does **not** use LAL:
``test_the_rotation_agrees_with_astropys_independent_precession_model`` compares the rotation against
Astropy's IAU 2006 frame transform. Everything else in this module measures fidelity to LAL, which
cannot distinguish a faithful port from a shared error.

``lalpulsar`` and DE405 ephemeris tables are required, so most of these skip where either is absent.
The ephemeris files are the same ones the continuous-wave tests use.
"""

from __future__ import annotations

import functools
import itertools
import pathlib

import numpy as np
import pytest
from astropy.time import Time

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

    Read as a statement about ``lalpulsar`` only. The ``precess=False`` figure is *also* the
    agreement with LAL's compact-binary convention, where it is not an error at all.
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


def test_each_convention_reproduces_its_own_lal_function():
    """Anchor both settings against LAL exactly, at the level of the delay itself.

    Not through a projected waveform: enabling precession changes the antenna coefficients as well
    as the delay, so any waveform-level comparison mixes the two. An earlier version of this test
    read the timing difference off a cross-correlation peak and came out 7.8% from LAL's own figure,
    which I attributed to the estimator's resolution -- an attribution I could not separate from
    antenna-coefficient contamination. Comparing the delays directly removes both effects, and the
    tolerance goes from 25% of the effect to the size of a named omission.

    The two references are LAL's own two functions, in the sense the projection applies:
    ``lal.TimeDelayFromEarthCenter`` for the compact-binary convention, and
    ``-erot`` from ``lalpulsar.Barycenter`` for the continuous-wave one.
    """
    lal = pytest.importorskip("lal", reason="lalsuite is not installed")

    from gwmock_signal.projection.network import _time_delay_from_earth_center_lal

    prefix, right_ascension, declination, t_gps = "H1", 1.1, 0.3, 1.75e9
    detector = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

    cbc_reference = lal.TimeDelayFromEarthCenter(detector.location, right_ascension, declination, t_gps)
    # `erot` is the opposite sense to `TimeDelayFromEarthCenter`. Getting this backwards leaves the
    # magnitudes agreeing and the sign inverted, which several weaker assertions would not catch.
    cw_reference = -_lal_erot(prefix, right_ascension, declination, t_gps)
    separation = cw_reference - cbc_reference
    assert abs(separation) > 1.0e-05, (
        f"LAL's two conventions differ by only {separation:.3e} s at this epoch, so the assertions "
        f"below cannot distinguish them; move the epoch further from J2000"
    )

    unprecessed = _time_delay_from_earth_center_lal(
        prefix, right_ascension=right_ascension, declination=declination, t_gps=t_gps
    )
    ra_of_date, dec_of_date = precess_to_epoch(right_ascension, declination, t_gps)
    precessed = _time_delay_from_earth_center_lal(
        prefix, right_ascension=float(ra_of_date), declination=float(dec_of_date), t_gps=t_gps
    )

    # The two bounds differ, and the difference is the point.
    #
    # Compact-binary side: 4.0e-08 s, measured. This package takes sidereal time from Astropy where
    # LAL uses its own implementation, and that is the whole residue -- three orders below the
    # 7.4e-05 s separating the conventions. Anything larger would be a geometry difference.
    assert abs(unprecessed - cbc_reference) < 5.0e-08, (
        f"without precession the delay is {unprecessed - cbc_reference:.3e} s from LAL's "
        f"compact-binary convention, which it is supposed to reproduce exactly up to sidereal time"
    )
    # Continuous-wave side: 5.4e-07 s, measured, and an order looser *for a stated reason* rather
    # than because the first bound would not fit. `Barycenter` applies precession **and nutation**;
    # `precess_to_epoch` applies precession only. Nutation's amplitude is ~17 arcseconds, which over
    # an Earth radius is ~5e-07 s -- so the residue is the size the omission predicts, and it is the
    # same quantity `_MAX_DELAY_DISAGREEMENT_SECONDS` bounds over the grid above.
    assert abs(precessed - cw_reference) < _MAX_DELAY_DISAGREEMENT_SECONDS, (
        f"with precession the delay is {precessed - cw_reference:.3e} s from LAL's continuous-wave "
        f"convention, above the {_MAX_DELAY_DISAGREEMENT_SECONDS:.1e} s omitted nutation accounts "
        f"for; the residue should be nutation-scale and nothing larger"
    )


def test_the_flag_selects_between_lals_two_conventions():
    """``precess_source_direction`` must route to the convention it names, through the projection.

    Exact rather than approximate: setting the flag must give the *same array* as pre-rotating the
    coordinates by hand and leaving the flag off. That pins the routing without needing to model
    what the rotation does to the output, which the test above measures separately.

    Worth having because the rotation is only useful if the switch reaches the projection, and an
    earlier revision of this work left a second entry point on the old frame.
    """
    pytest.importorskip("lal", reason="lalsuite is not installed")
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network

    prefix, right_ascension, declination = "H1", 1.1, 0.3
    t_gps, sampling_frequency, n_samples = 1.75e9, 4096.0, 1024
    times = np.arange(n_samples) / sampling_frequency
    taper = np.hanning(n_samples)
    polarizations = {
        "plus": TimeSeries(np.sin(2 * np.pi * 120.0 * times) * taper, t0=t_gps, sample_rate=sampling_frequency),
        "cross": TimeSeries(np.cos(2 * np.pi * 120.0 * times) * taper, t0=t_gps, sample_rate=sampling_frequency),
    }

    def project(*, ra, dec, precess):
        return project_polarizations_to_network(
            polarizations,
            [prefix],
            right_ascension=ra,
            declination=dec,
            polarization_angle=0.0,
            earth_rotation=False,
            precess_source_direction=precess,
        )[prefix].value

    ra_of_date, dec_of_date = precess_to_epoch(right_ascension, declination, t_gps)
    routed = project(ra=right_ascension, dec=declination, precess=True)
    by_hand = project(ra=float(ra_of_date), dec=float(dec_of_date), precess=False)
    untouched = project(ra=right_ascension, dec=declination, precess=False)

    scale = np.max(np.abs(untouched))
    assert scale > 0.0
    # This branch evaluates everything at one reference time, so the two routes differ only in
    # float64 round-off on the same arithmetic; measured at 0.0 exactly, bounded at 1e-14 rather
    # than asserting equality in case the two orderings ever stop being identical.
    assert np.max(np.abs(routed - by_hand)) < 1e-14 * scale, (
        "setting precess_source_direction did not reproduce pre-rotating the coordinates by hand, "
        "so the flag is not applying the rotation the tests above validated"
    )
    # And it must actually do something: measured at 6.8e-03 of peak here.
    assert np.max(np.abs(routed - untouched)) > 1e-04 * scale, (
        "precess_source_direction=True gave the same result as False, so the flag is a no-op"
    )


def test_the_projection_defaults_to_the_compact_binary_convention():
    """The default must be the convention compact-binary searches use.

    Pinned separately because it is the load-bearing part of the design and nothing else asserts it:
    every compact-binary path -- including the batched device one -- relies on the default rather
    than passing the argument, so flipping it would silently move every CBC injection ~0.43 degrees
    in right ascension while the tests above, which pass the flag explicitly, all stayed green.
    """
    import inspect

    from gwmock_signal.projection.network import project_polarizations_to_network

    default = inspect.signature(project_polarizations_to_network).parameters["precess_source_direction"].default
    assert default is False, (
        f"precess_source_direction defaults to {default!r}; it must default to False, the convention "
        f"lal.XLALTimeDelayFromEarthCenter and every compact-binary search use"
    )


def test_the_host_and_device_rotating_paths_agree_when_precessing():
    """The two ``earth_rotation=True`` implementations must agree with the rotation switched on.

    Closes a gap rather than repeating a check. Every other device-versus-host comparison in this
    suite pins ``right_ascension_rate = declination_rate = 0``, and every continuous-wave test runs
    the device backend, because ``ContinuousWaveSimulator`` defaults to it. So the host branch's
    per-sample position -- ``right_ascension_array`` and ``declination_array`` in
    ``project_polarizations_to_network`` -- was never evaluated with a non-zero rate by anything, and
    a regression in it would have passed the whole suite.

    It also pins the device branch's non-zero-rate path against an independent implementation. The
    continuous-wave coherence tests reach that path, but they compare the device against itself at
    different segmentations, so a rate applied wrongly but *consistently* survives them.
    """
    pytest.importorskip("jax", reason="jax not installed")
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network

    detectors = ["H1", "V1"]
    t_gps, sampling_frequency, n_samples = 1.75e9, 512.0, 4096
    times = np.arange(n_samples) / sampling_frequency
    polarizations = {
        "plus": TimeSeries(1e-24 * np.sin(2 * np.pi * 40.0 * times), t0=t_gps, sample_rate=sampling_frequency),
        "cross": TimeSeries(1e-24 * np.cos(2 * np.pi * 40.0 * times), t0=t_gps, sample_rate=sampling_frequency),
    }

    projected = {
        backend: project_polarizations_to_network(
            polarizations,
            detectors,
            right_ascension=1.1,
            declination=0.3,
            polarization_angle=0.4,
            earth_rotation=True,
            precess_source_direction=True,
            backend=backend,
        )
        for backend in ("numpy", "jax")
    }

    for name in detectors:
        host = projected["numpy"][name].value
        device = projected["jax"][name].value
        scale = float(np.max(np.abs(host)))
        assert scale > 0.0, f"{name} has a null response, so a relative bound means nothing"
        # atol=0.0 on purpose: this is strain of order 1e-24, and the default absolute tolerance of
        # any allclose-style comparison would make two arbitrary arrays of that size compare equal.
        residual = float(np.sqrt(np.mean((device - host) ** 2))) / scale
        # 6e-14 measured. Round-off from the same arithmetic associated differently -- one resampling
        # kernel and one sidereal model across both paths -- and eight orders below the 4.4e-03 of
        # peak that getting the sky frame wrong costs.
        assert residual < 1e-12, (
            f"{name}: host and device rotating projections differ by {residual:.3e} of peak with "
            f"precession on, which is too large to be round-off; the per-sample position has drifted "
            f"between the two implementations"
        )


def test_the_rotation_agrees_with_astropys_independent_precession_model():
    """Anchor the rotation against something that is not LAL.

    Every other test here compares against LAL, which makes them a check that the port is faithful
    rather than a check that the model is right. Astropy's ICRS-to-FK5-mean-equinox-of-date transform
    is an independent implementation of the same physical rotation, from the IAU 2006 model rather
    than the 1976 series LAL uses, and with no nutation -- so it is directly comparable.

    Over the grid the two agree to **4.2e-07 rad**, 0.087 arcseconds, worth 9.0e-09 s of delay over
    an Earth radius. That is two orders below the 8.7e-07 s nutation residue and four below the
    1.8e-04 s the rotation removes, so it confirms the rotation is a faithful frame-of-date
    conversion and not a transcription error in three polynomials.

    The disagreement grows monotonically with time from J2000 across the grid epochs -- 1.2e-07 rad
    at GPS 1.0e9 to 4.2e-07 at 1.75e9 -- which is the signature of two different precession models
    diverging, not of a bug. A constant offset, or one that did not grow, would be the worrying shape.

    What this does **not** anchor is the nutation that makes up the residue against ``lalpulsar``;
    that would need an apparent-place computation, and it remains LAL-only.
    """
    import astropy.units as u
    from astropy.coordinates import FK5, SkyCoord

    worst = 0.0
    per_epoch: dict[float, float] = {}
    for right_ascension, declination, t_gps in itertools.product(_RIGHT_ASCENSIONS, _DECLINATIONS, _EPOCHS):
        equinox = Time(t_gps, format="gps", scale="utc")
        reference = SkyCoord(ra=right_ascension * u.rad, dec=declination * u.rad, frame="icrs").transform_to(
            FK5(equinox=equinox)
        )
        ra_of_date, dec_of_date = precess_to_epoch(right_ascension, declination, t_gps)
        # Compared as an angular separation rather than per-coordinate: near the poles a right
        # ascension difference is not an angle on the sky, and a per-coordinate bound there would be
        # either vacuous or impossible.
        separation = (
            SkyCoord(ra=reference.ra, dec=reference.dec, frame="icrs")
            .separation(SkyCoord(ra=ra_of_date * u.rad, dec=dec_of_date * u.rad, frame="icrs"))
            .rad
        )
        worst = max(worst, separation)
        per_epoch[t_gps] = max(per_epoch.get(t_gps, 0.0), separation)

    # 4.2e-07 rad measured; the bound sits just above it rather than at a round number, so that a
    # regression in the polynomials cannot slip under it.
    assert worst < 6.0e-07, (
        f"the rotation differs from Astropy's independent precession model by {worst:.3e} rad over "
        f"{len(_RIGHT_ASCENSIONS) * len(_DECLINATIONS) * len(_EPOCHS)} combinations, more than the "
        f"1976-versus-2006 model difference accounts for"
    )
    # The shape matters as much as the size: two precession models diverge with time from J2000.
    # A disagreement that did not grow would point at a transcription error instead.
    ordered = [per_epoch[epoch] for epoch in sorted(per_epoch)]
    assert ordered == sorted(ordered), (
        f"the disagreement with Astropy does not grow with time from J2000 ({ordered}), which is "
        f"not how two precession models differ; suspect a constant error rather than a model gap"
    )
