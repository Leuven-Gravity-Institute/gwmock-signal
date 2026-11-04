"""The solar-system delay every source shares, and how a transient absorbs it.

Anchored to LALPulsar's own barycentring (``lalpulsar.Barycenter`` at the geocentre) and, for the part a
transient keeps, to an independent ephemeris: Astropy's, which shares no code or tables with the
LALPulsar ones.
"""

from __future__ import annotations

import dataclasses
import gzip
from pathlib import Path

import numpy as np
import pytest

from gwmock_signal.projection import solar_system
from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.network import _solar_timing

EPHEMERIS = solar_system.SolarSystemEphemeris()


@pytest.fixture(scope="module")
def tables():
    """The default tables, or a skip where they can neither be found nor downloaded."""
    try:
        return EPHEMERIS.tables()
    except FileNotFoundError as exc:
        pytest.skip(f"the default ephemeris is unavailable: {exc}")


def test_the_delay_is_lalpulsars_at_the_geocentre(tables):
    """``D`` equals ``deltaT`` of ``lalpulsar.Barycenter`` for a detector at the geocentre, and its rate ``tDot - 1``."""
    lal = pytest.importorskip("lal")
    lalpulsar = pytest.importorskip("lalpulsar")
    earth, sun = tables
    state = lalpulsar.InitBarycenter(earth.source, sun.source)
    rng = np.random.default_rng(0)
    for _ in range(50):
        right_ascension, declination = rng.uniform(0.0, 2.0 * np.pi), np.arcsin(rng.uniform(-1.0, 1.0))
        gps = lal.LIGOTimeGPS(1.0e9 + rng.uniform(0.0, 3.0e8))
        earth_state = lalpulsar.EarthState()
        lalpulsar.BarycenterEarth(earth_state, gps, state)
        given = lalpulsar.BarycenterInput()
        given.site, given.tgps, given.alpha, given.delta, given.dInv = (
            lal.Detector(),
            gps,
            right_ascension,
            declination,
            0.0,
        )
        emitted = lalpulsar.EmissionTime()
        lalpulsar.Barycenter(emitted, given, earth_state)
        value, rate = solar_system.delay(np.array([float(gps)]), right_ascension, declination, EPHEMERIS)
        assert value[0] == pytest.approx(emitted.deltaT, rel=0, abs=1e-12)
        assert rate[0] == pytest.approx(emitted.tDot - 1.0, rel=0, abs=1e-11)


def test_the_span_model_is_the_exact_delay(tables):
    """The model a projection evaluates per sample is the delay itself, over a BNS-long span."""
    del tables
    start = 1.25e9
    model = solar_system.DelayModel.build(start, start + 20000.0, 1.1, -0.3, EPHEMERIS)
    since = np.linspace(0.0, 20000.0, 4001)
    exact, _ = solar_system.delay(start + since, 1.1, -0.3, EPHEMERIS)
    assert np.max(np.abs(model.evaluate(since) - exact)) < 1e-12


def test_the_components_add_up(tables):
    """Roemer plus Einstein minus Shapiro is the whole; each switch removes exactly its own part."""
    del tables
    times = 1.3e9 + np.linspace(0.0, 86400.0, 25)
    whole, _ = solar_system.delay(times, 2.0, 0.4, EPHEMERIS)
    parts = [
        solar_system.delay(
            times, 2.0, 0.4, EPHEMERIS, **{name: name == part for name in ("roemer", "einstein", "shapiro")}
        )[0]
        for part in ("roemer", "einstein", "shapiro")
    ]
    np.testing.assert_allclose(parts[0] + parts[1] + parts[2], whole, rtol=0, atol=1e-12)
    assert np.max(np.abs(parts[0])) > 100.0  # hundreds of seconds of Roemer delay
    assert 1e-4 < np.max(np.abs(parts[1])) < 2e-3  # TDB - TT, about 1.7 ms
    assert 0.0 < np.max(np.abs(parts[2])) < 2e-4  # the Sun's delay, tens of microseconds away from the Sun


def test_a_span_outside_the_tables_is_refused(tables):
    """The default tables cover 2000 to 2040; outside them the delay is refused, as LALPulsar refuses it."""
    del tables
    with pytest.raises(ValueError, match="outside the ephemeris"):
        solar_system.delay(np.array([2.0e9]), 1.0, 0.0, EPHEMERIS)


def test_only_a_standard_name_is_ever_downloaded(tmp_path):
    """A missing path, or a name not of LALPulsar's standard form, is refused without a network request."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        solar_system.resolve_table(tmp_path / "earth00-40-DE405.dat.gz")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        solar_system.resolve_table("my-earth-table.dat")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("0 7200 2\n" + "0 " + "1 " * 9 + "\n", "declares 2 rows"),
        ("0 7200 2\n" + "0 " + "1 " * 9 + "\n" + "7000 " + "1 " * 9 + "\n", "not evenly spaced"),
    ],
)
def test_a_malformed_table_is_refused(tmp_path, body, message):
    """A header that disagrees with its rows, or rows that are not evenly spaced, are refused."""
    path = tmp_path / "earth-custom.dat.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(body)
    with pytest.raises(ValueError, match=message):
        solar_system.read_table.__wrapped__(str(path))


def test_a_transient_keeps_only_how_the_delay_changes_and_it_is_the_earths_acceleration(tables):
    """Absorbed at the reference time: zero there with zero rate, then the Earth's orbital acceleration.

    Anchored to Astropy's ephemeris, which shares no tables with LALPulsar's: over the 4.5 hours a binary
    neutron star spends above 5 Hz in the Einstein Telescope, the delay kept is 2.5 ms here, and the two
    agree on it to 6.4e-8 s. That is LALPulsar's own scheme -- second-order expansion about table entries
    7200 s apart, which misses the Earth's jerk by up to 5e-8 s between entries -- not the ephemerides:
    Astropy's built-in one and JPL's DE432s agree on the same quantity to 9e-9 s.
    """
    del tables
    astropy_time = pytest.importorskip("astropy.time")
    coordinates = pytest.importorskip("astropy.coordinates")
    units = pytest.importorskip("astropy.units")
    reference, span = 1.4e9, 16384.0
    right_ascension, declination = 0.7, 0.5
    roemer_only = ProjectionEffects(einstein_delay=False, shapiro_delay=False)
    timing = _solar_timing(
        roemer_only,
        EPHEMERIS,
        span=(reference - span, reference + 10.0),
        right_ascension=right_ascension,
        declination=declination,
        reference_time=reference,
        barycentric_offset_samples=None,
    )
    offsets = np.linspace(-span, 0.0, 65)
    kept = timing.delay(reference + offsets - timing.model.epoch)
    assert abs(kept[-1]) < 1e-12
    # Zero rate: one second later it has moved only by half the acceleration, ~1e-11 s.
    assert abs(timing.delay(np.array([reference + 1.0 - timing.model.epoch]))[0] - kept[-1]) < 1e-10

    source = solar_system.source_direction(right_ascension, declination)
    positions = coordinates.get_body_barycentric("earth", astropy_time.Time(reference + offsets, format="gps"))
    roemer = positions.xyz.to(units.m).value.T @ source / 299792458.0
    step = 1.0
    around = coordinates.get_body_barycentric(
        "earth", astropy_time.Time(reference + np.array([-step, step]), format="gps")
    )
    rate = np.diff(around.xyz.to(units.m).value.T @ source)[0] / (2.0 * step) / 299792458.0
    independent = roemer - roemer[-1] - rate * offsets
    assert np.max(np.abs(kept)) > 1e-4  # milliseconds over a BNS
    assert np.max(np.abs(kept - independent)) < 1e-7


def test_effects_without_solar_system_delays_build_no_timing():
    """With every solar-system delay off, nothing is evaluated, so any epoch is accepted."""
    effects = dataclasses.replace(ProjectionEffects(), roemer_delay=False, einstein_delay=False, shapiro_delay=False)
    assert not effects.solar_system
    assert (
        _solar_timing(
            effects,
            EPHEMERIS,
            span=(0.0, 1.0),
            right_ascension=0.0,
            declination=0.0,
            reference_time=0.5,
            barycentric_offset_samples=None,
        )
        is None
    )


def test_the_recorded_hashes_are_the_manifests():
    """The package's record of the default tables is the one CI caches by; the two cannot drift apart."""
    manifest = Path(__file__).parents[1] / "data" / "ephemeris.sha256"
    recorded = dict(
        reversed(line.split())
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert recorded == solar_system.RECORDED_SHA256


def _one_row_table(path: Path) -> Path:
    with gzip.open(path, "wt") as handle:
        handle.write("0 7200 1\n" + "0 " + "1 " * 9 + "\n")
    return path


def test_a_cached_default_table_with_other_content_is_refused(tmp_path, monkeypatch):
    """A default table served from the cache but holding anything else is refused, not read."""
    monkeypatch.setenv("GWMOCK_SIGNAL_CACHE_DIR", str(tmp_path))
    (tmp_path / "ephemeris").mkdir()
    _one_row_table(tmp_path / "ephemeris" / "earth00-40-DE405.dat.gz")
    with pytest.raises(ValueError, match=r"not the \w+ recorded for earth00-40-DE405\.dat\.gz"):
        solar_system.read_table.__wrapped__("earth00-40-DE405.dat.gz")


def test_a_default_name_in_the_working_directory_is_still_verified(tmp_path, monkeypatch):
    """A bare default name is the default table wherever it is found, the working directory included."""
    monkeypatch.chdir(tmp_path)
    _one_row_table(tmp_path / "sun00-40-DE405.dat.gz")
    with pytest.raises(ValueError, match=r"not the \w+ recorded for sun00-40-DE405\.dat\.gz"):
        solar_system.read_table.__wrapped__("sun00-40-DE405.dat.gz")


def test_a_table_passed_by_path_is_used_whatever_its_name(tmp_path):
    """A caller's own table is theirs: a default name on a path does not make it the recorded table."""
    table = solar_system.read_table.__wrapped__(str(_one_row_table(tmp_path / "earth00-40-DE405.dat.gz")))
    assert table.position.shape == (1, 3)


def test_a_cache_that_cannot_be_written_is_reported_as_such(tmp_path, monkeypatch):
    """Without a writable cache the error names the cache and what to do, not a raw OS error or the network."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("GWMOCK_SIGNAL_CACHE_DIR", str(blocker))
    with pytest.raises(FileNotFoundError, match="cannot be written to"):
        solar_system.resolve_table("earth00-40-DE405.dat.gz")


def test_the_default_reference_is_the_peak_of_the_unpadded_polarizations():
    """Without ``reference_time`` the delay is absorbed at the polarizations' peak power, as given.

    The peak is found on the input before it is padded against the geocentre delay; found on the padded
    buffer but labelled from the input's start, it came out late by the padding, 87 samples here.
    """
    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.timeseries import TimeSeries

    fs, start = 1024.0, 1.3e9
    t = np.arange(int(64 * fs)) / fs
    envelope = np.exp(-(((t - 40.0) / 4.0) ** 2))
    plus, cross = envelope * np.cos(2 * np.pi * 50.0 * t), envelope * np.sin(2 * np.pi * 50.0 * t)
    polarizations = {
        "plus": TimeSeries(plus, t0=start, sample_rate=fs),
        "cross": TimeSeries(cross, t0=start, sample_rate=fs),
    }
    common = {"right_ascension": 1.0, "declination": 0.2, "polarization_angle": 0.3}
    peak = start + int(np.argmax(plus**2 + cross**2)) / fs
    default = project_polarizations_to_network(polarizations, ["H1"], **common)
    explicit = project_polarizations_to_network(polarizations, ["H1"], reference_time=peak, **common)
    np.testing.assert_array_equal(default["H1"].data, explicit["H1"].data)
