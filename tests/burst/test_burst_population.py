"""Tests for the standard-candle burst population."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from gwmock_signal.burst import BurstSimulator, draw_standard_candle_population
from gwmock_signal.burst.population import (
    STANDARD_CANDLE_FAMILIES,
    STANDARD_CANDLE_LABEL,
    standard_candle_hrss,
)

START = 1_419_724_818.0
YEAR_S = 365.25 * 86400.0


def _draw(**overrides):
    kwargs = {
        "start_time": START,
        "end_time": START + 30 * 86400.0,
        "rate_density": 1.0e12,  # Gpc^-3 yr^-1: about 344 events within 1 Mpc in 30 days
        "energy": 1.0e-8,  # solar masses c^2
        "max_distance": 1.0,  # Mpc
        "rng": np.random.default_rng(2026),
    }
    kwargs.update(overrides)
    return draw_standard_candle_population(**kwargs)


def _isotropic_hrss_reference(energy, distance_mpc, frequency):
    """O3 all-sky search Eq. (2) inverted, with astropy's constants: an independent evaluation."""
    e = (energy * u.M_sun * const.c**2).to(u.J)
    r = (distance_mpc * u.Mpc).to(u.m)
    f = frequency * u.Hz
    hrss2 = e * const.G / (math.pi**2 * const.c**3 * r**2 * f**2)
    return math.sqrt(hrss2.to(u.Hz**-1).value)


# ---------------------------------------------------------------- the energy-to-hrss relation


@pytest.mark.parametrize("frequency", [70.0, 235.0, 1304.0])
def test_isotropic_hrss_matches_the_o3_relation(frequency):
    """A white-noise burst's hrss is Eq. (2) of the O3 all-sky search solved for hrss."""
    hrss = standard_candle_hrss(energy=1e-2, distance=10.0, frequency=frequency, inclination=0.3, elliptical=False)
    assert hrss == pytest.approx(_isotropic_hrss_reference(1e-2, 10.0, frequency), rel=1e-12, abs=0.0)


def test_elliptical_hrss_face_on_carries_the_rotating_source_factor():
    """Face on, E_rot = (2/5) E_iso(hrss), so hrss is sqrt(5/2) times the isotropic value."""
    hrss = standard_candle_hrss(energy=1e-2, distance=10.0, frequency=235.0, inclination=0.0, elliptical=True)
    expected = math.sqrt(2.5) * _isotropic_hrss_reference(1e-2, 10.0, 235.0)
    assert hrss == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_elliptical_hrss_follows_the_rotating_pattern():
    """hrss^2 follows ((1 + cos^2 i) / 2)^2 + cos^2 i, relative to face on (where it is 2)."""
    face_on = standard_candle_hrss(energy=1.0, distance=1.0, frequency=100.0, inclination=0.0, elliptical=True)
    for iota in (0.4, 1.0, math.pi / 2, 2.5, math.pi):
        c = math.cos(iota)
        pattern = ((1 + c**2) / 2) ** 2 + c**2
        hrss = standard_candle_hrss(energy=1.0, distance=1.0, frequency=100.0, inclination=iota, elliptical=True)
        assert (hrss / face_on) ** 2 == pytest.approx(pattern / 2, rel=1e-12, abs=0.0)


def test_elliptical_hrss_radiates_the_stated_energy_over_the_sphere():
    """The average of hrss^2 over isotropic orientations is the isotropic hrss^2: energy is conserved."""
    cos_iota, weights = np.polynomial.legendre.leggauss(16)  # exact for this polynomial in cos(iota)
    hrss2 = [
        standard_candle_hrss(energy=1e-3, distance=3.0, frequency=554.0, inclination=math.acos(c), elliptical=True) ** 2
        for c in cos_iota
    ]
    mean = float(np.dot(weights, hrss2)) / 2.0
    isotropic = _isotropic_hrss_reference(1e-3, 3.0, 554.0) ** 2
    assert mean == pytest.approx(isotropic, rel=1e-12, abs=0.0)


def test_hrss_scales_as_the_inverse_distance_and_root_energy():
    """Hrss is proportional to sqrt(E) / r."""
    base = standard_candle_hrss(energy=1e-4, distance=2.0, frequency=300.0, inclination=1.0, elliptical=True)
    assert standard_candle_hrss(
        energy=4e-4, distance=4.0, frequency=300.0, inclination=1.0, elliptical=True
    ) == pytest.approx(base, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"energy": 0.0}, "energy"),
        ({"energy": math.inf}, "energy"),
        ({"distance": -1.0}, "distance"),
        ({"frequency": math.nan}, "frequency"),
        ({"inclination": math.inf}, "inclination"),
    ],
)
def test_hrss_rejects_invalid_arguments(overrides, match):
    """Non-positive or non-finite inputs are refused."""
    kwargs = {"energy": 1e-2, "distance": 1.0, "frequency": 100.0, "inclination": 0.0, "elliptical": True}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        standard_candle_hrss(**kwargs)


# ---------------------------------------------------------------- the population


def test_population_count_is_poisson_with_the_volume_rate_mean():
    """The number of events is Poisson with mean R * (4/3) pi d^3 * T."""
    mean = 1.0e12 * 1e-9 * (4.0 / 3.0) * math.pi * 1.0**3 * (30 * 86400.0 / YEAR_S)
    assert mean == pytest.approx(344.0, rel=1e-3)
    draws = 200
    counts = [len(_draw(rng=np.random.default_rng(seed))) for seed in range(draws)]
    # Mean within 5 standard errors; a Poisson variance equals the mean (its sample variance has
    # relative standard deviation sqrt(2 / draws) = 0.1, so 0.5 is 5 sigma).
    assert abs(np.mean(counts) - mean) < 5 * math.sqrt(mean / draws)
    assert np.var(counts, ddof=1) == pytest.approx(mean, rel=0.5)


def test_population_distances_are_uniform_in_volume_and_times_in_the_span():
    """(d / d_max)^3 and the times are uniform; events come in time order."""
    events = _draw(rate_density=1.0e13)
    n = len(events)
    assert n > 2000
    d3 = np.sort([(e["distance"] / 1.0) ** 3 for e in events])
    times = np.array([e["peak_time"] for e in events])
    assert np.all(np.diff(times) >= 0)
    t = np.sort((times - START) / (30 * 86400.0))
    grid = np.arange(1, n + 1) / n
    # Kolmogorov distance well inside the 99.9% critical value 1.95 / sqrt(n).
    assert np.max(np.abs(d3 - grid)) < 1.95 / math.sqrt(n)
    assert np.max(np.abs(t - grid)) < 1.95 / math.sqrt(n)
    assert min(d3) > 0
    assert max(d3) <= 1.0
    assert times.min() >= START
    assert times.max() < START + 30 * 86400.0


def test_population_hrss_follows_from_each_events_energy_distance_and_orientation():
    """Every event's hrss is standard_candle_hrss of its own energy, distance, frequency and inclination."""
    for event in _draw():
        model = event["burst_model"]
        if model == "sine_gaussian":
            frequency, elliptical = event["frequency"], True
        else:
            frequency, elliptical = event["low_frequency"] + 0.5 * event["bandwidth"], False
        expected = standard_candle_hrss(
            energy=event["energy"],
            distance=event["distance"],
            frequency=frequency,
            inclination=event["inclination"],
            elliptical=elliptical,
        )
        assert event["hrss"] == expected
        assert event["energy"] == 1.0e-8


def test_population_draws_each_family_and_labels_every_event():
    """Every family appears; sine-Gaussian frequencies are log-uniform in range; events are labelled."""
    events = _draw()
    keys = {(e["burst_model"], e.get("quality_factor"), e.get("low_frequency")) for e in events}
    assert len(keys) == len(STANDARD_CANDLE_FAMILIES)
    assert all(e["injection_set"] == STANDARD_CANDLE_LABEL for e in events)
    freqs = np.log10([e["frequency"] for e in events if e["burst_model"] == "sine_gaussian"])
    assert freqs.min() >= math.log10(30.0)
    assert freqs.max() <= math.log10(1800.0)
    for event in events:
        assert ("seed" in event) == (event["burst_model"] == "white_noise_burst")
        assert -math.pi / 2 <= event["declination"] <= math.pi / 2
        assert 0 <= event["inclination"] <= math.pi


def test_population_is_reproducible():
    """The same seed gives the same events."""
    assert _draw() == _draw()


def test_population_may_be_empty():
    """A rate that expects almost nothing can draw nothing, and returns an empty list."""
    assert _draw(rate_density=1e-30) == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"end_time": START}, "end_time"),
        ({"rate_density": -1.0}, "rate_density"),
        ({"rate_density": math.nan}, "rate_density"),
        ({"energy": 0.0}, "energy"),
        ({"max_distance": 0.0}, "max_distance"),
        ({"max_distance": 150.0}, "100 Mpc"),
        ({"families": ()}, "families"),
        ({"families": ({"quality_factor": 3.0},)}, "burst_model"),
        ({"families": ({"burst_model": "gaussian", "duration": 1e-3},)}, "gaussian"),
        ({"families": ({"burst_model": "model_a"},)}, "model_a"),
        ({"sine_gaussian_frequency_range": (0.0, 10.0)}, "sine_gaussian_frequency_range"),
    ],
)
def test_population_rejects_invalid_configuration(overrides, match):
    """Invalid spans, rates, energies, distances and families are refused."""
    with pytest.raises(ValueError, match=match):
        _draw(**overrides)


@pytest.mark.requires_lalsuite
def test_population_events_simulate_at_their_drawn_hrss():
    """The simulator accepts the events and injects the drawn hrss."""
    events = _draw()
    simulator = BurstSimulator()
    for model in ("sine_gaussian", "white_noise_burst"):
        event = next(e for e in events if e["burst_model"] == model)
        hp, hc = simulator.generate_polarizations(event, sampling_frequency=8192.0, minimum_frequency=0.0)
        dt = 1.0 / 8192.0
        measured = math.sqrt(float(np.sum(np.asarray(hp.value) ** 2 + np.asarray(hc.value) ** 2)) * dt)
        assert measured == pytest.approx(event["hrss"], rel=1e-9, abs=0.0)
