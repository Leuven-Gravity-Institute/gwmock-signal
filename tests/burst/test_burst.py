"""Tests for the burst source: waveforms, file loader, simulator and test injection set."""

from __future__ import annotations

import math

import numpy as np
import pytest

import gwmock_signal
from gwmock_signal.burst import DRAFT_BURST_FAMILIES, BurstSimulator, draw_burst_injection_set
from gwmock_signal.burst.waveforms import gaussian, read_waveform_file, sine_gaussian, white_noise_burst

FS = 8192.0
PEAK = 1_000_000_000.0
SKY = {"right_ascension": 1.0, "declination": 0.3, "polarization_angle": 0.2}


def _rss(*series) -> float:
    """Root-sum-square strain, computed independently of LAL."""
    return math.sqrt(sum(float(np.sum(np.asarray(s.value) ** 2)) for s in series) / FS)


def _sg(**overrides):
    params = {"peak_time": PEAK, "hrss": 1e-21, "frequency": 235.0, "quality_factor": 9.0, "inclination": 0.0}
    params.update(overrides)
    return params


def _wnb(**overrides):
    params = {
        "peak_time": PEAK,
        "hrss": 1e-21,
        "low_frequency": 250.0,
        "bandwidth": 100.0,
        "duration": 0.1,
        "inclination": 0.4,
        "seed": 7,
    }
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Ad hoc waveforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generator", "params"),
    [
        (sine_gaussian, _sg()),
        (gaussian, {"peak_time": PEAK, "hrss": 1e-21, "duration": 2.5e-3}),
        (white_noise_burst, _wnb()),
    ],
)
def test_ad_hoc_families_are_normalised_to_the_requested_hrss(generator, params):
    """Every ad hoc family carries exactly the hrss the event asks for, measured without LAL."""
    hp, hc = generator(params, FS)
    assert _rss(hp, hc) == pytest.approx(params["hrss"], rel=1e-6)


@pytest.mark.parametrize("generator", [sine_gaussian, gaussian])
def test_waveform_peaks_at_peak_time(generator):
    """The waveform's envelope peaks on the event's peak_time, not at LAL's t = 0."""
    params = _sg() if generator is sine_gaussian else {"peak_time": PEAK, "hrss": 1e-21, "duration": 2.5e-3}
    hp, hc = generator(params, FS)
    envelope = np.asarray(hp.value) ** 2 + np.asarray(hc.value) ** 2
    peak_time = float(hp.times.value[np.argmax(envelope)])
    assert peak_time == pytest.approx(PEAK, abs=0.5 / FS)


@pytest.mark.parametrize("inclination", [0.0, 0.5, 1.0, math.pi / 2, 2.2, math.pi])
def test_sine_gaussian_polarization_ratio_follows_inclination(inclination):
    """The cross/plus amplitude ratio is the rotating-source ratio for the inclination."""
    # A rotating source has h_x / h_+ = 2 cos(iota) / (1 + cos^2 iota); Q = 100 makes the envelope
    # long enough that the sin/cos carriers carry equal power, so the rss ratio is the amplitude ratio.
    hp, hc = sine_gaussian(_sg(quality_factor=100.0, inclination=inclination), FS)
    cos_iota = math.cos(inclination)
    expected = 2 * abs(cos_iota) / (1 + cos_iota**2)
    assert _rss(hc) / _rss(hp) == pytest.approx(expected, abs=1e-6)


def test_inclination_beyond_half_pi_flips_the_cross_handedness():
    """Viewing from behind (iota -> pi - iota) keeps h_+ and negates h_x."""
    hp_front, hc_front = sine_gaussian(_sg(inclination=0.6), FS)
    hp_back, hc_back = sine_gaussian(_sg(inclination=math.pi - 0.6), FS)
    np.testing.assert_allclose(hp_back.value, hp_front.value, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(hc_back.value, -hc_front.value, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(hc_front.value)) > 0


def test_gaussian_is_linearly_polarized():
    """Gaussian pulses have no cross polarization."""
    _, hc = gaussian({"peak_time": PEAK, "hrss": 1e-21, "duration": 1e-4}, FS)
    assert not np.any(hc.value)


def test_white_noise_burst_is_reproducible_from_its_seed():
    """The seed fixes the noise realisation, and a different seed changes it."""
    a = white_noise_burst(_wnb(seed=7), FS)
    b = white_noise_burst(_wnb(seed=7), FS)
    c = white_noise_burst(_wnb(seed=8), FS)
    np.testing.assert_array_equal(a[0].value, b[0].value)
    assert not np.allclose(a[0].value, c[0].value, rtol=0.0, atol=0.0)


def test_white_noise_burst_power_is_centred_in_its_band():
    """The band is placed by its lower edge: power centres on low_frequency + bandwidth / 2."""
    hp, hc = white_noise_burst(_wnb(low_frequency=750.0, bandwidth=100.0), FS)
    power = np.abs(np.fft.rfft(hp.value)) ** 2 + np.abs(np.fft.rfft(hc.value)) ** 2
    freqs = np.fft.rfftfreq(len(hp), 1 / FS)
    centroid = float(np.sum(freqs * power) / np.sum(power))
    assert 760.0 < centroid < 840.0


@pytest.mark.parametrize(
    ("generator", "params", "match"),
    [
        (sine_gaussian, _sg(frequency=4096.0), "Nyquist"),
        (sine_gaussian, _sg(hrss=-1.0), "hrss"),
        (sine_gaussian, {"peak_time": PEAK, "hrss": 1e-21}, "frequency"),
        (white_noise_burst, _wnb(low_frequency=4000.0), "Nyquist"),
        (white_noise_burst, _wnb(duration=1e-3, bandwidth=10.0), "2/pi"),
        (white_noise_burst, _wnb(seed=-1), "seed"),
        (gaussian, {"peak_time": PEAK, "hrss": 1e-21, "duration": 0.0}, "duration"),
    ],
)
def test_invalid_burst_parameters_are_refused(generator, params, match):
    """Unrepresentable or malformed parameters raise a ValueError naming the problem."""
    with pytest.raises(ValueError, match=match):
        generator(params, FS)


# ---------------------------------------------------------------------------
# File waveforms
# ---------------------------------------------------------------------------


def _write_waveform(path, rate=16384.0, start=-0.05, n=2048, frequency=300.0):
    times = start + np.arange(n) / rate
    plus = 1e-20 * np.sin(2 * np.pi * frequency * times) * np.exp(-((times / 0.01) ** 2))
    cross = 0.5 * plus
    np.savetxt(path, np.column_stack([times, plus, cross]))
    return times, plus, cross


def test_file_waveform_is_scaled_placed_and_resampled(tmp_path):
    """A file waveform is scaled by distance, placed at peak_time and resampled to the target rate."""
    path = tmp_path / "model.txt"
    times, plus, _ = _write_waveform(path)
    simulator = BurstSimulator(waveform_files={"ccsn": path}, reference_distance=0.01)

    hp, hc = simulator.generate_polarizations(
        {"burst_model": "ccsn", "peak_time": PEAK, "distance": 0.04, **SKY}, FS, 10.0
    )

    assert hp.sample_rate.value == FS
    assert float(hp.t0.value) == pytest.approx(PEAK + times[0], abs=1e-6)
    # 16384 -> 8192 Hz keeps every other sample of a 300 Hz signal up to the anti-alias filter.
    np.testing.assert_allclose(hp.value, 0.25 * plus[::2], rtol=0.0, atol=1e-3 * np.max(np.abs(plus)))
    np.testing.assert_allclose(hc.value, 0.5 * hp.value, rtol=1e-12, atol=0.0)


def test_file_waveform_at_the_target_rate_is_used_unchanged(tmp_path):
    """A file already at the target rate is only scaled."""
    path = tmp_path / "model.txt"
    _, plus, _ = _write_waveform(path, rate=FS)
    simulator = BurstSimulator(waveform_files={"ccsn": path}, reference_distance=1.0)
    hp, _ = simulator.generate_polarizations({"burst_model": "ccsn", "peak_time": PEAK, "distance": 2.0}, FS, 10.0)
    np.testing.assert_allclose(hp.value, 0.5 * plus, rtol=1e-12, atol=0.0)


def test_waveform_file_must_be_uniformly_sampled(tmp_path):
    """A non-uniform time column is refused."""
    path = tmp_path / "bad.txt"
    np.savetxt(path, np.array([[0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [0.3, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="uniformly sampled"):
        read_waveform_file(path)


def test_waveform_file_must_have_three_columns(tmp_path):
    """A file without the t, h_+, h_x columns is refused."""
    path = tmp_path / "bad.txt"
    np.savetxt(path, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="three columns"):
        read_waveform_file(path)


def test_file_models_need_a_reference_distance(tmp_path):
    """Files cannot be scaled without the distance their strains are given at."""
    with pytest.raises(ValueError, match="reference_distance"):
        BurstSimulator(waveform_files={"ccsn": tmp_path / "x.txt"})


def test_file_models_may_not_shadow_ad_hoc_names(tmp_path):
    """A file model cannot silently replace an ad hoc family."""
    with pytest.raises(ValueError, match="sine_gaussian"):
        BurstSimulator(waveform_files={"sine_gaussian": tmp_path / "x.txt"}, reference_distance=1.0)


def test_file_model_needs_a_distance(tmp_path):
    """A file event without a distance is refused."""
    path = tmp_path / "model.txt"
    _write_waveform(path)
    simulator = BurstSimulator(waveform_files={"ccsn": path}, reference_distance=0.01)
    with pytest.raises(ValueError, match="distance"):
        simulator.generate_polarizations({"burst_model": "ccsn", "peak_time": PEAK}, FS, 10.0)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_burst_is_a_registered_source_type():
    """The burst simulator is reachable through the source-type registry and the package root."""
    assert gwmock_signal.resolve_simulator_backend("burst") is BurstSimulator
    assert "burst" in gwmock_signal.list_registered_source_types()
    assert gwmock_signal.BurstSimulator is BurstSimulator


def test_unknown_burst_model_is_refused():
    """An unknown burst_model raises rather than producing nothing."""
    with pytest.raises(ValueError, match="Unknown burst_model"):
        BurstSimulator().simulate(
            {"burst_model": "ringdown", "peak_time": PEAK, **SKY}, ["H1"], sampling_frequency=FS, minimum_frequency=10.0
        )


def test_simulate_projects_a_burst_onto_each_detector():
    """The full pipeline returns one finite channel per detector, peaking near peak_time."""
    event = {"burst_model": "sine_gaussian", **_sg(), **SKY}
    stack = BurstSimulator().simulate(event, ["H1", "L1"], sampling_frequency=FS, minimum_frequency=10.0)
    assert list(stack.detector_names) == ["H1", "L1"]
    for name in ("H1", "L1"):
        strain = stack[name]
        assert np.all(np.isfinite(strain.value))
        peak = float(strain.times.value[np.argmax(np.abs(strain.value))])
        # The light-travel delay from the geocentre is at most ~21 ms; the carrier adds half a cycle.
        assert abs(peak - PEAK) < 0.025


# ---------------------------------------------------------------------------
# Test injection set
# ---------------------------------------------------------------------------


def _draw(**overrides):
    kwargs = {
        "start_time": PEAK,
        "end_time": PEAK + 300_000.0,
        "interval": 300.0,
        "hrss_range": (1e-23, 1e-20),
        "rng": np.random.default_rng(2026),
    }
    kwargs.update(overrides)
    return draw_burst_injection_set(**kwargs)


def test_injection_set_has_a_fixed_rate_and_is_labelled_a_test_set():
    """Events sit on a fixed grid of one per interval and are labelled as a test set."""
    events = _draw(end_time=PEAK + 1000.0)
    assert [e["peak_time"] for e in events] == [PEAK + 150.0, PEAK + 450.0, PEAK + 750.0]
    assert all(e["injection_set"] == "test" for e in events)


def test_injection_set_draws_each_family_and_only_the_keys_it_uses():
    """Every draft family appears, and each event carries only the drawn keys its model uses."""
    events = _draw()
    models = {e["burst_model"] for e in events}
    assert models == {"sine_gaussian", "gaussian", "white_noise_burst"}
    q_values = {e["quality_factor"] for e in events if e["burst_model"] == "sine_gaussian"}
    assert q_values == {3.0, 9.0, 100.0}
    for event in events:
        assert ("inclination" in event) == (event["burst_model"] != "gaussian")
        assert ("seed" in event) == (event["burst_model"] == "white_noise_burst")
        assert ("frequency" in event) == (event["burst_model"] == "sine_gaussian")


def test_injection_set_distributions():
    """Hrss and frequency are log-uniform in range; sky and inclination are isotropic."""
    events = _draw()
    n = len(events)
    assert n == 1000
    log_hrss = np.log10([e["hrss"] for e in events])
    assert log_hrss.min() >= -23
    assert log_hrss.max() <= -20
    # Log-uniform: log10(hrss) is uniform on [-23, -20], mean -21.5, sd 3/sqrt(12) / sqrt(n).
    assert abs(log_hrss.mean() + 21.5) < 4 * (3 / math.sqrt(12)) / math.sqrt(n)
    frequencies = np.array([e["frequency"] for e in events if "frequency" in e])
    assert frequencies.min() >= 30.0
    assert frequencies.max() <= 1800.0
    log_f = np.log(frequencies)
    mid, width = (math.log(30.0) + math.log(1800.0)) / 2, math.log(1800.0 / 30.0)
    assert abs(log_f.mean() - mid) < 4 * width / math.sqrt(12 * len(log_f))
    # Isotropic sky: sin(dec) and cos(iota) uniform on [-1, 1], ra uniform on [0, 2 pi).
    sin_dec = np.sin([e["declination"] for e in events])
    assert abs(sin_dec.mean()) < 4 / math.sqrt(3 * n)
    assert abs(np.mean(sin_dec**2) - 1 / 3) < 0.05
    cos_iota = np.cos([e["inclination"] for e in events if "inclination" in e])
    assert abs(np.mean(cos_iota**2) - 1 / 3) < 0.05
    ra = np.array([e["right_ascension"] for e in events])
    assert ra.min() >= 0
    assert ra.max() < 2 * math.pi
    assert abs(ra.mean() - math.pi) < 4 * (2 * math.pi / math.sqrt(12)) / math.sqrt(n)


def test_injection_set_is_reproducible_and_family_values_override_draws():
    """A seeded generator reproduces the set, and a family's fixed value wins over a draw."""
    circular = ({"burst_model": "sine_gaussian", "quality_factor": 9.0, "inclination": 0.0},)
    first = _draw(families=circular, end_time=PEAK + 3000.0)
    second = _draw(families=circular, end_time=PEAK + 3000.0)
    assert first == second
    assert all(e["inclination"] == 0.0 for e in first)


def test_every_drawn_event_simulates():
    """Every event the draft set produces generates at its recorded hrss."""
    simulator = BurstSimulator()
    for event in _draw(end_time=PEAK + 300.0 * len(DRAFT_BURST_FAMILIES) * 3):
        hp, hc = simulator.generate_polarizations(event, FS, 10.0)
        assert _rss(hp, hc) == pytest.approx(event["hrss"], rel=1e-6)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"end_time": PEAK}, "end_time"),
        ({"interval": 0.0}, "interval"),
        ({"hrss_range": (1e-20, 1e-23)}, "hrss_range"),
        ({"hrss_range": (0.0, 1e-20)}, "hrss_range"),
        ({"sine_gaussian_frequency_range": (30.0, math.inf)}, "sine_gaussian_frequency_range"),
        ({"families": ()}, "families"),
        ({"families": ({"quality_factor": 9.0},)}, "burst_model"),
    ],
)
def test_injection_set_rejects_invalid_configuration(overrides, match):
    """Malformed spans, ranges and family lists are refused."""
    with pytest.raises(ValueError, match=match):
        _draw(**overrides)
