"""Projection keeps the signal the geocentre-to-detector delay moves past the buffer's ends.

Projecting delays the signal by the detector's light-travel time from the geocentre, up to ~21 ms.
A projection returning a series exactly as long as its input therefore loses whatever that delay
pushes past an end -- all of a 0.1 ms burst, whose buffer is 17 samples -- or, on the phase-shift
branch, wraps it round to the other end. These tests pin that every delayed sample is kept, against
a reference that shares nothing with the projection code: the detector tensor and position are
built here in closed form from the site's published geodetic coordinates, and the delay is applied
exactly, as a frequency-domain phase shift (constant delay) or a full unwindowed sinc sum
(time-dependent delay).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from gwmock_signal.network import Network
from gwmock_signal.projection.network import _project, project_polarizations_to_network
from gwmock_signal.projection.resampling import DEFAULT_KAISER_BETA, DEFAULT_SINC_TAPS, edge_padding
from gwmock_signal.timeseries import TimeSeries

FS = 8192.0
PEAK = 1_000_000_100.0
SKY = {"right_ascension": 1.0, "declination": 0.5, "polarization_angle": 0.3}
SPEED_OF_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def _detectors():
    return list(Network.from_preset("ET-Triangle-EMR").detector_names)


def _site(detector):
    """Return ``(tensor, position)`` of a detector from its geodetic site description.

    Closed form, WGS-84: the arm directions from azimuth (clockwise from north) and tilt in the
    local east-north-up frame, the response tensor ``(x x - y y) / 2``.
    """
    phi, lam, height = detector.latitude_rad, detector.longitude_rad, detector.elevation_m
    e2 = WGS84_F * (2.0 - WGS84_F)
    radius = WGS84_A / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    position = np.array(
        [
            (radius + height) * np.cos(phi) * np.cos(lam),
            (radius + height) * np.cos(phi) * np.sin(lam),
            ((1.0 - e2) * radius + height) * np.sin(phi),
        ]
    )
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

    def arm(azimuth, tilt):
        return np.cos(tilt) * (np.cos(azimuth) * north + np.sin(azimuth) * east) + np.sin(tilt) * up

    x = arm(detector.xarm_azimuth_rad, detector.xarm_tilt_rad)
    y = arm(detector.yarm_azimuth_rad, detector.yarm_tilt_rad)
    return 0.5 * (np.outer(x, x) - np.outer(y, y)), position


def _response(detector, t_gps):
    """Return ``(F+, Fx, delay)`` at GPS time(s) *t_gps*, delay = detector minus geocentre arrival."""
    tensor, position = _site(detector)
    hour_angle = np.atleast_1d(Time(t_gps, format="gps", scale="utc").sidereal_time("mean", "greenwich").rad)
    hour_angle = hour_angle - SKY["right_ascension"]
    dec, psi = SKY["declination"], SKY["polarization_angle"]
    ones = np.ones_like(hour_angle)
    sin_h, cos_h = np.sin(hour_angle), np.cos(hour_angle)
    x = np.stack(
        [
            -np.cos(psi) * sin_h - np.sin(psi) * cos_h * np.sin(dec),
            -np.cos(psi) * cos_h + np.sin(psi) * sin_h * np.sin(dec),
            np.sin(psi) * np.cos(dec) * ones,
        ]
    )
    y = np.stack(
        [
            np.sin(psi) * sin_h - np.cos(psi) * cos_h * np.sin(dec),
            np.sin(psi) * cos_h + np.cos(psi) * sin_h * np.sin(dec),
            np.cos(psi) * np.cos(dec) * ones,
        ]
    )
    direction = np.stack([np.cos(dec) * cos_h, -np.cos(dec) * sin_h, np.sin(dec) * ones])
    f_plus = np.einsum("ik,ij,jk->k", x, tensor, x) - np.einsum("ik,ij,jk->k", y, tensor, y)
    f_cross = 2.0 * np.einsum("ik,ij,jk->k", x, tensor, y)
    delay = -(position @ direction) / SPEED_OF_LIGHT
    return f_plus, f_cross, delay


def _gaussian_burst(duration: float):
    from gwmock_signal.burst import BurstSimulator

    params = {"burst_model": "gaussian", "duration": duration, "hrss": 1e-21, "peak_time": PEAK, "coa_time": PEAK}
    return BurstSimulator().generate_polarizations({**params, **SKY, "inclination": 0.0}, FS, 5.0)


def _phase_shift_reference(plus, cross, detector, pad):
    """Constant-pattern projection, delay applied exactly on the padded series.

    Evaluated at the midpoint of the span, as that branch documents.
    """
    midpoint = plus.t0 + 0.5 * (plus.n_samples - 1) / FS
    f_plus, f_cross, delay = (float(v[0]) for v in _response(detector, midpoint))
    n = plus.n_samples + 2 * pad
    phase = np.exp(-2j * np.pi * np.fft.rfftfreq(n, d=1.0 / FS) * delay)

    def shifted(series):
        return np.fft.irfft(np.fft.rfft(np.pad(np.asarray(series.data), (pad, pad))) * phase, n=n)

    return f_plus * shifted(plus) + f_cross * shifted(cross)


def _rotating_reference(plus, cross, detector, pad):
    """Per-sample pattern and delay, the delayed polarizations from the full band-limited sinc sum.

    The pattern and delay are evaluated at each output sample's own time, the coordinate the
    output is labelled in, as the rotating branch documents.
    """
    offsets = np.arange(-pad, plus.n_samples + pad) / FS  # output times relative to plus.t0
    f_plus, f_cross, delay = _response(detector, plus.t0 + offsets)
    kernel = np.sinc(FS * (offsets - delay)[:, None] - np.arange(plus.n_samples)[None, :])
    return f_plus * (kernel @ np.asarray(plus.data)) + f_cross * (kernel @ np.asarray(cross.data))


@pytest.fixture
def _sixty_four_bit_jax():
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("duration", [1e-4, 2.5e-3], ids=["0.1ms", "2.5ms"])
def test_the_phase_shift_branch_keeps_and_places_every_delayed_sample(duration):
    """A burst far shorter than the delay is projected, at the right time, not wrapped or lost.

    Same-length projection wrapped the delayed 0.1 ms burst to the far end of its 17-sample buffer.
    Measured residual 1.1e-09 of peak (0.1 ms) and 1.6e-10 (2.5 ms), from the closed-form geometry
    against the package's double-precision one.
    """
    plus, cross = _gaussian_burst(duration)
    pad = edge_padding(FS)
    projected = project_polarizations_to_network(
        {"plus": plus, "cross": cross}, _detectors(), earth_rotation=False, **SKY
    )
    for detector in _detectors():
        expected = _phase_shift_reference(plus, cross, detector, pad)
        actual = np.asarray(projected[detector.name].data)
        assert actual.shape == expected.shape
        peak = np.max(np.abs(expected))
        assert peak > 0.0
        assert np.max(np.abs(actual - expected)) < 1e-8 * peak, detector.name


@pytest.mark.requires_lalsuite
@pytest.mark.usefixtures("_sixty_four_bit_jax")
@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    ("duration", "tolerance"),
    [
        # Limited by the windowed-sinc kernel, not by the delay: a 0.1 ms Gaussian at 8192 Hz has
        # most of its power near Nyquist, where no finite kernel is exact. Measured 2.1e-04 to
        # 1.0e-03 of peak across the three detectors, halving when the tap count doubles.
        (1e-4, 2e-3),
        # Well inside the kernel's band: measured 5.2e-11 to 1.6e-10.
        (2.5e-3, 1e-9),
    ],
    ids=["0.1ms", "2.5ms"],
)
def test_the_rotating_branch_keeps_every_delayed_sample(backend, duration, tolerance):
    """Before the delay was given room, the 0.1 ms burst projected to exactly zero here."""
    plus, cross = _gaussian_burst(duration)
    pad = edge_padding(FS)
    projected = project_polarizations_to_network(
        {"plus": plus, "cross": cross}, _detectors(), earth_rotation=True, backend=backend, **SKY
    )
    for detector in _detectors():
        expected = _rotating_reference(plus, cross, detector, pad)
        actual = np.asarray(projected[detector.name].data)
        assert actual.shape == expected.shape
        peak = np.max(np.abs(expected))
        assert peak > 0.0
        assert np.max(np.abs(actual - expected)) < tolerance * peak, detector.name


@pytest.mark.requires_lalsuite
@pytest.mark.usefixtures("_sixty_four_bit_jax")
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_a_tapered_burst_near_nyquist_is_projected_to_the_exact_interpolant(backend):
    """Declared tapered, the 0.1 ms burst -- most of its power near Nyquist -- comes far closer to the sinc sum.

    The kernel alone leaves 2.1e-04 to 1.0e-03 of peak here (the case above); oversampled exactly first,
    1.8e-06 to 8.8e-06. What remains is the burst's own: sampled at 8192 Hz, a 0.1 ms Gaussian still has
    4% of its peak at Nyquist, so it is not band-limited, and its exact interpolant has tails falling as
    one over the distance that the padded buffer cuts off. A signal sampled with margin, such as the
    2.5 ms burst above, has none to cut.
    """
    plus, cross = _gaussian_burst(1e-4)
    pad = edge_padding(FS)
    projected = project_polarizations_to_network(
        {"plus": plus, "cross": cross}, _detectors(), earth_rotation=True, backend=backend, tapered=True, **SKY
    )
    for detector in _detectors():
        expected = _rotating_reference(plus, cross, detector, pad)
        actual = np.asarray(projected[detector.name].data)
        peak = np.max(np.abs(expected))
        assert np.max(np.abs(actual - expected)) < 2e-5 * peak, detector.name


@pytest.mark.requires_lalsuite
@pytest.mark.usefixtures("_sixty_four_bit_jax")
@pytest.mark.parametrize(
    ("earth_rotation", "backend", "tolerance"),
    [
        # An exact phase shift preserves energy exactly; measured 5e-11 to 3e-10, the geometry residual.
        (False, "numpy", 1e-8),
        # Measured -6.4e-06 to -1.3e-04: power near Nyquist the kernel attenuates (see above).
        (True, "numpy", 5e-4),
        (True, "jax", 5e-4),
    ],
)
def test_a_short_burst_keeps_its_energy(earth_rotation, backend, tolerance):
    """The projected strain carries the energy of ``F+ h+ + Fx hx``: nothing fell off an end.

    Same-length projection returned zero energy on the rotating branch. On the phase-shift branch
    it wrapped the burst round instead of dropping it, which conserves energy, so that branch's
    defect is caught by the placement test above and this case only guards against a loss.
    """
    plus, cross = _gaussian_burst(1e-4)
    projected = project_polarizations_to_network(
        {"plus": plus, "cross": cross}, _detectors(), earth_rotation=earth_rotation, backend=backend, **SKY
    )
    for detector in _detectors():
        f_plus, f_cross, delay = _response(detector, PEAK)
        if earth_rotation:  # evaluated in detector time, where the burst arrives
            f_plus, f_cross, _ = _response(detector, PEAK + float(delay[0]))
        expected = float(np.sum((f_plus[0] * np.asarray(plus.data) + f_cross[0] * np.asarray(cross.data)) ** 2))
        actual = float(np.sum(np.asarray(projected[detector.name].data) ** 2))
        # abs=0: the energies are ~1e-40, inside pytest.approx's default absolute tolerance of 1e-12,
        # which would accept the zero a same-length projection returned.
        assert expected > 0.0
        assert actual == pytest.approx(expected, rel=tolerance, abs=0.0), detector.name


@pytest.mark.parametrize("earth_rotation", [True, False])
def test_every_detector_shares_one_padded_span_on_the_input_lattice(earth_rotation):
    """The padding is the detector-independent bound, so every series starts and ends together."""
    n, t0, fs = 64, 1_000_000_000.0, 4096.0
    t = np.arange(n) / fs
    polarizations = {
        "plus": TimeSeries(np.sin(2 * np.pi * 100.0 * t) * np.hanning(n), t0=t0, sample_rate=fs),
        "cross": TimeSeries(np.cos(2 * np.pi * 100.0 * t) * np.hanning(n), t0=t0, sample_rate=fs),
    }
    pad = edge_padding(fs)
    projected = project_polarizations_to_network(
        polarizations, ["H1", "L1", "V1"], earth_rotation=earth_rotation, **SKY
    )
    for series in projected.values():
        assert series.n_samples == n + 2 * pad
        assert series.sample_rate == fs
        # A whole number of samples before the input, so the input's samples keep their times.
        assert (t0 - series.t0) * fs == pytest.approx(pad, abs=1e-6)


def _long_inspiral(fs: float, duration: float, t0: float):
    """A leading-order chirp from 20 Hz with a smooth ringdown-like end, tapered at its start."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    coalescence = duration - 0.5
    remaining = np.clip(1.0 - t / coalescence, 1e-4, None)
    phase = 2.0 * np.pi * 20.0 * coalescence * 1.6 * (1.0 - remaining**0.625)
    amplitude = remaining**-0.25
    end = int(coalescence * fs * (1.0 - 1e-4))
    amplitude[end:] = amplitude[end - 1] * np.exp(-(t[end:] - t[end]) / 0.004)
    taper = np.ones(n)
    ramp = int(0.5 * fs)
    taper[:ramp] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    return {
        "plus": TimeSeries(amplitude * np.cos(phase) * taper, t0=t0, sample_rate=fs),
        "cross": TimeSeries(amplitude * np.sin(phase) * taper, t0=t0, sample_rate=fs),
    }


@pytest.mark.usefixtures("_sixty_four_bit_jax")
@pytest.mark.parametrize(
    ("backend", "tolerance"),
    [
        # Measured 7.9e-14 of peak: only the time labels' rounding differs.
        ("numpy", 1e-12),
        # Measured 1.3e-11: the device path's linear sidereal model is anchored at the padded start.
        ("jax", 1e-10),
    ],
)
def test_padding_leaves_a_long_inspiral_unchanged_inside_its_span(backend, tolerance):
    """The fix adds samples outside the input's span and changes none inside it beyond rounding."""
    fs = 2048.0
    polarizations = _long_inspiral(fs, 64.0, 1_400_000_000.0)
    n = polarizations["plus"].n_samples
    pad = edge_padding(fs)
    common = {
        "earth_rotation": True,
        "precess_source_direction": False,
        "backend": backend,
        "sinc_taps": DEFAULT_SINC_TAPS,
        "kaiser_beta": DEFAULT_KAISER_BETA,
        **SKY,
    }
    padded = _project(polarizations, ["H1", "L1", "E1"], **common)
    unpadded = _project(polarizations, ["H1", "L1", "E1"], keep_delayed_samples=False, **common)
    for name, series in unpadded.items():
        reference = np.asarray(series.data)
        within = np.asarray(padded[name].data)[pad : pad + n]
        peak = np.max(np.abs(reference))
        assert np.max(np.abs(within - reference)) < tolerance * peak, name


@pytest.mark.requires_lalsuite
def test_a_burst_simulation_returns_and_injects_the_delayed_burst():
    """Through the simulator a caller uses, with and without a background to inject into.

    Same-length projection made both of these all zeros for the 0.1 ms burst.
    """
    from gwmock_signal.burst import BurstSimulator

    params = {"burst_model": "gaussian", "duration": 1e-4, "hrss": 1e-21, "peak_time": PEAK, "coa_time": PEAK}
    params = {**params, **SKY, "inclination": 0.0}
    detectors = _detectors()
    names = [detector.name for detector in detectors]

    stack = BurstSimulator().simulate(params, detectors, sampling_frequency=FS, minimum_frequency=5.0)
    energies = {name: float(np.sum(np.asarray(stack.to_dict()[name].data) ** 2)) for name in names}
    assert all(energy > 0.0 for energy in energies.values()), energies

    # A background a second either side of the peak, on the burst's own sample lattice.
    plus, _ = _gaussian_burst(1e-4)
    start = plus.t0 - round(1.0 * FS) / FS
    background = {name: TimeSeries(np.zeros(round(2.0 * FS)), t0=start, sample_rate=FS) for name in names}
    injected = (
        BurstSimulator().simulate(params, detectors, background, sampling_frequency=FS, minimum_frequency=5.0).to_dict()
    )
    for name in names:
        injected_energy = float(np.sum(np.asarray(injected[name].data) ** 2))
        assert injected_energy == pytest.approx(energies[name], rel=1e-12, abs=0.0), name
