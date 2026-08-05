"""Tests for the JAX detector-projection building blocks."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)  # GPS times / Julian dates need float64

from gwmock_signal.projection.geometry import reconstructed_geometry  # noqa: E402
from gwmock_signal.projection.jax_projection import (  # noqa: E402
    antenna_pattern,
    gmst_rad,
    project_polarizations_fd,
    time_delay_from_geocenter,
)
from gwmock_signal.projection.network import (  # noqa: E402
    _antenna_pattern_lal,
    _gmst_accurate,
    _time_delay_from_earth_center_lal,
)

# GPS times spanning the 36 s (pre-2017) and 37 s leap-second eras.
_GPS_TIMES = [1126259462.4, 1187008882.4, 1238166018.0, 1370000000.0]


def _astropy_reference(t_gps: float) -> tuple[float, float, float]:
    """Return (GMST, TAI-UTC, UT1-UTC) from Astropy for one GPS time."""
    t = Time(float(t_gps), format="gps", scale="utc", location=(0, 0))
    gmst = float(t.sidereal_time("mean").rad)
    jd_utc = t.utc.jd1 + t.utc.jd2
    tai_minus_utc = (t.tai.jd1 + t.tai.jd2 - jd_utc) * 86400.0
    dut1 = (t.ut1.jd1 + t.ut1.jd2 - jd_utc) * 86400.0
    return gmst, tai_minus_utc, dut1


def _wrapped_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a - b in radians."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize("t_gps", _GPS_TIMES)
def test_gmst_matches_astropy(t_gps: float) -> None:
    """Fed Astropy's leap seconds and DUT1, gmst_rad reproduces Astropy GMST to ~1e-6 rad.

    This anchors the JAX implementation against an external reference (Astropy's
    IAU sidereal time) rather than only internal consistency.
    """
    reference, tai_minus_utc, dut1 = _astropy_reference(t_gps)
    got = float(gmst_rad(t_gps, tai_minus_utc=tai_minus_utc, dut1=dut1))
    assert abs(_wrapped_diff(got, reference)) < 1e-6


def test_gmst_default_offsets_are_dut1_limited() -> None:
    """With default offsets (leap=37, dut1=0) a post-2017 time still matches to ~1e-4 rad.

    Documents the accuracy of the defaults: the only error is the neglected DUT1.
    """
    t_gps = 1370000000.0  # ~2023, leap-second era 37 s
    reference, _, _ = _astropy_reference(t_gps)
    got = float(gmst_rad(t_gps))  # defaults
    assert abs(_wrapped_diff(got, reference)) < 1e-4


def test_gmst_scalar_and_array_shapes() -> None:
    """gmst_rad returns a scalar for scalar input and preserves array shape."""
    assert gmst_rad(_GPS_TIMES[0]).shape == ()
    out = np.asarray(gmst_rad(np.array(_GPS_TIMES)))
    assert out.shape == (len(_GPS_TIMES),)
    assert ((out >= 0.0) & (out < 2.0 * np.pi)).all()


def test_gmst_is_jit_traceable() -> None:
    """gmst_rad is JAX-traceable (jit) and agrees with the eager result."""
    t_gps = _GPS_TIMES[0]
    eager = float(gmst_rad(t_gps))
    jitted = float(jax.jit(gmst_rad)(t_gps))
    # JIT may fuse float ops; agreement to 9 significant figures is far below the µs anchor.
    assert eager == pytest.approx(jitted, rel=1e-9)


# Sky positions (right_ascension, declination, polarization_angle) in radians.
_SKY = [(1.375, -1.211, 2.659), (0.1, 0.2, 0.3), (5.0, 1.0, 0.0)]
_DETECTORS = ["H1", "L1", "V1"]


@pytest.mark.parametrize("prefix", _DETECTORS)
@pytest.mark.parametrize(("ra", "dec", "psi"), _SKY)
def test_antenna_pattern_matches_numpy(prefix: str, ra: float, dec: float, psi: float) -> None:
    """JAX antenna pattern equals the NumPy implementation (same gmst and geometry)."""
    t_gps = 1370000000.0
    gmst = _gmst_accurate(t_gps)
    response, _ = reconstructed_geometry(prefix)
    fp_np, fc_np = _antenna_pattern_lal(
        prefix, right_ascension=ra, declination=dec, polarization_angle=psi, t_gps=t_gps
    )
    fp_jax, fc_jax = antenna_pattern(response, gmst, right_ascension=ra, declination=dec, polarization_angle=psi)
    assert float(fp_jax) == pytest.approx(fp_np, rel=1e-9, abs=1e-12)
    assert float(fc_jax) == pytest.approx(fc_np, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("prefix", _DETECTORS)
@pytest.mark.parametrize(("ra", "dec", "psi"), _SKY)
def test_time_delay_matches_numpy(prefix: str, ra: float, dec: float, psi: float) -> None:
    """JAX geocenter time delay equals the NumPy implementation (same gmst and geometry)."""
    t_gps = 1370000000.0
    gmst = _gmst_accurate(t_gps)
    _, location = reconstructed_geometry(prefix)
    tau_np = _time_delay_from_earth_center_lal(prefix, right_ascension=ra, declination=dec, t_gps=t_gps)
    tau_jax = time_delay_from_geocenter(location, gmst, right_ascension=ra, declination=dec)
    assert float(tau_jax) == pytest.approx(tau_np, rel=1e-9, abs=1e-15)


def test_antenna_pattern_array_matches_scalar() -> None:
    """An array of sidereal times gives per-element the scalar results (shape preserved)."""
    response, _ = reconstructed_geometry("H1")
    ra, dec, psi = _SKY[0]
    gmst_values = np.array([0.3, 1.1, 4.5, 6.0])
    fp_arr, fc_arr = antenna_pattern(response, gmst_values, right_ascension=ra, declination=dec, polarization_angle=psi)
    assert np.asarray(fp_arr).shape == gmst_values.shape
    for i, g in enumerate(gmst_values):
        fp_s, fc_s = antenna_pattern(response, float(g), right_ascension=ra, declination=dec, polarization_angle=psi)
        assert float(fp_arr[i]) == pytest.approx(float(fp_s), abs=1e-12)
        assert float(fc_arr[i]) == pytest.approx(float(fc_s), abs=1e-12)


def test_projection_primitives_are_jit_traceable() -> None:
    """antenna_pattern and time_delay_from_geocenter are JAX-traceable (jit)."""
    response, location = reconstructed_geometry("L1")
    ra, dec, psi = _SKY[1]
    gmst = _gmst_accurate(1370000000.0)

    fp = jax.jit(lambda r, g: antenna_pattern(r, g, right_ascension=ra, declination=dec, polarization_angle=psi))(
        response, gmst
    )
    tau = jax.jit(lambda loc, g: time_delay_from_geocenter(loc, g, right_ascension=ra, declination=dec))(location, gmst)
    assert np.isfinite(float(fp[0]))
    assert np.isfinite(float(tau))


def test_project_polarizations_fd_combines_and_delays() -> None:
    """With unit F+ and no delay, the projection is just the inverse FFT of plus."""
    fs = 1024.0
    n = 256
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    rng = np.random.default_rng(0)
    plus = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    cross = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))

    only_plus = np.asarray(
        project_polarizations_fd(
            freqs, plus, cross, f_plus=1.0, f_cross=0.0, time_delay=0.0, n_samples=n, sampling_frequency=fs
        )
    )
    plus_td = np.fft.irfft(plus, n=n) * fs
    assert np.max(np.abs(only_plus - plus_td)) < 1e-9 * np.max(np.abs(plus_td))

    # f_cross selects the cross polarization; a delay is a pure phase shift (norm preserved).
    only_cross = np.asarray(
        project_polarizations_fd(
            freqs, plus, cross, f_plus=0.0, f_cross=1.0, time_delay=0.0, n_samples=n, sampling_frequency=fs
        )
    )
    cross_td = np.fft.irfft(cross, n=n) * fs
    assert np.max(np.abs(only_cross - cross_td)) < 1e-9 * np.max(np.abs(cross_td))

    # An integer-sample delay equals a cyclic roll of the undelayed strain.
    shift = 5
    delayed = np.asarray(
        project_polarizations_fd(
            freqs, plus, cross, f_plus=1.0, f_cross=0.0, time_delay=shift / fs, n_samples=n, sampling_frequency=fs
        )
    )
    assert np.max(np.abs(delayed - np.roll(only_plus, shift))) < 1e-9 * np.max(np.abs(only_plus))


@pytest.mark.integration
def test_device_projection_matches_host_pipeline() -> None:
    """The on-device FD projection reproduces the host (NumPy) earth_rotation=False path.

    End-to-end check that ripple FD -> JAX antenna/delay/projection -> irfft equals
    project_polarizations_to_network for the same event, to a zero-lag overlap (so the
    delay timing is validated, not just the morphology).
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.waveform.backends import RippleBackend

    fs, f_min = 2048.0, 20.0
    ra, dec, psi = 1.375, -1.211, 2.659
    params = {
        "detector_frame_mass_1": 40.0,
        "detector_frame_mass_2": 31.0,
        "luminosity_distance": 400.0,
        "spin_1z": 0.5,
        "spin_2z": -0.2,
        "inclination": 0.9,
        "coa_phase": 0.3,
    }
    fd = RippleBackend().generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=fs, minimum_frequency=f_min, **params
    )
    n, dt = fd.n_samples, 1.0 / fs

    # Unplaced time-domain polarizations (coalescence at t=0), fed to the host path.
    t0 = 1126259462.0
    pols = {
        "plus": TimeSeries(np.fft.irfft(np.asarray(fd.plus), n=n) / dt, t0=t0, dt=dt),
        "cross": TimeSeries(np.fft.irfft(np.asarray(fd.cross), n=n) / dt, t0=t0, dt=dt),
    }
    host = project_polarizations_to_network(
        pols, ["H1"], right_ascension=ra, declination=dec, polarization_angle=psi, earth_rotation=False
    )["H1"]

    # Device path: F+, Fx and tau at the host's reference time (segment midpoint).
    times = pols["plus"].times.value
    reference_time = 0.5 * (times[0] + times[-1])
    gmst = _gmst_accurate(reference_time)
    response, location = reconstructed_geometry("H1")
    f_plus, f_cross = antenna_pattern(response, gmst, right_ascension=ra, declination=dec, polarization_angle=psi)
    tau = time_delay_from_geocenter(location, gmst, right_ascension=ra, declination=dec)
    device = np.asarray(
        project_polarizations_fd(
            fd.frequencies,
            fd.plus,
            fd.cross,
            f_plus=f_plus,
            f_cross=f_cross,
            time_delay=tau,
            n_samples=n,
            sampling_frequency=fs,
        )
    )

    a, b = host.value, device
    overlap = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    assert overlap > 0.9999, f"zero-lag overlap {overlap:.6f} below threshold"


def _chirp_polarizations(n_samples: int, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a tapered, band-limited chirp as (plus, cross).

    A chirp rather than a pure tone so the interpolation is exercised across a range
    of frequencies, and tapered so the projection's zero-fill at the edges does not
    dominate the comparison.

    The sweep is a fixed fraction of the sample rate (f_s/100 to f_s/20) rather than a
    fixed frequency band, so the signal stays band-limited and well oversampled at every
    rate these tests use. A hard-coded band would alias at the low sample rate the
    long-signal test needs to reach a multi-thousand-second duration cheaply.
    """
    t = np.arange(n_samples) / sampling_frequency
    duration = n_samples / sampling_frequency
    f_start = sampling_frequency / 100.0
    f_end = sampling_frequency / 20.0
    frequency = f_start + (f_end - f_start) * t / duration
    phase = 2.0 * np.pi * np.cumsum(frequency) / sampling_frequency
    envelope = np.hanning(n_samples)
    return envelope * np.cos(phase), envelope * np.sin(phase)


def test_rotating_projection_matches_numpy_path() -> None:
    """The device rotating projection reproduces the NumPy earth_rotation=True path."""
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    from gwmock_signal.projection.jax_projection import project_polarizations_td_rotating
    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate

    sampling_frequency = 2048.0
    n_samples = 2**16
    start_time = 1.4e9
    sky = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}

    plus, cross = _chirp_polarizations(n_samples, sampling_frequency)
    reference = project_polarizations_to_network(
        {
            "plus": GWpyTimeSeries(plus, t0=start_time, sample_rate=sampling_frequency),
            "cross": GWpyTimeSeries(cross, t0=start_time, sample_rate=sampling_frequency),
        },
        ["E1"],
        earth_rotation=True,
        **sky,
    )["E1"].value

    response, location = reconstructed_geometry("E1")
    anchors, rate = gmst_anchor_and_rate(start_time)
    # Zero rates, matching `precess_source_direction=False` in the reference above: whichever sky
    # convention the two sides use, they must use the *same* one, or this measures a frame offset
    # instead of the round-off it claims to. Mismatching them here reported 1.1% of peak.
    frozen_sky = {"right_ascension_rate": 0.0, "declination_rate": 0.0}
    device = np.asarray(
        project_polarizations_td_rotating(
            plus,
            cross,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            gmst_start=float(anchors[0]),
            gmst_rate=rate,
            **sky,
            **frozen_sky,
        )
    )

    scale = np.max(np.abs(reference))
    # The tolerance below is relative, so a null response would make it vacuous or
    # impossible; the sky position is fixed, but assert the premise rather than assume it.
    assert scale > 0.0
    # Round-off. Both paths resample with the same Kaiser-windowed sinc kernel and take
    # sidereal time from Astropy -- the device path via a host-computed anchor and rate,
    # which is linear to 6e-14 rad over these spans. Earlier revisions sat at 1e-3 (cubic
    # interpolation) and then 3.9e-5 (two different sidereal implementations); a tolerance
    # loose enough to pass those would no longer detect either regression.
    assert np.max(np.abs(device - reference)) < 1e-10 * scale


def test_rotating_projection_differs_from_static_for_long_signals() -> None:
    """Earth rotation changes the answer over an hour-long segment.

    Guards the reason this path exists: if the rotating and midpoint-only projections
    agreed, wiring the rotating one into the device path would be pointless.
    """
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    from gwmock_signal.projection.jax_projection import project_polarizations_td_rotating
    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate

    sampling_frequency = 64.0
    n_samples = 2**18  # 4096 s, the scale of a BNS inspiral in the ET band
    start_time = 1.4e9
    sky = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}

    plus, cross = _chirp_polarizations(n_samples, sampling_frequency)
    response, location = reconstructed_geometry("E1")
    anchors, rate = gmst_anchor_and_rate(start_time)

    rotating = np.asarray(
        project_polarizations_td_rotating(
            plus,
            cross,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            gmst_start=float(anchors[0]),
            gmst_rate=rate,
            **sky,
            # Zero, matching `project_polarizations_to_network`'s default: neither side precesses,
            # so Earth rotation is the only difference and it is what the mismatch below measures.
            right_ascension_rate=0.0,
            declination_rate=0.0,
        )
    )

    # Compare against the genuine earth_rotation=False projection rather than an
    # antenna-pattern-only expression: the static path still applies the midpoint
    # geocenter delay, and omitting it would let a |tau| <= 21 ms timing difference
    # masquerade as the Earth-rotation effect this test exists to detect.
    static = project_polarizations_to_network(
        {
            "plus": GWpyTimeSeries(plus, t0=start_time, sample_rate=sampling_frequency),
            "cross": GWpyTimeSeries(cross, t0=start_time, sample_rate=sampling_frequency),
        },
        ["E1"],
        earth_rotation=False,
        **sky,
    )["E1"].value

    mismatch = np.max(np.abs(rotating - static)) / np.max(np.abs(rotating))
    assert mismatch > 0.1, f"expected a large difference over 4096 s, got {mismatch:.3g}"


def test_projection_gathers_from_zero_padded_polarizations() -> None:
    """The *production* projection must zero-pad, not clamp, at the buffer edges.

    Routed through ``project_polarizations_td_rotating`` on purpose. Two earlier attempts at
    this test could not fail:

    * the first padded an array by hand and called the kernel helper directly, so it never
      exercised the production path -- which at the time still clamped;
    * the second compared the very first output sample against the interior, but with a
      positive geocenter delay that sample lands *outside* the signal entirely, where both
      treatments correctly give ~0.

    The discriminating samples are those just *inside* the buffer and within a kernel
    half-width of its start. There, clamping repeats the endpoint and reproduces a constant
    input exactly, while zero padding cannot, because part of the kernel support is empty.
    Measured for a DC input at 2048 Hz: clamping gives 1.000000 at every such sample, padding
    gives 0.9256 / 0.9771 / 1.0090 / 1.0013 at 1 / 5 / 10 / 20 samples in.
    """
    from gwmock_signal.projection.jax_projection import (
        project_polarizations_td_rotating,
        time_delay_from_geocenter,
    )
    from gwmock_signal.projection.resampling import edge_padding
    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate

    sampling_frequency = 2048.0
    n_samples = 8192
    start_time = 1.4e9
    # Zero rates: these exercise the resampler, the sidereal model and the guards, not
    # precession, so the sky position is held fixed deliberately rather than by default.
    sky = {
        "right_ascension": 0.4,
        "declination": 0.2,
        "polarization_angle": 0.0,
        "right_ascension_rate": 0.0,
        "declination_rate": 0.0,
    }

    response, location = reconstructed_geometry("E1")
    anchors, rate = gmst_anchor_and_rate(start_time)
    gmst_start = float(anchors[0])

    strain = np.asarray(
        project_polarizations_td_rotating(
            np.ones(n_samples),
            np.zeros(n_samples),
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            gmst_start=gmst_start,
            gmst_rate=rate,
            # Half a sample, so the kernel taps actually spread: at a whole-sample offset the
            # sinc weights vanish on every tap but the centre and no edge treatment is visible.
            extra_shift_samples=0.5,
            **sky,
        )
    )

    # Locate a sample that is inside the signal but within a kernel half-width of its start,
    # accounting for the geocenter delay, which shifts where the signal begins in the output.
    delay_samples = (
        float(
            time_delay_from_geocenter(location, gmst_start, **{k: sky[k] for k in ("right_ascension", "declination")})
        )
        * sampling_frequency
    )
    first_inside = int(np.ceil(delay_samples)) + 1
    edge_sample = first_inside + 4
    assert edge_sample < edge_padding(sampling_frequency, 127)

    interior = float(strain[n_samples // 2])
    assert abs(interior) > 0.0
    # F+ varies by ~1e-4 over the buffer, so require an order of magnitude more than that.
    assert abs(strain[edge_sample] - interior) > 1e-3 * abs(interior), (
        f"sample {edge_sample} is {strain[edge_sample] / interior:.6f} of the interior value; "
        "the production path appears to clamp out-of-range taps to the endpoint"
    )
    # Away from the edge the two treatments must be indistinguishable.
    assert abs(strain[n_samples // 2 - 1] - interior) < 0.1 * abs(interior)


def test_edge_taps_read_zeros_not_repeated_endpoints() -> None:
    """Out-of-range kernel taps must read zeros, not a repeated endpoint sample.

    The gather clamps its indices, so without an explicitly zero-padded source the taps that
    reach past either end read the first or last sample repeatedly. That *invents* a
    continuation of the signal: for a constant input, clamping returns the constant exactly,
    as though the buffer extended forever. Zero padding instead represents the truth that the
    strain is zero outside the buffer, and the resulting edge ringing is the honest
    consequence of a discontinuous input.

    Tested with a constant rather than a tapered inspiral on purpose. A tapered signal hides
    the difference, and this primitive is waveform-agnostic: it must be correct for waveforms
    with abrupt support too.

    The assertion is on the *difference between the two treatments* near the edge, and on their
    agreement in the interior, rather than on a specific edge value -- the edge value depends
    on the kernel's weight distribution and asserting a guessed number would test nothing.
    """
    import jax.numpy as jnp

    from gwmock_signal.projection.jax_projection import _interpolate_uniform_sinc

    n_samples = 8192
    pad = 109  # what the projection uses at 2048 Hz for a ground-based detector
    constant = jnp.ones(n_samples)
    zero_padded = jnp.pad(constant, (pad, pad))

    def padded_at(offset: float) -> float:
        return float(_interpolate_uniform_sinc(zero_padded, jnp.array([pad + offset]), n_samples + 2 * pad)[0])

    def clamped_at(offset: float) -> float:
        return float(_interpolate_uniform_sinc(constant, jnp.array([offset]), n_samples)[0])

    # Fractional offsets: at a whole-sample offset the sinc weights vanish on every tap but the
    # centre one, so no treatment of the edges is observable there at all.
    near_edge = 0.5
    interior = n_samples // 2 + 0.5

    assert abs(padded_at(near_edge) - clamped_at(near_edge)) > 1e-3, (
        "zero padding and endpoint clamping agree at the edge, so the source is not padded"
    )
    assert abs(padded_at(interior) - clamped_at(interior)) < 1e-9, (
        "the two treatments must be indistinguishable away from the edges"
    )


def test_sidereal_time_advances_linearly_across_the_buffer() -> None:
    """The response must track ``gmst_start + gmst_rate * sample / f_s`` across the buffer.

    This pins the primitive's own GMST progression, and nothing more. It does **not** cover the
    caller-side fix that anchors sidereal time at the *aligned* start rather than the requested
    one: it supplies ``gmst_start`` by hand with ``extra_shift_samples=0``, so it passes whether
    or not that correction is present -- which was the case when it was first written under a
    name claiming otherwise. ``test_sidereal_anchor_uses_the_aligned_start`` in
    ``tests/test_grid_aligned_assembly.py`` is the regression test for the correction.

    The rate is exaggerated so the linear progression is resolvable at all; at Earth's rate the
    change across a buffer is far below the tolerance any comparison here could use.
    """
    from gwmock_signal.projection.jax_projection import (
        antenna_pattern,
        project_polarizations_td_rotating,
    )

    sampling_frequency = 2048.0
    n_samples = 4096
    sky = {"right_ascension": 1.1, "declination": -0.3, "polarization_angle": 0.4}
    # Held fixed deliberately: this test isolates the *sidereal* anchor convention, and the
    # independent expectation below is `antenna_pattern` at one position. A non-zero precession rate
    # would move the position between the two and make the comparison test both effects at once.
    frozen_sky = {"right_ascension_rate": 0.0, "declination_rate": 0.0}
    response, location = reconstructed_geometry("E1")

    gmst_start = 0.7
    # Absurd rate: one radian per second, ~14000x Earth's, so a one-sample anchor error becomes
    # a ~5e-4 rad change in sidereal angle instead of ~3.6e-8.
    exaggerated_rate = 1.0

    plus = np.ones(n_samples)
    cross = np.zeros(n_samples)
    strain = np.asarray(
        project_polarizations_td_rotating(
            plus,
            cross,
            response=response,
            location=location,
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            gmst_start=gmst_start,
            gmst_rate=exaggerated_rate,
            extra_shift_samples=0.0,
            **sky,
            **frozen_sky,
        )
    )

    # Independent expectation for a sample well inside the buffer, where resampling is
    # pass-through in amplitude for a constant input: F evaluated at the anchor plus the
    # elapsed sidereal angle for that sample.
    probe = n_samples // 2
    gmst_at_probe = gmst_start + exaggerated_rate * probe / sampling_frequency
    f_plus, _ = antenna_pattern(response, gmst_at_probe, **sky)
    assert strain[probe] == pytest.approx(float(f_plus), rel=2e-3), (
        "the response is not evaluated at gmst_start + rate * (sample / f_s); the anchor convention has drifted"
    )


def test_production_projection_accepts_a_non_default_kernel() -> None:
    """A valid non-default tap/beta pair must work end to end, and size its padding from itself.

    Kernel validity couples taps and beta -- a large beta needs enough taps to hold its
    transition band -- so a non-default pair exercises a path the defaults never do. It also
    guards the ``edge_padding`` bug found in review, where the padding validated the *default*
    beta rather than the caller's: that both rejected valid small-taps/small-beta pairs and let
    an invalid large-beta pair through until after the buffer had been allocated.
    """
    from gwmock_signal.projection.jax_projection import project_polarizations_td_rotating
    from gwmock_signal.projection.resampling import edge_padding
    from gwmock_signal.projection.sidereal import gmst_anchor_and_rate

    sampling_frequency = 2048.0
    n_samples = 4096
    start_time = 1.4e9
    # Zero rates: these exercise the resampler, the sidereal model and the guards, not
    # precession, so the sky position is held fixed deliberately rather than by default.
    sky = {
        "right_ascension": 0.9,
        "declination": 0.1,
        "polarization_angle": 0.2,
        "right_ascension_rate": 0.0,
        "declination_rate": 0.0,
    }
    response, location = reconstructed_geometry("E1")
    anchors, rate = gmst_anchor_and_rate(start_time)

    taps, beta = 63, 15.0  # valid together: 63 >= 4 * 15 - 1
    assert edge_padding(sampling_frequency, taps, beta) < edge_padding(sampling_frequency, 127, 32.0)

    plus, cross = _chirp_polarizations(n_samples, sampling_frequency)
    common = {
        "response": response,
        "location": location,
        "sampling_frequency": sampling_frequency,
        "n_samples": n_samples,
        "gmst_start": float(anchors[0]),
        "gmst_rate": rate,
        **sky,
    }
    custom = np.asarray(project_polarizations_td_rotating(plus, cross, sinc_taps=taps, kaiser_beta=beta, **common))
    default = np.asarray(project_polarizations_td_rotating(plus, cross, **common))

    assert np.all(np.isfinite(custom))
    scale = np.max(np.abs(default))
    assert scale > 0.0
    # A shorter kernel is less accurate, but on an oversampled chirp it must still agree closely
    # with the default; a gross disagreement would mean the padding or index offset is wrong for
    # a non-default tap count.
    interior = slice(200, n_samples - 200)
    assert np.max(np.abs(custom[interior] - default[interior])) < 1e-6 * scale

    # An invalid pair must be refused before anything is allocated.
    with pytest.raises(ValueError, match="transition"):
        project_polarizations_td_rotating(plus, cross, sinc_taps=63, kaiser_beta=32.0, **common)
