"""A per-event projection put on a caller's lattice is the strain at that lattice's samples.

The reference needs no interpolation: the polarizations are an analytic band-limited signal, so the same
signal can be sampled on the lattice itself and projected there with no alignment at all. Projecting it
sampled off the lattice, with ``output_grid``, must give the same strain at the same times to the
kernel's accuracy. A shift applied with the wrong sign, or not applied, misplaces the strain by up to two
samples, which at these frequencies is a difference of order the signal itself.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.projection.resampling import DEFAULT_KAISER_BETA, DEFAULT_SINC_TAPS
from gwmock_signal.projection.solar_system import SolarSystemEphemeris
from gwmock_signal.sampling_grid import SamplingGrid
from gwmock_signal.timeseries import TimeSeries

_FS = 1024.0
_DURATION = 32
_EPOCH = 1_400_000_000.0
_SKY = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}
_ROTATING = ProjectionEffects.long_wavelength(earth_rotation=True)
#: Measured at 3e-14 to 5e-14 of peak for every path and shift here: the two sides make the same one
#: resampling at shifts that differ by a fraction of a sample, so they agree far below the kernel's own
#: 1e-11. The bound leaves a factor of 20 over that.
_TOLERANCE = 1e-12


def _polarizations(t0: float, rate: float = _FS) -> dict[str, TimeSeries]:
    """A Gaussian-windowed 40 Hz tone and its quadrature, sampled at ``t0 + k / rate``.

    The times are offsets from the window's centre, so the analytic values are evaluated without
    subtracting GPS-sized numbers.
    """
    centre = _EPOCH + 16.0
    offsets = (t0 - centre) + np.arange(int(_DURATION * rate)) / rate
    envelope = np.exp(-((offsets / 2.0) ** 2))
    phase = 2.0 * np.pi * 40.0 * offsets
    return {
        "plus": TimeSeries(envelope * np.cos(phase), t0=t0, sample_rate=rate),
        "cross": TimeSeries(envelope * np.sin(phase), t0=t0, sample_rate=rate),
    }


def _project(
    t0: float, effects: ProjectionEffects, backend: str, grid: SamplingGrid | None, rate: float = _FS, **kwargs
):
    return project_polarizations_to_network(
        _polarizations(t0, rate), ["E1"], effects=effects, backend=backend, output_grid=grid, **_SKY, **kwargs
    )["E1"]


def _compare(effects: ProjectionEffects, backend: str, fraction: float, rate: float = _FS, **kwargs) -> None:
    grid = SamplingGrid(_EPOCH, rate)
    off = _project(_EPOCH + 3.0 + fraction / rate, effects, backend, grid, rate, **kwargs)
    # The same signal sampled on the lattice sample preceding the off-lattice start, projected unaligned.
    on = _project(_EPOCH + 3.0, effects, backend, None, rate, **kwargs)

    assert grid.is_on_lattice(off.t0)
    # Measured equal; the bound only allows for the two starts being computed by different expressions.
    assert off.t0 == pytest.approx(on.t0, abs=1e-6 / rate)
    a, b = np.asarray(off.data), np.asarray(on.data)
    assert a.shape == b.shape
    error = float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))
    assert error < _TOLERANCE, error


@pytest.mark.parametrize("fraction", [0.25, 0.612, 0.999])
def test_numpy_projection_lands_on_the_lattice(fraction: float) -> None:
    """The host path's strain at the lattice samples, for shifts across a sample."""
    _compare(_ROTATING, "numpy", fraction)


def _analytic_error(strain: TimeSeries, rate: float) -> float:
    """The largest difference from the analytic strain at the series' own lattice times, of its peak.

    Each sample's time is counted in whole samples from the epoch, so no GPS-sized time is subtracted.
    The count is a different floating-point expression from the projection's own (which keeps whole
    samples and the sub-sample fraction apart), so at a rate whose spacing is not a dyadic time the
    reference itself is accurate only to a few times 1e-13 of peak at 40 Hz.
    """
    from gwmock_signal.projection.geometry import reconstructed_geometry
    from gwmock_signal.projection.kernels import antenna_pattern, time_delay_from_geocenter
    from gwmock_signal.projection.sidereal import gmst_rad_astropy

    first = round((float(strain.t0) - _EPOCH) * rate)
    since_epoch = (first + np.arange(len(strain.data))) / rate
    response, location = reconstructed_geometry("E1")
    gmst = gmst_rad_astropy(_EPOCH + since_epoch)
    sky = (_SKY["right_ascension"], _SKY["declination"])
    f_plus, f_cross = antenna_pattern(response, gmst, *sky, _SKY["polarization_angle"], xp=np)
    retarded = since_epoch - 16.0 - time_delay_from_geocenter(location, gmst, *sky, xp=np)
    envelope = np.exp(-((retarded / 2.0) ** 2))
    phase = 2.0 * np.pi * 40.0 * retarded
    expected = f_plus * envelope * np.cos(phase) + f_cross * envelope * np.sin(phase)
    return float(np.max(np.abs(np.asarray(strain.data) - expected))) / float(np.max(np.abs(expected)))


def test_a_rate_whose_padding_is_not_a_dyadic_time_lands_on_the_lattice() -> None:
    """At 3000 Hz the padding is not a dyadic time, so subtracting it from a GPS-sized start rounds.

    The rounding is up to half a unit in the last place, 1.2e-7 s. A lattice offset taken from the padded
    start was wrong by up to 4e-4 of a sample, 4e-6 of peak at 40 Hz. It is taken from the caller's own
    start, before padding, as ``simulate_cbc_batch`` does. An unaligned projection carries the rounded
    label too (9e-11 of peak here), so the reference is the analytic strain rather than another
    projection. The measured 5e-13 is that reference's own floor at this rate (see ``_analytic_error``),
    not the projection's; the bound is still four million times below the defect.
    """
    rate = 3000.0
    grid = SamplingGrid(_EPOCH, rate)
    strain = _project(_EPOCH + 3.0 + 0.612 / rate, _ROTATING, "numpy", grid, rate)
    assert grid.is_on_lattice(strain.t0)
    error = _analytic_error(strain, rate)
    assert error < _TOLERANCE, error


def test_a_caller_keeping_its_own_margin_is_placed_by_its_own_start() -> None:
    """Without the delay padding, the lattice start is the one preceding the caller's own start.

    Stepping it back by a padding the polarizations were not given would misplace the whole series.
    The caller's margin, zeros here, covers the delay and the sub-sample shift.
    """
    from gwmock_signal.projection.network import _project as project_privately

    grid = SamplingGrid(_EPOCH, _FS)
    strain = project_privately(
        _polarizations(_EPOCH + 3.0 + 0.612 / _FS),
        ["E1"],
        **_SKY,
        effects=_ROTATING,
        backend="numpy",
        sinc_taps=DEFAULT_SINC_TAPS,
        kaiser_beta=DEFAULT_KAISER_BETA,
        keep_delayed_samples=False,
        output_grid=grid,
    )["E1"]
    assert strain.t0 == pytest.approx(_EPOCH + 3.0, abs=1e-6 / _FS)
    error = _analytic_error(strain, _FS)
    assert error < _TOLERANCE, error


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_the_oversampled_tapered_path_lands_on_the_lattice(backend: str) -> None:
    """With ``tapered``, the kernel reads a series oversampled by a factor, and the shift is scaled with it."""
    if backend == "jax":
        jax = pytest.importorskip("jax", reason="jax not installed")
        jax.config.update("jax_enable_x64", True)
    # The window is 1.6e-28 of its peak at the buffer's ends: a whole signal tapered to zero.
    _compare(_ROTATING, backend, 0.612, tapered=True)


def test_the_constant_pattern_branch_lands_on_the_lattice() -> None:
    """The frequency-domain branch takes the shift into its phase as well."""
    _compare(ProjectionEffects.long_wavelength(), "numpy", 0.612)


def test_the_finite_arm_response_lands_on_the_lattice() -> None:
    """The finite-arm residual acts on the output's samples, which are now the lattice's."""
    _compare(dataclasses.replace(_ROTATING, finite_arm_length=True, source_precession=True), "numpy", 0.612)


def test_the_solar_system_delays_land_on_the_lattice() -> None:
    """The delays are evaluated at the output's own times, as the pattern and the site delay are."""
    ephemeris = SolarSystemEphemeris()
    try:
        ephemeris.tables()
    except FileNotFoundError as exc:
        pytest.skip(f"the default ephemeris is unavailable: {exc}")
    effects = dataclasses.replace(_ROTATING, roemer_delay=True, einstein_delay=True, shapiro_delay=True)
    # One reference time for both, so both absorb the same delay.
    _compare(effects, "numpy", 0.612, ephemeris=ephemeris, reference_time=_EPOCH + 16.0)


def test_the_device_projection_lands_on_the_lattice() -> None:
    """The device path passes the shift to its kernel, so it agrees with the reference too."""
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    _compare(dataclasses.replace(_ROTATING, finite_arm_length=True), "jax", 0.612)


def test_a_grid_at_another_rate_is_refused() -> None:
    """A lattice at another rate cannot be reached by a sub-sample shift."""
    with pytest.raises(ValueError, match="sample rate"):
        _project(_EPOCH + 3.3, _ROTATING, "numpy", SamplingGrid(_EPOCH, 2 * _FS))


def test_the_simulator_passes_the_grid_on() -> None:
    """``simulate`` returns a compact binary's strain on the lattice it is given.

    Forwarding only: this checks where the strain starts, not its values, which the projection tests
    above check for every branch.
    """
    pytest.importorskip("lal", reason="lalsuite not installed")
    from gwmock_signal.simulator import CBCSimulator

    params = {
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "distance": 400.0,
        "inclination": 0.3,
        "coa_phase": 0.1,
        "coa_time": _EPOCH + 100.123456,
        **_SKY,
    }
    grid = SamplingGrid(_EPOCH, _FS)
    simulator = CBCSimulator("IMRPhenomD")
    common = {"sampling_frequency": _FS, "minimum_frequency": 20.0, "effects": _ROTATING}
    aligned = simulator.simulate(params, ["E1"], output_grid=grid, **common)
    unaligned = simulator.simulate(params, ["E1"], **common)
    assert not grid.is_on_lattice(unaligned.t0)
    assert grid.is_on_lattice(aligned.t0)
    assert 0.0 < unaligned.t0 - aligned.t0 < 1.0 / _FS


@pytest.mark.parametrize("rate", [1024.0, 4000.0])
def test_a_projection_on_the_backgrounds_lattice_is_added_by_whole_samples(rate: float) -> None:
    """Built on the background's lattice, the strain is added to it without being resampled again.

    At 4000 Hz a lattice time ``epoch + k / 4000`` misses its sample by up to half a float64 step, 4.8e-4 of one at GPS 1.4e9,
    beyond the 1e-4 within which an undeclared series is placed by whole samples. With
    ``interpolate_if_offset=False`` an undeclared series is then skipped altogether, which is how the
    test sees whether it was declared.
    """
    pytest.importorskip("lal", reason="lalsuite not installed")
    from gwmock_signal.simulator import CBCSimulator

    params = {
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "distance": 400.0,
        "inclination": 0.3,
        "coa_phase": 0.1,
        "coa_time": _EPOCH + 100.123456,
        **_SKY,
    }
    grid = SamplingGrid(_EPOCH, rate)
    background = {"E1": TimeSeries(np.zeros(int(200 * rate)), t0=_EPOCH, sample_rate=rate)}
    simulator = CBCSimulator("IMRPhenomD")
    common = {"sampling_frequency": rate, "minimum_frequency": 20.0, "effects": _ROTATING}
    added = simulator.simulate(params, ["E1"], background, output_grid=grid, interpolate_if_offset=False, **common)
    alone = simulator.simulate(params, ["E1"], output_grid=grid, **common)

    first = round((float(alone.t0) - _EPOCH) * rate)
    placed = np.asarray(added.data)[0, first : first + alone.data.shape[-1]]
    np.testing.assert_array_equal(placed, np.asarray(alone.data)[0])


def test_a_background_on_another_lattice_is_placed_as_before() -> None:
    """A background off ``output_grid`` is not the lattice the strain was built on, so it is resampled onto it."""
    pytest.importorskip("lal", reason="lalsuite not installed")
    from gwmock_signal.simulator import CBCSimulator

    rate = 4000.0
    params = {
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "distance": 400.0,
        "inclination": 0.3,
        "coa_phase": 0.1,
        "coa_time": _EPOCH + 100.123456,
        **_SKY,
    }
    background = {"E1": TimeSeries(np.zeros(int(200 * rate)), t0=_EPOCH + 0.3 / rate, sample_rate=rate)}
    common = {"sampling_frequency": rate, "minimum_frequency": 20.0, "effects": _ROTATING}
    simulator = CBCSimulator("IMRPhenomD")
    skipped = simulator.simulate(
        params, ["E1"], background, output_grid=SamplingGrid(_EPOCH, rate), interpolate_if_offset=False, **common
    )
    resampled = simulator.simulate(params, ["E1"], background, output_grid=SamplingGrid(_EPOCH, rate), **common)
    assert not np.any(np.asarray(skipped.data))
    assert np.any(np.asarray(resampled.data))
