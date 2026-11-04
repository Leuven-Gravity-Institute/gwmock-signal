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
"""Tests for the continuous-wave simulator.

The load-bearing one is :func:`test_segmented_generation_matches_one_long_call`. A continuous wave
is generated one analysis segment at a time but searched coherently over the whole run, so a phase
discontinuity at a boundary destroys the signal while leaving every individual segment looking
entirely correct. Nothing about a single segment's output can reveal it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

from gwmock_signal.continuous import ContinuousWaveSimulator
from gwmock_signal.projection import network
from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.network import _project
from gwmock_signal.projection.resampling import edge_padding

_EARTH = "earth00-40-DE405.dat.gz"
_SUN = "sun00-40-DE405.dat.gz"
_EPOCH = 1577491218.0
_FS = 64.0
_DETECTORS = ["H1", "L1"]

_SOURCE = {
    "right_ascension": 1.1,
    "declination": 0.3,
    "frequency": 20.0,
    "initial_phase": 0.4,
    "amplitude_plus": 1.0e-24,
    "amplitude_cross": 7.0e-25,
    "polarization_angle": 0.2,
}


def _simulator(**overrides) -> ContinuousWaveSimulator:
    kwargs = {
        "earth_ephemeris": _EARTH,
        "sun_ephemeris": _SUN,
        "reference_time_ssb": _EPOCH,
        "spindowns": (-1.0e-10,),
    }
    kwargs.update(overrides)
    return ContinuousWaveSimulator(**kwargs)


def _zeros(epoch: float, n_samples: int) -> dict[str, TimeSeries]:
    return {name: TimeSeries(np.zeros(n_samples), t0=epoch, sample_rate=_FS, unit="strain") for name in _DETECTORS}


def _generate(simulator: ContinuousWaveSimulator, epoch: float, n_samples: int) -> np.ndarray:
    stack = simulator.simulate(
        _SOURCE,
        _DETECTORS,
        _zeros(epoch, n_samples),
        sampling_frequency=_FS,
        minimum_frequency=0.0,
    )
    return np.asarray(stack[_DETECTORS[0]], dtype=float)


class TestPhaseCoherenceAcrossSegments:
    """The property that makes a continuous wave usable, and the one nothing else would catch."""

    @pytest.mark.requires_gwpy
    def test_segmented_generation_matches_one_long_call(self):
        """Three consecutive segments must reconstruct the single long signal.

        This is the test the design exists for, and two separate mechanisms had to be right before
        it could pass.

        ``reference_time_ssb`` is fixed for the run: ripple derives it from the first sample it is
        given when left unset, which restarts the phase at every segment boundary. Measured that
        way, segments one and two disagreed with the long call by of order the signal amplitude
        itself while segment zero matched perfectly -- the signature of a per-segment phase origin,
        and invisible in any single segment.

        Polarizations are generated with a margin beyond both ends: the projection resamples with a
        windowed sinc kernel whose taps would otherwise read zero past a segment edge. Without the
        margin the worst disagreement was 1.4e-25 against a 4e-25 signal; with it, 6.0e-36.

        The tolerance is not bit-exactness, because the two are not the same computation: the long
        call FFTs 115200 samples and each segment 38400, and different lengths reassociate
        differently. That floor sits at ~1e-11 of peak. 1e-9 is loose enough to clear it and far
        too tight to admit a phase discontinuity, which shows up at order one.
        """
        segment_samples = int(600 * _FS)
        simulator = _simulator()

        whole = _generate(simulator, _EPOCH, segment_samples * 3)
        stitched = np.concatenate([_generate(simulator, _EPOCH + 600.0 * index, segment_samples) for index in range(3)])

        assert stitched.shape == whole.shape
        peak = float(np.max(np.abs(whole)))
        worst = float(np.max(np.abs(stitched - whole))) / peak
        assert worst < 1e-9, f"segments disagree with the continuous signal by {worst:.3e} of peak"

    def test_a_fractional_epoch_changes_the_signal(self):
        """The sub-second part of the start time must reach the phase.

        Every other test starts on a whole second, so a generator that floored its start would pass
        them all. Two starts 0.375 s apart within the same GPS second must give different output.
        """
        simulator = _simulator()
        n_samples = int(60 * _FS)

        # Deliberately against the private generator rather than `simulate`, whose output also differs
        # between two epochs through the antenna pattern, so it would pass either way.
        kwargs = {"n_samples": n_samples, "sampling_frequency": _FS}
        on_the_second, _ = simulator._ssb_polarizations(_SOURCE, start_ssb=_EPOCH, **kwargs)
        part_way_through, _ = simulator._ssb_polarizations(_SOURCE, start_ssb=_EPOCH + 0.375, **kwargs)

        assert not np.array_equal(on_the_second, part_way_through), (
            "starts 0.375 s apart within the same GPS second produced identical polarizations, so "
            "the sub-second part of the start is being discarded"
        )

    @pytest.mark.requires_gwpy
    def test_segments_join_up_on_a_fractional_epoch_too(self):
        """The seam property must hold when the run does not start on a whole second."""
        segment_samples = int(600 * _FS)
        epoch = _EPOCH + 0.375
        simulator = _simulator()

        whole = _generate(simulator, epoch, segment_samples * 2)
        stitched = np.concatenate([_generate(simulator, epoch + 600.0 * index, segment_samples) for index in range(2)])

        peak = float(np.max(np.abs(whole)))
        worst = float(np.max(np.abs(stitched - whole))) / peak
        assert worst < 1e-9, f"fractional-epoch segments disagree by {worst:.3e} of peak"

    @pytest.mark.requires_gwpy
    def test_a_later_segment_is_not_a_repeat_of_the_first(self):
        """Guards the opposite failure: agreement achieved by generating the same thing twice.

        A simulator that ignored the epoch entirely would satisfy the test above, because every
        segment would be identical and so would their concatenation with a signal generated the
        same wrong way.
        """
        simulator = _simulator()
        n_samples = int(600 * _FS)

        first = _generate(simulator, _EPOCH, n_samples)
        later = _generate(simulator, _EPOCH + 600.0, n_samples)

        assert not np.array_equal(first, later), "the segment content does not depend on its epoch"


class TestConstruction:
    """What the class refuses, and why."""

    def test_the_ssb_reference_is_required(self):
        """There is no default, because the tempting default is the one that breaks coherence."""
        with pytest.raises(TypeError):
            ContinuousWaveSimulator(earth_ephemeris=_EARTH, sun_ephemeris=_SUN)  # type: ignore[call-arg]

    def test_a_non_finite_ssb_reference_is_refused(self):
        """A NaN or infinite reference would poison every sample without saying why."""
        with pytest.raises(ValueError, match="finite"):
            _simulator(reference_time_ssb=float("nan"))

    @pytest.mark.requires_gwpy
    def test_required_parameters_are_named_when_missing(self):
        """The error names the absent key, rather than failing later inside ripple."""
        simulator = _simulator()
        incomplete = {k: v for k, v in _SOURCE.items() if k != "frequency"}

        with pytest.raises(ValueError, match="frequency"):
            simulator.simulate(
                incomplete,
                _DETECTORS,
                _zeros(_EPOCH, 64),
                sampling_frequency=_FS,
                minimum_frequency=0.0,
            )

    @pytest.mark.requires_gwpy
    def test_the_constant_antenna_pattern_branch_is_refused(self):
        """Refused, not warned about: no continuous wave is short enough for it to be valid.

        ``earth_rotation=False`` holds the pattern fixed at the midpoint of the span. That is a
        reasonable trade for a sub-second merger and meaningless for a signal running for months --
        and the output would look entirely normal while being wrong.
        """
        simulator = _simulator()

        with pytest.raises(ValueError, match="earth_rotation=False is not available"):
            simulator.simulate(
                _SOURCE,
                _DETECTORS,
                _zeros(_EPOCH, 64),
                sampling_frequency=_FS,
                minimum_frequency=0.0,
                effects=ProjectionEffects.long_wavelength(earth_rotation=False),
            )

    def test_non_finite_spindowns_are_refused(self):
        """A NaN spindown poisons every sample, with nothing in the output naming the cause."""
        with pytest.raises(ValueError, match="spindowns must all be finite"):
            _simulator(spindowns=(-1.0e-10, float("nan")))

    @pytest.mark.requires_gwpy
    def test_background_channels_must_share_a_grid(self):
        """The polarizations are generated once, for one epoch and length, and added to all.

        A channel describing a different stretch of time would silently receive a signal from the
        wrong interval, because only the first channel is consulted for the epoch and length.
        """
        simulator = _simulator()
        background = _zeros(_EPOCH, 640)
        background[_DETECTORS[1]] = TimeSeries(np.zeros(320), t0=_EPOCH, sample_rate=_FS, unit="strain")

        with pytest.raises(ValueError, match="must share a length"):
            simulator.simulate(_SOURCE, _DETECTORS, background, sampling_frequency=_FS, minimum_frequency=0.0)

    @pytest.mark.requires_gwpy
    def test_background_channels_must_share_an_epoch(self):
        """Same hazard as the length check, and equally invisible in the output."""
        simulator = _simulator()
        background = _zeros(_EPOCH, 640)
        background[_DETECTORS[1]] = TimeSeries(np.zeros(640), t0=_EPOCH + 100.0, sample_rate=_FS, unit="strain")

        with pytest.raises(ValueError, match="must share an epoch"):
            simulator.simulate(_SOURCE, _DETECTORS, background, sampling_frequency=_FS, minimum_frequency=0.0)

    @pytest.mark.requires_gwpy
    def test_a_sampling_frequency_disagreeing_with_the_background_is_refused(self):
        """The argument drives generation; the background defines the grid it is added to.

        A mismatch generates the wave at one rate and adds it to another elementwise, with no
        complaint from gwpy -- the signal comes out time-stretched by the ratio while every channel
        still looks individually plausible.
        """
        simulator = _simulator()

        with pytest.raises(ValueError, match="but the background is at"):
            simulator.simulate(
                _SOURCE,
                _DETECTORS,
                _zeros(_EPOCH, 640),
                sampling_frequency=_FS * 2,
                minimum_frequency=0.0,
            )

    @pytest.mark.requires_gwpy
    @pytest.mark.parametrize("key", sorted(_SOURCE.keys() - {"polarization_angle"}))
    def test_non_finite_source_parameters_are_refused(self, key: str):
        """A NaN anywhere in the source returns an all-NaN series with nothing naming the cause."""
        simulator = _simulator()
        params = dict(_SOURCE)
        params[key] = float("nan")

        with pytest.raises(ValueError, match=f"{key} must be finite"):
            simulator.simulate(params, _DETECTORS, _zeros(_EPOCH, 64), sampling_frequency=_FS, minimum_frequency=0.0)

    @pytest.mark.requires_gwpy
    @pytest.mark.parametrize(
        ("frequency", "expected"),
        [(0.0, "must be positive"), (-20.0, "must be positive"), (_FS / 2, "Nyquist"), (_FS, "Nyquist")],
    )
    def test_unrepresentable_frequencies_are_refused(self, frequency: float, expected: str):
        """Zero and negative are not signals; at or above Nyquist the tone aliases silently."""
        simulator = _simulator()
        params = dict(_SOURCE, frequency=frequency)

        with pytest.raises(ValueError, match=expected):
            simulator.simulate(params, _DETECTORS, _zeros(_EPOCH, 64), sampling_frequency=_FS, minimum_frequency=0.0)

    def test_a_background_of_plain_arrays_is_refused_clearly(self):
        """Otherwise it dies later with an AttributeError about `t0`, which names nothing useful."""
        simulator = _simulator()
        background = {name: np.zeros(64) for name in _DETECTORS}

        with pytest.raises(TypeError, match=r"must be gwmock_signal\.TimeSeries or GWpy TimeSeries"):
            simulator.simulate(_SOURCE, _DETECTORS, background, sampling_frequency=_FS, minimum_frequency=0.0)

    def test_a_background_is_required(self):
        """A continuous wave has no duration of its own; the segment comes from the background."""
        simulator = _simulator()

        with pytest.raises(ValueError, match="requires a background"):
            simulator.simulate(_SOURCE, _DETECTORS, None, sampling_frequency=_FS, minimum_frequency=0.0)


class TestOutput:
    """Shape, placement and amplitude of what comes back."""

    @pytest.mark.requires_gwpy
    def test_the_signal_is_added_to_the_background(self):
        """A continuous wave joins existing data rather than replacing it."""
        simulator = _simulator()
        n_samples = int(60 * _FS)
        offset = 1.0e-21
        background = {
            name: TimeSeries(np.full(n_samples, offset), t0=_EPOCH, sample_rate=_FS, unit="strain")
            for name in _DETECTORS
        }

        stack = simulator.simulate(_SOURCE, _DETECTORS, background, sampling_frequency=_FS, minimum_frequency=0.0)
        without = _generate(simulator, _EPOCH, n_samples)

        # Peak-relative, not a small absolute tolerance: adding a 1e-21 offset to a 1e-25 signal
        # and subtracting it again cancels catastrophically, losing about 1e-21 * eps ~ 1e-37 --
        # far above any atol that would look strict for strain of this size.
        recovered = np.asarray(stack[_DETECTORS[0]], dtype=float) - offset
        peak = float(np.max(np.abs(without)))
        assert float(np.max(np.abs(recovered - without))) / peak < 1e-9

    @pytest.mark.requires_gwpy
    def test_every_detector_gets_a_distinct_projection(self):
        """H1 and L1 see different antenna patterns and different delays."""
        simulator = _simulator()
        stack = simulator.simulate(
            _SOURCE,
            _DETECTORS,
            _zeros(_EPOCH, int(60 * _FS)),
            sampling_frequency=_FS,
            minimum_frequency=0.0,
        )

        h1 = np.asarray(stack["H1"], dtype=float)
        l1 = np.asarray(stack["L1"], dtype=float)

        assert not np.array_equal(h1, l1), "both detectors received identical strain"
        assert np.isfinite(h1).all()
        assert np.isfinite(l1).all()

    @pytest.mark.requires_gwpy
    def test_the_amplitude_is_of_the_configured_order(self):
        """Projection scales the polarizations by antenna factors below one, never above."""
        simulator = _simulator()
        strain = _generate(simulator, _EPOCH, int(600 * _FS))

        peak = float(np.max(np.abs(strain)))
        assert 0.0 < peak <= float(_SOURCE["amplitude_plus"]), f"peak {peak:.3e} is not a projected amplitude"


class TestTheSpanLimitIsMeasuredOnThePaddedSpan:
    """The device path's span limit applies to the buffer, not the nominal segment.

    `simulate` hands the projection a buffer padded at both ends by `edge_padding`, so the span
    the guard measures is longer than the segment the caller asked for. Documented in
    `ContinuousWaveSimulator`'s constructor docstring, and the reason a nominal-length segment at
    the limit is refused: the padding is what pushes it over.
    """

    @pytest.mark.requires_gwpy
    def test_the_buffer_the_simulator_generates_is_padded_at_both_ends(self, monkeypatch):
        """Pin the sizing itself, which is what makes the guard see a longer span.

        The polarizations handed to the projection start ``edge_padding`` samples before the
        background, and are that much longer at each end, and longer again by the range of the
        solar-system delay across the segment.
        """
        from gwmock_signal.continuous import simulator as module

        simulator = _simulator()
        seen: dict[str, float] = {}

        def spy(polarizations, *args, **kwargs):
            seen.update(t0=float(polarizations["plus"].t0), n_samples=len(polarizations["plus"].data))
            return _project(polarizations, *args, **kwargs)

        monkeypatch.setattr(module, "_project", spy)
        n_samples = int(600 * _FS)
        _generate(simulator, _EPOCH, n_samples)

        margin = edge_padding(_FS)
        assert margin > 0
        assert seen["n_samples"] >= n_samples + 2 * margin
        # Over 600 s the delay changes by at most 1e-4 s per second, 0.06 s, plus a sample at each end.
        assert seen["n_samples"] <= n_samples + 2 * margin + math.ceil(0.06 * _FS) + 2
        assert seen["t0"] == pytest.approx(_EPOCH - margin / _FS, rel=0, abs=1e-6)

    @pytest.mark.requires_gwpy
    def test_the_jax_guard_refuses_between_the_nominal_and_padded_spans(self, monkeypatch):
        """With the limit set between the two, only a padded-span guard refuses.

        This is the documented consequence, tested end to end rather than recomputed: the same
        call is refused when the limit sits above its nominal span but below its padded one.
        """
        pytest.importorskip("jax", reason="the [jax] extra is not installed")
        seconds = 120.0
        n_samples = int(seconds * _FS)
        margin = edge_padding(_FS)
        nominal = (n_samples - 1) / _FS
        padded = (n_samples + 2 * margin - 1) / _FS
        assert nominal < padded

        monkeypatch.setattr(network, "MAX_LINEAR_SIDEREAL_SPAN_SECONDS", 0.5 * (nominal + padded))
        simulator = _simulator()  # defaults to the device backend

        with pytest.raises(ValueError, match="accepts spans up to"):
            simulator.simulate(
                _SOURCE, _DETECTORS, _zeros(_EPOCH, n_samples), sampling_frequency=_FS, minimum_frequency=0.0
            )


class TestANonFinitePolarizationIsRefused:
    """A non-finite signal must be refused, not written.

    The phase is a polynomial in the time from the reference epoch, so an extreme spindown far from
    that epoch overflows it; ``_validate_source`` requires the parameters finite but does not bound them.
    """

    @staticmethod
    def _source() -> dict[str, float]:
        return {
            "right_ascension": 1.1,
            "declination": 0.3,
            "frequency": 20.0,
            "initial_phase": 0.4,
            "amplitude_plus": 1e-24,
            "amplitude_cross": 7e-25,
        }

    @pytest.mark.requires_gwpy
    def test_it_is_refused_through_the_public_entry_point(self):
        """Guards the placement, not just the condition: nothing non-finite reaches a caller.

        An extreme spindown far from the reference epoch overflows the phase; see
        ``test_an_out_of_range_spindown_overflows_the_phase`` for the mechanism.
        """
        simulator = _simulator(spindowns=(1e300,), reference_time_ssb=0.0)
        background = {
            name: TimeSeries(np.zeros(64), t0=_EPOCH, sample_rate=64.0, unit="strain") for name in ("H1", "L1")
        }

        with pytest.raises(RuntimeError, match="not all finite"):
            simulator.simulate(
                self._source(),
                ["H1", "L1"],
                background,
                sampling_frequency=64.0,
                minimum_frequency=5.0,
            )

    def test_a_finite_signal_is_returned_unchanged(self):
        """The guard must not reject an ordinary signal."""
        plus, cross = _simulator()._ssb_polarizations(
            self._source(), start_ssb=_EPOCH, n_samples=64, sampling_frequency=64.0
        )

        assert np.all(np.isfinite(plus))
        assert np.all(np.isfinite(cross))
        assert np.max(np.abs(plus)) > 0.0


def test_an_out_of_range_spindown_overflows_the_phase():
    """The guard's non-ripple cause, exercised rather than asserted.

    Its message leads with the source parameters, and until this test that claim rested on nothing:
    every other guard test monkeypatches ripple, so none showed the library producing NaN from
    finite input. A review pointed that out, and trying to reproduce it corrected the claim as well
    as substantiating it -- a large spindown alone is *not* enough. `f1 = 1e300` stays finite with
    the reference epoch beside the data, because the spindown term grows as the square of the time
    from it; the two together overflow.

    Absurd values, deliberately: the point is that ``_validate_source`` bounds finiteness and not
    magnitude, so this is reachable through the public interface rather than only in principle.
    """
    simulator = _simulator(spindowns=(1e300,), reference_time_ssb=0.0)

    with pytest.raises(RuntimeError, match="not all finite"):
        simulator._ssb_polarizations(
            {
                "right_ascension": 1.1,
                "declination": 0.3,
                "frequency": 20.0,
                "initial_phase": 0.4,
                "amplitude_plus": 1e-24,
                "amplitude_cross": 7e-25,
            },
            start_ssb=_EPOCH,
            n_samples=64,
            sampling_frequency=64.0,
        )


def test_the_same_spindown_is_fine_with_a_nearby_reference_epoch():
    """Pins the second half of the mechanism, so the test above cannot be misread.

    Without this, the case above reads as "a large spindown produces NaN", which is false and would
    send someone bounding spindowns rather than looking at the epoch difference.
    """
    simulator = _simulator(spindowns=(1e300,), reference_time_ssb=_EPOCH)

    plus, _ = simulator._ssb_polarizations(
        {
            "right_ascension": 1.1,
            "declination": 0.3,
            "frequency": 20.0,
            "initial_phase": 0.4,
            "amplitude_plus": 1e-24,
            "amplitude_cross": 7e-25,
        },
        start_ssb=_EPOCH,
        n_samples=64,
        sampling_frequency=64.0,
    )

    assert np.all(np.isfinite(plus))


@pytest.mark.requires_gwpy
def test_the_continuous_wave_path_precesses_the_source_direction():
    """This simulator must opt into the frame-of-date convention; the CBC path must not.

    LAL uses two sky conventions and they differ by 1.8e-04 s of geocentre-to-detector delay:
    ``lalpulsar.XLALBarycenter`` precesses, ``lal.XLALTimeDelayFromEarthCenter`` does not. A
    continuous wave needs the first, because the SSB-to-geocentre part of its phase already came from
    a barycentering routine that precessed, so the site term added to it has to match or the seam
    carries that offset -- across a months-long coherent integration, where it does not average away.

    Asserted on the call rather than on the strain because
    ``test_the_flag_selects_between_lals_two_conventions`` already measures what the flag *does*
    against both LAL references. What is unpinned without this is the wiring: which value this
    simulator passes. The projection's default is the compact-binary one, so the wiring is the whole
    difference, and silently losing it would leave every continuous-wave test still green.
    """
    from unittest.mock import patch

    simulator = _simulator()
    n_samples = int(8 * _FS)
    background = {
        name: TimeSeries(np.zeros(n_samples), t0=_EPOCH, sample_rate=_FS, unit="strain") for name in _DETECTORS
    }
    with patch(
        "gwmock_signal.continuous.simulator._project",
        side_effect=_project,
    ) as spy:
        simulator.simulate(_SOURCE, _DETECTORS, background, sampling_frequency=_FS, minimum_frequency=0.0)

    assert spy.call_count > 0, "the projection was never called, so this asserts nothing"
    for call in spy.call_args_list:
        assert call.kwargs["effects"].source_precession is True, (
            "the continuous-wave path must use the frame-of-date convention by default"
        )


@pytest.mark.requires_gwpy
def test_a_background_of_native_series_gives_the_same_strain_as_gwpy():
    """The package's own series and GWpy's, as background, give identical continuous-wave strain."""
    from gwmock_signal.timeseries import TimeSeries as NativeTimeSeries

    simulator = _simulator()
    n_samples = int(8 * _FS)
    native = {
        name: NativeTimeSeries(np.zeros(n_samples), t0=_EPOCH, sample_rate=_FS, unit="strain") for name in _DETECTORS
    }

    from_native = simulator.simulate(_SOURCE, _DETECTORS, native, sampling_frequency=_FS, minimum_frequency=0.0)
    from_gwpy = simulator.simulate(
        _SOURCE, _DETECTORS, _zeros(_EPOCH, n_samples), sampling_frequency=_FS, minimum_frequency=0.0
    )

    assert np.array_equal(from_native.data, from_gwpy.data)
    assert np.any(from_native.data != 0.0)


def test_the_finite_arm_option_reaches_the_projection(monkeypatch):
    """The effects are passed on to the projection."""
    seen = {}

    def recording(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("recorded")

    monkeypatch.setattr("gwmock_signal.continuous.simulator._project", recording)
    simulator = _simulator(projection_backend="numpy")
    with pytest.raises(RuntimeError, match="recorded"):
        simulator.simulate(
            _SOURCE,
            _DETECTORS,
            _zeros(_EPOCH, 64),
            sampling_frequency=_FS,
            minimum_frequency=0.0,
            effects=ProjectionEffects(shapiro_delay=False),
        )
    assert seen["effects"] == ProjectionEffects(shapiro_delay=False)


@pytest.mark.requires_gwpy
def test_the_signal_is_the_pulsar_phase_at_lalpulsars_barycentric_time():
    """End to end against LALPulsar: the phase evaluated at ``t - tau + deltaT``, sample by sample.

    ``deltaT`` comes from ``lalpulsar.Barycenter`` for a detector at the geocentre, the part every source
    now shares; the site delay and antenna pattern from the projection's kernels, whose own anchors are in
    ``tests/projection``. What is left is the resampling kernel's error at this frequency.
    """
    lal = pytest.importorskip("lal")
    lalpulsar = pytest.importorskip("lalpulsar")
    from gwmock_signal.projection import kernels
    from gwmock_signal.projection.geometry import reconstructed_geometry
    from gwmock_signal.projection.network import _gmst_accurate_array
    from gwmock_signal.projection.sidereal import precessed_sky_anchor_and_rate

    simulator = _simulator(projection_backend="numpy")
    earth, sun = simulator.ephemeris.tables()
    state = lalpulsar.InitBarycenter(earth.source, sun.source)
    n_samples = int(120 * _FS)
    # Finite arms left out: the reference treats the detector as its vertex.
    stack = simulator.simulate(
        _SOURCE,
        _DETECTORS,
        _zeros(_EPOCH, n_samples),
        sampling_frequency=_FS,
        minimum_frequency=0.0,
        effects=ProjectionEffects(finite_arm_length=False),
    )
    strain = np.asarray(stack[_DETECTORS[0]], dtype=float)

    offsets = np.arange(n_samples) / _FS
    (ra, dec), (d_ra, d_dec) = precessed_sky_anchor_and_rate(_SOURCE["right_ascension"], _SOURCE["declination"], _EPOCH)
    gmst = _gmst_accurate_array(_EPOCH + offsets)
    sky = (gmst, ra + d_ra * offsets, dec + d_dec * offsets)
    response, location = reconstructed_geometry(_DETECTORS[0])
    site = kernels.time_delay_from_geocenter(location, *sky, xp=np)
    f_plus, f_cross = kernels.antenna_pattern(response, *sky, _SOURCE["polarization_angle"], xp=np)

    delta = np.empty(n_samples)
    for index, offset in enumerate(offsets - site):
        gps = lal.LIGOTimeGPS(_EPOCH) + float(offset)
        earth_state = lalpulsar.EarthState()
        lalpulsar.BarycenterEarth(earth_state, gps, state)
        given = lalpulsar.BarycenterInput()
        given.site, given.tgps = lal.Detector(), gps
        given.alpha, given.delta, given.dInv = _SOURCE["right_ascension"], _SOURCE["declination"], 0.0
        emitted = lalpulsar.EmissionTime()
        lalpulsar.Barycenter(emitted, given, earth_state)
        delta[index] = emitted.deltaT
    tau = (_EPOCH - simulator.reference_time_ssb) + offsets - site + delta
    phase = _SOURCE["initial_phase"] + 2 * np.pi * (_SOURCE["frequency"] * tau + simulator.spindowns[0] * tau**2 / 2)
    expected = f_plus * _SOURCE["amplitude_plus"] * np.cos(phase) + f_cross * _SOURCE["amplitude_cross"] * np.sin(phase)

    interior = slice(n_samples // 20, -n_samples // 20)
    error = np.max(np.abs(strain[interior] - expected[interior])) / np.max(np.abs(expected))
    assert error < 1e-6, f"{error:.2e} of peak from the phase at LALPulsar's barycentric time"


@pytest.mark.requires_gwpy
def test_the_device_and_host_projections_agree_with_every_effect():
    """The default device projection carries every effect, the solar-system delay in full, as the host does.

    The difference left is float64 on a delay of up to 500 s, about 1e-13 s.
    """
    pytest.importorskip("jax", reason="the [jax] extra is not installed")
    n_samples = int(256 * _FS)
    device, host = (
        _generate(_simulator(projection_backend=backend), _EPOCH, n_samples) for backend in ("jax", "numpy")
    )
    assert np.max(np.abs(device - host)) < 1e-9 * np.max(np.abs(host))
