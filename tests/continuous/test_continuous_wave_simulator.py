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

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

from gwmock_signal.continuous import ContinuousWaveSimulator

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
                earth_rotation=False,
            )

    def test_a_background_is_required(self):
        """A continuous wave has no duration of its own; the segment comes from the background."""
        simulator = _simulator()

        with pytest.raises(ValueError, match="requires a background"):
            simulator.simulate(_SOURCE, _DETECTORS, None, sampling_frequency=_FS, minimum_frequency=0.0)


class TestOutput:
    """Shape, placement and amplitude of what comes back."""

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

    def test_the_amplitude_is_of_the_configured_order(self):
        """Projection scales the polarizations by antenna factors below one, never above."""
        simulator = _simulator()
        strain = _generate(simulator, _EPOCH, int(600 * _FS))

        peak = float(np.max(np.abs(strain)))
        assert 0.0 < peak <= float(_SOURCE["amplitude_plus"]), f"peak {peak:.3e} is not a projected amplitude"
