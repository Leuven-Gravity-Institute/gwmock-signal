"""Placing a burst in segmented data: the placement time, and the duration queries.

A burst has two times. ``peak_time`` is its physical reference and alone positions the waveform;
``coa_time`` is the key a caller writing segmented data places every transient by, and may lie
anywhere inside the burst's output. The caller asks how far the output reaches either side of the
placement time, so the segment claiming the event is the one its waveform starts in and the rest is
carried into the next. The property under test is agreement: the extent a simulator promises,
taken from the placement time, must be where the strain it returns starts and ends, for every
family, at more than one sample rate and for any placement time inside the output. A test
recomputing a sizing formula would pass against a simulator whose generation had drifted from what
it reports, which is the failure that matters.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from gwmock_signal.burst import BurstSimulator, WaveformFile
from gwmock_signal.projection.resampling import edge_padding

PEAK = 1_000_000_000.0
SKY = {"right_ascension": 1.0, "declination": 0.3, "polarization_angle": 0.2}

#: Every ad hoc morphology with the shape parameters that size its buffer, spanning the draft
#: families' extremes: the shortest and longest sine-Gaussians, both Gaussian widths, and the
#: three white-noise bands. Each is valid at both rates below.
_FAMILIES = {
    "sg-q3-high": {"burst_model": "sine_gaussian", "quality_factor": 3.0, "frequency": 900.0, "inclination": 0.3},
    "sg-q9": {"burst_model": "sine_gaussian", "quality_factor": 9.0, "frequency": 235.0, "inclination": 1.1},
    "sg-q100-30hz": {"burst_model": "sine_gaussian", "quality_factor": 100.0, "frequency": 30.0, "inclination": 2.0},
    "ga-0.1ms": {"burst_model": "gaussian", "duration": 1.0e-4},
    "ga-2.5ms": {"burst_model": "gaussian", "duration": 2.5e-3},
    "wnb-100": {
        "burst_model": "white_noise_burst",
        "low_frequency": 100.0,
        "bandwidth": 100.0,
        "duration": 0.1,
        "inclination": 0.4,
        "seed": 7,
    },
    "wnb-250": {
        "burst_model": "white_noise_burst",
        "low_frequency": 250.0,
        "bandwidth": 100.0,
        "duration": 0.1,
        "inclination": 0.4,
        "seed": 8,
    },
    "wnb-700": {
        "burst_model": "white_noise_burst",
        "low_frequency": 700.0,
        "bandwidth": 100.0,
        "duration": 0.1,
        "inclination": 0.4,
        "seed": 9,
    },
}


def _event(family: str, **overrides):
    event = {**_FAMILIES[family], "hrss": 1e-21, "peak_time": PEAK, **SKY}
    event.update(overrides)
    return event


#: Where a population may put ``coa_time``: on the peak (the simple choice), at the start of the
#: polarization buffer (a convention existing populations use), or at an arbitrary point inside.
_PLACEMENTS = ("peak", "buffer start", "interior")


def _placed(simulator, event, rate, placement):
    """Return *event* with ``coa_time`` set as *placement* describes."""
    if placement == "peak":
        return {**event, "coa_time": event["peak_time"]}
    hp, _ = simulator.generate_polarizations(event, rate, 5.0)
    if placement == "buffer start":
        return {**event, "coa_time": float(hp.t0)}
    return {**event, "coa_time": float(hp.t0) + 0.37 * hp.n_samples / hp.sample_rate}


def _assert_extent_matches_output(simulator, event, rate):
    """The extent promised from the placement time is where ``simulate`` output starts and ends."""
    placement = float(event.get("coa_time", event["peak_time"]))
    lead = simulator.pre_coalescence_duration(event, rate, 5.0)
    tail = simulator.post_coalescence_duration(event, rate, 5.0)
    hp, hc = simulator.generate_polarizations(event, rate, 5.0)
    strain = simulator.simulate(event, ["H1"], sampling_frequency=rate, minimum_frequency=5.0)["H1"]
    pad = edge_padding(rate) / rate

    assert lead >= 0
    assert tail >= 0
    # A bound, never shorter than the real extent: microseconds of slack for float rounding at GPS
    # epochs (1e9 s carries about 1e-7 s), far below one sample.
    slack = 1e-6
    assert placement - lead <= strain.t0 + slack
    assert strain.end <= placement + tail + slack
    # The padding is part of the promise: the polarization buffer alone sits that far inside it.
    for series in (hp, hc):
        assert placement - lead + pad <= series.t0 + slack
        assert series.end <= placement + tail - pad + slack
    # And exact: the promise is where the output starts and ends, not a generous margin.
    assert lead == pytest.approx(placement - strain.t0, abs=0.5 / rate)
    assert tail == pytest.approx(strain.end - placement, abs=0.5 / rate)


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("placement", _PLACEMENTS)
@pytest.mark.parametrize("rate", [8192.0, 2048.0])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_duration_queries_bound_every_ad_hoc_family(family, rate, placement):
    """Each family's start and end answers contain its projected output, padding included."""
    simulator = BurstSimulator()
    _assert_extent_matches_output(simulator, _placed(simulator, _event(family), rate, placement), rate)


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("placement", _PLACEMENTS)
@pytest.mark.parametrize("rate", [8192.0, 2048.0])
def test_duration_queries_bound_a_file_waveform(tmp_path, rate, placement):
    """A numerical waveform's extent follows its own time column, resampled or not."""
    file_rate = 4096.0
    times = np.arange(-0.3, 0.7, 1.0 / file_rate)
    envelope = np.exp(-(((times - 0.1) / 0.05) ** 2))
    plus = envelope * np.sin(2 * np.pi * 150.0 * times)
    cross = envelope * np.cos(2 * np.pi * 150.0 * times)
    path = tmp_path / "model.txt"
    np.savetxt(path, np.column_stack([times, plus, cross]))
    simulator = BurstSimulator(waveform_files={"model": WaveformFile(path, reference_distance=0.01)})
    event = {"burst_model": "model", "peak_time": PEAK, "distance": 0.01, **SKY}
    _assert_extent_matches_output(simulator, _placed(simulator, event, rate, placement), rate)


@pytest.mark.requires_lalsuite
def test_the_long_sine_gaussian_reaches_seconds_before_its_peak():
    """The case a segment boundary crops: a Q = 100 sine-Gaussian at 30 Hz starts 5.6 s early."""
    lead = BurstSimulator().pre_coalescence_duration(_event("sg-q100-30hz"), 8192.0, 5.0)
    assert 5.5 < lead < 5.7


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_moving_coa_time_shifts_the_answers_by_exactly_the_offset(family):
    """Measured from the placement time: ``coa_time = peak + d`` adds d to pre, removes it from post.

    Durations measured from ``peak_time`` regardless would leave both unchanged, and ones measured
    from the wrong end would move them the wrong way; either fails here.
    """
    simulator = BurstSimulator()
    event = _event(family)
    lead = simulator.pre_coalescence_duration(event, 8192.0, 5.0)
    tail = simulator.post_coalescence_duration(event, 8192.0, 5.0)
    # A power of two, so PEAK + offset is exact, and small enough to stay inside the shortest output.
    for offset in (2.0**-7, -(2.0**-7)):
        moved = {**event, "coa_time": PEAK + offset}
        assert simulator.pre_coalescence_duration(moved, 8192.0, 5.0) == pytest.approx(lead + offset, abs=1e-12)
        assert simulator.post_coalescence_duration(moved, 8192.0, 5.0) == pytest.approx(tail - offset, abs=1e-12)


def _calls(simulator, rate=4096.0):
    """Every entry point that must refuse the same placements, keyed by name."""
    return {
        "pre": lambda event: simulator.pre_coalescence_duration(event, rate, 5.0),
        "post": lambda event: simulator.post_coalescence_duration(event, rate, 5.0),
        "polarizations": lambda event: simulator.generate_polarizations(event, rate, 5.0),
        "simulate": lambda event: simulator.simulate(event, ["H1"], sampling_frequency=rate, minimum_frequency=5.0),
    }


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("call", ["pre", "post", "polarizations", "simulate"])
@pytest.mark.parametrize("family", ["sg-q9", "ga-0.1ms", "wnb-700"])
def test_a_coa_time_outside_the_output_is_refused_everywhere(family, side, call):
    """One sample outside the output is refused by the queries and by generation alike.

    A caller that treats a failed query as "duration unknown" and places by ``coa_time`` anyway
    then fails in generation instead of injecting a cropped burst.
    """
    simulator = BurstSimulator()
    event = _event(family)
    lead = simulator.pre_coalescence_duration(event, 4096.0, 5.0)
    tail = simulator.post_coalescence_duration(event, 4096.0, 5.0)
    coa_time = PEAK - lead - 1.0 / 4096.0 if side == "before" else PEAK + tail + 1.0 / 4096.0
    with pytest.raises(ValueError, match="lies outside this burst's output") as refused:
        _calls(BurstSimulator() if call in {"polarizations", "simulate"} else simulator)[call](
            {**event, "coa_time": coa_time}
        )
    # The same check, so the same message, whichever entry point raised it.
    with pytest.raises(ValueError, match="lies outside") as from_query:
        simulator.pre_coalescence_duration({**event, "coa_time": coa_time}, 4096.0, 5.0)
    assert str(refused.value) == str(from_query.value)


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("edge", ["first sample", "polarization start", "last sample"])
def test_a_coa_time_at_the_edge_of_the_output_is_accepted_everywhere(edge):
    """The output's first and last samples, and the polarization buffer's start, are valid placements."""
    simulator = BurstSimulator()
    event = _event("sg-q9")
    lead = simulator.pre_coalescence_duration(event, 4096.0, 5.0)
    tail = simulator.post_coalescence_duration(event, 4096.0, 5.0)
    hp, _ = simulator.generate_polarizations(event, 4096.0, 5.0)
    coa_time = {
        "first sample": PEAK - lead,
        "polarization start": float(hp.t0),
        "last sample": PEAK + tail - 1.0 / 4096.0,
    }[edge]
    placed = {**event, "coa_time": coa_time}
    expected = simulator.simulate(event, ["H1"], sampling_frequency=4096.0, minimum_frequency=5.0)["H1"]
    for call in _calls(simulator).values():
        call(placed)
    strain = simulator.simulate(placed, ["H1"], sampling_frequency=4096.0, minimum_frequency=5.0)["H1"]
    assert strain.t0 == expected.t0
    np.testing.assert_array_equal(np.asarray(strain.value), np.asarray(expected.value))


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("rate", [4000.0, 6000.0, 8192.0, 16000.0])
@pytest.mark.parametrize("epoch", [1.4e9, 2.0e9])
@pytest.mark.parametrize("edge", ["start", "end"])
def test_a_coa_time_on_the_returned_strains_own_edges_is_accepted(edge, epoch, rate):
    """A caller places ``coa_time`` on the strain's first sample, or one sample past its last, exactly.

    Computed the way a caller would -- ``t0`` of the strain ``simulate`` returned, and ``t0 + n / rate``
    -- at GPS epochs where one float64 step is 2.4e-7 s. At a rate whose sample spacing is not a power
    of two the check's own arithmetic rounds, and an exact edge comes out up to 1.3e-7 s outside
    (measured at 4000, 6000 and 16000 Hz); such a placement must be accepted, over a spread of peak
    times, and must leave the strain unchanged. 8192 Hz, where nothing rounds, is kept as a control.
    """
    simulator = BurstSimulator()
    refused = []
    for offset in np.linspace(0.0, 1.0, 9):
        event = _event("sg-q9", peak_time=epoch + 17.0 + offset * 0.987654321)
        strain = simulator.simulate(event, ["H1"], sampling_frequency=rate, minimum_frequency=5.0)["H1"]
        coa_time = strain.t0 if edge == "start" else strain.t0 + strain.n_samples / rate
        placed = {**event, "coa_time": coa_time}
        try:
            pre = simulator.pre_coalescence_duration(placed, rate, 5.0)
            post = simulator.post_coalescence_duration(placed, rate, 5.0)
            again = simulator.simulate(placed, ["H1"], sampling_frequency=rate, minimum_frequency=5.0)["H1"]
        except ValueError:
            refused.append(offset)
            continue
        assert pre >= 0
        assert post >= 0
        assert (pre if edge == "start" else post) < 1e-5
        assert again.t0 == strain.t0
        np.testing.assert_array_equal(np.asarray(again.value), np.asarray(strain.value))
    assert not refused, f"placements on the output's own {edge} refused for peak offsets {refused}"


@pytest.mark.requires_lalsuite
def test_queries_for_one_event_share_one_generation(mocker):
    """Start and end, and every placement time of one event, cost one generation between them."""
    simulator = BurstSimulator()
    event = _event("sg-q9")
    spy = mocker.spy(simulator, "_generate")
    lead = simulator.pre_coalescence_duration(event, 8192.0, 5.0)
    simulator.post_coalescence_duration(event, 8192.0, 5.0)
    simulator.pre_coalescence_duration({**event, "coa_time": PEAK + 0.01}, 8192.0, 5.0)
    assert simulator.pre_coalescence_duration(dict(event), 8192.0, 5.0) == lead
    assert spy.call_count == 1
    simulator.pre_coalescence_duration(event, 2048.0, 5.0)
    assert spy.call_count == 2
    # Generation with a coa_time checks it on the buffer it has just made: no second generation.
    simulator.generate_polarizations({**event, "coa_time": PEAK}, 4096.0, 5.0)
    assert spy.call_count == 3


# ---------------------------------------------------------------------------
# peak_time positions the waveform; coa_time only places it
# ---------------------------------------------------------------------------


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("family", ["sg-q100-30hz", "ga-2.5ms", "wnb-250"])
def test_a_valid_coa_time_never_changes_the_strain(family):
    """Any ``coa_time`` inside the output, or none, gives the strain ``peak_time`` alone gives."""
    simulator = BurstSimulator()
    event = _event(family)
    variants = {
        "peak_time only": event,
        "coa_time empty": {**event, "coa_time": None},
        **{placement: _placed(simulator, event, 4096.0, placement) for placement in _PLACEMENTS},
    }
    outputs = {
        name: simulator.simulate(variant, ["H1", "L1"], sampling_frequency=4096.0, minimum_frequency=5.0)
        for name, variant in variants.items()
    }
    reference = outputs["peak_time only"]
    for name, output in outputs.items():
        for detector in ("H1", "L1"):
            assert output[detector].t0 == reference[detector].t0, name
            np.testing.assert_array_equal(np.asarray(output[detector].value), np.asarray(reference[detector].value))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"peak_time": None, "coa_time": PEAK}, "peak_time must be a finite number"),
        ({"peak_time": math.nan}, "peak_time must be a finite number"),
        ({"peak_time": "noon"}, "peak_time must be a finite number"),
        ({"coa_time": math.inf}, "coa_time must be a finite number"),
        ({"coa_time": "noon"}, "coa_time must be a finite number"),
    ],
)
@pytest.mark.parametrize("call", ["pre", "post", "polarizations", "simulate"])
def test_an_unusable_time_is_refused(overrides, match, call):
    """No usable peak, or an unusable placement time, is refused rather than answered or generated.

    A ``coa_time`` alone is refused too: it may lie anywhere inside the buffer, so it does not say
    where the peak is.
    """
    event = _event("sg-q9", **overrides)
    # Generation reports an unusable peak_time in its own words, naming the key.
    with pytest.raises(ValueError, match=match if call in {"pre", "post"} or "coa" in match else "peak_time"):
        _calls(BurstSimulator())[call](event)


@pytest.mark.parametrize("peak_time", [None, math.nan, "noon"])
def test_simulate_still_refuses_an_event_without_a_usable_peak_time(peak_time):
    """``coa_time`` does not stand in for the peak in generation either, as before."""
    event = _event("sg-q9", coa_time=PEAK)
    if peak_time is None:
        del event["peak_time"]
    else:
        event["peak_time"] = peak_time
    with pytest.raises(ValueError, match="peak_time"):
        BurstSimulator().simulate(event, ["H1"], sampling_frequency=4096.0, minimum_frequency=5.0)


# ---------------------------------------------------------------------------
# End to end, through the orchestrator that places events into segments
# ---------------------------------------------------------------------------

#: Interpreter with the orchestrator installed. It is not a dependency of this package, so the
#: check runs only where it is present: in the environment running the tests, or in the one this
#: variable names. This package's ``src`` is put first on that interpreter's path, so the code under
#: test is this checkout's whatever version the environment itself carries.
_GWMOCK_PYTHON = os.environ.get("GWMOCK_PYTHON", sys.executable)
_SRC = Path(__file__).resolve().parents[2] / "src"

_POPULATION_MODULE = '''
import numpy as np

NAMES = ("coa_time", "peak_time", "burst_model", "hrss", "frequency", "quality_factor", "inclination",
         "right_ascension", "declination", "polarization_angle")


class OneBurst:
    """One Q = 100 sine-Gaussian at 30 Hz, placed by its peak or by the start of its buffer."""

    parameter_names = NAMES
    source_type = "burst"
    metadata = {}

    def __init__(self, *, peak_time, placement, sampling_frequency, source_type="burst"):
        self._peak = float(peak_time)
        self._placement = placement
        self._rate = float(sampling_frequency)

    def simulate(self, n_samples=None, seed=None, **_):
        event = dict(peak_time=self._peak, burst_model="sine_gaussian", hrss=1e-21,
                     frequency=30.0, quality_factor=100.0, inclination=0.4, right_ascension=1.0,
                     declination=0.3, polarization_angle=0.2)
        if self._placement == "peak":
            event["coa_time"] = self._peak
        else:
            from gwmock_signal.burst import BurstSimulator
            from gwmock_signal.projection.resampling import edge_padding

            plus, _ = BurstSimulator().generate_polarizations(event, self._rate, 0.0)
            # The projected output runs from the buffer's start less the padding to its end plus it.
            pad = edge_padding(self._rate) / self._rate
            event["coa_time"] = {
                "buffer start": float(plus.t0),
                "before output": float(plus.t0) - pad - 1.0 / self._rate,
                "after output": float(plus.t0) + plus.n_samples / self._rate + pad + 1.0 / self._rate,
            }[self._placement]
        return {name: np.array([event[name]], dtype=object) for name in NAMES}
'''

_CONFIG = """
globals:
    simulator-arguments:
        sampling-frequency: 1024
        duration: {duration}
        total-duration: 32
        start-time: 1000000000
    working-directory: .
    output-directory: out-{name}
    metadata-directory: meta-{name}
orchestration:
    population:
        backend: one_burst:OneBurst
        source-type: burst
        sort-by: coa_time
        arguments:
            peak_time: 1000000017.0
            placement: {placement}
            sampling_frequency: 1024
    signal:
        backend: burst
        minimum-frequency: 5.0
        earth-rotation: true
        detectors:
            - H1
        output:
            output_directory: signal
            file_name: S-{{{{ start_time }}}}-{{{{ duration }}}}.npy
"""


def _gwmock_available() -> bool:
    try:
        probe = subprocess.run(  # noqa: S603 -- the interpreter is the test environment's own, or one the user named
            [_GWMOCK_PYTHON, "-c", "import gwmock"], capture_output=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _run_gwmock(workdir: Path, name: str, duration: int, placement: str) -> subprocess.CompletedProcess:
    config = workdir / f"{name}.yaml"
    config.write_text(_CONFIG.format(name=name, duration=duration, placement=placement))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_SRC), str(workdir)]), "COLUMNS": "200"}
    command = "import sys; from gwmock.cli.main import app; sys.argv[0] = 'gwmock'; sys.exit(app())"
    result = subprocess.run(  # noqa: S603 -- as in _gwmock_available
        [_GWMOCK_PYTHON, "-c", command, "simulate", str(config)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    return result


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("placement", ["peak", "buffer start"])
def test_a_burst_across_a_segment_boundary_is_placed_whole_by_the_orchestrator(tmp_path, placement):
    """Peak 1 s after a boundary, 5.6 s of waveform before it: two segments sum to the whole.

    The orchestrator claims an event for the segment its waveform starts in, from the simulator's
    pre-coalescence answer measured from ``coa_time``, and carries the remainder into the next
    segment. With ``coa_time = peak_time`` and no answer it falls back to ``coa_time`` and the
    4.6 s before the boundary are discarded. ``coa_time`` at the buffer start is the convention
    populations used to avoid that, and must keep giving the same result.
    """
    if not _gwmock_available():
        pytest.skip(f"gwmock is not importable by {_GWMOCK_PYTHON}; set GWMOCK_PYTHON to an interpreter that has it")
    (tmp_path / "one_burst.py").write_text(textwrap.dedent(_POPULATION_MODULE))
    split = _run_gwmock(tmp_path, "split", 16, placement)
    whole_run = _run_gwmock(tmp_path, "whole", 32, placement)
    for run in (split, whole_run):
        assert run.returncode == 0, run.stdout + run.stderr

    first = np.load(tmp_path / "out-split" / "signal" / "S-1000000000-16.npy")
    second = np.load(tmp_path / "out-split" / "signal" / "S-1000000016-16.npy")
    whole = np.load(tmp_path / "out-whole" / "signal" / "S-1000000000-32.npy")
    summed = np.concatenate([first, second], axis=0)

    peak = np.max(np.abs(whole))
    assert peak > 0
    # The part before the boundary is really there, not merely matching an empty whole.
    assert np.max(np.abs(first)) > 0.05 * peak
    assert "Discarding" not in split.stdout + split.stderr
    np.testing.assert_allclose(summed, whole, rtol=0.0, atol=1e-12 * peak)


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("placement", ["before output", "after output"])
def test_a_coa_time_outside_the_output_fails_the_orchestrated_run(tmp_path, placement):
    """One sample outside the output: the run fails rather than writing a cropped burst.

    The orchestrator treats a failed duration query as "unknown", warns, and places the event by
    ``coa_time`` alone. Generation must then refuse the same placement, or the run completes with
    the part of the burst before the claiming segment silently missing.
    """
    if not _gwmock_available():
        pytest.skip(f"gwmock is not importable by {_GWMOCK_PYTHON}; set GWMOCK_PYTHON to an interpreter that has it")
    (tmp_path / "one_burst.py").write_text(textwrap.dedent(_POPULATION_MODULE))
    run = _run_gwmock(tmp_path, "split", 16, placement)
    assert run.returncode != 0, run.stdout + run.stderr
    assert "lies outside this burst's output" in run.stdout + run.stderr
