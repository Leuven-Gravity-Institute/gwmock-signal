"""The LAL and gwsignal buffers must start before the inspiral enters the band.

Both backends size through :func:`~gwmock_signal.waveform.backends.conditioning.segment_sample_count`.
It used the *0PN* chirp time plus a flat 2 s pad, rounded up to a power of two, so wherever the
rounding left less headroom than the 0PN term omits, the inspiral began before the buffer and
wrapped around it. That happens for asymmetric binaries with large aligned spins, which carry the
largest corrections: a 40+1.4 binary with spins of 0.99 at 7 Hz lasts 236.7 s and was given 230.4 s.

The reference here is independent of the sizing code: the time from ``minimum_frequency`` to
coalescence read from the stationary phase of LALSimulation's own IMRPhenomXAS,
``t(f) = -(1 / 2 pi) d arg h(f) / df``.
"""

from __future__ import annotations

import re
from pathlib import Path

import lal
import lalsimulation
import numpy as np
import pytest

from gwmock_signal.waveform.backends import conditioning
from gwmock_signal.waveform.backends.gwsignal import GWSignalBackend
from gwmock_signal.waveform.backends.lal import LALSimulationBackend

_FS = 1024.0

#: ``(mass1, mass2, chi, f_min)``: every case the previous sizing wrapped, among 420 combinations of
#: 1.4-40 + 1.2-5 solar masses, aligned spins 0-0.99 on both components, and 5-20 Hz.
_WRAPPED_CASES = (
    (10.0, 1.4, 0.9, 10.0),
    (10.0, 1.4, 0.99, 10.0),
    (20.0, 1.2, 0.9, 15.0),
    (20.0, 1.2, 0.99, 15.0),
    (40.0, 1.4, 0.9, 7.0),
    (40.0, 1.4, 0.99, 7.0),
)

#: ``(mass1, mass2, f_min)`` where the 1PN term exceeds the 10% floor, so the margin is the 1PN term
#: itself. None of the wrapped cases reaches that branch: their 1PN terms are all below 10%.
_STEEP_CASES = (
    (20.0, 1.2, 60.0),
    (40.0, 1.4, 50.0),
    (60.0, 3.0, 50.0),
)


def _stationary_phase_duration(mass1: float, mass2: float, chi: float, f_min: float, span: float) -> float:
    """Return the seconds from *f_min* to coalescence from IMRPhenomXAS's frequency-domain phase.

    ``delta_f = 1 / (8 span)`` resolves the phase of a signal up to eight times longer than *span*,
    so a buffer shorter than the inspiral cannot alias the reference.
    """
    delta_f = 1.0 / (8.0 * span)
    hp, _ = lalsimulation.SimInspiralChooseFDWaveform(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        0.0,
        0.0,
        chi,
        0.0,
        0.0,
        chi,
        100.0e6 * lal.PC_SI,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        delta_f,
        0.9 * f_min,
        3.0 * f_min,
        f_min,
        lal.CreateDict(),
        lalsimulation.GetApproximantFromString("IMRPhenomXAS"),
    )
    strain = hp.data.data
    frequencies = np.arange(strain.size) * delta_f
    band = (frequencies >= 0.95 * f_min) & (frequencies <= 1.05 * f_min) & (np.abs(strain) > 0.0)
    time = -np.gradient(np.unwrap(np.angle(strain[band])), frequencies[band]) / (2.0 * np.pi)
    return float(-np.interp(f_min, frequencies[band], time))


def _previous_pre_coalescence_seconds(mass1: float, mass2: float, f_min: float) -> float:
    """The room the 0PN-plus-2-s rule gave, kept to prove these cases genuinely discriminate."""
    chirp_seconds = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2 * conditioning.MTSUN_SI
    tau0 = (5.0 / 256.0) * (np.pi * f_min) ** (-8.0 / 3.0) * chirp_seconds ** (-5.0 / 3.0)
    seconds = 2.0 ** np.ceil(np.log2((tau0 + 2.0) / (1.0 - conditioning.DEFAULT_RINGDOWN_FRACTION)))
    n_samples = round(seconds * _FS)
    return conditioning.coalescence_placement(n_samples, _FS)[0] / _FS


def _source(mass1: float, mass2: float, chi: float) -> dict[str, float]:
    return {
        "detector_frame_mass_1": mass1,
        "detector_frame_mass_2": mass2,
        "spin_1z": chi,
        "spin_2z": chi,
        "luminosity_distance": 100.0,
    }


@pytest.mark.parametrize("backend_class", [LALSimulationBackend, GWSignalBackend])
@pytest.mark.parametrize(("mass1", "mass2", "chi", "f_min"), _WRAPPED_CASES)
def test_the_buffer_starts_before_the_inspiral(
    backend_class: type[LALSimulationBackend], mass1: float, mass2: float, chi: float, f_min: float
) -> None:
    """The room before coalescence must exceed the stationary-phase inspiral, with positive margin."""
    pre = backend_class().pre_coalescence_duration("IMRPhenomXAS", _FS, f_min, **_source(mass1, mass2, chi))
    assert pre is not None
    inspiral = _stationary_phase_duration(mass1, mass2, chi, f_min, pre)

    previous = _previous_pre_coalescence_seconds(mass1, mass2, f_min)
    assert previous < inspiral, (
        f"test premise broken: the previous rule already held the {mass1}+{mass2} chi={chi} inspiral "
        f"at {f_min} Hz ({previous:.1f} s for {inspiral:.1f} s), so this case cannot discriminate"
    )
    margin = pre / inspiral - 1.0
    assert margin > 0.0, (
        f"{mass1}+{mass2} chi={chi} at {f_min} Hz: {pre:.1f} s of room for a {inspiral:.1f} s inspiral "
        f"({margin:+.1%}); the inspiral wraps around the buffer"
    )


def _expected_requirement_seconds(mass1: float, mass2: float, f_min: float) -> float:
    """The 1PN chirp time grown by ``max(10%, 1PN term)`` plus 2 s, written out independently."""
    total_seconds = (mass1 + mass2) * conditioning.MTSUN_SI
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    chirp_seconds = total_seconds * eta**0.6
    tau0 = (5.0 / 256.0) * (np.pi * f_min) ** (-8.0 / 3.0) * chirp_seconds ** (-5.0 / 3.0)
    correction = (743.0 / 252.0 + 11.0 * eta / 3.0) * (np.pi * total_seconds * f_min) ** (2.0 / 3.0)
    return tau0 * (1.0 + correction) * (1.0 + max(0.10, correction)) + 2.0


@pytest.mark.parametrize(
    ("mass1", "mass2", "f_min"),
    [(mass1, mass2, f_min) for mass1, mass2, _, f_min in _WRAPPED_CASES] + list(_STEEP_CASES),
)
def test_the_shared_requirement_is_the_1pn_estimate_with_its_margin(mass1: float, mass2: float, f_min: float) -> None:
    """Pin the unrounded estimate, where dropping the 1PN term or the margin is visible.

    The buffer tests above cannot see either change on their own: the power-of-two rounding happens
    to leave enough room for these cases even then. The estimate is what must hold elsewhere.
    Spin does not enter it, so each mass pair and cutoff is checked once.
    """
    chirp_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    required = float(conditioning.inspiral_requirement_seconds(chirp_mass, eta, f_min))
    assert required == pytest.approx(_expected_requirement_seconds(mass1, mass2, f_min), rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("mass1", "mass2", "f_min"), _STEEP_CASES)
def test_the_margin_is_the_1pn_term_where_it_exceeds_the_floor(mass1: float, mass2: float, f_min: float) -> None:
    """Where the 1PN term is above 10%, the production requirement must grow by that term, not by 10%.

    Built from the production ``inspiral_seconds`` output, so a requirement that falls back to the
    flat floor is caught even if the independent formula above were wrong in the same way.
    """
    chirp_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    inspiral, correction = (float(value) for value in conditioning.inspiral_seconds(chirp_mass, eta, f_min))
    assert correction > conditioning.INSPIRAL_SAFETY_FRACTION, "test premise broken: the floor still applies"

    required = float(conditioning.inspiral_requirement_seconds(chirp_mass, eta, f_min))
    floor_only = inspiral * (1.0 + conditioning.INSPIRAL_SAFETY_FRACTION) + conditioning.SEGMENT_BUFFER_SECONDS
    assert required == pytest.approx(inspiral * (1.0 + correction) + conditioning.SEGMENT_BUFFER_SECONDS, rel=1e-12)
    assert required > floor_only, f"{mass1}+{mass2} at {f_min} Hz is sized with the 10% floor, not its 1PN term"


@pytest.mark.parametrize("backend_class", [LALSimulationBackend, GWSignalBackend])
@pytest.mark.parametrize(("mass1", "mass2", "chi", "f_min"), _WRAPPED_CASES)
def test_the_buffer_is_the_shortest_power_of_two_holding_the_requirement(
    backend_class: type[LALSimulationBackend], mass1: float, mass2: float, chi: float, f_min: float
) -> None:
    """Generation must size from the shared estimate: enough room for it, and half would not be."""
    source = _source(mass1, mass2, chi)
    pre = backend_class().pre_coalescence_duration("IMRPhenomXAS", _FS, f_min, **source)
    post = backend_class().post_coalescence_duration("IMRPhenomXAS", _FS, f_min, **source)
    assert pre is not None
    assert post is not None
    required = _expected_requirement_seconds(mass1, mass2, f_min)
    room_fraction = 1.0 - conditioning.DEFAULT_RINGDOWN_FRACTION
    assert pre >= required
    assert room_fraction * (pre + post) / 2.0 < required, "a buffer half as long would still hold the estimate"


def test_segment_sample_count_requires_the_mass_ratio() -> None:
    """``eta`` is required, so no caller can silently take an equal-mass underestimate."""
    with pytest.raises(TypeError):
        conditioning.segment_sample_count(1.2, 10.0, _FS)  # type: ignore[call-arg]


def test_a_pinned_segment_duration_still_overrides_the_estimate() -> None:
    """An explicitly pinned duration bypasses the estimate entirely, as before."""
    assert conditioning.segment_sample_count(1.2, 10.0, _FS, eta=0.25, segment_duration=64.0) == round(64.0 * _FS)


_PLACEMENT_PAGE = Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "segment-placement-losses.md"

#: One row of the page's "Measured under the previous LAL sizing" table.
_GEOMETRY_ROW = re.compile(
    r"^\s*\|\s*(?P<mass1>[\d.]+)\+(?P<mass2>[\d.]+)\s*\|\s*(?P<f_min>[\d.]+) Hz\s*\|"
    r"\s*(?P<lead_then>[\d.]+) s\s*\|\s*(?P<lead_now>[\d.]+) s\s*\|"
    r"\s*(?P<buffer_then>[\d.]+) s\s*\|\s*(?P<buffer_now>[\d.]+) s\s*\|\s*$",
    re.MULTILINE,
)


def _note_rows() -> list[dict[str, float]]:
    note = _PLACEMENT_PAGE.read_text(encoding="utf-8").split('!!! note "Measured under the previous LAL sizing"')[1]
    note = note.split("<!-- prettier-ignore-end -->")[0]
    return [{key: float(value) for key, value in match.groupdict().items()} for match in _GEOMETRY_ROW.finditer(note)]


def test_the_documented_geometry_change_matches_the_sizing() -> None:
    """The user guide's then/now table must be what the old rule and the current backend produce.

    Read from the page itself, so a sizing change that moves any quoted buffer fails here instead of
    leaving the page describing a geometry the backend no longer has.
    """
    rows = _note_rows()
    assert len(rows) == 4, f"expected the four documented rows, parsed {len(rows)}"
    for row in rows:
        source = {
            "detector_frame_mass_1": row["mass1"],
            "detector_frame_mass_2": row["mass2"],
            "luminosity_distance": 1.0,
        }
        backend = LALSimulationBackend()
        lead_now = backend.pre_coalescence_duration("IMRPhenomD", _FS, row["f_min"], **source)
        post_now = backend.post_coalescence_duration("IMRPhenomD", _FS, row["f_min"], **source)
        assert lead_now is not None
        assert post_now is not None
        lead_then = _previous_pre_coalescence_seconds(row["mass1"], row["mass2"], row["f_min"])
        label = f"{row['mass1']:g}+{row['mass2']:g} at {row['f_min']:g} Hz"
        # The page quotes leads to the precision shown, so compare at that precision.
        assert row["lead_now"] == pytest.approx(lead_now, abs=0.05), f"{label}: page says {row['lead_now']} s now"
        assert row["lead_then"] == pytest.approx(lead_then, abs=0.05), f"{label}: page says {row['lead_then']} s then"
        assert row["buffer_now"] == pytest.approx(lead_now + post_now, abs=0.01), f"{label}: buffer now"
        assert row["buffer_then"] == pytest.approx(lead_then / (1.0 - conditioning.DEFAULT_RINGDOWN_FRACTION), abs=0.01)


def test_the_documented_ringdown_fraction_lead_matches_the_sizing() -> None:
    """The note's ``ringdown_fraction`` 0.2 figure must be what the backend reports."""
    text = _PLACEMENT_PAGE.read_text(encoding="utf-8")
    match = re.search(r"With `ringdown_fraction` 0\.2, 30\+25 at 20 Hz now leads by ([\d.]+) s", text)
    assert match is not None, "the ringdown_fraction sentence is no longer on the page"
    source = {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 25.0, "luminosity_distance": 1.0}
    lead = LALSimulationBackend(ringdown_fraction=0.2).pre_coalescence_duration("IMRPhenomD", _FS, 20.0, **source)
    assert float(match.group(1)) == pytest.approx(lead, abs=0.0005)
