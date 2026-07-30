"""The analysis buffer must be long enough to hold the inspiral it is sized for.

``_segment_samples`` used the *leading-order Newtonian* chirp time plus a flat 2 s pad, then rounded
the duration up to a power of two. The 1PN correction to the chirp time is positive, so the 0PN term
always underestimates -- which meant the real safety margin was whatever the rounding happened to
leave. Measured across ordinary parameters that ranged from 2.8% to 256%, and where it fell below
the 1PN correction the inspiral wrapped around the buffer: a 10+1.4 binary at 10 Hz had 2.8% of room
against a 4.9% correction, and 1.8% of peak amplitude appeared in the region after the ringdown,
where nothing should be.

These tests pin the margin as a property of the *estimate* rather than of where the rounding lands.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed")
pytest.importorskip("ripplegw", reason="ripple not installed")

from gwmock_signal.waveform.backends.ripple import (
    _INSPIRAL_SAFETY_FRACTION,
    RippleBackend,
    _inspiral_margin,
    _inspiral_seconds,
)

_FS = 2048.0

#: Component masses and low-frequency cutoffs spanning BNS, NSBH and BBH, including the asymmetric
#: systems where the 1PN correction is largest.
_CASES = tuple(
    (mass1, mass2, f_min)
    for mass1, mass2 in ((1.4, 1.35), (1.2, 1.2), (10.0, 1.4), (30.0, 28.0), (5.0, 5.0), (60.0, 3.0), (2.0, 1.0))
    for f_min in (5.0, 10.0, 20.0)
)


def _newtonian_seconds(chirp_mass: float, f_min: float, mtsun: float) -> float:
    """The 0PN chirp time the old sizing rule used, for comparison."""
    return (5.0 / 256.0) * (np.pi * f_min) ** (-8.0 / 3.0) * (chirp_mass * mtsun) ** (-5.0 / 3.0)


def _chirp_mass_and_eta(mass1: float, mass2: float) -> tuple[float, float]:
    """Return ``(chirp_mass, eta)`` for a component-mass pair."""
    return (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2, mass1 * mass2 / (mass1 + mass2) ** 2


def test_the_1pn_estimate_always_exceeds_the_newtonian_one() -> None:
    """The 1PN correction is positive, so the 0PN term cannot bound the duration from above.

    This is the defect the sizing change addresses: the old rule treated an underestimate as if it
    were the whole duration.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    for mass1, mass2, f_min in _CASES:
        chirp_mass, eta = _chirp_mass_and_eta(mass1, mass2)
        newtonian = _newtonian_seconds(chirp_mass, f_min, mtsun)
        corrected = float(_inspiral_seconds(chirp_mass, eta, f_min, mtsun)[0])
        assert corrected > newtonian, f"{mass1}+{mass2} at {f_min} Hz: 1PN estimate is not longer"


def test_duration_depends_on_mass_ratio_at_fixed_chirp_mass() -> None:
    """At fixed chirp mass, a more asymmetric binary lasts longer.

    The total mass is ``Mc * eta^(-3/5)``, so lowering eta raises the 1PN term. This is why the
    duration estimate needs the mass ratio; whether it can also reorder two events is a separate and
    narrower question, covered below.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    equal = float(_inspiral_seconds(3.0, 0.25, 10.0, mtsun)[0])
    asymmetric = float(_inspiral_seconds(3.0, 0.10, 10.0, mtsun)[0])
    assert asymmetric > equal, "the asymmetric binary is not longer at equal chirp mass"


def test_a_heavier_but_more_asymmetric_binary_can_be_the_longest() -> None:
    """The lightest chirp mass in a batch is not always the longest inspiral.

    ``tau0 ~ Mc^(-5/3)`` dominates, so the lightest event usually *is* the longest and the window
    where this fails is narrow: at 20 Hz a heavier event overtakes a lighter equal-mass one only
    within about 3.5% in chirp mass. It is not exotic, though -- a 1:8 mass ratio flips it at +0.5%,
    which is an ordinary NSBH. Taking the maximum over every event is exact and costs nothing, so
    there is no reason to reason about a proxy at all.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    f_min = 20.0
    # Verified to straddle the crossover: nearly equal chirp masses, ordinary versus 1:8 mass ratio.
    lighter = (2.18, 0.25)
    heavier = (2.18 * 1.005, 0.10)

    lighter_seconds = float(_inspiral_seconds(*lighter, f_min, mtsun)[0])
    heavier_seconds = float(_inspiral_seconds(*heavier, f_min, mtsun)[0])
    assert heavier[0] > lighter[0], "test premise broken: the second event is not the heavier one"
    assert heavier_seconds > lighter_seconds, (
        "test premise broken: the heavier binary is not the longer one, so this cannot distinguish "
        "a maximum over durations from a reduction to the lightest chirp mass"
    )

    # An earlier version asserted only `both >= lightest_only`, which any implementation taking a
    # maximum satisfies -- including one maximising over the wrong quantity. The batch must be sized
    # for the *heavier* event specifically.
    both = backend._segment_samples(
        np.array([lighter[0], heavier[0]]), f_min, _FS, eta=np.array([lighter[1], heavier[1]])
    )
    assert both == backend._segment_samples(heavier[0], f_min, _FS, eta=heavier[1])


@pytest.mark.parametrize(("mass1", "mass2", "f_min"), _CASES)
def test_every_buffer_holds_its_inspiral_with_the_stated_margin(mass1: float, mass2: float, f_min: float) -> None:
    """The pre-coalescence room must exceed the 1PN inspiral by at least the safety fraction.

    Asserted against the *estimate*, not against the power-of-two rounding, so the guarantee does not
    depend on where a particular case happens to land. On the previous rule the 10+1.4 case at 10 Hz
    had 2.8% of room and fails this.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    chirp_mass, eta = _chirp_mass_and_eta(mass1, mass2)
    inspiral, relative_correction = _inspiral_seconds(chirp_mass, eta, f_min, mtsun)
    inspiral = float(inspiral)

    n_samples = backend._segment_samples(chirp_mass, f_min, _FS, eta=eta)
    room = (1.0 - backend._ringdown_fraction) * n_samples / _FS
    margin = room / inspiral - 1.0
    promised = _inspiral_margin(relative_correction)
    assert margin >= promised, (
        f"{mass1}+{mass2} at {f_min} Hz has {margin:.1%} of room over a {inspiral:.1f} s inspiral, "
        f"below the {promised:.1%} the sizing promises for a 1PN term of "
        f"{float(np.max(relative_correction)):.1%}"
    )


def test_a_pinned_segment_duration_is_still_honoured() -> None:
    """An explicitly pinned duration bypasses the estimate entirely, as before."""
    backend = RippleBackend().with_segment_duration(64.0)
    assert backend._segment_samples(1.2, 10.0, _FS, eta=0.25) == round(64.0 * _FS)


def test_the_nsbh_case_no_longer_wraps_its_inspiral() -> None:
    """The measured regression: 10+1.4 at 10 Hz used to leave 1.8% of peak after the ringdown.

    Wraparound is judged against a longer reference rather than by an absolute threshold, because
    the sharp ``minimum_frequency`` cutoff leaves ringing throughout the buffer that an absolute
    threshold cannot distinguish from wrapped signal. A reference four times longer cannot wrap, so a
    large *ratio* between the two is the signature. Before the fix that ratio was 70; the residual
    afterwards is the ringing, which is a separate matter.
    """
    backend = RippleBackend()

    def _post_ringdown_level(active: RippleBackend) -> tuple[float, float]:
        waveform = active.generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=_FS,
            minimum_frequency=10.0,
            mass1=10.0,
            mass2=1.4,
            luminosity_distance=400.0,
            inclination=0.3,
            coa_phase=0.0,
            spin1z=0.0,
            spin2z=0.0,
        )
        strain = np.asarray(waveform["plus"].value, dtype=float)
        n = strain.size
        merger = round((1.0 - active._ringdown_fraction) * n)
        # The second half of the post-coalescence pad: the ringdown of this system decays within
        # milliseconds, so anything here is either wrapped inspiral or cutoff ringing.
        tail = strain[merger + (n - merger) // 2 :]
        return n / _FS, float(np.max(np.abs(tail))) / float(np.max(np.abs(strain)))

    duration, level = _post_ringdown_level(backend)
    _, reference_level = _post_ringdown_level(backend.with_segment_duration(duration * 4.0))
    assert level / reference_level < 10.0, (
        f"the {duration:.0f} s buffer leaves {level:.2e} of peak after the ringdown against "
        f"{reference_level:.2e} for a 4x longer buffer, a ratio of {level / reference_level:.1f}; "
        f"the inspiral is still wrapping"
    )


def test_the_margin_grows_with_the_1pn_term_when_the_series_stops_converging() -> None:
    """Where the 1PN term is large the margin must grow with it, not stay at the 10% floor.

    A fixed fraction is only defensible while the PN series converges, and nothing restricts callers
    to that regime: a 60+3 binary at 512 Hz is accepted, and its expansion parameter is ~0.79, where
    the omitted terms are the same order as the one retained. The margin is therefore the larger of
    the floor and the 1PN term itself.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    chirp_mass, eta = _chirp_mass_and_eta(60.0, 3.0)

    _, gentle = _inspiral_seconds(chirp_mass, eta, 10.0, mtsun)
    _, severe = _inspiral_seconds(chirp_mass, eta, 512.0, mtsun)
    assert float(severe) > float(gentle), "test premise broken: the higher cutoff is not the harder case"

    assert _inspiral_margin(gentle) >= _INSPIRAL_SAFETY_FRACTION
    assert _inspiral_margin(severe) >= float(severe), (
        "where the 1PN term is large the margin must be at least as large as it, since the omitted "
        "terms are then the same order"
    )


@pytest.mark.parametrize(
    ("chirp_mass", "eta"),
    [
        (1.2, 0.0),  # eta must be positive
        (1.2, -0.1),
        (1.2, 0.3),  # 0.25 is the equal-mass maximum
        (1.2, float("nan")),
        (0.0, 0.25),  # chirp mass must be positive
        (-1.0, 0.25),
        (float("inf"), 0.25),
    ],
)
def test_unphysical_inputs_are_rejected(chirp_mass: float, eta: float) -> None:
    """Bad masses or ratios must raise, not produce a plausible-looking duration.

    Without this an ``eta`` above the equal-mass maximum, or a non-finite mass, yields a number that
    sizes a buffer -- and the resulting grid would be silently wrong rather than absent.
    """
    backend = RippleBackend()
    with pytest.raises(ValueError, match="must be"):
        _inspiral_seconds(chirp_mass, eta, 10.0, float(backend._constants.MTSUN))


@pytest.mark.parametrize("minimum_frequency", [0.0, -10.0, float("nan"), float("inf")])
def test_an_invalid_cutoff_frequency_is_rejected(minimum_frequency: float) -> None:
    """A non-positive or non-finite cutoff cannot define a chirp time."""
    backend = RippleBackend()
    with pytest.raises(ValueError, match="minimum_frequency"):
        _inspiral_seconds(1.2, 0.25, minimum_frequency, float(backend._constants.MTSUN))


def test_the_mass_ratio_must_be_supplied() -> None:
    """``eta`` is required, so no caller can silently take an equal-mass underestimate.

    An equal-mass default reads as harmless but underestimates the duration for every asymmetric
    binary, which is precisely the direction that wraps an inspiral around the buffer.
    """
    backend = RippleBackend()
    with pytest.raises(TypeError):
        backend._segment_samples(1.2, 10.0, _FS)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        backend.segment_duration_for(1.2, 10.0, _FS)  # type: ignore[call-arg]


def test_an_empty_batch_is_rejected_with_a_specific_message() -> None:
    """Sizing a grid for zero events must say so, not surface a numpy reduction error.

    ``np.all`` and ``np.any`` are vacuously true on an empty array, so the validation above would
    pass and the caller would see ``zero-size array reduction has no identity`` from the maximum
    inside ``_segment_samples`` -- which says nothing about the contract that was broken.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    empty = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        _inspiral_seconds(empty, empty, 10.0, mtsun)
    with pytest.raises(ValueError, match="non-empty"):
        backend._segment_samples(empty, 10.0, _FS, eta=empty)
