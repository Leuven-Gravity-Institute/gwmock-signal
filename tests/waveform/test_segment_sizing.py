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
        corrected = float(_inspiral_seconds(chirp_mass, eta, f_min, mtsun))
        assert corrected > newtonian, f"{mass1}+{mass2} at {f_min} Hz: 1PN estimate is not longer"


def test_duration_depends_on_mass_ratio_at_fixed_chirp_mass() -> None:
    """At fixed chirp mass, a more asymmetric binary lasts longer.

    This is why the batch path cannot size its grid from the lightest chirp mass alone. The total
    mass is ``Mc * eta^(-3/5)``, so lowering eta raises the 1PN term.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    equal = float(_inspiral_seconds(3.0, 0.25, 10.0, mtsun))
    asymmetric = float(_inspiral_seconds(3.0, 0.10, 10.0, mtsun))
    assert asymmetric > equal, "the asymmetric binary is not longer at equal chirp mass"


def test_a_heavier_but_more_asymmetric_binary_can_be_the_longest() -> None:
    """The lightest chirp mass in a batch is not necessarily the longest inspiral.

    ``_segment_samples`` therefore takes every event and uses the maximum. If it instead reduced to
    the lightest chirp mass, this batch would be sized for the wrong event.
    """
    backend = RippleBackend()
    light_equal = _chirp_mass_and_eta(2.0, 2.0)
    heavy_asymmetric = _chirp_mass_and_eta(20.0, 1.2)
    chirp_masses = np.array([light_equal[0], heavy_asymmetric[0]])
    etas = np.array([light_equal[1], heavy_asymmetric[1]])

    both = backend._segment_samples(chirp_masses, 10.0, _FS, eta=etas)
    lightest_only = backend._segment_samples(
        np.array([chirp_masses.min()]), 10.0, _FS, eta=np.array([etas[np.argmin(chirp_masses)]])
    )
    assert both >= lightest_only


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
    inspiral = float(_inspiral_seconds(chirp_mass, eta, f_min, mtsun))

    n_samples = backend._segment_samples(chirp_mass, f_min, _FS, eta=eta)
    room = (1.0 - backend._ringdown_fraction) * n_samples / _FS
    margin = room / inspiral - 1.0
    assert margin >= _INSPIRAL_SAFETY_FRACTION, (
        f"{mass1}+{mass2} at {f_min} Hz has {margin:.1%} of room over a {inspiral:.1f} s inspiral, "
        f"below the {_INSPIRAL_SAFETY_FRACTION:.0%} the sizing promises"
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
