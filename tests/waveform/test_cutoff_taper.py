"""The low-frequency cutoff must be tapered, not applied as a rectangular mask.

The out-of-band bins used to be zeroed with ``freqs >= minimum_frequency``. The waveform amplitude
at the cutoff is not zero, so that is rectangular truncation of a nonzero function, and its inverse
transform rings across the whole buffer. Measured in the post-coalescence region, where the ringdown
has long decayed and nothing should remain: 1.2e-2 of peak for a 30+28 system at 20 Hz.

The ringing's own spectrum peaks at exactly ``minimum_frequency``, which is what identifies it as a
cutoff artefact rather than anything physical, and is the property these tests use to tell the two
apart.

The ramp runs *below* the cutoff, so every bin the caller asked for keeps its amplitude. That puts
attenuated content below ``minimum_frequency``, which lengthens the inspiral -- hence
``signal_start_frequency`` and its use in the sizing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="jax not installed")
pytest.importorskip("ripplegw", reason="ripple not installed")

from gwmock_signal.waveform.backends.ripple import (
    _DEFAULT_TAPER_FRACTION,
    RippleBackend,
    _inspiral_margin,
    _inspiral_seconds,
    _next_smooth_even,
)

_FS = 2048.0
_PARAMETERS = {
    "luminosity_distance": 400.0,
    "inclination": 0.3,
    "coa_phase": 0.0,
    "spin1z": 0.0,
    "spin2z": 0.0,
}

#: (mass1, mass2, f_min). Spans the regime the ET work targets (BNS at 5-10 Hz) and the hard case
#: for a taper -- a heavy binary whose merger sits close to the cutoff.
_CASES = (
    (1.4, 1.35, 10.0),
    (1.4, 1.35, 20.0),
    (10.0, 1.4, 10.0),
    (5.0, 5.0, 20.0),
    (30.0, 28.0, 20.0),
)


def _post_coalescence(backend: RippleBackend, mass1: float, mass2: float, f_min: float) -> tuple[float, float]:
    """Return ``(peak-relative level, dominant frequency / f_min)`` for the post-ringdown region."""
    waveform = backend.generate_td_waveform(
        "IMRPhenomD",
        tc=0.0,
        sampling_frequency=_FS,
        minimum_frequency=f_min,
        mass1=mass1,
        mass2=mass2,
        **_PARAMETERS,
    )
    strain = np.asarray(waveform["plus"].value, dtype=float)
    n_samples = strain.size
    merger_index = round((1.0 - backend._ringdown_fraction) * n_samples)
    # Second half of the post-coalescence pad: the ringdown of these systems decays in milliseconds.
    tail = strain[merger_index + (n_samples - merger_index) // 2 :]
    level = float(np.max(np.abs(tail))) / float(np.max(np.abs(strain)))
    spectrum = np.abs(np.fft.rfft(tail))
    frequencies = np.fft.rfftfreq(tail.size, d=1.0 / _FS)
    return level, float(frequencies[int(np.argmax(spectrum))]) / f_min


@pytest.mark.parametrize(("mass1", "mass2", "f_min"), _CASES)
def test_the_taper_reduces_post_ringdown_contamination(mass1: float, mass2: float, f_min: float) -> None:
    """Every case must be cleaner after the ringdown with the taper than without it."""
    hard_level, _ = _post_coalescence(RippleBackend(taper_fraction=0.0), mass1, mass2, f_min)
    tapered_level, _ = _post_coalescence(RippleBackend(), mass1, mass2, f_min)
    assert tapered_level < hard_level, (
        f"{mass1}+{mass2} at {f_min} Hz: tapered level {tapered_level:.2e} is not below the "
        f"hard-cutoff level {hard_level:.2e}"
    )


def test_the_hard_cutoff_rings_at_the_cutoff_frequency() -> None:
    """The artefact is identified by *where* it sits, not only by its size.

    A rectangular truncation at ``minimum_frequency`` rings at that frequency. This is what
    separates the diagnosis from "there is some junk in the buffer", and it is the reason the fix
    targets the cutoff rather than, say, the buffer length.
    """
    _, dominant = _post_coalescence(RippleBackend(taper_fraction=0.0), 1.4, 1.35, 20.0)
    assert dominant == pytest.approx(1.0, abs=0.05), (
        f"hard-cutoff ringing peaks at {dominant:.2f} x f_min, so it is not the cutoff artefact this change is aimed at"
    )


def test_tapering_moves_the_residual_away_from_the_cutoff() -> None:
    """With the cutoff tapered, whatever remains is no longer a cutoff artefact.

    The residual moves to the top of the band, so the taper removes the artefact rather than
    shrinking it in place.
    """
    _, dominant = _post_coalescence(RippleBackend(), 1.4, 1.35, 20.0)
    assert dominant > 2.0, f"residual still peaks at {dominant:.2f} x f_min, so the cutoff still rings"


def test_the_taper_leaves_the_requested_band_untouched() -> None:
    """No bin at or above ``minimum_frequency`` may change.

    This is the whole reason the ramp is below the cutoff rather than above it. Tapering above
    suppresses the ringing equally well but removes in-band power -- measured at 4.2% for this
    width -- which is the one thing this backend should not trade away.
    """
    f_min = 20.0
    # Both pinned to one duration. Unpinned they would differ, because the taper lengthens the
    # inspiral and so the buffer -- a real consequence of the design, but one that would make this a
    # comparison of two different frequency grids rather than of the window.
    duration = 64.0
    hard = (
        RippleBackend(taper_fraction=0.0)
        .with_segment_duration(duration)
        .generate_fd_polarizations(
            "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=f_min, mass1=30.0, mass2=28.0, **_PARAMETERS
        )
    )
    tapered = (
        RippleBackend()
        .with_segment_duration(duration)
        .generate_fd_polarizations(
            "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=f_min, mass1=30.0, mass2=28.0, **_PARAMETERS
        )
    )
    frequencies = np.asarray(hard.frequencies, dtype=float)
    in_band = frequencies >= f_min
    assert np.array_equal(np.asarray(hard.frequencies), np.asarray(tapered.frequencies))
    assert np.array_equal(np.asarray(hard.plus)[in_band], np.asarray(tapered.plus)[in_band])
    assert np.array_equal(np.asarray(hard.cross)[in_band], np.asarray(tapered.cross)[in_band])


def test_the_taper_adds_content_below_the_cutoff() -> None:
    """The counterpart of the above: below the cutoff the strain gains attenuated content.

    This is the semantic change the ``taper_fraction`` parameter exists to make visible --
    ``minimum_frequency`` becomes the frequency of *full* amplitude, not of first content.
    """
    f_min = 20.0
    backend = RippleBackend()
    tapered = backend.with_segment_duration(64.0).generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=f_min, mass1=30.0, mass2=28.0, **_PARAMETERS
    )
    frequencies = np.asarray(tapered.frequencies, dtype=float)
    plus = np.asarray(tapered.plus)
    ramp = (frequencies >= backend.signal_start_frequency(f_min)) & (frequencies < f_min)
    below_ramp = (frequencies > 0.0) & (frequencies < backend.signal_start_frequency(f_min))
    assert np.any(plus[ramp] != 0.0), "the ramp region is empty, so no taper was applied"
    assert not np.any(plus[below_ramp] != 0.0), "content exists below the ramp, which should be zero"


def test_signal_start_frequency_is_the_ramp_edge() -> None:
    """The advertised signal start must match the window the conditioning applies."""
    backend = RippleBackend(taper_fraction=0.05)
    assert backend.signal_start_frequency(21.0) == pytest.approx(20.0)
    assert RippleBackend(taper_fraction=0.0).signal_start_frequency(20.0) == pytest.approx(20.0)


@pytest.mark.parametrize(("mass1", "mass2", "f_min"), _CASES)
def test_the_buffer_is_sized_from_the_signal_start_not_the_cutoff(mass1: float, mass2: float, f_min: float) -> None:
    """The margin must hold against the *tapered* inspiral, which starts below the cutoff.

    Sizing from ``minimum_frequency`` instead leaves a 1.4+1.35 system at 10 Hz with a negative
    margin and a post-ringdown level of 2.9e-3 -- worse than the hard cutoff the taper replaces --
    so this is the check that stops the fix reintroducing the wrap it was built after.
    """
    backend = RippleBackend()
    mtsun = float(backend._constants.MTSUN)
    chirp_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2

    inspiral, correction = _inspiral_seconds(chirp_mass, eta, backend.signal_start_frequency(f_min), mtsun)
    n_samples = backend._segment_samples(chirp_mass, f_min, _FS, eta=eta)
    room = (1.0 - backend._ringdown_fraction) * n_samples / _FS
    margin = room / float(inspiral) - 1.0
    assert margin >= _inspiral_margin(correction), (
        f"{mass1}+{mass2} at {f_min} Hz has {margin:.1%} of room over the tapered inspiral, below "
        f"the {_inspiral_margin(correction):.1%} promised"
    )


def test_zero_fraction_restores_the_hard_cutoff_exactly() -> None:
    """``taper_fraction=0.0`` must reproduce the previous behaviour bit-for-bit.

    Anyone whose results depend on the old conditioning needs an exact escape hatch, not an
    approximate one.
    """
    backend = RippleBackend(taper_fraction=0.0)
    fd = backend.generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=20.0, mass1=1.4, mass2=1.35, **_PARAMETERS
    )
    frequencies = np.asarray(fd.frequencies, dtype=float)
    plus = np.asarray(fd.plus)
    assert not np.any(plus[frequencies < 20.0] != 0.0), "a zero fraction still admitted sub-cutoff content"


def test_a_pinned_copy_keeps_the_taper() -> None:
    """``with_segment_duration`` must carry ``taper_fraction`` across.

    A pinned backend exists to make chunks of one catalogue share a grid; if the copy reverted to a
    hard cutoff it would change the conditioning of exactly those chunks.
    """
    backend = RippleBackend(taper_fraction=0.07)
    assert backend.with_segment_duration(64.0).taper_fraction == pytest.approx(0.07)
    assert RippleBackend(taper_fraction=0.0).with_segment_duration(64.0).taper_fraction == 0.0


@pytest.mark.parametrize("fraction", [-0.01, 1.0, 1.5, float("nan")])
def test_an_invalid_taper_fraction_is_rejected(fraction: float) -> None:
    """A fraction of 1 puts the ramp's lower edge at zero frequency; beyond that it is negative."""
    with pytest.raises(ValueError, match="taper_fraction"):
        RippleBackend(taper_fraction=fraction)


def test_the_default_fraction_is_documented_and_non_zero() -> None:
    """The taper is on by default, since the ringing it removes is a defect, not a preference."""
    assert _DEFAULT_TAPER_FRACTION > 0.0
    assert RippleBackend().taper_fraction == pytest.approx(_DEFAULT_TAPER_FRACTION)


@pytest.mark.parametrize("minimum", [1, 2, 3, 7, 1000, 12345, 999983, 2355200, 2**20 + 1])
def test_smooth_lengths_agree_with_scipy(minimum: int) -> None:
    """The local 5-smooth search must match scipy's, which is an independent implementation.

    Implemented locally so the sizing does not rest on a transitive dependency through gwpy, but
    checked against scipy so "5-smooth" means what scipy means by it.
    """
    from scipy.fft import next_fast_len

    expected = next_fast_len(minimum, real=True)
    expected += expected % 2
    assert _next_smooth_even(minimum) == expected


@pytest.mark.parametrize("minimum", [3, 100, 5000, 123457])
def test_smooth_lengths_are_even_and_five_smooth(minimum: int) -> None:
    """Independently of scipy: the result must be even, at least ``minimum``, and 2-3-5 factorable.

    Even because the real transform pair maps ``n`` samples to ``n // 2 + 1`` bins and back.
    """
    length = _next_smooth_even(minimum)
    assert length >= minimum
    assert length % 2 == 0
    residual = length
    for factor in (2, 3, 5):
        while residual % factor == 0:
            residual //= factor
    assert residual == 1, f"{length} is not 5-smooth"
