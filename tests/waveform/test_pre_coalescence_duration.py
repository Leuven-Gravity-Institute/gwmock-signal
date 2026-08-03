"""How long before ``tc`` a backend's buffer starts, asked before generating.

A caller placing signals into segmented data needs this in advance: a compact binary's inspiral
precedes its coalescence, so a buffer whose ``tc`` sits just past a segment boundary begins in an
earlier segment. Choosing the claiming segment from ``tc`` alone crops the start away -- 32% of a
30+25 solar-mass binary's strain-squared energy at 1024 Hz with 16 s segments, and 99.998% for a
binary neutron star whose buffer can begin before the run.

The property worth testing is not the arithmetic but the *agreement*: the number a backend promises
must be the number generation delivers. A test that recomputed the formula would pass against a
backend whose generation had drifted from its own sizing, which is the failure that matters.
"""

from __future__ import annotations

import pytest

_BBH = {"mass1": 30.0, "mass2": 25.0, "distance": 400.0, "inclination": 0.0}
_BNS = {"mass1": 1.4, "mass2": 1.35, "distance": 100.0, "inclination": 0.0}
_TC = 1577491312.5


def _lal_backend():
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    return LALSimulationBackend()


def _ripple_backend():
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    from gwmock_signal.waveform.backends.ripple import RippleBackend

    return RippleBackend()


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
@pytest.mark.parametrize("sampling_frequency", [1024.0, 2048.0])
def test_the_lal_backend_predicts_where_its_own_waveform_starts(params, sampling_frequency):
    """The promise must equal what generation delivers, not merely re-derive the formula."""
    backend = _lal_backend()

    predicted = backend.pre_coalescence_duration("IMRPhenomD", sampling_frequency, 20.0, **params)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, sampling_frequency, 20.0, **params)
    actual = _TC - float(generated["plus"].t0.value)

    assert predicted is not None
    # Exact: both come from the same sizing helpers, so any difference is a real divergence
    # between what the backend promises and what it produces, not floating-point slack.
    assert predicted == pytest.approx(actual, abs=0.5 / sampling_frequency)


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
def test_the_ripple_backend_predicts_where_its_own_waveform_starts(params):
    """Ripple sizes differently -- 5-smooth lengths, eta in the 1PN term -- and must still agree."""
    backend = _ripple_backend()

    predicted = backend.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **params)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **params)
    actual = _TC - float(generated["plus"].t0.value)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


def test_the_two_backends_disagree_and_that_is_the_point():
    """A single shared computation would be wrong for one of them.

    LAL rounds to a power of two from a leading-order chirp time; ripple rounds to a 5-smooth
    length and carries eta in a 1PN term. Pinned because it is the justification for asking the
    backend rather than computing this once in the caller -- if the two ever agreed exactly, that
    reasoning would need revisiting rather than silently remaining in the docstrings.
    """
    lal = _lal_backend().pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    ripple = _ripple_backend().pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)

    assert lal != ripple
    # Same order, though: a factor beyond this would mean one of them is wrong rather than merely
    # conditioning differently.
    assert 0.5 < lal / ripple < 2.0


def test_a_backend_that_cannot_say_returns_none_rather_than_zero():
    """``None`` means unknown. Zero would mean "starts at coalescence", which is never true.

    The base class default is deliberately unhelpful: a backend delegating conditioning to another
    library cannot answer without reimplementing it, and a caller that received ``0.0`` would place
    every event in the segment holding its ``tc`` -- exactly the behaviour this API exists to fix,
    but now with an authoritative-looking number behind it.
    """
    from gwmock_signal.waveform.backends.base import WaveformBackend

    class Unanswering(WaveformBackend):
        def available_approximants(self):
            return ["Whatever"]

        def generate_td_waveform(self, approximant, tc, sampling_frequency, minimum_frequency, **params):
            raise NotImplementedError

    assert Unanswering().pre_coalescence_duration("Whatever", 1024.0, 20.0, mass1=30.0, mass2=25.0) is None


def test_a_longer_signal_starts_earlier():
    """Sanity on the direction, which no amount of agreement testing would catch if inverted."""
    backend = _lal_backend()

    bbh = backend.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    bns = backend.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BNS)

    assert bns > bbh
    # And a lower cutoff means a longer inspiral, for the same source.
    lower_cutoff = backend.pre_coalescence_duration("IMRPhenomD", 1024.0, 10.0, **_BBH)
    assert lower_cutoff > bbh


@pytest.mark.parametrize("ringdown_fraction", [0.05, 0.25])
def test_a_non_default_ringdown_fraction_is_honoured(ringdown_fraction):
    """The backend's own fraction must reach the answer, not the module default.

    Added after a mutation survived: dropping ``self._ringdown_fraction`` from the query passed
    every other test here, because ``coalescence_placement`` defaults to the same 0.1 a
    default-constructed backend uses. The two are indistinguishable at that one value, so a
    backend configured with any other would have been told the wrong start time with nothing
    failing -- the fraction sets how much of the buffer sits *after* coalescence, so getting it
    wrong moves the start by up to a fifth of the buffer.
    """
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    backend = LALSimulationBackend(ringdown_fraction=ringdown_fraction)

    predicted = backend.pre_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **_BBH)
    actual = _TC - float(generated["plus"].t0.value)

    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)
