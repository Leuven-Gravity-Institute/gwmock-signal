"""How long after ``tc`` a backend's buffer runs, asked before generating.

The complement of :mod:`tests.waveform.test_pre_coalescence_duration`, and needed for the same
kind of decision from the other side: a caller that knows only where a buffer *starts* can tell
that an event begins before a segment, but not that it has finished before one. Without that, a
run beginning later than its population's first event has no way to know those earlier events
cannot contribute, and pulls every one of them into its first batch.

**The tail is not a rounding error.** It is a fixed fraction of the buffer, so it grows with the
buffer: sub-second for a stellar-mass binary at 20 Hz, tens of seconds for a BNS at the same
cutoff, and larger as the cutoff falls. Any caller tempted to approximate it with a constant --
"ringdown is milliseconds" -- would be wrong by orders of magnitude, which is why this is asked of
the backend.

As with the pre side, the property under test is *agreement*, not arithmetic: the number a backend
promises must be the number generation delivers. Recomputing the formula here would pass against a
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


def _end_relative_to_tc(generated, sampling_frequency: float) -> float:
    """Seconds from ``tc`` to one sample past the buffer's last sample."""
    plus = generated["plus"]
    return float(plus.t0.value) + len(plus.value) / sampling_frequency - _TC


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
@pytest.mark.parametrize("sampling_frequency", [1024.0, 2048.0])
def test_the_lal_backend_predicts_where_its_own_waveform_ends(params, sampling_frequency):
    """The promise must equal what generation delivers, not merely re-derive the formula."""
    backend = _lal_backend()

    predicted = backend.post_coalescence_duration("IMRPhenomD", sampling_frequency, 20.0, **params)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, sampling_frequency, 20.0, **params)
    actual = _end_relative_to_tc(generated, sampling_frequency)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / sampling_frequency)


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
def test_the_ripple_backend_predicts_where_its_own_waveform_ends(params):
    """Ripple sizes differently and must still agree with itself."""
    backend = _ripple_backend()

    predicted = backend.post_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **params)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **params)
    actual = _end_relative_to_tc(generated, 1024.0)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


@pytest.mark.parametrize("params", [_BBH, _BNS], ids=["bbh", "bns"])
def test_the_two_sides_account_for_the_whole_buffer(params):
    """``pre + post`` must be the buffer, or one of them is describing a different waveform.

    The two queries share their sizing, and this is what pins that they still do: a change to one
    that forgot the other would show up here rather than as a caller silently placing signals
    against half-updated arithmetic.
    """
    backend = _lal_backend()
    fs = 2048.0

    pre = backend.pre_coalescence_duration("IMRPhenomD", fs, 20.0, **params)
    post = backend.post_coalescence_duration("IMRPhenomD", fs, 20.0, **params)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, fs, 20.0, **params)
    buffer_seconds = len(generated["plus"].value) / fs

    assert pre is not None
    assert post is not None
    assert pre + post == pytest.approx(buffer_seconds, abs=1.0 / fs)


def test_the_tail_is_large_enough_that_no_caller_should_guess_it():
    """A constant would be wrong by orders of magnitude, which is why the query exists.

    Pinned as a fact about the shape of the answer rather than an exact figure: the tail scales
    with the buffer, so a BNS carries seconds of it where a stellar-mass binary carries a fraction
    of one. A caller approximating "ringdown" as a small constant would truncate the longer case.
    """
    backend = _lal_backend()

    bbh = backend.post_coalescence_duration("IMRPhenomD", 2048.0, 20.0, **_BBH)
    bns = backend.post_coalescence_duration("IMRPhenomD", 2048.0, 20.0, **_BNS)

    assert bbh is not None
    assert bns is not None
    assert bns > 10.0 * bbh, (bbh, bns)


def test_a_backend_that_cannot_say_returns_none_rather_than_zero():
    """``None`` means unknown. Zero would mean "ends at coalescence", which is never true.

    The mirror of the pre-side trap, and worse in this direction: a caller reading ``0.0`` as an
    answer concludes an event's content stops at its ``tc``, and would discard every event whose
    coalescence precedes a segment -- including the ones whose ringdown lands inside it.
    """
    from gwmock_signal.waveform.backends.base import WaveformBackend

    class Unanswering(WaveformBackend):
        def available_approximants(self):
            return ()

        def generate_td_waveform(self, *args, **kwargs):
            raise NotImplementedError

        def generate_fd_waveform(self, *args, **kwargs):
            raise NotImplementedError

    assert Unanswering().post_coalescence_duration("IMRPhenomD", 2048.0, 20.0, **_BBH) is None


def test_pycbc_cannot_say_either() -> None:
    """PyCBC delegates conditioning to its own library, so it answers neither side."""
    pytest.importorskip("pycbc", reason="the [pycbc] extra is not installed")
    from gwmock_signal.waveform.backends.pycbc import PyCBCBackend

    backend = PyCBCBackend()
    assert backend.post_coalescence_duration("IMRPhenomD", 2048.0, 20.0, **_BBH) is None
