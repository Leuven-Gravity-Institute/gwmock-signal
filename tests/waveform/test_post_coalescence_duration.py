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
    """A constant would be wrong by orders of magnitude -- a magnitude pin, nothing more.

    **This does not discriminate a post/pre swap**, and its earlier docstring implied it did: the
    ratio is roughly the same on both sides (~64 here, ~64 for pre), so returning the pre value
    would satisfy it unchanged. What actually catches that is the end-match test above, which
    compares the promise against generation. Kept because it pins the *scale* -- a caller
    substituting a small constant truncates the BNS case by tens of seconds -- and because the
    scale is the reason the query exists at all.
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
    coalescence precedes a segment -- including the ones whose tail lands inside it.
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


@pytest.mark.parametrize("ringdown_fraction", [0.05, 0.2, 0.3])
def test_a_non_default_ringdown_fraction_is_honoured(ringdown_fraction):
    """This fraction *is* the post side, so it must reach the answer.

    The pre-side version of this test exists because a mutation dropping ``self._ringdown_fraction``
    survived: the module default matches a default-constructed backend, so the two are
    indistinguishable at that one value. The same trap applies here and matters more -- the
    fraction defines how much of the buffer sits after coalescence, which is exactly what this
    query reports.
    """
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    backend = LALSimulationBackend(ringdown_fraction=ringdown_fraction)

    predicted = backend.post_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **_BBH)
    actual = _end_relative_to_tc(generated, 1024.0)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


@pytest.mark.parametrize("segment_duration", [8.0, 64.0])
def test_a_pinned_segment_duration_is_honoured(segment_duration):
    """A pinned duration bypasses the chirp-time estimate and must reach this path too."""
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    backend = LALSimulationBackend(segment_duration=segment_duration)

    predicted = backend.post_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **_BBH)
    actual = _end_relative_to_tc(generated, 1024.0)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


def test_the_gwsignal_backend_inherits_a_correct_answer():
    """It subclasses the LAL backend and overrides only the frequency-domain evaluation."""
    gwsignal = pytest.importorskip("lalsimulation.gwsignal", reason="gwsignal is not available in this lalsuite build")
    del gwsignal
    from gwmock_signal.waveform.backends.gwsignal import GWSignalBackend

    backend = GWSignalBackend()

    predicted = backend.post_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **_BBH)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **_BBH)
    actual = _end_relative_to_tc(generated, 1024.0)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


def test_ripple_carries_eta_into_the_answer():
    """Ripple's sizing carries eta, so an asymmetric binary must not be sized as a symmetric one."""
    backend = _ripple_backend()
    asymmetric = {"mass1": 60.0, "mass2": 6.0, "distance": 400.0, "inclination": 0.0}

    predicted = backend.post_coalescence_duration("IMRPhenomD", 1024.0, 20.0, **asymmetric)
    generated = backend.generate_td_waveform("IMRPhenomD", _TC, 1024.0, 20.0, **asymmetric)
    actual = _end_relative_to_tc(generated, 1024.0)

    assert predicted is not None
    assert predicted == pytest.approx(actual, abs=0.5 / 1024.0)


def test_the_simulator_and_factory_expose_the_query() -> None:
    """A caller holds a simulator, not a backend -- the same argument the pre side makes.

    Without this the only route to the tail is two private attributes across a package boundary,
    which breaks silently on an internal rename and bypasses custom-registration handling.
    """
    from gwmock_signal.simulator import CBCSimulator

    simulator = CBCSimulator(waveform_model="IMRPhenomD")
    # `_BBH` already carries `distance`; adding `luminosity_distance` too is refused as a mixed
    # alias, which is the backend being right rather than a problem to work around.
    params = {"coa_time": _TC, **_BBH}

    pre = simulator.pre_coalescence_duration(params, 1024.0, 20.0)
    post = simulator.post_coalescence_duration(params, 1024.0, 20.0)

    assert pre is not None
    assert post is not None
    assert post > 0.0


def test_a_custom_registered_model_answers_unknown() -> None:
    """A registered callable never reaches the backend, so the backend's sizing cannot describe it.

    The mirror of the pre-side case, and it fails worse on this side. Handing back the backend's
    tail would look authoritative while describing a waveform the simulator is not going to
    generate -- and the caller this query exists for uses the tail to decide an event is *finished*
    and can be dropped. A confident wrong tail therefore deletes signal, where the pre side's would
    only misplace it.
    """
    import numpy as np
    from gwpy.timeseries import TimeSeries

    from gwmock_signal.simulator import CBCSimulator

    def _flat(**kwargs):
        del kwargs
        return {
            "plus": TimeSeries(np.ones(128), t0=0.0, sample_rate=128.0),
            "cross": TimeSeries(np.zeros(128), t0=0.0, sample_rate=128.0),
        }

    simulator = CBCSimulator(waveform_model="MyFlatModel")
    simulator.register_waveform_model("MyFlatModel", _flat)

    assert simulator.post_coalescence_duration({"coa_time": _TC, **_BBH}, 1024.0, 20.0) is None


def test_an_unknown_model_raises_like_generation_would() -> None:
    """Unknown and unanswerable are different: one is a caller error, the other a capability gap.

    Both would be `None` if the query swallowed the lookup failure, and the caller cannot tell a
    typo'd approximant from a backend that declines to answer -- it would silently take the
    conservative branch forever.
    """
    from gwmock_signal.simulator import CBCSimulator

    simulator = CBCSimulator(waveform_model="NotAnApproximant")

    with pytest.raises(ValueError, match="not found"):
        simulator.post_coalescence_duration({"coa_time": _TC, **_BBH}, 1024.0, 20.0)
