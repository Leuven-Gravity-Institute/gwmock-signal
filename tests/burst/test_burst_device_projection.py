"""Bursts projected on the device: the backend option, and the fixed lengths the kernel is compiled for.

A burst injection set comes in thousands of buffer lengths, and the device kernel is compiled per length,
so each burst is padded with zeros to one of a few fixed lengths and the result cut back. These tests pin
that the padding changes nothing the burst contains -- exactly, where every step of the projection is
elementwise, and to rounding where the finite-arm residual is applied by FFT -- that bursts of different
lengths share a kernel, and that the padding is skipped where it would change the answer.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)

from gwmock_signal.burst.injection_set import draw_burst_injection_set  # noqa: E402
from gwmock_signal.burst.simulator import BurstSimulator  # noqa: E402
from gwmock_signal.network import Network  # noqa: E402
from gwmock_signal.projection import network  # noqa: E402
from gwmock_signal.projection.effects import ProjectionEffects  # noqa: E402
from gwmock_signal.projection.network import device_length  # noqa: E402
from gwmock_signal.projection.resampling import DEFAULT_KAISER_BETA, DEFAULT_SINC_TAPS  # noqa: E402
from gwmock_signal.projection.solar_system import DelayModel, SolarSystemEphemeris  # noqa: E402
from gwmock_signal.simulator import CBCSimulator  # noqa: E402

RATE = 8192.0

#: Every effect but the finite arms, whose FFT is the one step that is not elementwise.
_NO_ARMS = dataclasses.replace(ProjectionEffects(), finite_arm_length=False)


def _events() -> list[dict]:
    """Bursts of every family the injection set draws, from the start of a production-like stream."""
    drawn = draw_burst_injection_set(
        start_time=1e9,
        end_time=1e9 + 50.0 * 400,
        interval=50.0,
        hrss_range=(3.5e-24, 2.8e-22),
        rng=np.random.default_rng(1102),
    )
    return [{key: value for key, value in event.items() if value is not None} for event in drawn]


EVENTS = _events()
#: One burst of each family, and the white-noise bursts the length-dependent solar delay moved most.
SAMPLE = [next(e for e in EVENTS if e["burst_model"] == model) for model in ("gaussian", "sine_gaussian")] + [
    e for e in EVENTS if e["burst_model"] == "white_noise_burst"
][:6]


def _strain(simulator: BurstSimulator, event: dict, preset: str, effects: ProjectionEffects) -> list[np.ndarray]:
    detectors = list(Network.from_preset(preset).detector_names)
    stack = simulator.simulate(event, detectors, sampling_frequency=RATE, minimum_frequency=5.0, effects=effects)
    return [np.asarray(stack[i].data) for i in range(len(detectors))]


def _unpadded_simulator() -> BurstSimulator:
    simulator = BurstSimulator(projection_backend="jax")
    simulator.fixed_device_lengths = False
    return simulator


# The ladder


def test_the_ladder_has_two_lengths_per_octave_from_its_shortest():
    """``2**k`` and ``3 * 2**(k - 1)`` from 512: seventeen lengths up to 131,072 samples."""
    lengths = sorted({device_length(n) for n in range(1, 131_073)})
    assert lengths == [
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        49152,
        65536,
        98304,
        131072,
    ]


@pytest.mark.parametrize("n", [1, 511, 512, 513, 767, 768, 769, 1025, 17_683, 91_707, 98_305])
def test_a_length_is_padded_to_the_next_rung_by_less_than_half_of_itself(n):
    """No series grows by half its length or more, and a ladder length maps to itself."""
    length = device_length(n)
    assert length >= n
    assert length == 512 or length < 1.5 * n
    assert device_length(length) == length


def test_a_length_that_is_not_positive_is_refused():
    """An empty series has no length to pad to."""
    with pytest.raises(ValueError, match="positive"):
        device_length(0)


# The backend option


def test_the_burst_simulator_names_its_projection_backend():
    """Gwmock admits ``projection-backend`` only for a backend whose constructor names the parameter."""
    assert "projection_backend" in inspect.signature(BurstSimulator).parameters
    assert BurstSimulator().projection_backend == "numpy"
    assert BurstSimulator(projection_backend="jax").projection_backend == "jax"


def test_an_unknown_projection_backend_is_refused():
    """The name is checked as on the other simulators, at construction."""
    with pytest.raises(ValueError, match="projection_backend must be 'numpy' or 'jax'"):
        BurstSimulator(projection_backend="cuda")


def test_choosing_jax_switches_on_64_bit_mode():
    """The device projection refuses to run in 32-bit mode, so construction must turn 64-bit mode on."""
    jax.config.update("jax_enable_x64", False)
    try:
        BurstSimulator(projection_backend="jax")
        assert jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", True)


def test_only_bursts_ask_for_fixed_lengths():
    """A compact binary keeps compiling for its own length; the padding is the bursts' choice."""
    assert BurstSimulator.fixed_device_lengths is True
    assert CBCSimulator.fixed_device_lengths is False


# The padding changes nothing


@pytest.mark.parametrize("preset", ["ET-Triangle-EMR", "ET-2L-Misaligned"])
@pytest.mark.parametrize(
    "effects", [ProjectionEffects.long_wavelength(earth_rotation=True), _NO_ARMS], ids=["rotation", "no-arms"]
)
def test_padding_leaves_an_elementwise_projection_exactly_unchanged(preset, effects):
    """Every read past the polarizations' end is of a zero, padded or not, and every step is per sample.

    With the solar-system delay on, this needs the delay to be the same at every array length: summed by a
    compiled reduction it was not, and the padded bursts here moved by up to 3e-11 of their peak.
    """
    padded, unpadded = BurstSimulator(projection_backend="jax"), _unpadded_simulator()
    for event in SAMPLE:
        for a, b in zip(
            _strain(padded, event, preset, effects), _strain(unpadded, event, preset, effects), strict=True
        ):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("preset", ["ET-Triangle-EMR", "ET-2L-Misaligned"])
def test_padding_changes_a_projection_with_finite_arms_only_by_rounding(preset):
    """The finite-arm residual is a linear convolution evaluated by FFT on a length that grows with the series.

    Measured at most 4e-16 of the peak on these bursts.
    """
    padded, unpadded = BurstSimulator(projection_backend="jax"), _unpadded_simulator()
    for event in SAMPLE:
        for a, b in zip(
            _strain(padded, event, preset, ProjectionEffects()),
            _strain(unpadded, event, preset, ProjectionEffects()),
            strict=True,
        ):
            assert a.shape == b.shape
            assert np.max(np.abs(a - b)) <= 2e-15 * np.max(np.abs(b))


@pytest.mark.parametrize("preset", ["ET-Triangle-EMR", "ET-2L-Misaligned"])
def test_padded_device_bursts_agree_with_the_host_projection(preset):
    """The padded device path stays within the two backends' agreement for these bursts, measured at 7e-10 of peak."""
    device, host = BurstSimulator(projection_backend="jax"), BurstSimulator()
    for event in SAMPLE:
        for a, b in zip(
            _strain(device, event, preset, ProjectionEffects()),
            _strain(host, event, preset, ProjectionEffects()),
            strict=True,
        ):
            assert a.shape == b.shape
            assert np.max(np.abs(a - b)) <= 2e-9 * np.max(np.abs(b))


def test_the_solar_system_delay_is_the_same_at_every_array_length():
    """A sample's delay must not depend on how many other samples it is evaluated with, on either backend."""
    import jax.numpy as jnp

    model = DelayModel.build(1e9, 1e9 + 4.0, 1.3, -0.4, SolarSystemEphemeris())
    evaluate = jax.jit(lambda t: model.evaluate(t, xp=jnp))
    short = np.asarray(evaluate(jnp.arange(17_683) / RATE))
    for longer in (17_684, 24_576):
        np.testing.assert_array_equal(np.asarray(evaluate(jnp.arange(longer) / RATE))[:17_683], short)
    np.testing.assert_array_equal(
        model.evaluate(np.arange(24_576) / RATE)[:17_683], model.evaluate(np.arange(17_683) / RATE)
    )


# Kernels are shared, and padding is skipped where it would change the answer


def _compiled_lengths(monkeypatch) -> list[int]:
    """Record the length each device kernel is asked for."""
    seen: list[int] = []
    original = network._compiled_rotating_projection

    def recording(sampling_frequency, n_samples, *args):
        seen.append(n_samples)
        return original(sampling_frequency, n_samples, *args)

    monkeypatch.setattr(network, "_compiled_rotating_projection", recording)
    return seen


def test_bursts_of_different_lengths_on_one_rung_share_a_kernel(monkeypatch):
    """Two burst lengths on one rung ask for one kernel length; unpadded, they ask for two."""
    simulator = BurstSimulator(projection_backend="jax")
    padding = 2 * network.edge_padding(RATE)
    by_rung: dict[int, dict[int, dict]] = {}
    for event in EVENTS:
        if event["burst_model"] != "sine_gaussian":
            continue
        hp, _ = simulator.generate_polarizations(event, RATE, 5.0)
        by_rung.setdefault(device_length(hp.n_samples + padding), {}).setdefault(hp.n_samples, event)
    pair = next(list(events.values())[:2] for events in by_rung.values() if len(events) >= 2)
    detector = [next(iter(Network.from_preset("ET-Triangle-EMR").detector_names))]
    seen = _compiled_lengths(monkeypatch)
    for event in pair:
        simulator.simulate(event, detector, sampling_frequency=RATE, minimum_frequency=5.0)
    assert len(seen) == 2
    assert len(set(seen)) == 1
    assert seen[0] == device_length(seen[0])
    simulator.fixed_device_lengths = False
    for event in pair:
        simulator.simulate(event, detector, sampling_frequency=RATE, minimum_frequency=5.0)
    assert len(set(seen[2:])) == 2


@pytest.mark.parametrize(
    ("tapered", "keep_delayed_samples"), [(True, True), (False, False)], ids=["tapered", "no-delay-margin"]
)
def test_fixed_lengths_are_skipped_where_padding_would_change_the_result(monkeypatch, tapered, keep_delayed_samples):
    """A tapered input is oversampled over its whole length; without the delay margin its end is not zero."""
    seen = _compiled_lengths(monkeypatch)
    simulator = BurstSimulator()
    hp, hc = simulator.generate_polarizations(SAMPLE[1], RATE, 5.0)
    network._project(
        {"plus": hp, "cross": hc},
        [next(iter(Network.from_preset("ET-Triangle-EMR").detector_names))],
        right_ascension=1.3,
        declination=-0.4,
        polarization_angle=0.7,
        effects=ProjectionEffects(),
        backend="jax",
        sinc_taps=DEFAULT_SINC_TAPS,
        kaiser_beta=DEFAULT_KAISER_BETA,
        keep_delayed_samples=keep_delayed_samples,
        tapered=tapered,
        fixed_lengths=True,
    )
    expected = hp.n_samples + (2 * network.edge_padding(RATE) if keep_delayed_samples else 0)
    assert seen == [expected]
    assert device_length(expected) != expected


def test_the_kernel_cache_keeps_a_burst_set_on_a_four_detector_network():
    """One kernel per ladder length and per pair of arm lengths, which the finite arms make static.

    The production-like set at 8192 Hz spans 497 to about 92,000 samples once padded for the delay: 16
    lengths. H1, L1, V1 and K1 have four distinct pairs of arm lengths, so the set needs 64 kernels.
    """
    pairs = {network._arm_lengths(["H1", "L1", "V1", "K1"])[name] for name in ("H1", "L1", "V1", "K1")}
    assert len(pairs) == 4
    lengths = {device_length(n) for n in range(497, 92_000)}
    assert len(lengths) == 16
    assert len(lengths) * len(pairs) <= network._KERNEL_CACHE_SIZE
    assert network._compiled_rotating_projection.cache_info().maxsize == network._KERNEL_CACHE_SIZE
