"""Tests for device-memory estimation, chunk sizing and the batch preflight.

The point of this machinery is not precision — the underlying coefficients are calibrated
from one measurement — but that a catalogue too large for the device fails with an
actionable message instead of a bare XLA ``RESOURCE_EXHAUSTED``, and that the default
chunk size is derived from the grid actually selected rather than left unbounded.

The consistency test below is the one that matters most: it checks the model against every
A100 observation available, including the configuration that actually ran out of memory.
"""

from __future__ import annotations

import pytest

from gwmock_signal import jax_batch
from gwmock_signal.jax_batch import (
    estimate_batch_memory_bytes,
    recommend_chunk_size,
)

_A100_BYTES = 80 * 2**30


def test_estimate_scales_linearly_in_events_and_samples() -> None:
    """Peak memory is proportional to the total sample count of the batch."""
    base = estimate_batch_memory_bytes(100, 3, 1024)
    assert estimate_batch_memory_bytes(200, 3, 1024) == pytest.approx(2 * base, rel=1e-12)
    assert estimate_batch_memory_bytes(100, 3, 2048) == pytest.approx(2 * base, rel=1e-12)


def test_estimate_grows_with_detectors_but_not_proportionally() -> None:
    """Only the projection buffers scale with detector count; generation does not.

    Guards the modelling decision: treating the whole estimate as proportional to detector
    count would under-estimate single-detector batches, which is the dangerous direction.
    """
    one = estimate_batch_memory_bytes(100, 1, 1024)
    three = estimate_batch_memory_bytes(100, 3, 1024)
    assert one < three < 3 * one


def test_rotating_path_is_estimated_larger() -> None:
    """The rotating path holds more per-detector buffers, so it must chunk smaller."""
    static = estimate_batch_memory_bytes(100, 3, 1024, earth_rotation=False)
    rotating = estimate_batch_memory_bytes(100, 3, 1024, earth_rotation=True)
    assert rotating > static
    assert recommend_chunk_size(3, 1024, earth_rotation=True, available_bytes=_A100_BYTES) <= (
        recommend_chunk_size(3, 1024, earth_rotation=False, available_bytes=_A100_BYTES)
    )


@pytest.mark.parametrize(
    ("n_events", "n_samples", "fits"),
    [
        # The measurement the coefficients come from: this asked XLA for 85.4 GiB and failed.
        (16384, 8192, False),
        # Chunk sizes that were run successfully on an A100.
        (2048, 8192, True),
        (8, 4194304, True),  # BNS, f_min 10 Hz
        (2, 33554432, True),  # BNS, f_min 5 Hz
    ],
)
def test_model_agrees_with_every_a100_observation(n_events: int, n_samples: int, fits: bool) -> None:
    """The estimate must classify all four observed configurations correctly."""
    estimate = estimate_batch_memory_bytes(n_events, 3, n_samples, earth_rotation=False)
    assert (estimate <= _A100_BYTES) is fits, f"{estimate / 2**30:.1f} GiB vs {_A100_BYTES / 2**30:.0f} GiB"


def test_recommended_chunk_fits_the_budget() -> None:
    """A recommended chunk must fit within the fraction it was sized against."""
    for n_samples in (8192, 131072, 4194304, 33554432):
        chunk = recommend_chunk_size(3, n_samples, available_bytes=_A100_BYTES, memory_fraction=0.6)
        assert chunk >= 1
        assert estimate_batch_memory_bytes(chunk, 3, n_samples) <= 0.6 * _A100_BYTES


def test_recommendation_is_none_when_the_device_limit_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU device reports no limit, and that must read as "cannot check", not "no limit"."""
    # Patched rather than relying on whatever device the test host happens to have, so the
    # assertion is about behaviour and not about the runner.
    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: None)
    assert recommend_chunk_size(3, 8192) is None
    # An explicit zero limit means the same thing.
    assert recommend_chunk_size(3, 8192, available_bytes=0) is None


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, 2.0, float("nan")])
def test_memory_fraction_outside_the_unit_interval_rejected(fraction: float) -> None:
    """A fraction above 1 would recommend a chunk larger than the device it is sized for."""
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        recommend_chunk_size(3, 8192, available_bytes=_A100_BYTES, memory_fraction=fraction)


def test_preflight_is_silent_when_the_limit_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never block a run because the device could not be queried."""
    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: None)
    jax_batch._check_batch_fits(10**9, 3, 10**6, earth_rotation=True)


def test_preflight_passes_a_batch_that_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch inside the limit must not raise."""
    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: _A100_BYTES)
    jax_batch._check_batch_fits(1024, 3, 8192, earth_rotation=True)


def test_preflight_error_names_the_numbers_and_a_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole reason this exists: the message must say what to do next."""
    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: _A100_BYTES)
    with pytest.raises(MemoryError) as excinfo:
        jax_batch._check_batch_fits(16384, 3, 8192, earth_rotation=False)
    message = str(excinfo.value)
    assert "GiB" in message
    assert "chunk_size=" in message
    assert "16384 events" in message


def test_the_free_remedy_comes_before_the_one_that_costs_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two remedies, one of them free, and the message must not present them as equivalent.

    `chunk_size` splits the batch and costs wall-clock alone: the output is identical. Raising
    `minimum_frequency` shortens the buffer by discarding the early inspiral, which is a *different
    simulation*, not a smaller one. A user reading "or" as "equivalently" damages every waveform in
    the run to fit a memory limit -- and gets no warning that they did.

    Order is asserted, not merely presence: the previous version of this test asserted that
    `minimum_frequency` appeared at all, which any phrasing satisfies including the defective one.
    """
    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: _A100_BYTES)
    with pytest.raises(MemoryError) as excinfo:
        jax_batch._check_batch_fits(16384, 3, 8192, earth_rotation=False)
    message = str(excinfo.value)

    assert message.index("chunk_size=") < message.index("minimum_frequency"), (
        "the physics-altering remedy is offered before the free one"
    )
    # The cost has to be stated where the remedy is offered, not left to the reader's knowledge of
    # what a low-frequency cutoff does.
    tail = message[message.index("minimum_frequency") - 200 :]
    assert any(word in tail for word in ("discard", "removes", "loses", "changes what")), (
        f"raising minimum_frequency is offered without saying it changes the simulated signal: {tail!r}"
    )


def test_preflight_runs_before_waveform_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check must fire before anything large is allocated.

    Most of the estimate *is* the waveform-generation buffers, so a preflight placed after
    generation could never fire for a batch that exhausts memory while generating -- which
    defeats the purpose. Asserted by making generation explode if it is ever reached.
    """
    pytest.importorskip("jax", reason="jax not installed")
    pytest.importorskip("ripplegw", reason="ripple not installed")
    import numpy as np

    from gwmock_signal.waveform.backends.ripple import RippleBackend

    monkeypatch.setattr(jax_batch, "available_device_memory_bytes", lambda: 1024)  # absurdly small

    class _ExplodingBackend(RippleBackend):
        def generate_fd_polarizations_batch(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("waveform generation ran before the memory preflight")

    parameters = {
        "detector_frame_mass_1": np.array([30.0, 25.0]),
        "detector_frame_mass_2": np.array([28.0, 22.0]),
        "luminosity_distance": np.array([900.0, 1200.0]),
        "inclination": np.array([0.3, 1.1]),
        "coa_phase": np.array([0.0, 2.0]),
        "right_ascension": np.array([1.3, 4.0]),
        "declination": np.array([-0.4, 0.6]),
        "polarization_angle": np.array([0.7, 2.1]),
        "coa_time": np.array([1.4e9, 1.4e9 + 300.0]),
    }
    with pytest.raises(MemoryError, match="chunk_size="):
        jax_batch.simulate_cbc_batch(
            "IMRPhenomD",
            ["E1", "E2", "E3"],
            sampling_frequency=1024.0,
            minimum_frequency=25.0,
            parameters=parameters,
            backend=_ExplodingBackend(),
        )


@pytest.mark.parametrize(("events", "detectors", "samples"), [(0, 3, 8), (1, 0, 8), (1, 3, 0), (-1, 3, 8)])
def test_degenerate_shapes_rejected(events: int, detectors: int, samples: int) -> None:
    """Zero or negative shapes are a caller error, not a zero-memory batch."""
    with pytest.raises(ValueError, match=">= 1"):
        estimate_batch_memory_bytes(events, detectors, samples)


def test_catalogue_forwards_earth_rotation() -> None:
    """``simulate_cbc_catalogue`` must expose Earth rotation.

    It did not when the rotating path first landed, so the production entry point silently
    took the new default with no way to turn it off.
    """
    import inspect

    signature = inspect.signature(jax_batch.simulate_cbc_catalogue)
    assert "earth_rotation" in signature.parameters
    assert signature.parameters["earth_rotation"].default is True
