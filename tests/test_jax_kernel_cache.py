"""The batched kernels must be compiled once per configuration, not once per call.

Both batched entry points used to build ``jax.jit`` around closures defined inside the
function body, so every call handed XLA a new callable and re-paid tracing, lowering and
compilation — about 121 s per call for IMRPhenomXPHM on an A100, which made the batched path
slower than the per-event LAL loop it exists to replace.

These tests assert reuse through the caches' own hit counters rather than by timing, so they
do not depend on the speed of the machine running them. The failure they guard against is
specific and has happened in this codebase before: reintroducing a closure, or closing over a
per-call value, silently restores the recompilation.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("ripplegw", reason="ripple not installed")

from gwmock_signal.jax_batch import (  # noqa: E402
    _rotating_projection_kernel,
    _static_projection_kernel,
    _time_domain_kernel,
    simulate_cbc_batch,
)
from gwmock_signal.waveform.backends.ripple import (  # noqa: E402
    RippleBackend,
    _batched_polarization_kernel,
)

_PARAMETERS = {
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
_WAVEFORM_KEYS = ("detector_frame_mass_1", "detector_frame_mass_2", "luminosity_distance", "inclination", "coa_phase")


def _waveform_parameters() -> dict[str, np.ndarray]:
    return {k: _PARAMETERS[k] for k in _WAVEFORM_KEYS}


def test_repeated_waveform_batches_reuse_one_kernel() -> None:
    """A second call with the same configuration must hit the cache, not recompile."""
    backend = RippleBackend()
    kwargs = {"sampling_frequency": 1024.0, "minimum_frequency": 25.0, "parameters": _waveform_parameters()}
    backend.generate_fd_polarizations_batch("IMRPhenomD", **kwargs)
    before = _batched_polarization_kernel.cache_info().hits
    backend.generate_fd_polarizations_batch("IMRPhenomD", **kwargs)
    assert _batched_polarization_kernel.cache_info().hits == before + 1


def test_a_new_backend_instance_reuses_the_kernel() -> None:
    """The cache is keyed on configuration, not on backend identity.

    Callers construct backends freely (``simulate_cbc_batch`` does when none is passed), so a
    per-instance cache would miss on every call in normal use.
    """
    kwargs = {"sampling_frequency": 1024.0, "minimum_frequency": 25.0, "parameters": _waveform_parameters()}
    RippleBackend().generate_fd_polarizations_batch("IMRPhenomD", **kwargs)
    before = _batched_polarization_kernel.cache_info().hits
    RippleBackend().generate_fd_polarizations_batch("IMRPhenomD", **kwargs)
    assert _batched_polarization_kernel.cache_info().hits == before + 1


def test_different_presets_get_different_kernels() -> None:
    """Two preset configurations must not share one kernel.

    The cache is cleared first: other test modules populate it with the same configurations,
    so asserting on cache growth without clearing would pass or fail depending on test order.
    The frequency grid is no longer part of the key -- it is an argument -- so the distinction
    tested here is the preset, which is what the key actually covers.
    """
    _batched_polarization_kernel.cache_clear()
    backend = RippleBackend()
    for approximant in ("IMRPhenomD", "IMRPhenomXAS"):
        backend.generate_fd_polarizations_batch(
            approximant,
            sampling_frequency=1024.0,
            minimum_frequency=25.0,
            parameters=_waveform_parameters(),
        )
    assert _batched_polarization_kernel.cache_info().currsize == 2


def test_varying_only_the_grid_reuses_one_kernel() -> None:
    """A different frequency grid must *not* build a new kernel.

    This is the payoff of taking the grid as an argument instead of keying on it. Varied via
    the sample rate rather than the low-frequency cutoff: with no explicit ``f_ref`` the
    backend uses ``minimum_frequency`` as the reference frequency, so changing the cutoff
    genuinely changes the ripple preset and *should* build a new kernel. The sample rate
    changes the grid alone.
    """
    _batched_polarization_kernel.cache_clear()
    backend = RippleBackend()
    for sampling_frequency in (1024.0, 2048.0, 4096.0):
        backend.generate_fd_polarizations_batch(
            "IMRPhenomD",
            sampling_frequency=sampling_frequency,
            minimum_frequency=25.0,
            parameters=_waveform_parameters(),
        )
    info = _batched_polarization_kernel.cache_info()
    assert info.currsize == 1, info
    assert info.hits == 2, info


def test_reference_frequency_is_part_of_the_key() -> None:
    """A different reference frequency changes the preset, so it must not share a kernel.

    Complements the test above: the grid is deliberately outside the key, ``f_ref`` is
    deliberately inside it, and with the default backend the cutoff sets ``f_ref``.
    """
    _batched_polarization_kernel.cache_clear()
    backend = RippleBackend()
    for minimum_frequency in (25.0, 30.0):
        backend.generate_fd_polarizations_batch(
            "IMRPhenomD",
            sampling_frequency=1024.0,
            minimum_frequency=minimum_frequency,
            parameters=_waveform_parameters(),
        )
    assert _batched_polarization_kernel.cache_info().currsize == 2


@pytest.mark.parametrize("earth_rotation", [False, True])
def test_repeated_batches_reuse_the_projection_kernels(earth_rotation: bool) -> None:
    """Every kernel the batch path uses must be reused across calls."""
    caches = [_static_projection_kernel] if not earth_rotation else [_rotating_projection_kernel, _time_domain_kernel]
    kwargs = {
        "sampling_frequency": 1024.0,
        "minimum_frequency": 25.0,
        "parameters": _PARAMETERS,
        "earth_rotation": earth_rotation,
    }
    simulate_cbc_batch("IMRPhenomD", ["E1", "E2", "E3"], **kwargs)
    before = [c.cache_info().hits for c in caches]
    simulate_cbc_batch("IMRPhenomD", ["E1", "E2", "E3"], **kwargs)
    after = [c.cache_info().hits for c in caches]
    assert all(a > b for a, b in zip(after, before, strict=True)), (before, after)


@pytest.mark.parametrize("earth_rotation", [False, True])
def test_reuse_does_not_change_the_output(earth_rotation: bool) -> None:
    """Caching is an optimisation: repeated calls must be bit-identical."""
    kwargs = {
        "sampling_frequency": 1024.0,
        "minimum_frequency": 25.0,
        "parameters": _PARAMETERS,
        "earth_rotation": earth_rotation,
    }
    first = np.asarray(simulate_cbc_batch("IMRPhenomD", ["E1", "E2", "E3"], **kwargs).strain)
    second = np.asarray(simulate_cbc_batch("IMRPhenomD", ["E1", "E2", "E3"], **kwargs).strain)
    assert np.array_equal(first, second)


def test_sidereal_rate_is_not_part_of_the_cache_key() -> None:
    """Two different epochs must share one rotating kernel.

    The sidereal rate is derived from Astropy per call. If it were closed over rather than
    passed as an argument, every epoch would be a fresh cache key and the recompilation this
    cache removes would come straight back.
    """
    kwargs = {"sampling_frequency": 1024.0, "minimum_frequency": 25.0, "earth_rotation": True}
    simulate_cbc_batch("IMRPhenomD", ["E1"], parameters=_PARAMETERS, **kwargs)
    before = _rotating_projection_kernel.cache_info()
    shifted = dict(_PARAMETERS)
    # Half a year later: a clearly different sidereal rate evaluation.
    shifted["coa_time"] = _PARAMETERS["coa_time"] + 1.5e7
    simulate_cbc_batch("IMRPhenomD", ["E1"], parameters=shifted, **kwargs)
    after = _rotating_projection_kernel.cache_info()
    assert after.currsize == before.currsize, "a new epoch created a new kernel"
    assert after.hits > before.hits
