"""Convergence and transfer tests for the band-limited resampling kernel.

Accuracy here is a claim about convergence, so it is measured against a reference with no
interpolation error at all: a pure tone, whose value at any shifted time is known
analytically.

An earlier version of these tests drew several *random* frequencies below a stated
fraction of Nyquist and reported that as the signal band. That measures the wrong thing —
with a fixed seed the drawn frequencies never approach the bound, so the kernel looked
accurate right up to Nyquist when it is not. Single tones at exactly the stated fraction
are used instead, which is what makes the band limit below trustworthy.

Comparisons are restricted to the interior of the series: the resampler zero-fills outside
``[0, n - 1]`` while a tone does not vanish there, so the edges compare two deliberately
different things. Boundary behaviour is covered in ``test_jax_projection.py``.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from gwmock_signal.projection.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    resample_uniform_sinc,
    validate_kernel,
)

_N = 8192
#: Fraction of Nyquist up to which the default kernel is accurate. Above this, no
#: windowed-sinc kernel of practical length works, which is why the projection paths
#: require the strain to be oversampled with margin.
USABLE_BAND_FRACTION = 0.8


def _tone_error(frequency_over_nyquist: float, *, taps: int, beta: float, shift: float = 0.5) -> float:
    """Return the max resampling error for one tone, over the interior of the series.

    Args:
        frequency_over_nyquist: Tone frequency as a fraction of the Nyquist frequency.
        taps: Kernel taps.
        beta: Kaiser window shape parameter.
        shift: Sub-sample shift to resample by.

    Returns:
        Maximum absolute error; the tone has unit amplitude, so this is also relative.
    """
    frequency = 0.5 * frequency_over_nyquist  # cycles per sample
    phase = 0.3
    samples = np.cos(2.0 * np.pi * frequency * np.arange(_N, dtype=float) + phase)

    index = np.arange(_N, dtype=float) - shift
    got = resample_uniform_sinc(samples, index, taps=taps, beta=beta)
    want = np.cos(2.0 * np.pi * frequency * index + phase)

    interior = slice(taps, _N - taps)
    return float(np.max(np.abs(got[interior] - want[interior])))


@pytest.mark.parametrize("frequency_over_nyquist", [0.1, 0.5, USABLE_BAND_FRACTION])
def test_default_kernel_is_accurate_across_the_usable_band(frequency_over_nyquist: float) -> None:
    """The shipped default resamples to ~1e-11 or better up to 0.8 x Nyquist.

    This is the number that justifies replacing the previous cubic, whose error on the
    same projection was 1.9e-3 RMS and 5.9e-2 at worst.
    """
    error = _tone_error(frequency_over_nyquist, taps=DEFAULT_SINC_TAPS, beta=DEFAULT_KAISER_BETA)
    assert error < 1e-11, error


@pytest.mark.parametrize("shift", [0.5, 0.25, 0.1, 1.7, -0.6])
def test_accuracy_holds_for_any_sub_sample_shift(shift: float) -> None:
    """Accuracy must not depend on where between samples the delay lands."""
    error = _tone_error(0.5, taps=DEFAULT_SINC_TAPS, beta=DEFAULT_KAISER_BETA, shift=shift)
    assert error < 1e-11, error


def test_error_falls_with_tap_count_at_fixed_beta() -> None:
    """Adding taps must reduce the error until the beta-set floor is reached."""
    errors = [_tone_error(0.5, taps=taps, beta=16.0) for taps in (63, 127, 255)]
    assert all(later <= earlier for earlier, later in pairwise(errors)), errors


def test_beta_sets_the_error_floor_not_the_tap_count() -> None:
    """At fixed taps, a larger beta is what buys accuracy.

    Pins the finding that drove the default: raising taps at beta = 16 plateaus around
    1e-9, while raising beta at 127 taps reaches 1e-12.
    """
    low_beta = _tone_error(0.5, taps=DEFAULT_SINC_TAPS, beta=16.0)
    high_beta = _tone_error(0.5, taps=DEFAULT_SINC_TAPS, beta=DEFAULT_KAISER_BETA)
    assert high_beta < low_beta / 100.0, (low_beta, high_beta)


def test_accuracy_degrades_above_the_usable_band() -> None:
    """No windowed sinc of practical length works up against Nyquist.

    The projection docstrings tell users to oversample with margin; this pins the reason,
    so the advice cannot quietly stop being true.
    """
    near_nyquist = _tone_error(0.99, taps=DEFAULT_SINC_TAPS, beta=DEFAULT_KAISER_BETA)
    in_band = _tone_error(USABLE_BAND_FRACTION, taps=DEFAULT_SINC_TAPS, beta=DEFAULT_KAISER_BETA)
    assert in_band < 1e-11
    assert near_nyquist > 1e-3, f"expected content at 0.99 Nyquist to fail badly; got {near_nyquist:.3e}"


def test_integer_shift_reproduces_the_samples() -> None:
    """A whole-sample shift must return the samples themselves."""
    samples = np.cos(2.0 * np.pi * 0.1 * np.arange(_N, dtype=float))
    got = resample_uniform_sinc(samples, np.arange(_N, dtype=float) - 3.0)
    interior = slice(DEFAULT_SINC_TAPS, _N - DEFAULT_SINC_TAPS)
    assert np.allclose(got[interior], np.roll(samples, 3)[interior], rtol=0.0, atol=1e-12)


def test_device_and_host_kernels_agree() -> None:
    """The JAX and NumPy evaluations of one specification must agree to round-off.

    They are separate implementations; if they drift, every device-versus-host projection
    comparison silently inherits the difference.
    """
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    from gwmock_signal.projection.jax_projection import _interpolate_uniform_sinc

    samples = np.cos(2.0 * np.pi * 0.2 * np.arange(_N, dtype=float) + 0.4)
    index = np.arange(_N, dtype=float) - 0.37
    host = resample_uniform_sinc(samples, index)
    device = np.asarray(_interpolate_uniform_sinc(samples, index, _N))
    assert np.max(np.abs(host - device)) < 1e-12


@pytest.mark.parametrize("taps", [2, 4, 128, 0, -1])
def test_even_or_tiny_tap_counts_rejected(taps: int) -> None:
    """The kernel must be symmetric, so even and degenerate tap counts are errors."""
    with pytest.raises(ValueError, match="odd integer"):
        validate_kernel(taps, 4.0)


def test_negative_beta_rejected() -> None:
    """A negative Kaiser beta is not a valid window."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_kernel(DEFAULT_SINC_TAPS, -1.0)


def test_beta_too_large_for_taps_rejected() -> None:
    """Rejected rather than warned: this combination fails quietly, not loudly."""
    with pytest.raises(ValueError, match="transition"):
        validate_kernel(63, DEFAULT_KAISER_BETA)


def test_defaults_are_self_consistent() -> None:
    """The shipped defaults must satisfy the kernel's own validity bound."""
    assert validate_kernel(DEFAULT_SINC_TAPS, DEFAULT_KAISER_BETA) == (
        DEFAULT_SINC_TAPS,
        DEFAULT_KAISER_BETA,
    )
