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
    EARTH_RADIUS_M,
    SPEED_OF_LIGHT_M_S,
    edge_padding,
    require_shift_within_padding,
    require_terrestrial_location,
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


@pytest.mark.parametrize("shift", [0.5, 0.25, 0.1, 1.7, -0.6, 2.0**-20, 1.0 - 2.0**-20])
def test_accuracy_holds_for_any_sub_sample_shift(shift: float) -> None:
    """Accuracy must not depend on where between samples the delay lands.

    The last two shifts sit a millionth of a sample from a whole sample, on either side. The
    kernel reduces its sine argument about the nearest sample, so those are the two ends of the
    reduced interval; a reduction that lost accuracy at either end would show up here.
    """
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


def _resample_with_a_sinc_per_tap(samples: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Evaluate the same kernel with ``np.sinc`` called once per tap.

    Deliberately the naive form: one transcendental per tap, no identity applied. The kernel
    itself hoists the sine out of the loop, which is exact in real arithmetic but a different
    computation in floating point, so it needs something independent to be checked against.
    """
    half = (DEFAULT_SINC_TAPS - 1) // 2
    denominator = half + 1.0
    n = samples.shape[0]
    base = np.floor(index)
    frac = index - base
    base_int = base.astype(np.int64)

    total = np.zeros(index.shape, dtype=float)
    weight_sum = np.zeros(index.shape, dtype=float)
    for offset in range(-half, half + 1):
        x = frac - offset
        window = np.i0(DEFAULT_KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - (x / denominator) ** 2)))
        weight = np.sinc(x) * window / np.i0(DEFAULT_KAISER_BETA)
        total += weight * samples[np.clip(base_int + offset, 0, n - 1)]
        weight_sum += weight

    interpolated = np.where(weight_sum != 0.0, total / weight_sum, 0.0)
    return np.where((index >= 0.0) & (index <= n - 1), interpolated, 0.0)


@pytest.mark.parametrize("shift", [0.37, 0.5, 2.0**-20, 1.0 - 2.0**-20, 0.0])
def test_hoisted_sine_reproduces_a_sinc_per_tap(shift: float) -> None:
    """Hoisting the sine out of the tap loop must not change what the kernel computes.

    ``sin(pi*(frac - offset)) = (-1)**offset * sin(pi*frac)`` is exact in the reals, so the two
    evaluations may differ only by round-off. Dropping the per-tap ``(-1)**offset`` was checked
    to break this by O(1), which is the failure the change could plausibly introduce.

    What it deliberately does *not* claim to guard is the one-per-position ``(-1)**nearest``
    sign: that negates the tap sum and the weight sum alike, so it cancels in their quotient and
    no output-level test can see it.

    ``shift = 0`` covers the one tap whose argument is exactly zero, where the quotient
    ``sin(pi*x)/(pi*x)`` is replaced by 1.
    """
    samples = np.cos(2.0 * np.pi * 0.21 * np.arange(_N, dtype=float) + 0.7)
    index = np.arange(_N, dtype=float) - shift

    got = resample_uniform_sinc(samples, index)
    want = _resample_with_a_sinc_per_tap(samples, index)

    interior = slice(DEFAULT_SINC_TAPS, _N - DEFAULT_SINC_TAPS)
    # Absolute, with atol pinned: the signal is O(1) here, and a relative-only tolerance would
    # pass trivially wherever the resampled value happens to sit near a zero crossing.
    assert np.allclose(got[interior], want[interior], rtol=0.0, atol=1e-14)


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


@pytest.mark.parametrize("taps", [127.5, 126.9, 3.5])
def test_fractional_tap_counts_rejected(taps: float) -> None:
    """A fractional tap count must not be silently truncated to an integer."""
    with pytest.raises(ValueError, match="odd integer"):
        validate_kernel(taps, 4.0)


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_beta_rejected(beta: float) -> None:
    """NaN slips past a bare `beta < 0` check and would poison every output sample."""
    with pytest.raises(ValueError, match="finite"):
        validate_kernel(DEFAULT_SINC_TAPS, beta)


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


def test_padding_covers_the_largest_terrestrial_delay() -> None:
    """The delay term must bound the light travel time to the furthest ground-based detector.

    Checked against the constants rather than a hard-coded number, since the point of the bound
    is the physics: a detector on the equator is ``R_earth / c`` = 21.3 ms from the geocentre, so
    at any sample rate the padding must exceed that many samples.
    """
    sampling_frequency = 4096.0
    delay_samples = EARTH_RADIUS_M / SPEED_OF_LIGHT_M_S * sampling_frequency
    assert delay_samples == pytest.approx(87.2, abs=0.1)
    padding = edge_padding(sampling_frequency, DEFAULT_SINC_TAPS, DEFAULT_KAISER_BETA)
    assert padding > delay_samples + (DEFAULT_SINC_TAPS - 1) // 2


@pytest.mark.parametrize("sampling_frequency", [0.0, -1.0, float("nan"), float("inf")])
def test_padding_rejects_an_invalid_sample_rate(sampling_frequency: float) -> None:
    """A non-positive or non-finite rate would size the padding nonsensically."""
    with pytest.raises(ValueError, match="sampling_frequency"):
        edge_padding(sampling_frequency)


def test_padding_validates_the_callers_beta_not_the_default() -> None:
    """Regression: the kernel actually passed must be the one validated.

    ``edge_padding`` originally validated ``DEFAULT_KAISER_BETA`` regardless of the ``beta``
    argument, which both rejected the valid pair below and let an invalid one reserve a buffer
    before failing inside the interpolation.
    """
    assert edge_padding(63, taps=63, beta=15.0) > 0
    with pytest.raises(ValueError, match="transition"):
        edge_padding(4096.0, taps=63, beta=DEFAULT_KAISER_BETA)


@pytest.mark.parametrize("shift", [0.0, 0.5, 0.999999])
def test_valid_alignment_shifts_accepted(shift: float) -> None:
    """``split_index`` returns a remainder in [0, 1), so the whole range must pass."""
    require_shift_within_padding(np.array([shift]))


@pytest.mark.parametrize("shift", [-1e-9, -0.5, 1.0, 2.5, float("nan")])
def test_out_of_range_alignment_shifts_rejected(shift: float) -> None:
    """A shift outside [0, 1) reads past the padding; NaN would poison every sample."""
    with pytest.raises(ValueError, match=r"\[0, 1.0\) samples"):
        require_shift_within_padding(np.array([shift]))


def test_shift_error_names_the_offending_entry() -> None:
    """Regression: the reported entry must be an *offender*, not the largest in magnitude.

    Selecting by largest absolute value named the valid 0.9 here while the actual offender was
    the -0.1 at index 0, sending the reader to inspect a correct input.
    """
    with pytest.raises(ValueError, match="entry 0") as raised:
        require_shift_within_padding(np.array([-0.1, 0.9]))
    message = str(raised.value)
    assert "entry 0" in message
    assert "-0.1" in message
    assert "0.9" not in message.replace("-0.1", "")


def test_terrestrial_location_accepts_a_real_detector() -> None:
    """A LIGO Hanford-scale geocentre distance must pass."""
    require_terrestrial_location(np.array([-2.16e6, -3.83e6, 4.60e6]))


def test_non_terrestrial_location_rejected() -> None:
    """The padding bound assumes a ground-based detector, so a distant one is an error."""
    with pytest.raises(ValueError, match="equatorial radius"):
        require_terrestrial_location(np.array([0.0, 0.0, 4.0e8]), name="location of LISA-like")
