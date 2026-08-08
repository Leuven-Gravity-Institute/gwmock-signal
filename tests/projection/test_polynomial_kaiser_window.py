# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Evaluating the Kaiser window as a polynomial instead of an ``i0`` per tap.

The device resample kernel evaluated ``i0(beta * sqrt(1 - v)) / i0(beta)`` once per tap, 127 times
per output sample, in the kernel R3 measured at about 87% of the whole pipeline. Because ``i0`` is
even, that composition is analytic in ``v = (x / denominator) ** 2``, so a polynomial in ``v``
replaces **both** the ``i0`` and the ``sqrt``.

Measured on an RTX 3060 in float64, 2,000,000 output samples at 127 taps: **347.873 ns per output
sample against 159.891, a 2.18x speedup**, with the error against an analytic sinusoid unchanged at
4.027e-12 of peak and the two kernels differing by 2.2e-15.

What these tests pin is the part that could silently rot: that the fit meets its accuracy target for
every beta a caller can reach, that the window itself matches the exact one, and that the kernel's
agreement with an **analytic** signal is not degraded. The speedup is not asserted -- a timing
assertion on shared hardware fails for reasons that have nothing to do with this code.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_signal.projection.resampling import (
    _WINDOW_FIT_TOLERANCE,
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    kaiser_window_chebyshev,
)

pytestmark = pytest.mark.unit


def _exact_window(beta: float, v: np.ndarray) -> np.ndarray:
    """The window the kernel must reproduce, as a function of ``v``."""
    return np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - v))) / np.i0(beta)


def _clenshaw(coefficients: tuple[float, ...], v: np.ndarray) -> np.ndarray:
    """Evaluate the Chebyshev series on ``[0, 1]``, the way the device kernel does."""
    t = 2.0 * v - 1.0
    b_kp1 = np.zeros_like(t)
    b_kp2 = np.zeros_like(t)
    for coefficient in coefficients[:0:-1]:
        b_kp1, b_kp2 = 2.0 * t * b_kp1 - b_kp2 + coefficient, b_kp1
    return t * b_kp1 - b_kp2 + coefficients[0]


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 4.0, 8.0, 16.0, 24.0, 32.0])
def test_the_fit_meets_its_target_for_every_reachable_beta(beta: float) -> None:
    """A polynomial fitted for one beta would silently apply the wrong window to every other one.

    ``validate_kernel`` accepts any finite non-negative beta small enough for the tap count, so the
    fit is per-beta and its accuracy has to hold across the range rather than at the default.
    """
    coefficients = kaiser_window_chebyshev(beta)
    assert coefficients is not None, f"no degree reached the target for beta={beta}"

    v = np.unique(np.concatenate([np.linspace(0.0, 1.0, 3001), 1.0 - np.logspace(-12.0, -1.0, 300)]))
    v = v[(v >= 0.0) & (v <= 1.0)]
    error = float(np.max(np.abs(_clenshaw(coefficients, v) - _exact_window(beta, v))))

    assert error <= _WINDOW_FIT_TOLERANCE, f"beta={beta} fit is off by {error:.3e}"


def test_the_degree_grows_with_beta_rather_than_being_fixed() -> None:
    """A fixed degree would either waste work at small beta or miss the target at large beta.

    Asserted as a property rather than as specific degrees, so a better fitting method is free to
    change the numbers without failing this.
    """
    degrees = [len(kaiser_window_chebyshev(beta)) - 1 for beta in (1.0, 8.0, 16.0, 32.0)]

    assert degrees == sorted(degrees), f"degree is not monotone in beta: {degrees}"
    assert degrees[-1] > degrees[0], f"degree did not grow with beta at all: {degrees}"


def test_the_window_is_clamped_beyond_the_kernel_support() -> None:
    """Past the support the exact window is ``i0(0) / i0(beta)`` -- small, and not zero.

    The kernel clamps ``v`` at 1 to reproduce the exact form's ``maximum(0.0, 1 - v)``. Dropping the
    clamp would evaluate the polynomial outside the interval it was fitted on, where a Chebyshev
    series diverges quickly, and the edge taps carry that error into every output sample.
    """
    beta = float(DEFAULT_KAISER_BETA)
    coefficients = kaiser_window_chebyshev(beta)
    assert coefficients is not None

    at_edge = float(_clenshaw(coefficients, np.array([1.0]))[0])
    expected = 1.0 / float(np.i0(beta))

    assert at_edge == pytest.approx(expected, abs=_WINDOW_FIT_TOLERANCE)
    assert at_edge > 0.0, "the window vanished at the support edge, which drops the outer taps"


def test_an_unreachable_target_returns_none_rather_than_a_worse_window() -> None:
    """The caller must be able to keep ``i0``; silently accepting a worse window is the failure.

    Asked for an accuracy no polynomial can deliver in float64 -- the evaluation itself bottoms out
    near 2e-14 at this beta -- so this pins the refusal rather than a degree.
    """
    assert kaiser_window_chebyshev(float(DEFAULT_KAISER_BETA), tolerance=1e-18) is None


def test_the_resampled_output_matches_an_analytic_signal() -> None:
    """The check that matters: accuracy against something neither implementation computed.

    A band-limited sinusoid has a known value at any fractional offset, so this measures the kernel
    rather than comparing it with its own sibling. The bound is the kernel's own error at these
    settings -- 4.027e-12 of peak measured on device -- not the window's, because the window
    approximation contributes 2.2e-15 and is invisible here. That is the point: it must stay
    invisible.
    """
    pytest.importorskip("jax", reason="jax not installed")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gwmock_signal.projection.jax_projection import _interpolate_uniform_sinc

    n_samples = 1 << 14
    sampling_frequency = 2048.0
    frequency = 137.0
    times = np.arange(n_samples) / sampling_frequency
    samples = np.sin(2.0 * np.pi * frequency * times)

    half = (int(DEFAULT_SINC_TAPS) - 1) // 2
    index = np.random.default_rng(20260808).uniform(half + 1, n_samples - half - 2, 20000)
    truth = np.sin(2.0 * np.pi * frequency * (index / sampling_frequency))

    got = np.asarray(_interpolate_uniform_sinc(jnp.asarray(samples), jnp.asarray(index), n_samples))
    error = float(np.max(np.abs(got - truth)) / np.max(np.abs(truth)))

    assert error < 1e-10, f"resampled output is off by {error:.3e} of peak against the analytic value"


@pytest.mark.parametrize("beta", [16.046, 15.5, 17.25, 23.9, 31.4, 63.7, 128.3, 201.5, 260.0, 269.1])
def test_the_fit_holds_at_beta_between_the_documented_values(beta: float) -> None:
    """Off-grid beta, checked off-grid in v, because both were validated circularly before.

    A reviewer found the hole: at beta = 16.046 the fit's own grid saw 9.97e-14 and accepted, while an
    independent search found 1.03e-13 at v = 3.15e-07 -- reachable, since that v corresponds to
    x = 0.036 which `frac - offset` produces. 1,243 points exceeded the target, **all near v = 0**,
    where the original grid had no logarithmic clustering: it clustered at v -> 1, where the window is
    smallest, rather than where the polynomial struggles.

    Two things were circular and both are fixed: the fit is now accepted against a *different*,
    denser grid at half the tolerance, and this test samples beta values that are not the round ones
    the other test uses. The chosen beta values sit near the degree transitions, which is where the
    search is most likely to stop one step early.
    """
    coefficients = kaiser_window_chebyshev(beta)
    if coefficients is None:
        # A legal outcome, and **not portable**: CI returned `None` at beta = 269.1 where this
        # machine accepted a degree-64 fit at 0.567x the target. Acceptance inside the flicker
        # band above beta ~250 depends on float64 least-squares details, which differ with the
        # BLAS in use, so asserting *which* beta is served encodes one platform's arithmetic.
        # The contract is "None, or accurate"; the accuracy half is what this test is for.
        pytest.skip(f"beta={beta} falls back to i0 here, which the contract permits")

    v = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 40001),
                np.logspace(-14.0, -1.0, 2000),
                1.0 - np.logspace(-14.0, -1.0, 2000),
            ]
        ).clip(0.0, 1.0)
    )
    error = np.abs(_clenshaw(coefficients, v) - _exact_window(beta, v))
    worst = int(np.argmax(error))

    assert float(error[worst]) <= _WINDOW_FIT_TOLERANCE, (
        f"beta={beta} is off by {error[worst]:.3e} at v={v[worst]:.6e}, over the {_WINDOW_FIT_TOLERANCE:.0e} target"
    )


def test_an_int_and_a_float_beta_share_one_cache_entry() -> None:
    """``32`` and ``32.0`` are the same window; ``lru_cache`` disagrees unless the key is normalised.

    A reviewer found the split. The cost is small -- a duplicate fit and a wasted cache slot -- but it
    is the kind of thing that turns into "the fit runs on every call" once a caller happens to pass a
    numpy scalar, which has the same problem.
    """
    assert kaiser_window_chebyshev(32) is kaiser_window_chebyshev(32.0)
    assert kaiser_window_chebyshev(np.float64(16.0)) is kaiser_window_chebyshev(16.0)
    assert kaiser_window_chebyshev(-0.0) is kaiser_window_chebyshev(0.0)


def test_a_large_beta_may_fall_back_and_that_is_not_an_error() -> None:
    """Above about beta = 250 acceptance alternates, and a caller there gets ``i0``.

    Found in review, and it contradicts a claim of mine that ``None`` was only reachable for
    tolerances no caller would pass. Degree 64's error oscillates with beta instead of decreasing, so
    the accept/reject boundary is not monotone -- 262 and 266-270 fall back while 260, 264 and 272 are
    served. Such a beta needs roughly 1000 taps, so it is far from the default, but it is legal.

    Pinned as *behaviour*, not as a specific set: what must hold is that a fallback is a clean `None`
    the kernel can act on, not an exception and not a worse window.
    """
    # The invariant, not the split: for every beta the answer is either `None` or a fit meeting the
    # target. Which beta lands where is platform-dependent -- CI and this machine disagree at 269.1 --
    # so a test naming the served set passes in one place and fails in another.
    v = np.unique(np.concatenate([np.linspace(0.0, 1.0, 20001), np.logspace(-14.0, -1.0, 1500)]).clip(0.0, 1.0))
    for beta in (250.0, 260.0, 262.0, 264.0, 266.0, 268.0, 270.0, 272.0, 280.0):
        coefficients = kaiser_window_chebyshev(beta)
        if coefficients is None:
            continue
        error = float(np.max(np.abs(_clenshaw(coefficients, v) - _exact_window(beta, v))))
        assert error <= _WINDOW_FIT_TOLERANCE, f"beta={beta} was served a fit off by {error:.3e}"

    # And the default must still be served, or the fit has stopped working where it matters.
    assert kaiser_window_chebyshev(float(DEFAULT_KAISER_BETA)) is not None
