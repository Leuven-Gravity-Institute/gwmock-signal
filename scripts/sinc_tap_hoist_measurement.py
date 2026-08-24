"""Measure hoisting the sine out of the resampling tap loop, at the site itself.

The Kaiser-windowed sinc kernel evaluates ``sinc(x) = sin(pi*x)/(pi*x)`` once per tap, at
``x = frac - offset`` with ``frac`` in ``[0, 1)`` and ``offset`` running over the taps. Since
``sin(pi*(frac - offset)) = (-1)**offset * sin(pi*frac)`` exactly in real arithmetic, the sine
can be evaluated once per output sample instead of once per tap -- 127 transcendentals become
one, and every tap after that is a sign flip (exact) and a division.

This script measures what that substitution actually buys and costs *in situ*, because neither
quantity can be inferred from the per-tap arithmetic:

* **Speed** -- the tap loop also evaluates a window per tap. In the NumPy path that window is
  ``i0(beta*sqrt(1 - v))``, which is far more expensive than the sine, so the sine is a small
  share of the loop; in the device path the window is a Chebyshev polynomial, so the sine's
  share is larger. The saving is therefore a property of the path, not of the sine, and both
  paths are timed separately.
* **Accuracy** -- a resampled sample is a *normalised* weighted sum of 127 taps. The Kaiser
  taper suppresses exactly the large-``|x|`` taps where the direct sine is least accurate, and
  the normalisation cancels part of what is left, so the per-tap ulp differences do not carry
  over. Both forms are compared against an mpmath reference at 50 decimal digits that evaluates
  the same normalised sum with an exact sinc and an exact Kaiser window.

The hoist is applied with **symmetric** reduction -- ``n = round(frac)``, ``r = frac - n`` in
``[-1/2, 1/2]``, ``sin(pi*(frac - offset)) = (-1)**(n - offset) * sin(pi*r)`` -- rather than
reduction to ``[0, 1)``. Both cost one sine; the symmetric one avoids evaluating the sine next
to ``pi``, where rounding the product ``fl(pi)*frac`` costs the result its leading digits.

Run it with mpmath supplied on the fly, so it need not become a dependency::

    uv run --with mpmath python scripts/sinc_tap_hoist_measurement.py

The report is written to stdout as markdown. Every number it prints is stamped with the commit
that produced it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from mpmath import mp, mpf

from gwmock_signal.projection.resampling import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_TAPS,
    resample_uniform_sinc,
    validate_kernel,
)

#: Working precision of the reference. 50 decimal digits is ~166 bits, so the reference carries
#: ~113 bits beyond float64 and its own error is negligible against the ~1e-16 effects measured.
REFERENCE_DPS = 50

#: The kernel's own truncation error, from the module that defines it: 127 taps at beta = 32
#: reach 4.027e-12 of peak against an analytic sinusoid. Quoted here as the yardstick the
#: arithmetic difference has to be judged against -- a float64 difference far below this changes
#: nothing about the kernel's accuracy, whatever it looks like in ulps.
KERNEL_TRUNCATION_ERROR = 4.027e-12

#: Tone frequency for the test signal, as a fraction of Nyquist. Inside the kernel's usable band
#: (the tests put that at 0.8), so the measurement is taken where the kernel is meant to be used.
TONE_FRACTION_OF_NYQUIST = 0.5


def _producing_commit() -> str:
    """Return ``<sha>`` (or ``<sha>-dirty``) for the checkout this ran from.

    Every number in the report is stamped with this, so a figure can never be quoted without
    the code that produced it. A dirty tree is reported as such rather than silently attributed
    to the last commit.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - resolved from PATH deliberately
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return f"{sha}-dirty" if dirty else sha


def resample_direct(
    samples: np.ndarray,
    index: np.ndarray,
    *,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """The shipped NumPy tap loop, transcribed so the timing isolates the sine.

    Held identical to :func:`gwmock_signal.projection.resampling.resample_uniform_sinc` except
    that nothing else in the loop may differ from :func:`resample_hoisted`. Fidelity is not
    assumed: `check_transcription_is_faithful` requires this to reproduce the shipped function
    bit for bit before any timing is reported.
    """
    taps, beta = validate_kernel(taps, beta)
    samples = np.asarray(samples, dtype=float)
    index = np.asarray(index, dtype=float)
    n = samples.shape[0]
    half = (taps - 1) // 2

    base = np.floor(index)
    frac = index - base
    base_int = base.astype(np.int64)

    total = np.zeros(index.shape, dtype=float)
    weight_sum = np.zeros(index.shape, dtype=float)
    denominator = half + 1.0
    for offset in range(-half, half + 1):
        x = frac - offset
        window = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (x / denominator) ** 2))) / np.i0(beta)
        weight = np.sinc(x) * window
        gathered = samples[np.clip(base_int + offset, 0, n - 1)]
        total += weight * gathered
        weight_sum += weight

    interpolated = np.where(weight_sum != 0.0, total / weight_sum, 0.0)
    in_range = (index >= 0.0) & (index <= n - 1)
    return np.where(in_range, interpolated, 0.0)


def resample_hoisted(
    samples: np.ndarray,
    index: np.ndarray,
    *,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """:func:`resample_direct` with the sine hoisted out of the tap loop, symmetrically reduced."""
    taps, beta = validate_kernel(taps, beta)
    samples = np.asarray(samples, dtype=float)
    index = np.asarray(index, dtype=float)
    n = samples.shape[0]
    half = (taps - 1) // 2

    base = np.floor(index)
    frac = index - base
    base_int = base.astype(np.int64)

    # The one transcendental. `nearest` is 0 or 1, so `reduced` lands in [-1/2, 1/2] and the
    # subtraction is exact (Sterbenz for frac >= 1/2, trivially exact below).
    nearest = np.round(frac)
    reduced = frac - nearest
    sine = np.sin(np.pi * reduced)
    # (-1)**nearest, folded in once rather than per tap.
    sine = np.where(nearest == 0.0, sine, -sine)

    total = np.zeros(index.shape, dtype=float)
    weight_sum = np.zeros(index.shape, dtype=float)
    denominator = half + 1.0
    for offset in range(-half, half + 1):
        x = frac - offset
        window = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (x / denominator) ** 2))) / np.i0(beta)
        # sin(pi*x) = (-1)**offset * sin(pi*frac); the sign is exact in IEEE 754.
        numerator = sine if offset % 2 == 0 else -sine
        # x == 0 only when frac == 0 and offset == 0, where sinc is 1. Divided by a substituted
        # 1.0 rather than guarded after the fact, so the invalid division never happens.
        safe = np.where(x == 0.0, 1.0, x)
        sinc = np.where(x == 0.0, 1.0, numerator / (np.pi * safe))
        weight = sinc * window
        gathered = samples[np.clip(base_int + offset, 0, n - 1)]
        total += weight * gathered
        weight_sum += weight

    interpolated = np.where(weight_sum != 0.0, total / weight_sum, 0.0)
    in_range = (index >= 0.0) & (index <= n - 1)
    return np.where(in_range, interpolated, 0.0)


def reference_resample(
    samples: np.ndarray,
    index: np.ndarray,
    *,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
    exact_window: bool = True,
) -> list[mpf]:
    """The same normalised tap sum at ``REFERENCE_DPS`` digits, with an exact sinc and window.

    The float64 samples are exact inputs, so the only approximation left is the arithmetic --
    which is what the comparison is about.

    Args:
        samples: The series being resampled.
        index: Fractional positions to evaluate at.
        taps: Kernel taps.
        beta: Kaiser window shape parameter.
        exact_window: With ``True`` the Kaiser window is evaluated exactly, so a float64 form's
            error against this reference includes its own window error -- which is the same code
            in both forms, and large enough here to hide the sine underneath it. With ``False``
            the window is taken from the float64 loop and treated as exact, which leaves the sinc
            and the summation as the only approximations and so isolates the effect under test.
            Both are reported, because the first is the honest end-to-end error and the second is
            the honest attribution.

    Returns:
        One high-precision value per position.
    """
    taps, beta = validate_kernel(taps, beta)
    half = (taps - 1) // 2
    denominator = half + 1
    n = samples.shape[0]

    out: list[mpf] = []
    with mp.workdps(REFERENCE_DPS):
        beta_hp = mpf(beta)
        normalisation = mp.besseli(0, beta_hp)
        for position in index:
            base = int(np.floor(position))
            frac = mpf(float(position)) - base  # exact: float64 minus its own floor
            total = mpf(0)
            weight_sum = mpf(0)
            for offset in range(-half, half + 1):
                x = frac - offset
                if exact_window:
                    argument = beta_hp * mp.sqrt(max(mpf(0), 1 - (x / denominator) ** 2))
                    window = mp.besseli(0, argument) / normalisation
                else:
                    # Bit-identical to what the float64 loops compute, so the window term
                    # contributes zero difference and the comparison sees only the sinc.
                    x_f64 = np.float64(float(position)) - np.floor(np.float64(float(position))) - offset
                    window = mpf(
                        float(
                            np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (x_f64 / float(denominator)) ** 2)))
                            / np.i0(beta)
                        )
                    )
                weight = mp.sincpi(x) * window
                gathered = mpf(float(samples[min(max(base + offset, 0), n - 1)]))
                total += weight * gathered
                weight_sum += weight
            out.append(total / weight_sum)
    return out


def check_transcription_is_faithful(samples: np.ndarray, index: np.ndarray) -> str:
    """Require one of the two transcriptions to reproduce the shipped kernel bit for bit.

    Without this the comparison would be measuring a transcription that had drifted from the
    shipped code, and the difference would be attributed to the hoist. Which of the two matches
    also says, without the reader having to open the source, which form is currently shipped --
    so the same script is valid on both sides of the change.

    Args:
        samples: A series to resample.
        index: Positions to resample at.

    Returns:
        ``"direct"`` or ``"hoisted"``, whichever the shipped kernel reproduces exactly.

    Raises:
        RuntimeError: If the shipped kernel matches neither transcription.
    """
    shipped = resample_uniform_sinc(samples, index)
    candidates = {"direct": resample_direct(samples, index), "hoisted": resample_hoisted(samples, index)}
    for name, values in candidates.items():
        if np.array_equal(shipped, values):
            print(
                f"Transcription check: the shipped `resample_uniform_sinc` reproduces the "
                f"transcribed **{name}** loop bit for bit at all {shipped.size} positions, so the "
                "two transcriptions differ from each other only by the hoist."
            )
            return name
    worst = {name: float(np.max(np.abs(shipped - values))) for name, values in candidates.items()}
    raise RuntimeError(
        "the shipped resample_uniform_sinc matches neither transcription bit for bit "
        f"(max |difference|: {worst}); the comparison below would not be measuring the hoist alone"
    )


def _band_limited_samples(count: int, rng: np.random.Generator) -> np.ndarray:
    """A test signal inside the kernel's usable band, so the kernel is used as intended."""
    frequency = 0.5 * TONE_FRACTION_OF_NYQUIST  # cycles per sample
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.cos(2.0 * np.pi * frequency * np.arange(count, dtype=float) + phase)


def _best_of(function: Callable[[], np.ndarray], repeats: int) -> float:
    """Fastest wall-clock time over ``repeats`` runs, in seconds.

    The minimum, not the mean: the fastest run is the one least perturbed by whatever else the
    machine was doing, and this is a shared host.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def report_numpy_speed(rng: np.random.Generator, repeats: int) -> None:
    """Time the NumPy tap loop with and without the hoist, across problem sizes."""
    print("\n### NumPy path\n")
    print(f"Fastest of {repeats} runs per cell; {DEFAULT_SINC_TAPS} taps, beta = {DEFAULT_KAISER_BETA}.\n")
    print("| output samples | direct (s) | hoisted (s) | speedup | shipped (s) | max \\|difference\\| |")
    print("| --- | --- | --- | --- | --- | --- |")
    for exponent in (14, 16, 18):
        count = 2**exponent
        samples = _band_limited_samples(count + 2 * DEFAULT_SINC_TAPS, rng)
        index = np.arange(count, dtype=float) + DEFAULT_SINC_TAPS - 0.37
        direct_time = _best_of(lambda s=samples, i=index: resample_direct(s, i), repeats)
        hoisted_time = _best_of(lambda s=samples, i=index: resample_hoisted(s, i), repeats)
        shipped_time = _best_of(lambda s=samples, i=index: resample_uniform_sinc(s, i), repeats)
        difference = float(np.max(np.abs(resample_direct(samples, index) - resample_hoisted(samples, index))))
        print(
            f"| {count} | {direct_time:.4f} | {hoisted_time:.4f} | "
            f"**{direct_time / hoisted_time:.3f}x** | {shipped_time:.4f} | {difference:.3e} |"
        )
    print(
        "\nThe `shipped` column is the kernel as it stands in this checkout, timed the same way: it "
        "should track whichever transcription the check above matched."
    )


def _build_device_kernel(jax_module: Any, jnp: Any, coefficients_host: tuple[float, ...]) -> Callable[..., Any]:
    """Return the shipped device tap loop, parameterised by whether the sine is hoisted.

    One body, two variants selected by a static flag, so the loop is structurally identical
    apart from the sine -- Chebyshev window included. Substituting the exact ``i0`` window into
    one side would hide the sine's saving behind a far larger cost.

    Args:
        jax_module: The imported ``jax`` module (imported by the caller, since it is optional).
        jnp: The imported ``jax.numpy`` module.
        coefficients_host: Chebyshev coefficients of the normalised Kaiser window.

    Returns:
        ``kernel(samples, index, n_samples, hoist)``, ready to be ``jax.jit``-ed.
    """
    half = (DEFAULT_SINC_TAPS - 1) // 2
    denominator = half + 1.0

    def kernel(samples: Any, index: Any, n_samples: int, hoist: bool) -> Any:
        samples = jnp.asarray(samples, dtype=jnp.float64)
        index = jnp.asarray(index, dtype=jnp.float64)
        # No 1/i0(beta) factor here: the Chebyshev fit approximates the already-normalised
        # window, which is why the shipped code divides by i0(beta) only on its exact-i0 branch.
        coefficients = jnp.asarray(coefficients_host, dtype=jnp.float64)

        def window(x: Any) -> Any:
            v = jnp.minimum(1.0, (x / denominator) ** 2)
            t = 2.0 * v - 1.0
            b_kp1 = jnp.zeros_like(t)
            b_kp2 = jnp.zeros_like(t)
            for coefficient in coefficients[:0:-1]:
                b_kp1, b_kp2 = 2.0 * t * b_kp1 - b_kp2 + coefficient, b_kp1
            return t * b_kp1 - b_kp2 + coefficients[0]

        base = jnp.floor(index)
        frac = index - base
        base_int = base.astype(jnp.int32)
        nearest = jnp.round(frac)
        sine = jnp.where(nearest == 0.0, 1.0, -1.0) * jnp.sin(jnp.pi * (frac - nearest))

        def accumulate(step: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            total, weight_sum = carry
            offset = step - half
            x = frac - offset
            if hoist:
                numerator = jnp.where(offset % 2 == 0, sine, -sine)
                safe = jnp.where(x == 0.0, 1.0, x)
                sinc = jnp.where(x == 0.0, 1.0, numerator / (jnp.pi * safe))
            else:
                sinc = jnp.sinc(x)
            weight = sinc * window(x)
            gathered = samples[jnp.clip(base_int + offset, 0, n_samples - 1)]
            return total + weight * gathered, weight_sum + weight

        zeros = jnp.zeros_like(index)
        total, weight_sum = jax_module.lax.fori_loop(0, DEFAULT_SINC_TAPS, accumulate, (zeros, zeros))
        interpolated = jnp.where(weight_sum != 0.0, total / weight_sum, 0.0)
        in_range = (index >= 0.0) & (index <= n_samples - 1)
        return jnp.where(in_range, interpolated, 0.0)

    return kernel


def report_device_speed(rng: np.random.Generator, repeats: int) -> None:
    """Time the device tap loop with and without the hoist, on whatever backend JAX finds.

    The two variants come from ONE transcribed body selected by a static flag, so the loop is
    structurally identical apart from the sine -- including the Chebyshev window the shipped
    device path uses, which would otherwise make the comparison meaningless: substituting the
    exact ``i0`` window into one side would hide the sine's saving behind a far larger cost.
    """
    print("\n### Device path\n")
    try:
        import jax  # noqa: PLC0415 - optional dependency, absent in a minimal install
        import jax.numpy as jnp  # noqa: PLC0415
    except ImportError:
        print("JAX is not installed in this environment, so the device path was not timed.")
        return

    jax.config.update("jax_enable_x64", True)

    from gwmock_signal.projection.jax_projection import _interpolate_uniform_sinc  # noqa: PLC0415
    from gwmock_signal.projection.resampling import kaiser_window_chebyshev  # noqa: PLC0415

    backend = jax.default_backend()
    print(
        f"JAX {jax.__version__} on the **{backend}** backend. Compilation is excluded (each variant is "
        f"run once to compile, then timed as the fastest of {repeats} runs).\n"
    )
    if backend == "cpu":
        print(
            "> This host has no NVIDIA GPU, so this is the CPU backend: it shows whether the hoist\n"
            "> removes work, not what a GPU would do with it. The GPU figure is **not measured here**.\n"
        )

    coefficients_host = kaiser_window_chebyshev(DEFAULT_KAISER_BETA)
    if coefficients_host is None:  # pragma: no cover - the shipped default fits
        print("The Chebyshev window fit did not converge at this beta, so the device timing is skipped.")
        return

    transcribed = _build_device_kernel(jax, jnp, coefficients_host)

    shipped_jit = jax.jit(_interpolate_uniform_sinc, static_argnums=(2,))
    direct_jit = jax.jit(transcribed, static_argnums=(2, 3))
    hoisted_jit = jax.jit(transcribed, static_argnums=(2, 3))

    print("| output samples | direct (s) | hoisted (s) | speedup | max \\|difference\\| | shipped matches |")
    print("| --- | --- | --- | --- | --- | --- |")
    for exponent in (14, 16, 18):
        count = 2**exponent
        padded = count + 2 * DEFAULT_SINC_TAPS
        samples = jnp.asarray(_band_limited_samples(padded, rng))
        index = jnp.asarray(np.arange(count, dtype=float) + DEFAULT_SINC_TAPS - 0.37)
        shipped_value = np.asarray(jax.block_until_ready(shipped_jit(samples, index, padded)))
        direct_value = np.asarray(jax.block_until_ready(direct_jit(samples, index, padded, False)))
        hoisted_value = np.asarray(jax.block_until_ready(hoisted_jit(samples, index, padded, True)))
        direct_time = _best_of(
            lambda s=samples, i=index, n=padded: jax.block_until_ready(direct_jit(s, i, n, False)), repeats
        )
        hoisted_time = _best_of(
            lambda s=samples, i=index, n=padded: jax.block_until_ready(hoisted_jit(s, i, n, True)), repeats
        )
        difference = float(np.max(np.abs(direct_value - hoisted_value)))
        if np.array_equal(shipped_value, hoisted_value):
            matches = "`hoisted`"
        elif np.array_equal(shipped_value, direct_value):
            matches = "`direct`"

        else:
            matches = "**neither**"
        print(
            f"| {count} | {direct_time:.4f} | {hoisted_time:.4f} | "
            f"**{direct_time / hoisted_time:.3f}x** | {difference:.3e} | {matches} |"
        )
    print(
        "\nThe last column is which transcription the shipped device kernel reproduces bit for bit. "
        "`neither` would mean the transcriptions had drifted and the speedup no longer isolated the "
        "sine; it is the same check the NumPy path makes above."
    )


def _fmt(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.3g}"


def _accuracy_table(direct: np.ndarray, hoisted: np.ndarray, reference: list[mpf]) -> None:
    """Print one comparison of both float64 forms against one high-precision reference."""
    with mp.workdps(REFERENCE_DPS):
        err_direct = np.array([float(abs(mpf(float(v)) - r)) for v, r in zip(direct, reference, strict=True)])
        err_hoisted = np.array([float(abs(mpf(float(v)) - r)) for v, r in zip(hoisted, reference, strict=True)])
        reference_f64 = np.array([float(r) for r in reference])

    ulp = np.spacing(np.abs(reference_f64))
    usable = ulp > 0.0
    count = err_direct.size

    print("\n| metric | direct | hoisted |")
    print("| --- | --- | --- |")
    print(f"| max absolute error | {_fmt(float(np.max(err_direct)))} | {_fmt(float(np.max(err_hoisted)))} |")
    print(
        f"| RMS absolute error | {_fmt(float(np.sqrt(np.mean(err_direct**2))))} "
        f"| {_fmt(float(np.sqrt(np.mean(err_hoisted**2))))} |"
    )
    print(f"| median absolute error | {_fmt(float(np.median(err_direct)))} | {_fmt(float(np.median(err_hoisted)))} |")
    print(
        f"| max error, ulps of output | {_fmt(float(np.max(err_direct[usable] / ulp[usable])))} "
        f"| {_fmt(float(np.max(err_hoisted[usable] / ulp[usable])))} |"
    )
    print(
        f"| median error, ulps of output | {_fmt(float(np.median(err_direct[usable] / ulp[usable])))} "
        f"| {_fmt(float(np.median(err_hoisted[usable] / ulp[usable])))} |"
    )
    closer_hoisted = int(np.sum(err_hoisted < err_direct))
    closer_direct = int(np.sum(err_direct < err_hoisted))
    print(
        f"\nCloser to the reference: hoisted at {closer_hoisted} positions "
        f"({100.0 * closer_hoisted / count:.1f}%), direct at {closer_direct} "
        f"({100.0 * closer_direct / count:.1f}%), identical at {count - closer_hoisted - closer_direct}."
    )


def report_end_to_end_accuracy(rng: np.random.Generator, positions: int) -> None:
    """Compare both forms against high-precision evaluations of the same tap sum.

    Two references are used, because they answer different questions: one with the Kaiser window
    evaluated exactly (the honest end-to-end error of each form, window arithmetic included), and
    one with the window taken from the float64 loop (which removes a term common to both forms
    and so attributes the remaining difference to the sine).

    Args:
        rng: Source of the test signal and positions.
        positions: How many positions to evaluate.
    """
    print("\n## End-to-end accuracy of the resampled output\n")
    count = 4096
    samples = _band_limited_samples(count, rng)

    # Interior positions only: the kernel clamps taps at the ends, which is a property of the
    # boundary handling rather than of the sine, and would swamp the comparison.
    whole = rng.integers(DEFAULT_SINC_TAPS, count - DEFAULT_SINC_TAPS, size=positions).astype(float)
    # Fractions spanning the interval, with both branch boundaries of the reduction included
    # exactly rather than approached by luck.
    fractions = np.concatenate(
        [
            rng.uniform(0.0, 1.0, size=max(0, positions - 8)),
            np.array([0.0, np.nextafter(0.0, 1.0), 0.5, np.nextafter(0.5, 0.0), 0.25, 0.75]),
            np.array([np.nextafter(1.0, 0.0), 1.0 - 2.0**-20]),
        ]
    )[:positions]
    index = whole + fractions

    direct = resample_direct(samples, index)
    hoisted = resample_hoisted(samples, index)

    print(
        f"{positions} positions inside a {count}-sample band-limited series at "
        f"{TONE_FRACTION_OF_NYQUIST:g} x Nyquist, unit amplitude; reference at {REFERENCE_DPS} digits.\n"
    )

    print("### Against an exact reference (end-to-end error, window arithmetic included)")
    _accuracy_table(direct, hoisted, reference_resample(samples, index))

    print("\n### Against a reference sharing the float64 window (isolates the sine)")
    _accuracy_table(direct, hoisted, reference_resample(samples, index, exact_window=False))

    difference = np.abs(direct - hoisted)
    scale = np.spacing(np.abs(direct))
    usable = scale > 0.0
    in_ulps = difference[usable] / scale[usable]
    print(
        f"\nDisagreement between the two forms: max {_fmt(float(np.max(difference)))} absolute "
        f"({_fmt(float(np.max(in_ulps)))} ulps of output), median {_fmt(float(np.median(in_ulps)))} ulps, "
        f"RMS {_fmt(float(np.sqrt(np.mean(difference**2))))} absolute. Bit-identical at "
        f"{int(np.sum(difference == 0.0))} of {positions} positions."
    )
    worst = float(max(np.max(difference), 0.0))
    print(
        f"\n**Against the kernel's own error floor.** The largest arithmetic effect above is "
        f"{_fmt(worst)} on a unit-amplitude signal, while this kernel's truncation error at these "
        f"settings is {KERNEL_TRUNCATION_ERROR:.3e} of peak -- larger by a factor of "
        f"{KERNEL_TRUNCATION_ERROR / worst:.3g}. Neither form's arithmetic is what limits the "
        "kernel's accuracy, so the choice between them cannot be justified on output accuracy."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the measurement and write the report to stdout.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260824, help="seed for the signal and positions")
    parser.add_argument("--repeats", type=int, default=5, help="timing runs per cell; the fastest is reported")
    parser.add_argument("--positions", type=int, default=400, help="positions in the accuracy comparison")
    parser.add_argument("--skip-speed", action="store_true", help="skip the timings")
    parser.add_argument("--skip-accuracy", action="store_true", help="skip the high-precision comparison")
    arguments = parser.parse_args(argv)

    mp.dps = REFERENCE_DPS
    # A generator per section, derived from the one seed: sharing one generator made the accuracy
    # figures depend on whether --skip-speed had consumed draws first, so the same seed reported
    # different numbers for the same code.
    speed_rng = np.random.default_rng(arguments.seed)
    accuracy_rng = np.random.default_rng(arguments.seed + 1)

    print("# Hoisting the sine out of the resampling tap loop, measured in situ\n")
    print(
        f"Produced by commit **{_producing_commit()}**; numpy {np.__version__}, "
        f"Python {sys.version.split()[0]}, seed {arguments.seed}.\n"
    )

    check_transcription_is_faithful(
        _band_limited_samples(4096, np.random.default_rng(arguments.seed)),
        np.arange(200, 3000, 7, dtype=float) + 0.31,
    )

    if not arguments.skip_speed:
        print("\n## Speed\n")
        report_numpy_speed(speed_rng, arguments.repeats)
        report_device_speed(speed_rng, arguments.repeats)
    if not arguments.skip_accuracy:
        report_end_to_end_accuracy(accuracy_rng, arguments.positions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
