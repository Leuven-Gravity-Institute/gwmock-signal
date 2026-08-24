# Hoisting the sine out of the resampling tap loop

The Kaiser-windowed sinc kernel evaluated `sinc(x)` once per tap, at
`x = frac - offset` for each of its 127 integer offsets. Because

```text
sin(pi*(frac - offset)) = (-1)**offset * sin(pi*frac)
```

is an identity in real arithmetic, all 127 transcendentals follow from one: each
tap keeps a sign flip, exact in IEEE 754, and the division `sinc` already
performed.

This document records what that actually bought, measured at the site rather
than inferred from the arithmetic. **Every number below was produced by commit
`eaf428d4d0e809a8ad39a048b937cb4e954b8169`** via
`scripts/sinc_tap_hoist_measurement.py` (numpy 2.3.5, CPython 3.14.7, x86-64
glibc, two threads, 127 taps, beta = 32).

## Bottom line

**Speed is the justification; accuracy is not.**

|                                          | direct (one sine per tap) | hoisted (one per position) |
| ---------------------------------------- | ------------------------- | -------------------------- |
| NumPy path, 2^18 samples                 | 1.40 s                    | 1.23 s (**1.14x**)         |
| device path, 2^18 samples                | 0.138 s                   | 0.039 s (**3.51x**)        |
| RMS output error vs a 50-digit reference | 6.33e-16                  | 6.01e-16                   |
| closer to that reference                 | 32.0% of positions        | 49.0% of positions         |

The two paths differ by that much because of the **window, not the sine**: the
NumPy path still evaluates `i0(beta*sqrt(1 - v))` per tap and spends most of its
time there, so removing the sine moves a small share of the total; the device
path's window is a Chebyshev polynomial, which had left the sine as the loop's
dominant cost.

On accuracy the honest answer is **"unchanged, with a slight tilt in the hoist's
favour"** — not the decisive win the per-tap arithmetic suggests. A resampled
sample is a _normalised_ sum of 127 taps, the Kaiser taper suppresses exactly
the large-`|x|` taps where the direct sine is least accurate, and the
normalisation cancels part of what survives. Both forms land about **4500x below
this kernel's own truncation error** (4.027e-12 of peak), so neither is what
limits it.

The change is therefore worth making for the device path's 3.5x, and the
accuracy argument should not be leaned on in either direction.

## Speed

Fastest of 5 runs per cell. Run-to-run spread across three repetitions of the
whole measurement was about ±0.1x, so these are quoted to two figures, not
three.

### NumPy path

| output samples | direct (s) | hoisted (s) | speedup   | shipped (s) |
| -------------- | ---------- | ----------- | --------- | ----------- |
| 16384          | 0.0648     | 0.0496      | **1.31x** | 0.0479      |
| 65536          | 0.2920     | 0.2290      | **1.28x** | 0.2280      |
| 262144         | 1.4039     | 1.2278      | **1.14x** | 1.1746      |

Across three repetitions the range was 1.14x–1.33x, decreasing with size as the
per-tap `i0` comes to dominate. The `shipped` column is the kernel as committed,
timed identically; it tracks the `hoisted` transcription, which is the
cross-check that the timing compares what it claims to.

### Device path

| output samples | direct (s) | hoisted (s) | speedup   | shipped matches |
| -------------- | ---------- | ----------- | --------- | --------------- |
| 16384          | 0.0237     | 0.0045      | **5.30x** | `hoisted`       |
| 65536          | 0.0365     | 0.0111      | **3.27x** | `hoisted`       |
| 262144         | 0.1378     | 0.0393      | **3.51x** | `hoisted`       |

JAX 0.11.0, **CPU backend** — this host has no NVIDIA GPU. Compilation is
excluded. The 5.3x at the smallest size is partly fixed overhead; 3.3x–3.5x at
the larger sizes is the figure to believe.

### How the comparison is kept honest

Both variants come from **one transcribed body** per backend, selected by a
flag, so the loops are structurally identical apart from the sine — Chebyshev
window included. Substituting the exact `i0` window into one side would have
hidden the sine's saving behind a far larger cost. The script then requires the
shipped kernel to reproduce one of its two transcriptions **bit for bit** before
reporting any timing, and names which; `neither` would mean a transcription had
drifted and the speedup no longer isolated the sine.

## Accuracy of the resampled output

400 positions inside a 4096-sample band-limited series at 0.5 x Nyquist, unit
amplitude, interior only (the ends clamp taps, which is boundary handling rather
than arithmetic). The reference evaluates the same normalised tap sum at 50
decimal digits.

### Against an exact reference — the end-to-end error of each form

| metric                       | direct   | hoisted  |
| ---------------------------- | -------- | -------- |
| max absolute error           | 2.31e-15 | 2.08e-15 |
| RMS absolute error           | 6.33e-16 | 6.01e-16 |
| median absolute error        | 4.00e-16 | 3.33e-16 |
| max error, ulps of output    | 57.4     | 38.8     |
| median error, ulps of output | 5.03     | 4.70     |

Closer to the reference: **hoisted at 196 positions (49.0%), direct at 128
(32.0%)**, identical at 76 (19.0%).

### Against a reference sharing the float64 window — isolates the sine

Each form's error above includes its own Kaiser-window arithmetic, which is the
same code in both and large enough to sit on top of the sine. Taking the window
from the float64 loop and treating it as exact removes that common term:

| metric                       | direct   | hoisted  |
| ---------------------------- | -------- | -------- |
| max absolute error           | 1.50e-15 | 1.19e-15 |
| RMS absolute error           | 3.94e-16 | 3.71e-16 |
| median absolute error        | 2.10e-16 | 2.00e-16 |
| max error, ulps of output    | 46.9     | 10.8     |
| median error, ulps of output | 2.96     | 2.55     |

Closer to the reference: hoisted at 182 positions (45.5%), direct at 142
(35.5%), identical at 76.

So the hoist is better by 6% in RMS and 5% in the median once the window is
factored out, and better at roughly half the positions against a third — a tilt,
not a separation. Read this as "accuracy is not harmed", which is all the change
needs.

### The two forms against each other

Max **8.88e-16** absolute (46 ulps of output), median **2 ulps**, RMS 2.33e-16,
bit-identical at 76 of 400 positions. This is the quantity a bit-comparability
bar would look at, and it is the end-to-end counterpart of the much larger
per-tap ulp differences: the tap loop's own structure absorbs most of them.

### Against the kernel's own error floor

The largest arithmetic effect anywhere above is 8.88e-16 on a unit-amplitude
signal. This kernel's truncation error at 127 taps and beta = 32 is **4.027e-12
of peak** — larger by a factor of 4.5e3. Changing the arithmetic here cannot
move the kernel's accuracy; only taps and beta can.

## Why the sine is reduced to `[-1/2, 1/2]`

`nearest = round(frac)`, `r = frac - nearest`, and
`sin(pi*(frac - offset)) = (-1)**(nearest - offset) * sin(pi*r)`. The
alternative — leaving the argument in `[0, 1)` — costs exactly the same one
sine, but `frac` near 1 puts that argument next to pi, where rounding the
product `fl(pi)*frac` costs the result its leading digits. A separate sweep of
the sine alone, over 45837 inputs against a 50-digit reference, put the
symmetric form within **1.53 ulps everywhere** while the `[0, 1)` form reached
**1.3e3 ulps** at that boundary. The difference does not survive into the output
at this kernel's settings — see above — but it costs nothing to avoid, and it
would matter to any caller that used the tap weights unnormalised.

One consequence worth stating because it is easy to misread in the source: the
per-position `(-1)**nearest` factor is **not load-bearing**. It negates the tap
sum and the weight sum alike, so it cancels exactly in their quotient — verified
bit-identical with and without it. It is kept so that each weight is the kernel
weight rather than its negation; the per-tap `(-1)**offset` parity is the part
that matters, and dropping that was confirmed to break the tests by O(1).

## What is _not_ measured here

- **GPU.** The device figures are the CPU backend. What the hoist does to a real
  GPU kernel — where the transcendental/bandwidth balance is different — is
  unmeasured, and 3.5x should not be assumed to carry over.
- **float32.** Everything here is float64. Nothing in this document speaks to a
  reduced-precision path.
- **Production shapes.** The largest size timed is 2^18 output samples; full
  segments are larger, and the NumPy path's speedup was still falling with size
  at that point.
- **One signal.** A single tone at 0.5 x Nyquist, one amplitude. The accuracy
  figures are properties of that signal as well as of the kernel.
- **One libm.** x86-64 glibc via numpy 2.3.5. The mechanism behind the sine's
  error is argument-side and should carry to other libms, but the sub-ulp
  figures will not be identical.

## Reproducing it

```console
uv run --extra jax --with mpmath python scripts/sinc_tap_hoist_measurement.py
```

`mpmath` is supplied on the fly and is deliberately not a project dependency.
The run takes about 45 s. `--seed` fixes the signal and the positions (the speed
and accuracy sections draw from separate generators, so a section's numbers do
not depend on which other sections ran); `--skip-speed` and `--skip-accuracy`
select the halves.
