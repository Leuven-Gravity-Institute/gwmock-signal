# Plan: GPU-native data simulation via a JAX/ripple backend

Status: **draft / proposal** — no code written yet.

**Goal (clarified):** enable gravitational-wave _data simulation on GPU_ when
the [ripple](https://github.com/GW-JAX-Team/ripple) (JAX) waveform backend is
selected, while keeping LAL/PyCBC working on CPU. This is **not** merely "add
another waveform source" — the target is keeping the simulation on-device
end-to-end.

**Confirmed scope decisions (2026-06-18):**

- **Workload:** many injections — typically a **1-year catalogue**.
  Batched/`vmap`-ped on-device simulation (Phase 3) is therefore the real
  payoff, not single-event speed.
- **Sequencing:** build the **CPU ripple backend first** to lock in ripple
  compatibility, _then_ proceed to the GPU adaptation.
- **Delivery process:** this repo is **branch + PR** (recorded in the workspace
  git-push-policy). Every PR must be **narrow-scoped — change one thing at a
  time.** The phases below are therefore decomposed into individual PRs in §7.

---

## 1. The core problem: only the waveform stage is GPU-capable today

ripple produces JAX (`jnp`) **frequency-domain** arrays that live on the GPU.
But every downstream stage in gwmock-signal is CPU/host-bound:

| Stage                   | File                        | Implementation                                         | Device  |
| ----------------------- | --------------------------- | ------------------------------------------------------ | ------- |
| Waveform (ripple)       | _new_                       | JAX, frequency-domain                                  | **GPU** |
| Projection to detectors | `projection/network.py:195` | numpy + scipy `interp1d` + astropy GMST + LAL geometry | CPU     |
| Strain injection        | `injection/core.py:29`      | numpy + scipy `interp1d`, time-domain superposition    | CPU     |
| Stacking / output       | `multichannel/stack.py`     | GWpy `TimeSeries`, h5py                                | host    |
| Network SNR             | `snr/_network.py:23`        | numpy frequency-domain inner product                   | CPU     |

So a waveform generated on the GPU is immediately copied to host as soon as it
enters projection. **The bottleneck for GPU simulation is the pipeline, not the
waveform.**

### What "GPU-native" actually buys (and the workload that matters)

ripple's own paper notes that _serial_ single-waveform evaluation is only ~on
par with LAL's C code; the >10× speedups come from **batched, `vmap`-ped,
JIT-compiled** evaluation. For a mock-data generator the high-value capability
is therefore **simulating a whole population of injections in one batched
on-device call**, not one event at a time.

> **Open question for the user — please confirm:** is the workload "generate
> many injections (a catalog)" or "one event at a time"? If the latter, GPU
> offers little benefit and this whole effort's payoff is small. The rest of
> this plan assumes the batched-catalog workload.

This matters for the strategy: a CPU adapter that converts ripple → time-domain
→ GWpy `TimeSeries` (the v1 approach) **cannot** be `vmap`-ped or JIT-ed,
because GWpy `TimeSeries` and the numpy/scipy projection are not JAX-traceable.
It is a useful correctness oracle but a dead-end for the GPU goal — you cannot
"gradually" morph the numpy pipeline into a batched device pipeline; a parallel
JAX compute path is required at some point.

---

## 2. ripple API (confirmed from local source `/srv/weave/projects/software/ripple`)

Modern, class-based interface (`ripplegw.interfaces`):

```python
from ripplegw import IMRPhenomD          # or waveform_preset["IMRPhenomD"]
import jax.numpy as jnp

wf = IMRPhenomD(f_ref=20.0)
freqs = jnp.arange(f_min, f_max + df, df)
out = wf(freqs, {"M_c": ..., "eta": ..., "s1_z": ..., "s2_z": ...,
                 "d_L": ..., "phase_c": ..., "iota": ...})
hp, hc = out["p"], out["c"]              # jnp complex arrays, FREQUENCY domain
```

- Output keys are `"p"`/`"c"` (plus/cross), **frequency-domain `jnp` arrays** on
  the supplied `freqs` grid. (Exception: `SineGaussian` is time-domain.)
- `waveform_preset: dict[str, type[Waveform]]` maps model name → class — use
  this for `available_approximants()` instead of hardcoding.
- Each model exposes `.parameter_names` (ordered tuple) — drives parameter
  mapping.
- Package/import name is **`ripplegw`** (pip `rippleGW`, `rippleGW[cuda]` for
  GPU).

### Per-model parameter names (from `.parameter_names`)

| Family          | Models                                                                  | ripple param keys                                                  |
| --------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Aligned-spin    | `TaylorF2`, `IMRPhenomD`, `IMRPhenomHM`, `IMRPhenomXAS`, `IMRPhenomXHM` | `M_c, eta, s1_z, s2_z, d_L, phase_c, iota`                         |
| Aligned + tides | `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3`                        | …+ `lambda_1, lambda_2` (or `lambda_tilde, delta_lambda_tilde`)    |
| Precessing      | `IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomXPHM`                          | `M_c, eta, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z, d_L, phase_c, iota` |

### Mapping gwmock-pop canonical names → ripple keys

(gwmock canonical names come from `simulator.py:341` `_REQUIRED` and the
`_pop_alias` translations in the LAL/PyCBC backends.)

| gwmock canonical (alias)                           | ripple key               | Notes                                                                         |
| -------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| `detector_frame_mass_1` (`mass1`) + `_2` (`mass2`) | `M_c`, `eta`             | via `ripplegw.ms_to_Mc_eta` (component masses → chirp mass + symmetric ratio) |
| `luminosity_distance` (`distance`)                 | `d_L`                    | Mpc                                                                           |
| `spin_1z` (`spin1z`) / `spin_2z` (`spin2z`)        | `s1_z` / `s2_z`          |                                                                               |
| `spin_1x/1y`, `spin_2x/2y`                         | `s1_x/s1_y`, `s2_x/s2_y` | precessing models only; aligned models must validate ≈0                       |
| `inclination`                                      | `iota`                   |                                                                               |
| `coa_phase`                                        | `phase_c`                |                                                                               |
| `coa_time` (`tc`)                                  | via epoch, not ripple    | place coalescence in the output time grid                                     |
| `lambda_1` (`tidal_1`) / `lambda_2` (`tidal_2`)    | `lambda_1` / `lambda_2`  | NRTidal variants only                                                         |
| `minimum_frequency`                                | `f_ref` (default)        | or a separate `--f-ref`                                                       |

A per-model spec (param keys + constructor kwargs) should be derived from each
model's `.parameter_names` rather than one monolithic translator.

---

## 3. Two strategies (the decision you raised)

**Option A — redesign the pipeline first**, making every stage
backend/device-aware, then plug ripple in as the GPU path.

- _Pro:_ clean target architecture, no throwaway code.
- _Con:_ large up-front design with **no working integration to validate
  against** — you'd be designing the JAX projection/injection in the abstract,
  high risk of the wrong abstraction, and a long lead time before any GPU
  result.

**Option B — integrate ripple first** as a CPU time-domain backend (the original
PLAN v1), then incrementally JAX-ify the pipeline.

- _Pro:_ fast, ships a usable feature, produces a validated correctness oracle
  and test fixtures, settles packaging.
- _Con:_ delivers **zero GPU benefit** in itself; the CPU adapter is a stepping
  stone, and there's a real risk that "incremental" small steps on the numpy
  pipeline never actually reach a `vmap`-able device pipeline (each step doesn't
  move toward batching/JIT).

### Recommendation: a hybrid — oracle-first, but architect the seam up front

Neither pure option is ideal. Take the cheap, high-value parts of B as a
_correctness oracle_, decide the compute-backend seam (the valuable part of A)
**before** sinking effort into JAX projection, then build the JAX-native
frequency-domain path as the real deliverable. Rationale: the FD→TD CPU adapter
is genuinely useful as the golden-reference oracle that the GPU path is
validated against (it lets us reuse the battle-tested numpy projection physics
as ground truth), but we commit to the JAX FD path as the actual goal rather
than pretending the numpy pipeline can be morphed.

---

## 4. Recommended phased plan

### Phase 0 — ripple as a CPU/time-domain backend (the oracle) — _low risk, ships_

- New `RippleBackend(WaveformBackend)` in `waveform/backends/ripple.py`,
  lazy-importing `ripplegw` + `jax` (mirror `PyCBCBackend.__init__`,
  `backends/pycbc.py:30`).
- `available_approximants()` → `list(ripplegw.waveform_preset)` (minus models
  without a param mapping/test).
- `generate_td_waveform(...)`: map params (§2) → build FD grid → call ripple →
  **FD→TD conditioning** (window/taper, `irfft`, epoch + `tc` placement) →
  `jnp`→numpy → GWpy `TimeSeries`. Match LAL's epoch convention (`lal.py:97`).
- CLI: add `"ripple"` to `--backend` (`cli/inject.py:167`), wrap `ImportError`
  like PyCBC.
- Packaging: optional extra `jax = ["rippleGW>=<pin>"]`; verify `uv` lock
  resolves alongside lalsuite/gwpy; confirm Python 3.12/3.13 (pyproject line
  19).
- **Anchor test (mandatory):** ripple-vs-LAL `IMRPhenomD` time-domain overlap >
  0.99 and matching peak time, same params/`f_min`/`sampling_frequency`. This is
  the external reference proving the FD→TD conventions are right, not just
  plausible.
- **Outcome:** a usable ripple backend on CPU, a validated oracle, fixtures. _No
  GPU yet — state this explicitly so expectations are managed._

### Phase 1 — define the compute-backend seam — _design, validated by Phase 0_

- Decide the **internal representation**: frequency-domain,
  array-namespace-agnostic (functions take an array module / accept `np` or
  `jnp`). This mirrors the existing exact FD phase-shift path
  (`network.py:340`), which is already the right shape.
- Decide the **numpy↔jax / device boundary**: public API still returns GWpy
  `TimeSeries` on the host; only the _compute core_ (waveform → projection →
  injection → final `irfft`) runs in the chosen namespace, transferring to host
  once, at output.
- Specify the `simulate(...)` extension so a `CBCSimulator` can run either the
  numpy (CPU, current) or jax (GPU) core, selected by the backend.

### Phase 2 — JAX-native FD pipeline, single event, `earth_rotation=False` — _first GPU milestone_

- This path is **exact and fully JAX-traceable**: FD time-shift is a closed-form
  phase multiply (no interpolation), matching `network.py:342`.
- Reimplement in JAX: antenna-pattern `F+/F×` and geocenter time delay. **Single
  source of truth for detector geometry** — extract the LAL response tensor +
  location once (`_reconstructed_geometry`, `network.py:80`) into a static table
  consumed by both the numpy and jax paths, so the physics is not encoded twice.
- Need a JAX-friendly **GMST**; anchor it against astropy's value
  (`_gmst_accurate`, `network.py:70`) to ~µs.
- Keep everything on device: ripple FD `hp/hc` → `F+ hp + F× hc` with FD delay
  phase → `irfft` only at the very end → host transfer for output.
- **Cross-validation (guards duplicated physics):** require jax-path vs
  numpy-path overlap > 0.999 on a parameter grid.

### Phase 3 — batched / `vmap` simulation — _where the GPU win materializes_

- `vmap` (and `jit`) the Phase-2 core over a batch of source parameters →
  simulate a whole injection catalog in one device call.
- Benchmark vs the CPU loop; report speedup decomposed by stage (waveform vs
  projection vs FFT) and anchored against ripple's published batched timings.

### Phase 4 — `earth_rotation=True` on device — _hardest, possibly deferred_

- The current time-dependent antenna pattern uses TD cubic interpolation
  (`network.py:285`), which conflicts with a pure-FD on-device path. Options:
  segmented FD response, or a differentiable resampling. May stay CPU-only
  initially; document the limitation.

---

## 5. Risks & flags

1. **Single-waveform GPU is not faster than CPU** — the payoff is batched `vmap`
   (Phase 3). If the workload is one-event-at-a-time, reconsider scope (see §1
   open question).
2. **Duplicated physics** — antenna pattern, time delay, and GMST would exist in
   _both_ numpy (existing, trusted) and JAX (new). This is exactly the "same
   physics in two places" latent-inconsistency risk. Mitigation: numpy path is
   the reference oracle; cross-test every JAX result against it (Phase 2); one
   shared static source for detector-geometry constants.
3. **FD→TD conventions** (FFT normalization, cyclic roll, epoch sign) — the most
   bug-prone part; mitigated by the Phase-0 anchor test vs LAL.
4. **`earth_rotation=True`** has no clean FD/GPU form yet (Phase 4).
5. **Per-model `theta`/param packing** — precessing XP/XPHM differ; build from
   each model's `.parameter_names`, and confirm against the released
   `GW-JAX-Team/ripple` (the local checkout) rather than the upstream
   `tedwards2412` fork.
6. **`uv`/dependency coexistence** — jax/jaxlib alongside lalsuite/gwpy; current
   conflict rule (pyproject line 32) only blocks `pycbc` ⊻ `sgwb`. Verify the
   lock and the CI matrix; gate ripple tests behind the extra so the default job
   stays green.
7. **API instability** — ripple is pre-1.0 (README warns); pin the version.

---

## 6. Suggested first step

Build **PR 1 + PR 2 below for `IMRPhenomD` only** and get the anchor test
(overlap vs LAL > 0.99) green. That de-risks the parameter mapping, the FD→TD
conditioning, and the packaging, and produces the oracle the GPU path will be
validated against — without committing to the larger redesign before we've
proven the foundation.

---

## 7. PR decomposition (narrow-scoped, one change per PR)

This repo requires branch + PR, **one thing per PR**. Each PR below should be
independently reviewable, land green, and not bundle unrelated changes. Later
PRs are intentionally vague — re-plan them once the earlier ones land and inform
the design.

### Stage A — CPU ripple backend (Phase 0)

- **PR 1 — packaging only.** Add the optional `jax = ["rippleGW>=<pin>"]` extra
  to `pyproject.toml`, update `uv.lock`, confirm it resolves alongside
  lalsuite/gwpy on Python 3.12/3.13. No source code. (Pure dependency change —
  easy to review/revert.)
- **PR 2 — `RippleBackend` for `IMRPhenomD` only.** New
  `waveform/backends/ripple.py` with lazy import, param mapping (§2), FD→TD
  conditioning, and the **anchor test vs LAL** (overlap > 0.99). Register in
  `backends/__init__.py` + `waveform/__init__.py`. Single model keeps the diff
  small and the conditioning logic isolated.
- **PR 3 — CLI `--backend ripple`.** Wire the choice in `cli/inject.py:167` +
  help text + `ImportError` handling. Thin, user-facing only.
- **PR 4 — aligned-spin models.** Extend `available_approximants()` /param specs
  to `TaylorF2, IMRPhenomXAS, IMRPhenomHM, IMRPhenomXHM` (same param shape as
  D), one test each. (Could split further if review prefers.)
- **PR 5 — NRTidal variants.** `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3`
  (`lambda_1/lambda_2` routing).
- **PR 6 — precessing models.** `IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomXPHM`
  (6-component spin packing) — separate because the param mapping genuinely
  differs.
- **PR 7 — docs.** User-guide + `--backend` docs for ripple, model list, install
  extra, and the explicit "CPU only / no GPU yet" note.

_Stage A exit:_ ripple usable on CPU, validated oracle + fixtures in place.

### Stage B — compute-backend seam (Phase 1)

- **PR 8 — design doc / ADR** (docs only): the FD internal representation, the
  numpy↔jax boundary, and the `simulate(...)` extension contract. Review the
  _design_ before any code so Stage C PRs stay narrow.

### Stage C — JAX FD pipeline, `earth_rotation=False` (Phase 2) — re-plan after PR 8

Sketch (split further when we get there):

- **PR 9 —** extract detector geometry constants into one shared static table
  consumed by the existing numpy path (no behaviour change; pure refactor toward
  single source of truth).
- **PR 10 —** JAX GMST helper + test anchored against astropy.
- **PR 11 —** JAX antenna-pattern + FD time-delay, cross-validated vs numpy
  (>0.999).
- **PR 12 —** wire the JAX FD compute core into `simulate` for the ripple
  backend (single event), behind the seam from PR 8.

### Stage D — batching (Phase 3) & earth-rotation-on-device (Phase 4)

- **PR 13+ —** `vmap`/`jit` over a parameter batch (the 1-year-catalogue
  payoff) + benchmarks; then the harder `earth_rotation=True` on-device path.
  Re-plan after Stage C. </content>
