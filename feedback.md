# Documentation Feedback Report

## Phase 1: Documentation Review (No Source Code)

### Overall Assessment

The documentation for `gwmock-signal` is well-structured, thorough, and
professional. The README provides a strong overview, and the user guide follows
a logical pipeline (waveforms → projection → injection → multichannel) that
mirrors how a user would actually use the library. The separation between
narrative examples (user guide) and authoritative API reference (auto-generated
from docstrings) is a smart design choice.

However, several important features and workflows are undocumented or
under-documented, which would leave new users confused or unaware of
functionality available to them.

---

### Completeness

#### What is Covered Well

- **Installation**: Comprehensive, covering both PyPI and source installs with
  `uv`, including development dependencies and the `[pycbc]` optional extra.
- **CLI reference**: Well-documented with parameter tables, examples, and
  network resolution logic.
- **Core pipeline**: Waveform generation, detector projection, strain injection,
  and multichannel stacking all have dedicated user guide pages with narrative
  examples.
- **Custom backends**: Good coverage of the stable `GWSimulator.simulate`
  contract for downstream packages.
- **Troubleshooting**: The `docs/dev/troubleshooting.md` is thorough for
  contributor/developer scenarios.

#### What is Missing or Under-Documented

1. **`CBCSimulator.write()` method is undocumented**: This is a high-level
   convenience method that simulates a CBC injection, writes the result to disk
   (HDF5/GWF/NPY/TXT), and auto-generates a JSON sidecar with the injection
   parameters. This is likely a common user workflow, yet it is completely
   absent from the user guide. Users who want to save results to disk must
   discover this in the API reference or source code.

2. **`DetectorStrainStack.write()` / `read()` not shown in user guide**: The
   multichannel strains page (multi-channel-strains.md) shows only basic
   stacking operations. It never demonstrates how to save a stack to disk (HDF5,
   NPY, GWF, or TXT format) or read one back. A user who wants to persist
   results has no guidance.

3. **`DetectorStrainStack` properties undocumented in user guide**: Properties
   like `t0`, `sample_rate`, and `detector_names` are available but never shown
   in any user guide example. A user inspecting a stack for the first time has
   to go directly to the API reference.

4. **Network class programmatic usage not in user guide**: The CLI page
   describes `--network` resolution, but there is **no user guide page showing
   how to use the `Network` class programmatically**. Methods like
   `Network.from_file()`, `Network.from_detectors()`, `Network.from_name()`,
   `Network.list_names()`, and `Network.list_lal_detectors()` are all
   undocumented in the narrative user guide. There is a brief mention of
   `Network.from_file` in the CLI page, but no standalone example showing the
   YAML/JSON network file format users need to write.

5. **No reference or documentation for network file format (YAML/JSON)**: Users
   who want to define a custom detector network in a file must reverse-engineer
   the format from the source code's `Network.from_file` docstring. There should
   be a user guide page (or section) showing a complete example YAML/JSON
   network definition, including both simple LAL code entries and full custom
   detector geometries.

6. **`CustomDetector` has no dedicated user guide example**: The class is
   mentioned briefly in "Example 3" of the detector projection page with a vague
   suggestion ("pass `gwmock_signal.detector.CustomDetector` instances"), but
   there is no complete working example showing how to construct a
   `CustomDetector` with specific coordinates, arm orientations, and elevation.
   The nine required fields (`latitude_rad`, `longitude_rad`, `elevation_m`,
   `xarm_azimuth_rad`, `yarm_azimuth_rad`, etc.) are not walked through.

7. **`TransientSimulator.register_waveform_model()` is undocumented**: The user
   guide's custom backends page (custom-backends.md) describes `GWSimulator`
   subclasses and the stable contract, but it never mentions the
   `register_waveform_model` method on `TransientSimulator`. This is a key
   extensibility point for users adding custom waveform generators (NR, ROM,
   burst models) to a CBC-style simulator instance.

8. **`inject_cbc_signal()` has no user guide code example**: The pipeline API
   reference documents this function, and the CLI page mentions it by reference
   link, but there is no Python code snippet in the user guide showing how to
   use `inject_cbc_signal()` directly with existing `TimeSeries` data.

9. **`io` module (Bilby `.interferometer` compatibility) is undocumented**: The
   `gwmock_signal.io` subpackage provides `read_interferometer_config`,
   `interferometer_config_to_custom_detector`, and
   `resolve_interferometer_config_path` for reading legacy Bilby detector
   configs. These are completely absent from any user guide. Users migrating
   from Bilby will not know this helper exists.

10. **Logging utilities (`setup_logger`, `get_version_information`) not in user
    guide**: The `gwmock_signal.utils` module provides programmatic logging
    setup, but no user guide shows how to configure logging from Python (as
    opposed to the CLI's `--verbose` flag).

11. **`WaveformFactory.get_model()` not shown in examples**: Users can look up
    individual registered waveform generators, but there is no example
    demonstrating this.

12. **`CBCSimulator.waveform_model` property not mentioned**: This read-only
    property that exposes the waveform model name used by a simulator instance
    is not shown in any example.

13. **`Network.from_preset()` not mentioned in docs**: While bundled presets
    (ET-Triangle-Sardinia, ET-Triangle-EMR, etc.) are listed in the CLI page
    table, the `Network.from_preset()` classmethod for programmatic loading is
    not documented anywhere in the user guide.

14. **The `examples/` directory is unreferenced**: The repository ships
    `examples/gw150914_like.json` and `examples/networks/` (with
    `et_triangle.yaml` and `hlvk.yaml`), but the user guide never tells users
    about these. New users might not discover them.

15. **`python -m gwmock_signal` not documented**: The package has a
    `__main__.py` that sets up logging, but this is never mentioned as an
    alternative entry point.

16. **`--seed` flag in CLI lacks explanation**: The CLI page's table lists
    `--seed` as an optional random seed, but doesn't explain _what_ it seeds
    (NumPy's global random state) or _why_ a user would set it (reproducibility
    when generating zero-noise backgrounds for the same injection parameters).

---

### Example Validation

I verified all documented code snippets against their expected behavior based on
the documented API signatures:

#### Correct Snippets

- **Waveform Example 1** (waveform.md lines 97-117): The
  `WaveformFactory().generate()` call with `tc` in the parameters dict and
  `sampling_frequency`/`minimum_frequency` as extra kwargs correctly matches the
  `generate` method's merge semantics.
- **Waveform Example 2** (PyCBC backend, lines 121-141): The PyCBC parameter
  names (`mass1`, `mass2`, `spin1z`, `spin2z`) match what
  `PyCBCBackend.generate_td_waveform` expects via `_pop_alias`.
- **Waveform Example 3** (`pycbc_waveform_wrapper`, lines 150-166): Direct call
  syntax is correct.
- **Waveform Example 4** (custom model registration, lines 175-213): The custom
  callable signature matches the expected contract.
- **CLI example** (cli.md line 112):
  `gwmock-signal inject cbc --params cbc.json --network H1L1V1 --output injected.h5`
  — correct given default values for `--backend`, `--sample-rate`, `--f-min`,
  `--duration`, and `--approximant`.
- **Detector projection Example 1** (detector-projection.md lines 33-70):
  Correct usage of `project_polarizations_to_network`.
- **Strain injection Example 1** (strain-injection.md lines 32-75): Correct
  `inject_strain` call.
- **Multichannel Example 1** (multi-channel-strains.md lines 32-69): Correct
  `DetectorStrainStack.from_mapping` usage.
- **Custom backends example** (custom-backends.md lines 50-95): The
  `ConstantBurstSimulator` correctly implements the `GWSimulator` protocol.

#### Issues Found

1. **Waveform Example 1 — `sampling_frequency` type**: The example passes
   `sampling_frequency=4096.0` (float) and `minimum_frequency=20.0` (float).
   These are correct for the API, but there should be a note that
   `sampling_frequency` in the CLI is an `int` (exposed via `--sample-rate`),
   while the Python API expects `float`. This discrepancy between CLI
   (`--sample-rate 4096` is int) and Python API (`sampling_frequency=4096.0` is
   float) could confuse users switching between the two interfaces.

2. **CLI documentation table lists wrong default type for `--sample-rate`**: The
   table lists `--sample-rate` with a default of `4096`, implying an integer.
   The Python API consistently uses `float` for `sampling_frequency`. The CLI
   code also uses `int` for the Typer parameter annotation
   (`sample_rate: Annotated[int, ...]`). This is a minor inconsistency, but
   documentation should clarify that the CLI types `--sample-rate` as integer
   while the Python API types `sampling_frequency` as float.

---

### Organization and Clarity

#### Strengths

- The pipeline metaphor (waveforms → projection → injection → multichannel) is
  intuitive for GW analysts.
- Cross-referencing is excellent — every page has "See also" links and API
  reference pointers.
- The separation between "narrative examples" and "authoritative API reference"
  is clearly communicated.
- The CLI page's table-driven parameter documentation is clean and scannable.
- The backend comparison table (LAL vs PyCBC) is helpful.

#### Areas for Improvement

1. **No sidebar entry for "Examples" showing example files**: The sidebar
   structure is: "Home", "Installation", "Quick Start", "Command-line
   interface", "Examples" (with sub-pages: Waveforms, Detector projection,
   Strain injection, Multichannel strains), "Custom backends", "API", ... This
   is good, but "Examples" points to narrative how-to pages, not to the bundled
   example data files. Consider adding a small section or page that points users
   to `examples/gw150914_like.json` and `examples/networks/`.

2. **The "Quick Start" page is very thin**: It has only two verification
   commands and a "next steps" pointer. After verifying the install, a user has
   no immediate working demo. Consider adding a 10-line `inject_cbc_signal`
   example that produces a result from the bundled GW150914-like parameters.

3. **No single-page overview of the top-level API**: Users see
   `from gwmock_signal import CBCSimulator` in the README but there is no page
   listing all top-level exports in one place. The API index mentions "Top-level
   package exports" but doesn't enumerate them.

4. **The custom backends page buries the `source_type` registration feature**:
   The registry functions (`register_simulator_backend`,
   `resolve_simulator_backend`) are critical for downstream packages but are
   only mentioned in the "Registration by source_type" subsection at the end of
   the custom backends page. This should be given equal prominence alongside the
   `GWSimulator` subclassing instructions.

---

## Phase 2: Source Code Consistency Check

### Undocumented Features

The following features exist in the source code but are **not mentioned in any
user guide page**. (API reference pages generated from docstrings may cover
them, but narrative user guide examples are absent.)

| Feature                                                  | Location                             | Description                                                                                                                                                               |
| -------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CBCSimulator.write()`                                   | `simulator.py:410-458`               | Simulate + write to disk (HDF5/GWF/NPY/TXT) + auto-generate JSON sidecar. This is a top-level convenience that users are likely to need.                                  |
| `DetectorStrainStack.write()`                            | `multichannel/stack.py:190-246`      | Write stack to HDF5, NPY, GWF, or TXT with metadata.                                                                                                                      |
| `DetectorStrainStack.read()`                             | `multichannel/stack.py:248-317`      | Read back a stack from HDF5 or NPY.                                                                                                                                       |
| `DetectorStrainStack.t0` property                        | `multichannel/stack.py:133-135`      | GPS start time of the first sample.                                                                                                                                       |
| `DetectorStrainStack.sample_rate` property               | `multichannel/stack.py:138-144`      | Sample rate as a GWpy `Quantity`.                                                                                                                                         |
| `DetectorStrainStack.detector_names` property            | `multichannel/stack.py:128-130`      | Immutable tuple of detector names in channel order.                                                                                                                       |
| `TransientSimulator.register_waveform_model()`           | `simulator.py:205-248`               | Per-instance custom waveform registration. Supports callables returning `(plus, cross)` tuples or `{"plus": ..., "cross": ...}` dicts.                                    |
| `CBCSimulator.waveform_model` property                   | `simulator.py:366-368`               | Exposes the waveform model name.                                                                                                                                          |
| `WaveformFactory.get_model()`                            | `waveform/factory.py:112-126`        | Look up a specific registered generator callable by name.                                                                                                                 |
| `WaveformFactory.register_model()` (string-based import) | `waveform/factory.py:79-110`         | The factory supports registering models via import strings (`"module.path:callable"` or `"package.module.callable"`). This is not demonstrated in any user guide example. |
| `Network.from_preset()`                                  | `network.py:269-292`                 | Load a bundled YAML/JSON detector preset.                                                                                                                                 |
| `Network.list_lal_detectors()`                           | `network.py:379-386`                 | List all available LAL detector codes at runtime.                                                                                                                         |
| `list_lal_detectors()` (module-level)                    | `network.py:134-136`                 | Standalone helper function.                                                                                                                                               |
| `read_interferometer_config()`                           | `io/interferometer_format.py:80-84`  | Read a Bilby `.interferometer` file.                                                                                                                                      |
| `interferometer_config_to_custom_detector()`             | `io/interferometer_format.py:95-117` | Convert `.interferometer` to `CustomDetector`.                                                                                                                            |
| `resolve_interferometer_config_path()`                   | `io/interferometer_format.py:28-33`  | Resolve and validate config path.                                                                                                                                         |
| `setup_logger()`                                         | `utils/log.py:34-83`                 | Programmatic logging setup with optional file output.                                                                                                                     |
| `get_version_information()`                              | `utils/log.py:24-31`                 | Return package version string.                                                                                                                                            |
| `python -m gwmock_signal`                                | `__main__.py:18-21`                  | Package can be run as a module to set up logging and print version.                                                                                                       |
| `CustomDetector` class                                   | `detector.py:49-164`                 | Has examples in the API reference but no dedicated narrative walkthrough in user guide.                                                                                   |

### Documentation–Code Mismatches

#### 1. CLI `--earth-rotation` flag is not exposed

The CLI `inject cbc` command does not expose an `--earth-rotation` flag. The
internal call chain always uses `earth_rotation=True` (via the
`TransientSimulator.simulate` and `project_polarizations_to_network` defaults).
However, the user guide's detector projection page mentions "For very short
waveforms where Earth rotation over the segment is negligible" and demonstrates
`earth_rotation=False`. Users following the CLI workflow have no way to access
this optimization.

**Impact**: CLI users with short-duration waveforms cannot disable Earth
rotation even though the feature exists in the Python API. The documentation's
guidance about when to use `earth_rotation=False` is unreachable from the CLI.

**Suggestion**: Add `--earth-rotation` / `--no-earth-rotation` flags to the CLI
`inject cbc` command.

#### 2. CLI `--interpolate-if-offset` flag is not exposed

Similarly, `interpolate_if_offset` is always `True` in the CLI path. The strain
injection user guide shows
`inject_strain(target, h1, interpolate_if_offset=False)` as an optimization, but
CLI users cannot access this.

**Impact**: Same as above — documented feature unreachable from CLI.

**Suggestion**: Add `--interpolate-if-offset` / `--no-interpolate-if-offset`
flags to the CLI `inject cbc` command.

#### 3. README `CBCSimulator` import example may mislead

The README shows:

```python
from gwmock_signal import CBCSimulator, LALSimulationBackend
sim = CBCSimulator("IMRPhenomD", waveform_backend=LALSimulationBackend())
```

This is correct but incomplete — the user sees an instantiated `CBCSimulator`
but is not shown how to call `sim.simulate(...)` or what `DetectorStrainStack`
is. There is a 5-line logical gap between "I have a simulator" and "I have
results." The README could benefit from a concise end-to-end snippet using the
bundled example parameters file.

#### 4. Custom backends example imports `GWSimulator` from `gwmock_signal`

The custom backends page shows
`from gwmock_signal import DetectorStrainStack, GWSimulator`. This works because
of the lazy imports in `__init__.py`, but the actual definition location is
`gwmock_signal.simulator`. The user guide could note the canonical submodule
path for users who prefer explicit imports.

#### 5. Network preset table in CLI page is incomplete

The CLI page table lists presets `H1L1`, `H1L1V1`, `HLVK`, `ET-triangle`,
`ET-L`, `ET-Triangle-Sardinia`, `ET-Sardinia`, `ET-Triangle-EMR`, `ET-EMR`,
`ET-2L-Aligned`, `ET-2L-Misaligned`. However, the table only lists 6 rows and
uses aliases. It doesn't show the entries for `ET-Sardinia` (alias of
`ET-Triangle-Sardinia`) or `ET-EMR` (alias of `ET-Triangle-EMR`). Users might
not realize these shorter names work.

#### 6. `--output` help text in CLI says "HDF5 output file"

The CLI help says "HDF5 output file path" but the written file is actually a
GWpy `TimeSeriesDict` HDF5 (not a generic HDF5). This is a minor precision
issue.

#### 7. `inject_cbc_signal` docstring says `interpolate_if_offset=False` "skip the injection silently"

The docstring in `pipeline.py:71` says: "If `False`, skip the injection silently
when off-grid." However, in `inject_strain` (injection/core.py:78-80), when
`interpolate_if_offset=False` and the offset is non-integer, it returns
`target.copy()` — i.e., the background unchanged. The word "silently" is
accurate (only a debug log), but the behavior (returning an unmodified copy vs.
raising an error vs. mutating) could be clarified more explicitly for users who
need to detect when injections are skipped.

#### 8. `DetectorStrainStack` from injection page imports from `gwmock_signal.multichannel`

The multichannel user guide shows
`from gwmock_signal.multichannel import DetectorStrainStack`. This is correct,
but the top-level `__init__.py` also exports it, so
`from gwmock_signal import DetectorStrainStack` works too. The user guide should
be consistent — either always use the top-level import or always use the
submodule path.

---

### Instances Where Source Code Would Have Resolved Documentation Ambiguities

During Phase 1, I encountered the following points where looking at the source
code was necessary to fully understand behavior:

1. **What exactly does `--network` resolve to?**: The CLI page describes a
   3-step resolution (file → preset → comma-split codes), but the exact order
   and fallback logic could only be confirmed by reading
   `cli/inject.py:106-128`. The documentation's description is accurate, but a
   flowchart or code snippet would improve clarity.

2. **What happens when `--output` is omitted?**: The CLI page says "one line per
   detector is printed to stdout (RMS and duration)." The code
   (`cli/inject.py:182-186`) confirms the exact format:
   `f"{name}  rms={rms:.4e}  duration={ts.duration.value:.1f}s"`. The
   documentation does not show example output, which would help users validate
   correctness.

3. **How does `CBCSimulator.generate_polarizations` filter projection keys?**:
   The source (`simulator.py:395-398`) shows that `right_ascension`,
   `declination`, `polarization_angle`, and `coa_time` are excluded from
   waveform parameters. The user guide waveform examples don't explain this
   filtering, which could lead users to assume these keys are passed to the
   waveform backend.

4. **What parameters does `LALSimulationBackend` accept?**: The user guide says
   "LAL... accepts only the parameters documented for LAL time-domain generation
   (unknown keys error)." The source (`backends/lal.py:49-62`) shows the exact
   accepted parameters and their aliases. A reference table in the user guide
   would be more helpful than the vague description.

---

## Phase 3: Suggested Features

### Suggestion 1: CLI `--earth-rotation` and `--interpolate-if-offset` flags

**Description**: Expose the `earth_rotation` and `interpolate_if_offset` boolean
parameters as CLI flags (`--earth-rotation`/`--no-earth-rotation`,
`--interpolate-if-offset`/`--no-interpolate-if-offset`).

**Rationale**: These features are documented in the user guide as useful
optimizations for short waveforms, but they are inaccessible to CLI users. This
creates a feature gap between the CLI and Python API.

**Implementation approach**: Add two `Annotated[bool, typer.Option(...)]`
parameters to the `cbc` command in `cli/inject.py`, defaulting to `True` to
match current behavior. Pass them through to `inject_cbc_signal`.

---

### Suggestion 2: Add a built-in example parameter loading helper

**Description**: Provide a convenience function
`gwmock_signal.examples.load_gw150914_like_params()` that loads the bundled
`examples/gw150914_like.json` as a dict.

**Rationale**: New users need a zero-configuration way to get a working
parameter set. The bundled example file is unreferenced in documentation and
must be discovered manually.

**Implementation approach**: Add a small module `gwmock_signal.examples` with a
function that uses `importlib.resources` to locate the JSON file and return the
parsed dict. Document in Quick Start.

---

### Suggestion 3: Network file format user guide page

**Description**: Add a user guide page titled "Network configuration files" that
shows the YAML/JSON schema for `Network.from_file()`, with examples for both
simple LAL-code-only networks and full custom-detector geometry networks.

**Rationale**: Users who want to define a custom detector network (e.g., a
future third-generation observatory, or a specific observing run's subset)
currently have no documentation to follow. The source code's `Network.from_file`
docstring describes the schema but doesn't show a complete, runnable example
file.

**Implementation approach**: Create `docs/user_guide/network-config.md` with:

- A simple example (3-detector LIGO-Virgo using `H1`, `L1`, `V1` codes)
- A full example (a custom detector with `latitude_deg`, `longitude_deg`,
  `elevation_m`, arm azimuths/tilts)
- A table documenting all recognized angle keys (`_deg` vs `_rad` variants)
- Note about the deprecated `.interferometer` format migration

---

### Suggestion 4: `DetectorStrainStack` I/O user guide example

**Description**: Add an example to the multichannel strains user guide page
demonstrating `write()` and `read()` with HDF5 format (the default).

**Rationale**: Writing results to disk is a core workflow that is currently
undocumented. Users need to know how to persist and reload `DetectorStrainStack`
objects.

**Implementation approach**: Add an "Example 4 — Save and reload a stack"
section to `docs/user_guide/multi-channel-strains.md`:

```python
# Save
stack.write("my_injection.h5", format="hdf5")

# Reload
reloaded = DetectorStrainStack.read("my_injection.h5", format="hdf5")
assert reloaded.detector_names == stack.detector_names
```

---

### Suggestion 5: User guide page for `CBCSimulator.write()`

**Description**: Add a user guide page (or a subsection in the
CLI/strain-injection pages) showing how to use `CBCSimulator.write()` for a
one-step simulate-and-save workflow.

**Rationale**: `CBCSimulator.write()` is the highest-level convenience method in
the library and is not demonstrated anywhere.

**Implementation approach**: Add a section showing:

```python
from gwmock_signal import CBCSimulator

sim = CBCSimulator("IMRPhenomD")
result = sim.write(
    "output.h5",
    params={
        "detector_frame_mass_1": 36.0, ...
        "right_ascension": 1.375, ...
    },
    detector_names=["H1", "L1"],
    background={...},
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
    format="hdf5",
)
# Also writes output_params.json sidecar
```

---

### Suggestion 6: Quick Start end-to-end demo

**Description**: Expand the Quick Start page (quick_start.md) with a 10-line
working demo that generates a waveform, projects it, and prints a summary.

**Rationale**: The current Quick Start page only verifies installation. A user
who succeeds at `gwmock-signal --help` has no immediate next step that produces
a tangible result. A minimal working demo reduces
time-to-first-meaningful-output.

**Implementation approach**: Add a code block after "Verify the install" that
loads the bundled example params and runs the pipeline:

```python
import json
from pathlib import Path
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.network import Network
from gwpy.timeseries import TimeSeries
import numpy as np

# Load example parameters (bundled with package)
params = json.loads(Path("examples/gw150914_like.json").read_text())
net = Network.from_name("H1L1")
bg = {name: TimeSeries(np.zeros(8192), t0=params["coa_time"] - 1, sample_rate=4096) for name in net.detector_names}
result = inject_cbc_signal("IMRPhenomD", params, net.detector_names, bg, sampling_frequency=4096, minimum_frequency=20)
for name in result.detector_names:
    print(f"{name}: rms={np.sqrt(np.mean(result[name].value**2)):.2e}")
```

---

### Suggestion 7: CLI multi-injection support

**Description**: Extend `gwmock-signal inject cbc` to accept a JSON array of
parameter objects and inject each event sequentially into the same zero-noise
background.

**Rationale**: Users running mock data challenges with multiple injections
currently need to write Python scripts. Supporting multiple injections from a
single CLI invocation simplifies batch workflows.

**Implementation approach**: If `--params` points to a JSON array (detected by
`isinstance(cbc_params, list)`), iterate over each entry and call
`inject_strains_sequential` with all projected injections before writing/output.
Add a `--max-injections N` flag to limit processing.

---

### Suggestion 8: Noise generation utility

**Description**: Add an optional utility function
`gwmock_signal.utils.generate_gaussian_noise(duration, sample_rate, seed=None) -> TimeSeries`
that produces a stationary Gaussian noise background.

**Rationale**: The user guide repeatedly mentions "noise generation can live in
a separate package" and the CLI generates zero-noise backgrounds. For quick
prototyping, users shouldn't need to install an external noise package just to
test injection into something more realistic than zeros. This keeps scope small
(pure Python, no PSD modeling).

**Implementation approach**: A simple function using `numpy.random.default_rng`
to generate white Gaussian noise at unit amplitude. Users scale externally.
Place in `gwmock_signal.utils.noise`.

---

### Suggestion 9: `CustomDetector` user guide walkthrough

**Description**: Add a dedicated user guide page (or major section in the
detector projection page) showing how to construct a `CustomDetector` for a
non-standard observatory, add it to a `Network`, and use it in projection.

**Rationale**: Currently, `CustomDetector` has no narrative documentation. Users
wanting to simulate a third-generation detector or a proposed observatory have
to read the docstrings or source code.

**Implementation approach**: Create `docs/user_guide/custom-detectors.md` with:

- Explanation of the 8 geometry parameters (name, lat, lon, elevation, 2 arm
  azimuths, 2 arm tilts)
- An example: a hypothetical "ET-site" detector
- Integration with `Network.from_detectors()` and
  `project_polarizations_to_network`
- Note about the automatic LAL prefix generation

---

### Suggestion 10: `WaveformFactory.register_model()` string-based import in user guide

**Description**: Document the string-based import syntax for `register_model()`
(`"module.path:callable"` and `"package.module.callable"`).

**Rationale**: This feature (waveform/factory.py:96-104) allows
configuration-file-driven waveform registration without importing the callable
at registration time. It is a powerful feature for pipeline automation that is
completely undocumented.

**Implementation approach**: Add an example to the Waveforms user guide page
showing:

```python
factory = WaveformFactory()
factory.register_model("my_nr_model", "mypackage.waveforms:my_nr_generator")
```

---

### Summary

The `gwmock-signal` documentation is already at a high quality level. The main
gaps are: (1) several convenience methods (`CBCSimulator.write`,
`DetectorStrainStack.write/read`) are undocumented in the user guide despite
being likely the first things a new user would want to do after generating
results; (2) the CLI is missing flags for features that exist in the Python API
(`earth_rotation`, `interpolate_if_offset`); and (3) several extension points
(`CustomDetector`, `register_waveform_model`, network file format, string-based
model registration) lack narrative walkthroughs. Addressing these gaps would
make the package significantly more accessible to new users and more useful in
downstream pipelines.
