"""What works with only the core dependencies: no GWpy, no h5py, and no lalsuite.

This is the installation Python 3.15 gets today, where lalsuite and h5py publish no wheel. CI runs
this module there, with only the package and pytest installed; it also runs with everything
installed, where it checks the same paths never import the optional packages.

Each test runs its scenario in a fresh interpreter, so what the package imports can be observed:
``sys.modules`` in the test process already holds whatever other tests imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

#: Packages none of these paths may import.
_OPTIONAL = ("gwpy", "h5py", "lal", "lalsimulation", "jax", "pycbc")


def _run(code: str) -> dict:
    """Run ``code`` in a fresh interpreter; it must print one JSON object as its last line."""
    script = textwrap.dedent(code) + textwrap.dedent(
        f"""
        import json as _json, sys as _sys
        _loaded = sorted({{name.split(".")[0] for name in _sys.modules}} & set({_OPTIONAL!r}))
        print(_json.dumps({{"result": result, "optional_loaded": _loaded}}))
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_package_and_its_cli_import_without_optional_packages():
    """Importing the package, its time series, its I/O module and the CLI loads none of them."""
    outcome = _run(
        """
        import gwmock_signal, gwmock_signal.cli.main, gwmock_signal.io, gwmock_signal.projection.network
        from gwmock_signal import TimeSeries
        result = gwmock_signal.__version__
        """
    )

    assert outcome["optional_loaded"] == []


def test_projection_and_injection_work_without_optional_packages():
    """Built-in and custom detectors, with and without Earth rotation, then injection."""
    outcome = _run(
        """
        import numpy as np
        from gwmock_signal import CustomDetector, TimeSeries
        from gwmock_signal.injection import inject_strain
        from gwmock_signal.projection import project_polarizations_to_network

        t0, fs = 1_400_000_000.0, 512.0
        t = np.arange(1024) / fs
        tone = np.exp(-0.5 * ((t - 1.0) / 0.1) ** 2) * np.sin(2 * np.pi * 40.0 * t)
        polarizations = {
            "plus": TimeSeries(tone, t0=t0, sample_rate=fs),
            "cross": TimeSeries(0.5 * tone, t0=t0, sample_rate=fs),
        }
        custom = CustomDetector(
            name="X1", latitude_rad=0.76, longitude_rad=0.18, elevation_m=50.0,
            xarm_azimuth_rad=0.34, yarm_azimuth_rad=1.39,
        )
        peaks = {}
        for rotation in (True, False):
            strains = project_polarizations_to_network(
                polarizations, ["H1", custom], right_ascension=1.1, declination=-0.3,
                polarization_angle=0.4, earth_rotation=rotation,
            )
            for name, strain in strains.items():
                peaks[f"{name}-{rotation}"] = float(np.max(np.abs(strain.data)))
        background = TimeSeries(np.zeros(4096), t0=t0 - 1.0, sample_rate=fs)
        injected = inject_strain(background, strains["H1"])
        result = {"peaks": peaks, "injected_peak": float(np.max(np.abs(injected.data)))}
        """
    )

    assert outcome["optional_loaded"] == []
    assert all(peak > 0.0 for peak in outcome["result"]["peaks"].values())
    assert outcome["result"]["injected_peak"] > 0.0


def test_simulation_with_a_registered_model_works_without_optional_packages():
    """The CBC simulator with a waveform model the caller registers, which needs no lalsuite."""
    outcome = _run(
        """
        import numpy as np
        from gwmock_signal import CBCSimulator, TimeSeries
        from gwmock_signal.waveform.backends.base import WaveformBackend

        class NoBackend(WaveformBackend):
            def available_approximants(self):
                return []

            def generate_td_waveform(self, approximant, tc, sampling_frequency, minimum_frequency, **params):
                raise NotImplementedError

        def burst(*, tc, sampling_frequency, **_):
            t = np.arange(256) / sampling_frequency
            envelope = np.exp(-0.5 * ((t - 0.25) / 0.05) ** 2)
            return (
                TimeSeries(envelope * np.sin(2 * np.pi * 60.0 * t), t0=tc - 0.25, sample_rate=sampling_frequency),
                TimeSeries(envelope * np.cos(2 * np.pi * 60.0 * t), t0=tc - 0.25, sample_rate=sampling_frequency),
            )

        simulator = CBCSimulator(waveform_model="burst", waveform_backend=NoBackend())
        simulator.register_waveform_model("burst", burst)
        params = {
            "detector_frame_mass_1": 30.0, "detector_frame_mass_2": 25.0, "distance": 400.0, "inclination": 0.0,
            "coa_time": 1_400_000_000.0, "right_ascension": 1.0, "declination": 0.2,
            "polarization_angle": 0.3,
        }
        background = {name: TimeSeries(np.zeros(1024), t0=1_399_999_999.0, sample_rate=1024.0) for name in ("H1", "L1")}
        stack = simulator.simulate(params, ["H1", "L1"], background, sampling_frequency=1024.0, minimum_frequency=20.0)
        result = {"names": list(stack.detector_names), "peak": float(np.max(np.abs(stack.data)))}
        """
    )

    assert outcome["optional_loaded"] == []
    assert outcome["result"]["names"] == ["H1", "L1"]
    assert outcome["result"]["peak"] > 0.0


def test_text_and_numpy_files_need_no_optional_package(tmp_path):
    """A stack round-trips through the NumPy format with core dependencies only."""
    path = tmp_path / "stack.npy"
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_signal import DetectorStrainStack, TimeSeries

        channels = {{name: TimeSeries(np.arange(8.0), t0=10.0, sample_rate=4.0) for name in ("H1", "L1")}}
        DetectorStrainStack.from_mapping(["H1", "L1"], channels).write({str(path)!r}, format="npy")
        back = DetectorStrainStack.read({str(path)!r}, format="npy")
        result = {{"t0": back.t0, "rate": back.sample_rate, "equal": bool(np.array_equal(back.data[0], np.arange(8.0)))}}
        """
    )

    assert outcome["optional_loaded"] == []
    assert outcome["result"] == {"t0": 10.0, "rate": 4.0, "equal": True}


def test_hdf5_without_h5py_names_the_extra():
    """Asking for HDF5 without h5py says how to install it, rather than failing at import."""
    pytest.importorskip("numpy")
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        from gwmock_signal import DetectorStrainStack, TimeSeries

        stack = DetectorStrainStack.from_mapping(["H1"], {"H1": TimeSeries([0.0], t0=0.0, sample_rate=1.0)})
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock-signal\[hdf5\]'"):
            stack.write("unused.h5")
    else:
        pytest.skip("h5py is installed; tests/io/test_hdf5.py covers this case")


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
def test_the_cli_defers_what_only_its_commands_need():
    """From Python 3.15, building the command-line app imports neither the simulator nor astropy.

    They are declared in ``__lazy_modules__``, so the CLI and ``--help`` start without them; a
    command imports them when it runs.
    """
    outcome = _run(
        """
        import sys
        import gwmock_signal.cli.main
        heavy = ("astropy", "numpy", "gwmock_signal.pipeline", "gwmock_signal.simulator")
        result = sorted(name for name in heavy if name in sys.modules)
        """
    )

    assert outcome["result"] == []


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
def test_projection_defers_astropy_until_sidereal_time_is_needed():
    """Importing the projection does not load astropy; computing sidereal time does."""
    outcome = _run(
        """
        import sys
        from gwmock_signal.projection.sidereal import gmst_rad_astropy
        before = "astropy" in sys.modules
        gmst_rad_astropy(1_400_000_000.0)
        result = [before, "astropy" in sys.modules]
        """
    )

    assert outcome["result"] == [False, True]


def test_the_version_is_read_when_asked_for():
    """Importing the package does not read its installed metadata; ``__version__`` does, once."""
    outcome = _run(
        """
        import sys
        import gwmock_signal
        before = "gwmock_signal.version" in sys.modules
        version = gwmock_signal.__version__
        result = [before, "gwmock_signal.version" in sys.modules, isinstance(version, str) and bool(version)]
        """
    )

    assert outcome["result"] == [False, True, True]
