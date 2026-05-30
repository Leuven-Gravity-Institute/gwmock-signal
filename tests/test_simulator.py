"""Unit tests for GWSimulator ABC and CBCSimulator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

import gwmock_signal
from gwmock_signal import GWSimulator, register_simulator_backend, resolve_simulator_backend
from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.simulator import CBCSimulator, TransientSimulator, _json_default
from gwmock_signal.waveform import LALSimulationBackend

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zeros(n: int = 128, fs: float = 4096.0, t0: float = 1_000_000.0) -> TimeSeries:
    return TimeSeries(np.zeros(n), t0=t0, sample_rate=fs)


_MINIMAL_PARAMS: dict = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "coa_time": 1_126_259_462.4,
    "distance": 410.0,
    "inclination": 0.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0,
}


def _toy_registered_waveform(
    *,
    tc: float,
    sampling_frequency: float,
    waveform_model: str,
    **_: object,
) -> tuple[TimeSeries, TimeSeries]:
    dt = 1.0 / sampling_frequency
    hp = TimeSeries(np.ones(8), t0=tc, dt=dt, name=f"{waveform_model}-plus")
    hc = TimeSeries(np.zeros(8), t0=tc, dt=dt, name=f"{waveform_model}-cross")
    return hp, hc


# ---------------------------------------------------------------------------
# GWSimulator ABC
# ---------------------------------------------------------------------------


class TestGWSimulatorAbstract:
    """Tests that GWSimulator enforces abstract-class semantics."""

    def test_direct_instantiation_raises_type_error(self):
        """GWSimulator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GWSimulator()  # type: ignore[abstract]

    def test_concrete_subclass_without_simulate_raises(self):
        """Subclass missing simulate raises TypeError."""

        class Incomplete(GWSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                return frozenset()

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_stochastic_simulator_stub_no_transient_required(self):
        """A GWSimulator subclass implementing only simulate and required_params is valid."""

        class _StochasticStub(GWSimulator):
            @property
            def required_params(self):
                return frozenset()

            def simulate(self, params) -> DetectorStrainStack:  # type: ignore[override]
                raise NotImplementedError

        stub = _StochasticStub()
        assert isinstance(stub, GWSimulator)


# ---------------------------------------------------------------------------
# _validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """Tests for GWSimulator._validate_params."""

    def _make_concrete(self, required: frozenset[str]) -> GWSimulator:
        """Return a minimal concrete subclass with the given required_params."""

        class Concrete(GWSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                return required

            def simulate(self, params) -> DetectorStrainStack:  # type: ignore[override]
                raise NotImplementedError

        return Concrete()

    def test_all_present_does_not_raise(self):
        """No exception when all required keys are present (extras are allowed)."""
        sim = self._make_concrete(frozenset({"a", "b"}))
        sim._validate_params({"a": 1, "b": 2, "c": 3})  # extra key is fine

    def test_missing_single_key_raises_value_error(self):
        """Single missing key names that key in the error."""
        sim = self._make_concrete(frozenset({"detector_frame_mass_1", "detector_frame_mass_2"}))
        with pytest.raises(ValueError, match="detector_frame_mass_2"):
            sim._validate_params({"detector_frame_mass_1": 10.0})

    def test_missing_multiple_keys_names_all(self):
        """All missing keys appear in the error message."""
        sim = self._make_concrete(frozenset({"a", "b", "c"}))
        with pytest.raises(ValueError, match="a") as exc_info:
            sim._validate_params({})
        msg = str(exc_info.value)
        assert "b" in msg
        assert "c" in msg


# ---------------------------------------------------------------------------
# CBCSimulator
# ---------------------------------------------------------------------------


class TestCBCSimulatorRequiredParams:
    """Tests for CBCSimulator.required_params."""

    def test_required_params_is_frozenset(self):
        """required_params returns a frozenset."""
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        assert isinstance(sim.required_params, frozenset)

    def test_required_params_includes_cbc_keys(self):
        """CBC waveform keys are in required_params."""
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        for key in ("detector_frame_mass_1", "detector_frame_mass_2", "coa_time", "distance", "inclination"):
            assert key in sim.required_params, f"Expected '{key}' in required_params"

    def test_required_params_includes_projection_keys(self):
        """Projection sky-position keys are in required_params."""
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        for key in ("right_ascension", "declination", "polarization_angle"):
            assert key in sim.required_params, f"Expected '{key}' in required_params"

    def test_waveform_model_property_returns_constructor_value(self):
        """waveform_model property exposes the model passed at construction."""
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        assert sim.waveform_model == "IMRPhenomD"

    @patch("gwmock_signal.simulator.WaveformFactory")
    def test_constructor_passes_waveform_backend_to_factory(self, mock_factory_cls):
        """An explicit waveform backend is forwarded into WaveformFactory."""
        backend = LALSimulationBackend()
        CBCSimulator(waveform_model="IMRPhenomD", waveform_backend=backend)
        mock_factory_cls.assert_called_once_with(backend=backend)


class TestRegisterWaveformModel:
    """Tests for the public per-instance waveform registration hook."""

    @staticmethod
    def _make_simulator() -> CBCSimulator:
        backend = MagicMock(spec=LALSimulationBackend)
        backend.available_approximants.return_value = []
        return CBCSimulator(waveform_model="toy_burst", waveform_backend=backend)

    def test_register_waveform_model_allows_custom_callable(self) -> None:
        """A registered callable waveform can be selected by ``waveform_model``."""
        sim = self._make_simulator()

        sim.register_waveform_model("toy_burst", _toy_registered_waveform)

        hp, hc = sim.generate_polarizations(
            _MINIMAL_PARAMS,
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )

        assert isinstance(hp, TimeSeries)
        assert isinstance(hc, TimeSeries)
        assert len(hp) == 8
        assert len(hc) == 8
        assert np.allclose(hp.value, 1.0)
        assert np.allclose(hc.value, 0.0)

    def test_register_waveform_model_reregister_same_factory_is_noop(self) -> None:
        """Re-registering the same callable under the same name is harmless."""
        sim = self._make_simulator()

        sim.register_waveform_model("toy_burst", _toy_registered_waveform)
        sim.register_waveform_model("toy_burst", _toy_registered_waveform)

        hp, hc = sim.generate_polarizations(
            _MINIMAL_PARAMS,
            sampling_frequency=256.0,
            minimum_frequency=20.0,
        )

        assert len(hp) == 8
        assert len(hc) == 8

    def test_register_waveform_model_conflicting_reregistration_raises(self) -> None:
        """Reusing a name for a different factory raises ``ValueError``."""

        def other_waveform(
            *,
            tc: float,
            sampling_frequency: float,
            waveform_model: str,
            **_: object,
        ) -> tuple[TimeSeries, TimeSeries]:
            dt = 1.0 / sampling_frequency
            hp = TimeSeries(np.full(8, 2.0), t0=tc, dt=dt, name=f"{waveform_model}-plus")
            hc = TimeSeries(np.full(8, -1.0), t0=tc, dt=dt, name=f"{waveform_model}-cross")
            return hp, hc

        sim = self._make_simulator()
        sim.register_waveform_model("toy_burst", _toy_registered_waveform)

        with pytest.raises(ValueError, match="already registered"):
            sim.register_waveform_model("toy_burst", other_waveform)


class TestCBCSimulatorMissingParam:
    """Tests that missing required params raise ValueError with the key named."""

    def test_missing_mass1_raises_value_error_naming_key(self):
        """Missing detector_frame_mass_1 raises ValueError that names the key."""
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        incomplete = {k: v for k, v in _MINIMAL_PARAMS.items() if k != "detector_frame_mass_1"}
        with pytest.raises(ValueError, match="detector_frame_mass_1"):
            sim._validate_params(incomplete)


class TestCBCSimulatorSimulate:
    """Mock the full pipeline to test CBCSimulator.simulate returns DetectorStrainStack."""

    @patch("gwmock_signal.simulator.project_polarizations_to_network")
    @patch("gwmock_signal.simulator.inject_strain")
    @patch("gwmock_signal.simulator.WaveformFactory")
    def test_simulate_returns_detector_strain_stack(
        self,
        mock_factory_cls,
        mock_inject,
        mock_project,
    ):
        """Simulate returns a DetectorStrainStack with the requested detector order."""
        fs = 4096.0
        n = 128
        t0 = _MINIMAL_PARAMS["coa_time"]

        hp = TimeSeries(np.ones(n), t0=t0, sample_rate=fs)
        hc = TimeSeries(np.zeros(n), t0=t0, sample_rate=fs)
        strain_h1 = _zeros(n, fs, t0)
        strain_l1 = _zeros(n, fs, t0)

        mock_factory_cls.return_value.generate.return_value = {"plus": hp, "cross": hc}
        mock_project.return_value = {"H1": strain_h1, "L1": strain_l1}
        mock_inject.side_effect = lambda target, injection, **kw: _zeros(n, fs, t0)

        detector_names = ["H1", "L1"]
        background = {name: _zeros(n, fs, t0) for name in detector_names}

        sim = CBCSimulator(waveform_model="IMRPhenomD")
        result = sim.simulate(
            _MINIMAL_PARAMS,
            detector_names,
            background,
            sampling_frequency=fs,
            minimum_frequency=20.0,
        )

        assert isinstance(result, DetectorStrainStack)
        assert result.detector_names == tuple(detector_names)

    @patch("gwmock_signal.simulator.project_polarizations_to_network")
    @patch("gwmock_signal.simulator.inject_strain")
    @patch("gwmock_signal.simulator.WaveformFactory")
    def test_simulate_calls_pipeline_steps_in_order(
        self,
        mock_factory_cls,
        mock_inject,
        mock_project,
    ):
        """Verify _validate_params → generate_polarizations → project → inject order."""
        fs = 4096.0
        n = 64
        t0 = _MINIMAL_PARAMS["coa_time"]

        hp = TimeSeries(np.ones(n), t0=t0, sample_rate=fs)
        hc = TimeSeries(np.zeros(n), t0=t0, sample_rate=fs)
        projected_strain = _zeros(n, fs, t0)

        call_order: list[str] = []

        def _gen(*args, **kwargs):
            call_order.append("generate")
            return {"plus": hp, "cross": hc}

        def _project(*args, **kwargs):
            call_order.append("project")
            return {"H1": projected_strain}

        def _inject(*args, **kwargs):
            call_order.append("inject")
            return _zeros(n, fs, t0)

        mock_factory_cls.return_value.generate.side_effect = _gen
        mock_project.side_effect = _project
        mock_inject.side_effect = _inject

        background = {"H1": _zeros(n, fs, t0)}
        CBCSimulator(waveform_model="IMRPhenomD").simulate(
            _MINIMAL_PARAMS,
            ["H1"],
            background,
            sampling_frequency=fs,
            minimum_frequency=20.0,
        )

        mock_factory_cls.return_value.generate.assert_called_once()
        mock_project.assert_called_once()
        mock_inject.assert_called_once()
        assert call_order == ["generate", "project", "inject"]

    @patch("gwmock_signal.simulator.WaveformFactory")
    def test_simulate_missing_param_raises_before_waveform(self, mock_factory_cls):
        """Missing required parameter raises ValueError before any waveform call."""
        incomplete = {k: v for k, v in _MINIMAL_PARAMS.items() if k != "coa_time"}
        sim = CBCSimulator(waveform_model="IMRPhenomD")
        with pytest.raises(ValueError, match="coa_time"):
            sim.simulate(
                incomplete,
                ["H1"],
                {"H1": _zeros()},
                sampling_frequency=4096.0,
                minimum_frequency=20.0,
            )
        mock_factory_cls.return_value.generate.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end pipeline (mocked only at get_td_waveform boundary)
# ---------------------------------------------------------------------------


class TestCBCSimulatorEndToEnd:
    """Non-mocked assembly test: real project_polarizations_to_network and inject_strain."""

    @patch("gwmock_signal.waveform.pycbc_wrapper.get_td_waveform")
    def test_gw150914_like_params_on_two_detector_network(self, mock_get_td):
        """CBCSimulator.simulate returns a DetectorStrainStack for GW150914-like params.

        WaveformFactory/PyCBC waveform generation is mocked at the get_td_waveform
        boundary; project_polarizations_to_network and inject_strain run unpatched,
        exercising the full assembled pipeline with real geometry and injection logic.
        The plus polarization is set to a unit-amplitude signal so that the antenna
        response at the GW150914 sky position produces a non-zero projected strain,
        confirming that the projection/injection path actually modifies the data.
        """
        pytest.importorskip("pycbc", reason="pycbc not installed")
        from pycbc.types import TimeSeries as PyCBCTimeSeries

        tc = _MINIMAL_PARAMS["coa_time"]
        fs = 4096.0
        n = 256

        # Non-zero plus polarization; pycbc_waveform_wrapper adds tc to start_time
        # so the resulting GWpy series starts at tc and overlaps the background fully.
        pycbc_hp = PyCBCTimeSeries(np.ones(n, dtype=float), delta_t=1.0 / fs, epoch=0.0)
        pycbc_hc = PyCBCTimeSeries(np.zeros(n, dtype=float), delta_t=1.0 / fs, epoch=0.0)
        mock_get_td.return_value = (pycbc_hp, pycbc_hc)

        # Zero-noise background covers the post-injection epoch [tc, tc + n/fs)
        detector_names = ["H1", "L1"]
        background = {name: TimeSeries(np.zeros(n), t0=tc, sample_rate=fs) for name in detector_names}

        result = CBCSimulator(waveform_model="IMRPhenomD").simulate(
            _MINIMAL_PARAMS,
            detector_names,
            background,
            sampling_frequency=fs,
            minimum_frequency=20.0,
        )

        assert isinstance(result, DetectorStrainStack)
        assert result.detector_names == tuple(detector_names)
        assert len(result) == len(detector_names)
        # Each channel should be a TimeSeries with the same length as the background.
        for name in detector_names:
            assert len(result[name]) == n
        # The projected plus polarization at the GW150914 sky position is non-zero for
        # at least one detector; verify the injection path modified the background.
        assert any(np.any(result[name].value != 0.0) for name in detector_names), (
            "Expected non-zero strain in at least one detector after injection"
        )

    def test_lal_backend_simulate_runs_without_pycbc_for_builtin_detectors(self) -> None:
        """The CBC+LAL backend path works when PyCBC imports are blocked."""
        code = """
import builtins
import numpy as np
from gwpy.timeseries import TimeSeries

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("pycbc"):
        raise ModuleNotFoundError("No module named 'pycbc'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from gwmock_signal.simulator import CBCSimulator
from gwmock_signal.waveform.backends import LALSimulationBackend

params = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "coa_time": 1126259462.4,
    "distance": 410.0,
    "inclination": 0.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 0.0,
}
background = {
    "H1": TimeSeries(np.zeros(8192), t0=params["coa_time"] - 1.0, sample_rate=4096.0),
}
result = CBCSimulator("IMRPhenomD", waveform_backend=LALSimulationBackend()).simulate(
    params,
    ["H1"],
    background,
    sampling_frequency=4096.0,
    minimum_frequency=20.0,
)
assert len(result["H1"]) == 8192
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


# ---------------------------------------------------------------------------
# Public import contract
# ---------------------------------------------------------------------------


class TestPublicImport:
    """Tests that GWSimulator and CBCSimulator are importable from the top-level package."""

    def test_gw_simulator_importable_from_top_level(self):
        """GWSimulator is reachable via gwmock_signal.GWSimulator."""
        assert hasattr(gwmock_signal, "GWSimulator")

    def test_transient_simulator_importable_from_top_level(self):
        """TransientSimulator is reachable via gwmock_signal.TransientSimulator."""
        assert hasattr(gwmock_signal, "TransientSimulator")

    def test_cbc_simulator_importable_from_top_level(self):
        """CBCSimulator is reachable via gwmock_signal.CBCSimulator."""
        assert hasattr(gwmock_signal, "CBCSimulator")

    def test_source_type_registry_helpers_importable_from_top_level(self):
        """Source-type registry helpers are reachable via the top-level package."""
        assert hasattr(gwmock_signal, "resolve_simulator_backend")
        assert hasattr(gwmock_signal, "register_simulator_backend")
        assert hasattr(gwmock_signal, "list_registered_source_types")


class TestSourceTypeRegistry:
    """Tests for public source-type backend resolution."""

    @pytest.fixture(autouse=True)
    def _isolate_source_type_registry(self):
        import gwmock_signal.registry as registry_module

        snapshot = dict(registry_module._SOURCE_TYPE_REGISTRY)
        try:
            yield
        finally:
            registry_module._SOURCE_TYPE_REGISTRY.clear()
            registry_module._SOURCE_TYPE_REGISTRY.update(snapshot)

    def test_resolve_bbh_without_importing_cbc_simulator_at_call_site(self):
        """Downstream code can resolve the BBH backend without a direct CBC import."""
        backend = resolve_simulator_backend("bbh")
        assert issubclass(backend, GWSimulator)
        assert backend.__name__ == "CBCSimulator"

    def test_resolve_bns_returns_cbc_simulator(self):
        """BNS source type resolves to CBCSimulator."""
        backend = resolve_simulator_backend("bns")
        assert issubclass(backend, GWSimulator)
        assert backend.__name__ == "CBCSimulator"

    def test_resolve_nsbh_returns_cbc_simulator(self):
        """NSBH source type resolves to CBCSimulator."""
        backend = resolve_simulator_backend("nsbh")
        assert issubclass(backend, GWSimulator)
        assert backend.__name__ == "CBCSimulator"

    @pytest.mark.parametrize("source_type", ["bns", "nsbh"])
    def test_compact_binary_variants_case_insensitive(self, source_type: str):
        """BNS and NSBH lookups are case-insensitive."""
        assert resolve_simulator_backend(source_type.upper()) is resolve_simulator_backend(source_type)

    @pytest.mark.parametrize("source_type", ["bns", "nsbh"])
    def test_list_registered_source_types_includes_compact_binary_variants(self, source_type: str):
        """BNS and NSBH are present in the default public registry."""
        assert source_type in gwmock_signal.list_registered_source_types()

    def test_bns_and_nsbh_resolve_to_same_backend_as_bbh(self):
        """BNS and NSBH share the CBCSimulator backend with BBH."""
        assert resolve_simulator_backend("bns") is resolve_simulator_backend("bbh")
        assert resolve_simulator_backend("nsbh") is resolve_simulator_backend("bbh")

    def test_source_type_lookup_is_case_insensitive(self):
        """Source-type normalization is lowercase and trims whitespace."""
        assert resolve_simulator_backend("  BBH  ") is resolve_simulator_backend("bbh")

    def test_list_registered_source_types_includes_bbh(self):
        """The default public registry exposes the built-in BBH backend."""
        assert "bbh" in gwmock_signal.list_registered_source_types()

    def test_register_backend_allows_future_source_families_without_contract_change(self):
        """Future source families extend the registry by registration alone."""

        class _StubSimulator(GWSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                return frozenset()

            def simulate(self, strain, params, detector) -> DetectorStrainStack:  # type: ignore[override]
                raise NotImplementedError

        register_simulator_backend("stochastic", _StubSimulator)

        assert resolve_simulator_backend("stochastic") is _StubSimulator

    def test_register_backend_rejects_conflicting_duplicate_mapping(self):
        """A registered source type cannot be silently rebound to a different backend."""

        class _FirstStub(GWSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                return frozenset()

            def simulate(self, strain, params, detector) -> DetectorStrainStack:  # type: ignore[override]
                raise NotImplementedError

        class _SecondStub(GWSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                return frozenset()

            def simulate(self, strain, params, detector) -> DetectorStrainStack:  # type: ignore[override]
                raise NotImplementedError

        register_simulator_backend("burst", _FirstStub)

        with pytest.raises(ValueError, match="already registered"):
            register_simulator_backend("burst", _SecondStub)

    def test_resolve_unknown_source_type_raises_key_error(self):
        """Unknown source types fail with a clear lookup error."""
        with pytest.raises(KeyError, match="source_type='unknown'"):
            resolve_simulator_backend("unknown")


class TestJsonDefault:
    """Unit tests for simulator JSON serialization helper."""

    def test_numpy_scalar_converts_to_python_scalar(self):
        """NumPy scalars are converted via item()."""
        value = np.float64(1.25)
        out = _json_default(value)
        assert isinstance(out, float)
        assert out == pytest.approx(1.25)

    def test_numpy_array_converts_to_list(self):
        """NumPy arrays are converted via tolist()."""
        out = _json_default(np.array([1.0, 2.0, 3.0]))
        assert out == [1.0, 2.0, 3.0]

    def test_unsupported_type_raises_type_error(self):
        """Non-NumPy objects without item()/tolist() raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default({"not": "serializable by helper"})


class TestTransientProjectionValidation:
    """Tests for projection-key validation in TransientSimulator.simulate."""

    def test_missing_projection_keys_raise_value_error(self):
        """Non-CBC transient subclasses still require sky-position projection keys."""

        class _TransientStub(TransientSimulator):
            @property
            def required_params(self) -> frozenset[str]:
                # Intentionally exclude projection keys to exercise the guard.
                return frozenset({"tc"})

            def generate_polarizations(
                self, params: dict, sampling_frequency: float, minimum_frequency: float
            ) -> tuple[TimeSeries, TimeSeries]:
                n = 8
                return (
                    TimeSeries(np.zeros(n), t0=float(params["tc"]), sample_rate=sampling_frequency),
                    TimeSeries(np.zeros(n), t0=float(params["tc"]), sample_rate=sampling_frequency),
                )

        sim = _TransientStub()
        params = {"tc": 100.0}  # right_ascension/declination/polarization_angle absent
        background = {"H1": _zeros(n=8, fs=8.0, t0=100.0)}
        with pytest.raises(ValueError, match="right_ascension"):
            sim.simulate(
                params,
                ["H1"],
                background,
                sampling_frequency=8.0,
                minimum_frequency=20.0,
            )


class TestCBCSimulatorWrite:
    """Tests for CBCSimulator.write side effects and return value."""

    def test_write_saves_stack_and_json_sidecar(self, tmp_path):
        """write() calls simulate, writes output, and stores params JSON."""
        fs = 256.0
        n = 64
        t0 = _MINIMAL_PARAMS["coa_time"]
        params = {
            **_MINIMAL_PARAMS,
            "distance": np.float64(_MINIMAL_PARAMS["distance"]),
        }
        detector_names = ["H1", "L1"]
        background = {name: TimeSeries(np.zeros(n), t0=t0, sample_rate=fs) for name in detector_names}

        out_path = tmp_path / "simulated.hdf5"
        expected = DetectorStrainStack.from_mapping(detector_names, background)

        sim = CBCSimulator(waveform_model="IMRPhenomD")
        with patch.object(sim, "simulate", return_value=expected) as mock_simulate:
            result = sim.write(
                out_path,
                params,
                detector_names,
                background,
                sampling_frequency=fs,
                minimum_frequency=20.0,
                format="hdf5",
            )

        mock_simulate.assert_called_once()
        assert result is expected
        assert out_path.exists()

        params_sidecar = tmp_path / "simulated_params.json"
        assert params_sidecar.exists()
        saved = json.loads(params_sidecar.read_text())
        assert saved["distance"] == pytest.approx(_MINIMAL_PARAMS["distance"])
