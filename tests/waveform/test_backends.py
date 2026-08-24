"""Tests for waveform backends."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import LALSimulationBackend, PyCBCBackend

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lalsimulation_backend_available_approximants_includes_imrphenomd() -> None:
    """The default LAL backend exposes standard TD approximants."""
    models = LALSimulationBackend().available_approximants()
    assert "IMRPhenomD" in models


def test_lalsimulation_backend_generates_timeseries_dict() -> None:
    """A minimal LAL waveform call returns GWpy time series."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=36.0,
        detector_frame_mass_2=29.0,
        luminosity_distance=410.0,
        inclination=0.0,
        coa_phase=0.0,
    )
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)
    assert isinstance(result["cross"], TimeSeries)


def test_pycbc_backend_raises_helpful_import_error_when_pycbc_missing() -> None:
    """PyCBCBackend fails at instantiation time with installation guidance."""
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "pycbc.waveform":
            raise ImportError("pycbc unavailable")
        return real_import_module(name, package)

    with (
        patch("gwmock_signal.waveform.backends.pycbc.importlib.import_module", side_effect=_import_module),
        pytest.raises(ImportError, match=r"gwmock-signal\[pycbc\]"),
    ):
        PyCBCBackend()


def test_pycbc_backend_available_approximants_when_installed() -> None:
    """PyCBCBackend works normally when PyCBC is present."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    assert "IMRPhenomD" in PyCBCBackend().available_approximants()


def test_lal_backend_accepts_tidal_params_for_nrtidal_approximant() -> None:
    """lambda_1/lambda_2 do not raise ValueError for NRTidal waveforms."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomPv2_NRTidalv2",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=1.4,
        detector_frame_mass_2=1.4,
        luminosity_distance=40.0,
        lambda_1=500.0,
        lambda_2=300.0,
    )
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)


def test_lal_backend_accepts_tidal_aliases() -> None:
    """tidal_1/tidal_2 are accepted as aliases for lambda_1/lambda_2."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomPv2_NRTidalv2",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=1.4,
        detector_frame_mass_2=1.4,
        luminosity_distance=40.0,
        tidal_1=500.0,
        tidal_2=300.0,
    )
    assert set(result) == {"plus", "cross"}


def test_lal_backend_nontidal_unaffected_by_default_lambdas() -> None:
    """Non-tidal waveforms work without tidal params (lambda=0 is safe)."""
    result = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=1_126_259_462.4,
        sampling_frequency=4096.0,
        minimum_frequency=20.0,
        detector_frame_mass_1=36.0,
        detector_frame_mass_2=29.0,
        luminosity_distance=410.0,
    )
    assert set(result) == {"plus", "cross"}


@pytest.mark.parametrize("param", ["lambda_1", "lambda_2"])
def test_lal_backend_negative_tidal_param_raises(param: str) -> None:
    """Negative tidal deformability raises ValueError before reaching LAL."""
    with pytest.raises(ValueError, match=f"{param} must be >= 0"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=1_126_259_462.4,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=36.0,
            detector_frame_mass_2=29.0,
            luminosity_distance=410.0,
            **{param: -1.0},
        )


def test_lal_backend_tidal_alias_conflict_raises() -> None:
    """Passing lambda_1 and tidal_1 simultaneously raises ValueError."""
    with pytest.raises(ValueError, match="Do not mix aliases"):
        LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=1_126_259_462.4,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=36.0,
            detector_frame_mass_2=29.0,
            luminosity_distance=410.0,
            lambda_1=500.0,
            tidal_1=500.0,
        )


_COMMON_PARAMS: dict[str, object] = {
    "mass1": 1.4,
    "mass2": 1.4,
    "distance": 100.0,
}


def _fake_result() -> dict[str, TimeSeries]:
    import numpy as np

    ts = TimeSeries(np.zeros(10), t0=0.0, dt=1.0 / 4096)
    return {"plus": ts, "cross": ts}


def test_pycbc_backend_translates_lambda_1_to_lambda1() -> None:
    """lambda_1/lambda_2 (underscore) are renamed to lambda1/lambda2 for pycbc."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    with patch(
        "gwmock_signal.waveform.pycbc_wrapper.pycbc_waveform_wrapper",
        return_value=_fake_result(),
    ) as mock_wrapper:
        PyCBCBackend().generate_td_waveform(
            "IMRPhenomPv2_NRTidalv2",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            lambda_1=500.0,
            lambda_2=300.0,
            **_COMMON_PARAMS,
        )
    kw = mock_wrapper.call_args.kwargs
    assert kw["lambda1"] == 500.0
    assert kw["lambda2"] == 300.0
    assert "lambda_1" not in kw
    assert "lambda_2" not in kw


def test_pycbc_backend_tidal_aliases_accepted() -> None:
    """tidal_1/tidal_2 are accepted as aliases and mapped to lambda1/lambda2."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    with patch(
        "gwmock_signal.waveform.pycbc_wrapper.pycbc_waveform_wrapper",
        return_value=_fake_result(),
    ) as mock_wrapper:
        PyCBCBackend().generate_td_waveform(
            "IMRPhenomPv2_NRTidalv2",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            tidal_1=500.0,
            tidal_2=300.0,
            **_COMMON_PARAMS,
        )
    kw = mock_wrapper.call_args.kwargs
    assert kw["lambda1"] == 500.0
    assert kw["lambda2"] == 300.0
    assert "tidal_1" not in kw
    assert "tidal_2" not in kw


def test_pycbc_backend_tidal_defaults_to_zero() -> None:
    """Omitting tidal params passes lambda1=0.0, lambda2=0.0 to pycbc."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    with patch(
        "gwmock_signal.waveform.pycbc_wrapper.pycbc_waveform_wrapper",
        return_value=_fake_result(),
    ) as mock_wrapper:
        PyCBCBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            **_COMMON_PARAMS,
        )
    kw = mock_wrapper.call_args.kwargs
    assert kw["lambda1"] == 0.0
    assert kw["lambda2"] == 0.0


def test_pycbc_backend_tidal_alias_conflict_raises() -> None:
    """Passing lambda_1 and tidal_1 simultaneously raises ValueError."""
    pytest.importorskip("pycbc", reason="pycbc not installed")
    with pytest.raises(ValueError, match="Do not mix aliases"):
        PyCBCBackend().generate_td_waveform(
            "IMRPhenomPv2_NRTidalv2",
            tc=0.0,
            sampling_frequency=4096.0,
            minimum_frequency=20.0,
            lambda_1=500.0,
            tidal_1=500.0,
            **_COMMON_PARAMS,
        )


def test_top_level_backend_import_succeeds_without_pycbc() -> None:
    """Top-level backend exports do not require importing PyCBC."""
    code = """
import builtins

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("pycbc"):
        raise ModuleNotFoundError("No module named 'pycbc'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
from gwmock_signal import LALSimulationBackend, WaveformBackend
assert WaveformBackend.__name__ == "WaveformBackend"
assert LALSimulationBackend.__name__ == "LALSimulationBackend"
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
        # Generous because it guards against a hang, not against slowness: the subprocess pays a
        # cold interpreter start plus the astropy and lalsuite imports, which on a loaded macOS
        # runner sharing cores with the xdist workers has exceeded 30 s. What the test asserts is
        # that the import *succeeds* without PyCBC; how long it takes is not the claim.
        timeout=300,
    )
    assert result.returncode == 0, result.stderr or result.stdout


# --- Coalescence-time convention -------------------------------------------------
# The LAL backend evaluates the waveform in the frequency domain (bilby's approach)
# so coalescence sits at the FD phase reference, matching the ripple backend. These
# tests pin that convention: they fail if the backend regresses to
# SimInspiralChooseTDWaveform, whose epoch is re-pinned to the amplitude peak.

_TC = 1_126_259_462.4
_FS = 2048.0
_F_MIN = 20.0
_CONVENTION_SOURCE: dict[str, float] = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "luminosity_distance": 410.0,
    "spin_1z": 0.4,
    "spin_2z": -0.3,
    "inclination": 0.5,
    "coa_phase": 1.2,
}


def test_lal_fd_td_roundtrip() -> None:
    """FFT of the conditioned TD waveform reproduces LAL's FD waveform to machine precision.

    Guards the coalescence-placement convention: the LAL backend must place
    coalescence at the FD phase reference (like the ripple backend and bilby), so a
    signal injected at ``tc`` Fourier-transforms back to the frequency-domain
    template. ``SimInspiralChooseTDWaveform`` (amplitude-peak epoch) would fail this.
    """
    import lal
    import lalsimulation

    from gwmock_signal.waveform.backends import conditioning

    td = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD", tc=_TC, sampling_frequency=_FS, minimum_frequency=_F_MIN, **_CONVENTION_SOURCE
    )["plus"]

    # Reconstruct LAL's FD template on the same grid the backend used.
    m1, m2 = _CONVENTION_SOURCE["detector_frame_mass_1"], _CONVENTION_SOURCE["detector_frame_mass_2"]
    chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    n_samples = conditioning.segment_sample_count(chirp_mass, _F_MIN, _FS)
    delta_f = _FS / n_samples
    hp_fd, _ = lalsimulation.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0,
        0.0,
        _CONVENTION_SOURCE["spin_1z"],
        0.0,
        0.0,
        _CONVENTION_SOURCE["spin_2z"],
        _CONVENTION_SOURCE["luminosity_distance"] * lal.PC_SI * 1e6,
        _CONVENTION_SOURCE["inclination"],
        _CONVENTION_SOURCE["coa_phase"],
        0.0,
        0.0,
        0.0,
        delta_f,
        _F_MIN,
        _FS / 2,
        _F_MIN,
        lal.CreateDict(),
        lalsimulation.GetApproximantFromString("IMRPhenomD"),
    )
    template = np.asarray(hp_fd.data.data)
    freqs = np.arange(len(template)) * delta_f
    in_band = freqs >= _F_MIN

    epoch = float(td.t0.value) - _TC
    # FFT the TD samples and undo the epoch shift; this must recover the FD template.
    recon = np.exp(-2j * np.pi * freqs * epoch) * np.fft.rfft(td.value, n=n_samples) / _FS
    a, b = recon[in_band], template[in_band]
    overlap = np.real(np.sum(a * np.conj(b))) / np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    assert 1.0 - overlap < 1e-6, f"FFT(TD) vs FD overlap {overlap:.8f}"


def test_lal_coalescence_at_fd_phase_reference_not_amplitude_peak() -> None:
    """LAL places coalescence at the FD phase reference, so the |h| peak precedes tc.

    Regression guard against reverting to ``SimInspiralChooseTDWaveform``, which
    would pin the amplitude peak onto ``tc`` (peak offset ~0) and trip this test.
    """
    hp = LALSimulationBackend().generate_td_waveform(
        "IMRPhenomD", tc=_TC, sampling_frequency=_FS, minimum_frequency=_F_MIN, **_CONVENTION_SOURCE
    )["plus"]
    peak_offset = float(hp.times.value[int(np.argmax(np.abs(hp.value)))]) - _TC
    assert peak_offset < -2.0e-3, (
        f"LAL peak offset {peak_offset * 1e3:+.3f} ms should be well before tc; "
        "coalescence must sit at the FD phase reference, not the amplitude peak"
    )


def test_lal_and_ripple_agree_on_coalescence() -> None:
    """LAL and ripple place the same source at the same sample (one shared tc convention).

    Both backends reference coalescence to the FD phase reference and share the same
    segment sizing, so the amplitude peaks coincide to within a sample -- the
    mass-dependent ~12 M offset from the old ChooseTDWaveform epoch is gone.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")

    from gwmock_signal.waveform.backends import RippleBackend

    common = {"tc": _TC, "sampling_frequency": _FS, "minimum_frequency": _F_MIN, **_CONVENTION_SOURCE}
    lal_hp = LALSimulationBackend().generate_td_waveform("IMRPhenomD", **common)["plus"]
    rip_hp = RippleBackend().generate_td_waveform("IMRPhenomD", **common)["plus"]

    # Same auto-sized grid (MTSUN and segment sizing are shared).
    assert lal_hp.value.size == rip_hp.value.size
    lal_peak = float(lal_hp.times.value[int(np.argmax(np.abs(lal_hp.value)))])
    rip_peak = float(rip_hp.times.value[int(np.argmax(np.abs(rip_hp.value)))])
    assert abs(lal_peak - rip_peak) < 1.5 / _FS, (
        f"LAL and ripple coalescence disagree by {(lal_peak - rip_peak) * 1e3:.3f} ms "
        "(> 1 sample); the two backends must share the FD-phase-reference convention"
    )
