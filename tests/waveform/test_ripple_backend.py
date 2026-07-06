"""Tests for the ripple (JAX) waveform backend."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.waveform.backends import RippleBackend

_BBH_PARAMS: dict[str, float] = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "luminosity_distance": 410.0,
    "inclination": 0.4,
}
_TC = 1_126_259_462.4
_FS = 2048.0
_F_MIN = 20.0


def _generate(**overrides: object) -> dict[str, TimeSeries]:
    """Run a default IMRPhenomD ripple waveform with optional parameter overrides."""
    params: dict[str, object] = {**_BBH_PARAMS, **overrides}
    return RippleBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=_TC,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        **params,
    )


def test_ripple_backend_raises_helpful_import_error_when_ripple_missing() -> None:
    """RippleBackend fails at instantiation time with installation guidance."""
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name.startswith("ripplegw") or name == "jax":
            raise ImportError(f"{name} unavailable")
        return real_import_module(name, package)

    with (
        patch("gwmock_signal.waveform.backends.ripple.importlib.import_module", side_effect=_import_module),
        pytest.raises(ImportError, match=r"gwmock-signal\[jax\]"),
    ):
        RippleBackend()


def test_ripple_backend_rejects_invalid_ringdown_fraction() -> None:
    """ringdown_fraction outside (0, 1) raises ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match="ringdown_fraction"):
        RippleBackend(ringdown_fraction=1.0)


def test_ripple_backend_available_approximants() -> None:
    """The backend advertises every supported model."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    assert set(RippleBackend().available_approximants()) == {
        "IMRPhenomD",
        "IMRPhenomHM",
        "IMRPhenomXAS",
        "IMRPhenomXHM",
        "TaylorF2",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS_NRTidalv3",
        "IMRPhenomPv2",
        "IMRPhenomXP",
        "IMRPhenomXPHM",
    }


def test_ripple_backend_rejects_unsupported_approximant() -> None:
    """An unsupported approximant raises a helpful ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match="does not support approximant"):
        RippleBackend().generate_td_waveform(
            "SineGaussian",
            tc=_TC,
            sampling_frequency=_FS,
            minimum_frequency=_F_MIN,
            **_BBH_PARAMS,
        )


def test_ripple_backend_generates_timeseries_dict() -> None:
    """A minimal ripple waveform call returns GWpy time series with the right grid."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = _generate()
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)
    assert isinstance(result["cross"], TimeSeries)
    assert result["plus"].dt.value == pytest.approx(1.0 / _FS)


def test_ripple_backend_places_coalescence_at_tc() -> None:
    """The plus-polarization peak lands near the requested coalescence time."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    hp = _generate()["plus"]
    peak_time = float(hp.times.value[int(np.argmax(np.abs(hp.value)))])
    assert peak_time == pytest.approx(_TC, abs=0.1)


def test_ripple_backend_rejects_in_plane_spin() -> None:
    """IMRPhenomD is aligned-spin only; nonzero in-plane spin raises ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match="in-plane spins must be zero"):
        _generate(spin_1x=0.3)


def test_ripple_backend_precessing_accepts_in_plane_spin() -> None:
    """A precessing model accepts in-plane spins and returns time series."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = RippleBackend().generate_td_waveform(
        "IMRPhenomPv2",
        tc=_TC,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        detector_frame_mass_1=40.0,
        detector_frame_mass_2=30.0,
        luminosity_distance=400.0,
        spin_1x=0.3,
        spin_1y=0.1,
        spin_1z=0.2,
        spin_2x=-0.1,
        spin_2y=0.2,
        spin_2z=0.1,
        inclination=0.6,
    )
    assert set(result) == {"plus", "cross"}
    assert np.all(np.isfinite(result["plus"].value))


def test_ripple_backend_rejects_tidal_params() -> None:
    """IMRPhenomD has no tidal sector; nonzero lambda raises ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match="does not support tidal"):
        _generate(lambda_1=500.0)


def test_ripple_backend_rejects_unknown_param() -> None:
    """Unrecognized waveform parameters raise ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match="Unsupported ripple waveform parameters"):
        _generate(not_a_param=1.0)


_BNS_PARAMS: dict[str, float] = {
    "detector_frame_mass_1": 1.6,
    "detector_frame_mass_2": 1.4,
    "luminosity_distance": 100.0,
    "inclination": 0.6,
}


def _generate_tidal(approximant: str, **overrides: object) -> dict[str, TimeSeries]:
    """Run a default BNS tidal waveform with optional parameter overrides."""
    params: dict[str, object] = {**_BNS_PARAMS, **overrides}
    return RippleBackend().generate_td_waveform(
        approximant,
        tc=_TC,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        **params,
    )


def test_ripple_backend_tidal_model_accepts_lambda() -> None:
    """An NRTidal model accepts lambda_1/lambda_2 and returns time series."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = _generate_tidal("IMRPhenomD_NRTidalv2", lambda_1=400.0, lambda_2=500.0)
    assert set(result) == {"plus", "cross"}
    assert isinstance(result["plus"], TimeSeries)


def test_ripple_backend_tidal_accepts_tidal_aliases() -> None:
    """tidal_1/tidal_2 are accepted as aliases for lambda_1/lambda_2."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = _generate_tidal("IMRPhenomXAS_NRTidalv3", tidal_1=400.0, tidal_2=500.0)
    assert set(result) == {"plus", "cross"}


@pytest.mark.parametrize("param", ["lambda_1", "lambda_2"])
def test_ripple_backend_negative_tidal_raises(param: str) -> None:
    """Negative tidal deformability raises ValueError before reaching ripple."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    with pytest.raises(ValueError, match=f"{param} must be >= 0"):
        _generate_tidal("IMRPhenomD_NRTidalv2", **{param: -1.0})


def _match(a: np.ndarray, b: np.ndarray, sampling_frequency: float, f_min: float) -> float:
    """White (flat-PSD) match between two real time series, maximized over time and phase."""
    n = 1 << (int(np.ceil(np.log2(max(len(a), len(b))))) + 1)
    spectrum_a = np.fft.rfft(a, n=n)
    spectrum_b = np.fft.rfft(b, n=n)
    in_band = np.fft.rfftfreq(n, d=1.0 / sampling_frequency) >= f_min
    spectrum_a = np.where(in_band, spectrum_a, 0.0)
    spectrum_b = np.where(in_band, spectrum_b, 0.0)
    cross = spectrum_a * np.conj(spectrum_b)
    full = np.zeros(n, dtype=complex)
    full[: len(cross)] = cross
    correlation = np.fft.ifft(full) * n  # complex overlap as a function of time shift
    norm = np.sqrt(np.sum(np.abs(spectrum_a) ** 2) * np.sum(np.abs(spectrum_b) ** 2))
    return float(np.max(np.abs(correlation)) / norm)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("approximant", "mass1", "mass2", "chi1", "chi2", "iota"),
    [
        # One representative configuration per supported aligned-spin model.
        ("IMRPhenomD", 40.0, 31.0, 0.5, -0.2, 0.9),
        ("IMRPhenomHM", 40.0, 31.0, 0.5, -0.2, 0.9),
        ("IMRPhenomXAS", 40.0, 31.0, 0.5, -0.2, 0.9),
        ("IMRPhenomXHM", 40.0, 31.0, 0.5, -0.2, 0.9),
        # Extra mass/spin coverage on the baseline model.
        ("IMRPhenomD", 36.0, 29.0, 0.0, 0.0, 0.4),
        ("IMRPhenomD", 1.6, 1.3, 0.0, 0.0, 0.7),
    ],
)
def test_ripple_matches_lal(  # noqa: PLR0913
    approximant: str, mass1: float, mass2: float, chi1: float, chi2: float, iota: float
) -> None:
    """Ripple agrees with the LAL implementation of the same approximant (white match > 0.99).

    This is the anchor that validates the frequency->time conditioning against an
    external reference rather than only internal consistency.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    from gwmock_signal.waveform.backends import LALSimulationBackend

    common = {
        "tc": _TC,
        "sampling_frequency": _FS,
        "minimum_frequency": _F_MIN,
        "detector_frame_mass_1": mass1,
        "detector_frame_mass_2": mass2,
        "luminosity_distance": 400.0,
        "spin_1z": chi1,
        "spin_2z": chi2,
        "inclination": iota,
    }
    ripple = RippleBackend().generate_td_waveform(approximant, **common)
    lal = LALSimulationBackend().generate_td_waveform(approximant, **common)

    for pol in ("plus", "cross"):
        match = _match(ripple[pol].value, lal[pol].value, _FS, _F_MIN)
        assert match > 0.99, f"{approximant} {pol} match {match:.4f} below threshold"


# f_min=40 Hz keeps the BNS segment short enough for a fast integration test.
_TIDAL_F_MIN = 40.0


@pytest.mark.integration
@pytest.mark.parametrize("approximant", ["IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3"])
def test_ripple_tidal_matches_lal(approximant: str) -> None:
    """Ripple NRTidal models agree with LAL, including the tidal sector (white match > 0.99)."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    from gwmock_signal.waveform.backends import LALSimulationBackend

    common = {
        "tc": _TC,
        "sampling_frequency": _FS,
        "minimum_frequency": _TIDAL_F_MIN,
        "detector_frame_mass_1": 1.6,
        "detector_frame_mass_2": 1.4,
        "luminosity_distance": 100.0,
        "spin_1z": 0.02,
        "spin_2z": -0.01,
        "inclination": 0.6,
        "lambda_1": 400.0,
        "lambda_2": 500.0,
    }
    ripple = RippleBackend().generate_td_waveform(approximant, **common)
    lal = LALSimulationBackend().generate_td_waveform(approximant, **common)

    for pol in ("plus", "cross"):
        match = _match(ripple[pol].value, lal[pol].value, _FS, _TIDAL_F_MIN)
        assert match > 0.99, f"{approximant} {pol} match {match:.4f} below threshold"


def _fd_match(spectrum_a: np.ndarray, spectrum_b: np.ndarray, in_band: np.ndarray) -> float:
    """White FD match between two spectra over ``in_band``, maximized over time and phase."""
    spectrum_a = np.where(in_band, spectrum_a, 0.0)
    spectrum_b = np.where(in_band, spectrum_b, 0.0)
    cross = spectrum_a * np.conj(spectrum_b)
    n = 2 * (len(cross) - 1)
    full = np.zeros(n, dtype=complex)
    full[: len(cross)] = cross
    correlation = np.fft.ifft(full) * n
    norm = np.sqrt(np.sum(np.abs(spectrum_a) ** 2) * np.sum(np.abs(spectrum_b) ** 2))
    return float(np.max(np.abs(correlation)) / norm)


def test_ripple_taylorf2_generates_finite_timeseries() -> None:
    """TaylorF2 conditions to a finite time series despite ripple's above-ISCO NaNs."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    result = _generate_tidal("TaylorF2", lambda_1=400.0, lambda_2=500.0)
    assert set(result) == {"plus", "cross"}
    assert np.all(np.isfinite(result["plus"].value))
    assert np.all(np.isfinite(result["cross"].value))


@pytest.mark.integration
def test_ripple_taylorf2_matches_lal_fd() -> None:
    """Ripple TaylorF2 agrees with LAL's frequency-domain TaylorF2 (match > 0.99).

    LAL provides no time-domain TaylorF2 generator, so the backend's conditioned
    time series is transformed back to the frequency domain and compared against
    ``SimInspiralChooseFDWaveform`` over the inspiral band (up to the ISCO).
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import lal
    import lalsimulation

    mass1, mass2 = 1.6, 1.4
    chi1, chi2, lambda_1, lambda_2, iota, phic = 0.02, -0.01, 400.0, 500.0, 0.6, 0.2
    hp = RippleBackend().generate_td_waveform(
        "TaylorF2",
        tc=_TC,
        sampling_frequency=_FS,
        minimum_frequency=_TIDAL_F_MIN,
        detector_frame_mass_1=mass1,
        detector_frame_mass_2=mass2,
        luminosity_distance=100.0,
        spin_1z=chi1,
        spin_2z=chi2,
        inclination=iota,
        coa_phase=phic,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
    )["plus"]

    n = len(hp.value)
    delta_f = _FS / n
    ripple_fd = np.fft.rfft(hp.value)
    freqs = np.fft.rfftfreq(n, d=1.0 / _FS)

    pars = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(pars, lambda_1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(pars, lambda_2)
    lal_hp, _ = lalsimulation.SimInspiralChooseFDWaveform(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        0.0,
        0.0,
        chi1,
        0.0,
        0.0,
        chi2,
        100.0 * lal.PC_SI * 1e6,
        iota,
        phic,
        0.0,
        0.0,
        0.0,
        delta_f,
        _TIDAL_F_MIN,
        _FS / 2,
        _TIDAL_F_MIN,
        pars,
        lalsimulation.GetApproximantFromString("TaylorF2"),
    )
    lal_fd = np.asarray(lal_hp.data.data)

    n_bins = min(len(ripple_fd), len(lal_fd))
    # TaylorF2 terminates at the Schwarzschild ISCO; compare only the inspiral band.
    f_isco = 1.0 / (6.0**1.5 * np.pi * (mass1 + mass2) * lal.MTSUN_SI)
    in_band = (freqs[:n_bins] >= _TIDAL_F_MIN) & (freqs[:n_bins] <= min(f_isco, _FS / 2))
    match = _fd_match(ripple_fd[:n_bins], lal_fd[:n_bins], in_band)
    assert match > 0.99, f"TaylorF2 FD match {match:.4f} below threshold"


@pytest.mark.integration
@pytest.mark.parametrize("approximant", ["IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"])
def test_ripple_precessing_matches_lal(approximant: str) -> None:
    """Ripple precessing models agree with LAL, including in-plane spins (white match > 0.99)."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    from gwmock_signal.waveform.backends import LALSimulationBackend

    common = {
        "tc": _TC,
        "sampling_frequency": _FS,
        "minimum_frequency": _F_MIN,
        "detector_frame_mass_1": 40.0,
        "detector_frame_mass_2": 30.0,
        "luminosity_distance": 400.0,
        "spin_1x": 0.3,
        "spin_1y": 0.1,
        "spin_1z": 0.2,
        "spin_2x": -0.1,
        "spin_2y": 0.2,
        "spin_2z": 0.1,
        "inclination": 0.6,
        "coa_phase": 0.2,
    }
    ripple = RippleBackend().generate_td_waveform(approximant, **common)
    lal = LALSimulationBackend().generate_td_waveform(approximant, **common)

    for pol in ("plus", "cross"):
        match = _match(ripple[pol].value, lal[pol].value, _FS, _F_MIN)
        assert match > 0.99, f"{approximant} {pol} match {match:.4f} below threshold"


def test_generate_fd_polarizations_grid_and_masking() -> None:
    """generate_fd_polarizations returns on-device FD arrays on a valid one-sided grid."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import jax

    fd = RippleBackend().generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, **_BBH_PARAMS
    )
    assert fd.sampling_frequency == _FS
    assert fd.frequencies.shape == (fd.n_samples // 2 + 1,)
    assert fd.plus.shape == fd.frequencies.shape == fd.cross.shape
    assert isinstance(fd.plus, jax.Array)  # stays on device (not converted to NumPy)

    freqs = np.asarray(fd.frequencies)
    assert freqs[0] == 0.0
    assert np.allclose(np.diff(freqs), _FS / fd.n_samples)
    # Out-of-band bins (including DC) are zeroed and the result is finite.
    assert np.all(np.asarray(fd.plus)[freqs < _F_MIN] == 0.0)
    assert np.all(np.isfinite(np.asarray(fd.plus)))


def test_generate_fd_polarizations_conditions_to_td_waveform() -> None:
    """Inverse-FFTing and placing the FD output reproduces generate_td_waveform exactly."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    ringdown_fraction = 0.1
    backend = RippleBackend(ringdown_fraction=ringdown_fraction)
    fd = backend.generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, **_BBH_PARAMS
    )
    td = backend.generate_td_waveform(
        "IMRPhenomD", tc=_TC, sampling_frequency=_FS, minimum_frequency=_F_MIN, **_BBH_PARAMS
    )

    dt = 1.0 / _FS
    merger_index = round((1.0 - ringdown_fraction) * fd.n_samples)
    for pol, key in ((fd.plus, "plus"), (fd.cross, "cross")):
        reconstructed = np.roll(np.fft.irfft(np.asarray(pol), n=fd.n_samples) / dt, merger_index)
        np.testing.assert_allclose(reconstructed, td[key].value, rtol=0.0, atol=0.0)


@pytest.mark.integration
def test_generate_fd_polarizations_matches_lal_fd() -> None:
    """Ripple's frequency-domain IMRPhenomD agrees with LAL's FD IMRPhenomD (match > 0.99)."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import lal
    import lalsimulation

    mass1, mass2, chi1, chi2, iota, phic = 40.0, 31.0, 0.5, -0.2, 0.9, 0.3
    fd = RippleBackend().generate_fd_polarizations(
        "IMRPhenomD",
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        detector_frame_mass_1=mass1,
        detector_frame_mass_2=mass2,
        luminosity_distance=400.0,
        spin_1z=chi1,
        spin_2z=chi2,
        inclination=iota,
        coa_phase=phic,
    )
    freqs = np.asarray(fd.frequencies)
    delta_f = _FS / fd.n_samples
    lal_hp, _ = lalsimulation.SimInspiralChooseFDWaveform(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        0.0,
        0.0,
        chi1,
        0.0,
        0.0,
        chi2,
        400.0 * lal.PC_SI * 1e6,
        iota,
        phic,
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
    lal_fd = np.asarray(lal_hp.data.data)
    n_bins = min(len(freqs), len(lal_fd))
    in_band = freqs[:n_bins] >= _F_MIN
    match = _fd_match(np.asarray(fd.plus)[:n_bins], lal_fd[:n_bins], in_band)
    assert match > 0.99, f"FD match {match:.4f} below threshold"


_BATCH_EVENTS = [
    {
        "detector_frame_mass_1": 40.0,
        "detector_frame_mass_2": 31.0,
        "luminosity_distance": 400.0,
        "spin_1z": 0.5,
        "spin_2z": -0.2,
        "inclination": 0.9,
        "coa_phase": 0.3,
    },
    {
        "detector_frame_mass_1": 36.0,
        "detector_frame_mass_2": 29.0,
        "luminosity_distance": 410.0,
        "spin_1z": 0.0,
        "spin_2z": 0.0,
        "inclination": 0.4,
        "coa_phase": 0.0,
    },
    {
        "detector_frame_mass_1": 55.0,
        "detector_frame_mass_2": 48.0,
        "luminosity_distance": 800.0,
        "spin_1z": 0.2,
        "spin_2z": 0.1,
        "inclination": 1.2,
        "coa_phase": 1.0,
    },
]


def _as_struct_of_arrays(events: list[dict]) -> dict:
    return {key: np.array([event[key] for event in events]) for key in events[0]}


def test_generate_fd_polarizations_batch_matches_per_event() -> None:
    """Each row of the batched FD evaluation equals the per-event call on the same grid."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    backend = RippleBackend(segment_duration=8.0)  # fixed grid so both paths share n_samples
    batch = backend.generate_fd_polarizations_batch(
        "IMRPhenomD",
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=_as_struct_of_arrays(_BATCH_EVENTS),
    )
    assert batch.plus.shape == (len(_BATCH_EVENTS), batch.n_samples // 2 + 1)
    for i, event in enumerate(_BATCH_EVENTS):
        single = backend.generate_fd_polarizations(
            "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, **event
        )
        for batched_pol, single_pol in ((batch.plus[i], single.plus), (batch.cross[i], single.cross)):
            diff = np.max(np.abs(np.asarray(batched_pol) - np.asarray(single_pol)))
            assert diff < 1e-9 * np.max(np.abs(np.asarray(single_pol)))


def test_generate_fd_polarizations_batch_sizes_worst_case() -> None:
    """Without a fixed segment_duration the batch grid matches the longest (lightest) event."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    backend = RippleBackend()
    batch = backend.generate_fd_polarizations_batch(
        "IMRPhenomD",
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        parameters=_as_struct_of_arrays(_BATCH_EVENTS),
    )
    # The lightest pair (36 + 29) has the longest chirp time and sets the segment length.
    lightest = min(_BATCH_EVENTS, key=lambda e: e["detector_frame_mass_1"] + e["detector_frame_mass_2"])
    single = backend.generate_fd_polarizations(
        "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, **lightest
    )
    assert batch.n_samples == single.n_samples


def test_generate_fd_polarizations_batch_rejects_in_plane_spin_for_aligned() -> None:
    """An aligned-spin model rejects any nonzero in-plane spin in the batch."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    params = _as_struct_of_arrays(_BATCH_EVENTS)
    params["spin_1x"] = np.array([0.0, 0.3, 0.0])  # one event has in-plane spin
    with pytest.raises(ValueError, match="spin_1x must be zero"):
        RippleBackend().generate_fd_polarizations_batch(
            "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, parameters=params
        )


def test_generate_fd_polarizations_batch_rejects_length_mismatch() -> None:
    """Mismatched parameter array lengths raise ValueError."""
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    params = _as_struct_of_arrays(_BATCH_EVENTS)
    params["spin_2z"] = np.array([0.0, 0.0])  # wrong length
    with pytest.raises(ValueError, match="expected"):
        RippleBackend().generate_fd_polarizations_batch(
            "IMRPhenomD", sampling_frequency=_FS, minimum_frequency=_F_MIN, parameters=params
        )


# Frequency-domain <-> time-domain consistency: the Fourier transform of the
# conditioned TD waveform (injected at tc) must reproduce the FD waveform, so
# simulated data stays consistent with frequency-domain inference. LAL's
# ChooseTDWaveform re-pins the epoch to the amplitude peak and does NOT satisfy
# this; gwmock-signal places coalescence at the FD phase reference and must.
_ROUNDTRIP_CASES = [
    ("IMRPhenomD", {"spin_1z": 0.5, "spin_2z": -0.2}, 20.0),
    ("IMRPhenomHM", {"spin_1z": 0.5, "spin_2z": -0.2}, 20.0),
    ("IMRPhenomXAS", {"spin_1z": 0.5, "spin_2z": -0.2}, 20.0),
    ("IMRPhenomXHM", {"spin_1z": 0.5, "spin_2z": -0.2}, 20.0),
    (
        "IMRPhenomD_NRTidalv2",
        {"detector_frame_mass_1": 1.6, "detector_frame_mass_2": 1.4, "lambda_1": 400.0, "lambda_2": 500.0},
        40.0,
    ),
    ("IMRPhenomPv2", {"spin_1x": 0.3, "spin_1y": 0.1, "spin_1z": 0.2, "spin_2x": -0.1, "spin_2y": 0.2}, 20.0),
    ("IMRPhenomXP", {"spin_1x": 0.3, "spin_1y": 0.1, "spin_1z": 0.2, "spin_2x": -0.1, "spin_2y": 0.2}, 20.0),
]


@pytest.mark.parametrize(
    ("approximant", "extra", "f_min"), _ROUNDTRIP_CASES, ids=[case[0] for case in _ROUNDTRIP_CASES]
)
def test_ripple_fd_td_roundtrip(approximant: str, extra: dict, f_min: float) -> None:
    """FFT of the conditioned TD waveform reproduces the FD waveform to machine precision.

    Guards the coalescence-placement convention: a signal injected at ``tc`` must
    Fourier-transform back to the frequency-domain template (``FFT(TD) == FD``), so
    simulated data is consistent with frequency-domain inference.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    backend = RippleBackend()
    params = {**_BBH_PARAMS, **extra}
    common = {"sampling_frequency": _FS, "minimum_frequency": f_min}
    fd = backend.generate_fd_polarizations(approximant, **common, **params)
    td = backend.generate_td_waveform(approximant, tc=_TC, **common, **params)

    freqs = np.asarray(fd.frequencies)
    in_band = freqs >= f_min
    for polarization in ("plus", "cross"):
        template = np.asarray(getattr(fd, polarization))[in_band]
        series = td[polarization]
        epoch = float(series.t0.value) - _TC
        # FFT the TD samples and undo the epoch shift; this must recover the FD waveform.
        recon = (np.exp(-2j * np.pi * freqs * epoch) * np.fft.rfft(series.value, n=fd.n_samples) / _FS)[in_band]
        overlap = np.real(np.sum(recon * np.conj(template))) / np.sqrt(
            np.sum(np.abs(recon) ** 2) * np.sum(np.abs(template) ** 2)
        )
        assert 1.0 - overlap < 1e-6, f"{approximant} {polarization}: FFT(TD) vs FD overlap {overlap:.8f}"


def test_ripple_coalescence_sits_at_fd_phase_reference_not_amplitude_peak() -> None:
    """Pin the deliberate coalescence convention: ``tc`` is the FD phase reference.

    This is the flip side of :func:`test_ripple_fd_td_roundtrip`. gwmock-signal
    places coalescence at ripple's frequency-domain phase reference (so that
    ``FFT(TD) == FD`` for frequency-domain inference), which is **not** the (2,2)
    amplitude peak: for a heavy BBH the peak lands earlier than ``tc`` by a
    mass-scaled merger/ringdown offset (~12 M).

    Regression guard: a future edit that "aligns" ripple to the amplitude peak
    (e.g. rolling by the peak index in ``_to_time_domain``) would silently break
    the FD/TD roundtrip guarantee. That edit moves the ripple peak onto ``tc`` and
    trips this test. Do not "fix" this peak offset; it is intentional, and the LAL
    backend deliberately uses the same convention (see
    ``test_lal_and_ripple_agree_on_coalescence`` in ``test_backends.py``).
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")

    hp = RippleBackend().generate_td_waveform(
        "IMRPhenomD",
        tc=_TC,
        sampling_frequency=_FS,
        minimum_frequency=_F_MIN,
        # A heavy BBH keeps the peak-vs-phase-reference offset many samples wide.
        detector_frame_mass_1=36.0,
        detector_frame_mass_2=29.0,
        luminosity_distance=410.0,
        inclination=0.4,
    )["plus"]

    peak_offset = float(hp.times.value[int(np.argmax(np.abs(hp.value)))]) - _TC
    # The peak sits well before tc; coalescence is the FD phase reference, not the peak.
    assert peak_offset < -2.0e-3, (
        f"ripple peak offset {peak_offset * 1e3:+.3f} ms should be well before tc; "
        "coalescence must stay at the FD phase reference, not the amplitude peak"
    )
