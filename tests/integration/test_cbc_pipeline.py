"""End-to-end integration tests for the CBC injection pipeline."""

from __future__ import annotations

try:
    import lalsimulation
except ModuleNotFoundError as exc:  # Python 3.15: lalsuite does not install yet; its tests are skipped
    if exc.name != "lalsimulation":
        raise
    lalsimulation = None
import numpy as np
import pytest

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

from gwmock_signal.network import Network
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.projection.network import _time_delay_from_earth_center_lal
from gwmock_signal.waveform.backends import LALSimulationBackend

# SciPy arrives with GWpy, which does not install on Python 3.15 yet; every test here also needs lalsuite.
correlate = pytest.importorskip("scipy.signal").correlate

# ---------------------------------------------------------------------------
# GW150914-like injection parameters
# ---------------------------------------------------------------------------

FS = 4096.0  # Hz
FMIN = 20.0  # Hz
DURATION = 8.0  # seconds
POST_MERGER_PADDING = 0.05  # 50 ms — accommodates time_delay_from_earth_center offsets

PARAMS: dict = {
    "detector_frame_mass_1": 36.0,
    "detector_frame_mass_2": 29.0,
    "spin_1z": 0.0,
    "spin_2z": 0.0,
    "distance": 410.0,
    "right_ascension": 1.375,
    "declination": -1.211,
    "polarization_angle": 2.659,
    "inclination": 2.5,
    "coa_phase": 0.0,
    "coa_time": 1126259462.4,
}

RA = PARAMS["right_ascension"]
DEC = PARAMS["declination"]
TC = PARAMS["coa_time"]

# Historical PyCBC reference for the H1 matched-filter SNR against the aLIGO
# design PSD (P1200087). The LAL-only regression keeps the same value to better
# than 1e-6 relative precision.
_PYCBC_REFERENCE_SNR = 4.884168e01
_REFERENCE_SNR = 4.884168e01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_background(detector_names: tuple[str, ...]) -> dict[str, TimeSeries]:
    """Return a zero-noise background for each detector in the tuple.

    Extends 50 ms past TC so that the signal peak (which arrives at TC +
    time_delay_from_earth_center, up to ~21 ms) is not truncated.
    """
    n_samples = int((DURATION + POST_MERGER_PADDING) * FS)
    t0 = TC - DURATION
    return {name: TimeSeries(np.zeros(n_samples), t0=t0, sample_rate=FS) for name in detector_names}


def _matched_filter_sigma_lal(ts: TimeSeries) -> float:
    """Return the matched-filter sigma using the LAL design PSD helper."""
    data = np.asarray(ts.value, dtype=float)
    dt = ts.dt
    freqs = np.fft.rfftfreq(len(data), d=dt)
    if len(freqs) < 2:
        return 0.0
    hf = np.fft.rfft(data) * dt
    df = freqs[1] - freqs[0]
    psd = np.array(
        [lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(float(f)) if f > 0 else np.inf for f in freqs],
        dtype=float,
    )
    mask = freqs >= FMIN
    sigma_sq = 4.0 * np.sum(np.abs(hf[mask]) ** 2 / psd[mask]) * df
    return float(np.sqrt(sigma_sq))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_gwpy
@pytest.mark.integration
def test_cbc_h1l1v1_returns_stack():
    """inject_cbc_signal on H1L1V1 returns a 3-channel stack with non-zero RMS."""
    detectors = Network.from_name("H1L1V1").detector_names
    background = _make_background(detectors)

    stack = inject_cbc_signal(
        "IMRPhenomD",
        PARAMS,
        detectors,
        background,
        sampling_frequency=FS,
        minimum_frequency=FMIN,
        waveform_backend=LALSimulationBackend(),
    )

    assert len(stack) == 3
    for name in detectors:
        rms = float(np.sqrt(np.mean(stack[name].value ** 2)))
        assert rms > 0.0, f"RMS of {name} channel is zero"


@pytest.mark.requires_gwpy
@pytest.mark.integration
def test_h1l1_time_delay_consistent():
    """Cross-correlation lag between H1 and L1 agrees with analytic time delay."""
    detectors = ("H1", "L1")
    background = _make_background(detectors)

    stack = inject_cbc_signal(
        "IMRPhenomD",
        PARAMS,
        detectors,
        background,
        sampling_frequency=FS,
        minimum_frequency=FMIN,
        waveform_backend=LALSimulationBackend(),
    )

    h1_values = stack["H1"].value
    l1_values = stack["L1"].value

    # Use the amplitude envelope (Hilbert) to remove phase-mixing between h+/hx
    # polarizations that would otherwise shift the raw CC peak.
    from scipy.signal import hilbert

    h1_env = np.abs(hilbert(h1_values))
    l1_env = np.abs(hilbert(l1_values))

    cc = correlate(h1_env, l1_env, mode="full")
    n = len(h1_values)
    lags = np.arange(-(n - 1), n)
    # Restrict search to +-60 samples (~14.6 ms) -- safely bounds the H1-L1 delay
    mid = n - 1
    search = slice(mid - 60, mid + 61)
    peak_lag_seconds = float(lags[search][np.argmax(cc[search])]) / FS

    expected_delay = _time_delay_from_earth_center_lal(
        "H1",
        right_ascension=RA,
        declination=DEC,
        t_gps=TC,
    ) - _time_delay_from_earth_center_lal(
        "L1",
        right_ascension=RA,
        declination=DEC,
        t_gps=TC,
    )

    assert abs(peak_lag_seconds - expected_delay) <= 1.0 / FS


@pytest.mark.requires_gwpy
@pytest.mark.requires_lalsuite
@pytest.mark.integration
def test_snr_regression():
    """Matched-filter SNR on H1 channel matches the LAL-only regression reference."""
    detectors = Network.from_name("H1L1V1").detector_names
    background = _make_background(detectors)

    stack = inject_cbc_signal(
        "IMRPhenomD",
        PARAMS,
        detectors,
        background,
        sampling_frequency=FS,
        minimum_frequency=FMIN,
        waveform_backend=LALSimulationBackend(),
    )

    snr_computed = _matched_filter_sigma_lal(stack["H1"])

    assert _REFERENCE_SNR > 0.0, f"_REFERENCE_SNR is not set. Update it to {snr_computed:.6e} in this file."
    assert abs(_REFERENCE_SNR - _PYCBC_REFERENCE_SNR) / _PYCBC_REFERENCE_SNR < 1e-6
    assert abs(snr_computed - _REFERENCE_SNR) / _REFERENCE_SNR < 1e-6


@pytest.mark.requires_gwpy
@pytest.mark.integration
def test_cbc_et_triangle_returns_stack():
    """inject_cbc_signal on ET-triangle returns a 3-channel stack."""
    network = Network.from_name("ET-triangle")
    background = _make_background(network.detector_names)

    stack = inject_cbc_signal(
        "IMRPhenomD",
        PARAMS,
        network.detector_names,
        background,
        sampling_frequency=FS,
        minimum_frequency=FMIN,
        waveform_backend=LALSimulationBackend(),
    )

    assert len(stack) == 3
