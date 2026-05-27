"""Integration regression test for gwmock_signal.snr.network_optimal_snr.

Validates that network_optimal_snr on a GW150914-like H1L1 zero-noise injection
equals sqrt(rho_H1² + rho_L1²) — the uncorrelated quadrature sum — to within
1e-6 relative tolerance, confirming Eq. 19 of Cireddu et al. 2025.
"""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.network import Network
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.snr._network import network_optimal_snr
from gwmock_signal.waveform.backends import LALSimulationBackend

# ---------------------------------------------------------------------------
# GW150914-like injection parameters (kept in sync with test_cbc_pipeline.py)
# ---------------------------------------------------------------------------

FS = 4096.0
FMIN = 20.0
DURATION = 8.0
POST_MERGER_PADDING = 0.05

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

TC = PARAMS["coa_time"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_background(detector_names: tuple[str, ...]) -> dict[str, TimeSeries]:
    n_samples = int((DURATION + POST_MERGER_PADDING) * FS)
    t0 = TC - DURATION
    return {name: TimeSeries(np.zeros(n_samples), t0=t0, sample_rate=FS) for name in detector_names}


def _aligo_psd_numpy(n_freq: int, delta_f: float) -> np.ndarray:
    """Return aLIGO design PSD (P1200087) as a numpy array via pycbc.

    pycbc zeros both the DC bin and the Nyquist bin by convention.  Replacing
    those zeros with inf keeps the noise matrix invertible; the inverse of a
    diagonal inf matrix is the zero matrix, so those bins contribute nothing to
    the inner product sum.
    """
    from pycbc.psd import analytical

    psd_pycbc = analytical.aLIGODesignSensitivityP1200087(n_freq, delta_f, FMIN)
    psd = np.array(psd_pycbc.data, dtype=float)
    psd[psd == 0.0] = np.inf
    return psd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_network_snr_h1l1_equals_quadrature_sum() -> None:
    """network_optimal_snr (no cross-PSDs) on H1L1 equals sqrt(rho_H1² + rho_L1²)."""
    pytest.importorskip("pycbc")

    detectors = Network.from_name("H1L1").detector_names
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

    ref_ts = stack[detectors[0]]
    n_samples = len(ref_ts.value)
    dt = float(ref_ts.dt.value)
    delta_f = 1.0 / (n_samples * dt)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    n_freq = len(freqs)

    psd_arr = _aligo_psd_numpy(n_freq, delta_f)
    psds = {det: psd_arr.copy() for det in detectors}

    # Reference: quadrature sum using the same formula as _network.py
    mask = freqs >= FMIN
    rho_sq_sum = 0.0
    for det in detectors:
        s_tilde = np.fft.rfft(stack[det].value) * dt
        rho_sq_sum += 4.0 * delta_f * (np.abs(s_tilde[mask]) ** 2 / psd_arr[mask]).sum()

    expected = float(np.sqrt(rho_sq_sum))
    result = network_optimal_snr(stack, psds, low_frequency_cutoff=FMIN)

    assert result > 0.0
    assert result == pytest.approx(expected, rel=1e-6)
