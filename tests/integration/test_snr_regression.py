"""Integration regression test for gwmock_signal.snr.optimal_snr.

Validates that optimal_snr on the H1 channel of a GW150914-like zero-noise
injection matches the historical PyCBC reference SNR to within 1e-4 relative
tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.network import Network
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.snr._pycbc import matched_filter_snr, optimal_snr
from gwmock_signal.waveform.backends import LALSimulationBackend

# ---------------------------------------------------------------------------
# GW150914-like injection parameters  (kept in sync with test_cbc_pipeline.py)
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

# H1 optimal SNR against aLIGO design PSD (P1200087), originally computed
# with pycbc.filter.sigma.
_PYCBC_REFERENCE_SNR = 4.884168e01

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_background(detector_names: tuple[str, ...]) -> dict[str, TimeSeries]:
    n_samples = int((DURATION + POST_MERGER_PADDING) * FS)
    t0 = TC - DURATION
    return {name: TimeSeries(np.zeros(n_samples), t0=t0, sample_rate=FS) for name in detector_names}


def _aligop1200087_psd(n_freq: int, delta_f: float) -> object:
    """Return aLIGO design PSD (P1200087) as a pycbc FrequencySeries."""
    from pycbc.psd import analytical

    return analytical.aLIGODesignSensitivityP1200087(n_freq, delta_f, FMIN)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_optimal_snr_h1_matches_reference() -> None:
    """optimal_snr on H1 matches the historical pycbc reference within 1e-4."""
    pytest.importorskip("pycbc")
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

    h1_strain = stack["H1"]
    pycbc_ts = h1_strain.to_pycbc()
    htilde = pycbc_ts.to_frequencyseries()
    psd = _aligop1200087_psd(len(htilde), htilde.delta_f)

    snr = optimal_snr(h1_strain, psd, low_frequency_cutoff=FMIN)

    assert snr > 0.0
    assert abs(snr - _PYCBC_REFERENCE_SNR) / _PYCBC_REFERENCE_SNR < 1e-4


@pytest.mark.integration
def test_matched_filter_snr_peak_consistent_with_optimal() -> None:
    """Peak matched-filter SNR (template == data) agrees with optimal_snr."""
    pytest.importorskip("pycbc")
    detectors = ("H1",)
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

    h1_strain = stack["H1"]
    pycbc_ts = h1_strain.to_pycbc()
    htilde = pycbc_ts.to_frequencyseries()
    psd = _aligop1200087_psd(len(htilde), htilde.delta_f)

    opt = optimal_snr(h1_strain, psd, low_frequency_cutoff=FMIN)
    snr_ts = matched_filter_snr(h1_strain, h1_strain, psd, low_frequency_cutoff=FMIN)

    assert isinstance(snr_ts, TimeSeries)
    assert snr_ts.value.max() == pytest.approx(opt, rel=1e-4)
