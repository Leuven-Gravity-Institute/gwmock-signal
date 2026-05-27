"""Unit tests for gwmock_signal.snr._network.network_optimal_snr."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.multichannel.stack import DetectorStrainStack
from gwmock_signal.snr._network import network_optimal_snr

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

_FS = 512.0  # Hz — low enough to keep tests fast
_DURATION = 4.0  # seconds — gives 0.25 Hz frequency resolution
_N = int(_FS * _DURATION)
_FMIN = 20.0  # Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stack(strains: dict[str, TimeSeries]) -> DetectorStrainStack:
    return DetectorStrainStack.from_mapping(list(strains.keys()), strains)


def _sine_ts(freq: float = 50.0, amplitude: float = 1.0) -> TimeSeries:
    """Pure-tone gwpy TimeSeries at an integer-cycle frequency (zero leakage)."""
    t = np.arange(_N) / _FS
    return TimeSeries(amplitude * np.sin(2 * np.pi * freq * t), sample_rate=_FS, t0=0.0)


def _flat_psd(n_freq: int | None = None) -> np.ndarray:
    n = n_freq if n_freq is not None else _N // 2 + 1
    return np.ones(n)


def _single_det_snr_numpy(ts: TimeSeries, psd: np.ndarray, fmin: float = _FMIN) -> float:
    """Reference single-detector SNR using the same formula as _network.py."""
    s = ts.value
    n_samples = len(s)
    dt = float(ts.dt.value)
    delta_f = 1.0 / (n_samples * dt)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    s_tilde = np.fft.rfft(s) * dt
    mask = freqs >= fmin
    rho_sq = 4.0 * delta_f * (np.abs(s_tilde[mask]) ** 2 / psd[mask]).sum()
    return float(np.sqrt(max(rho_sq, 0.0)))


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestQuadratureSumEquivalence:
    """With cross_psds=None, result equals sqrt(sum of squared per-detector SNRs)."""

    def test_two_detectors_different_signals(self) -> None:
        """Two detectors with different sine frequencies and amplitudes."""
        h1 = _sine_ts(freq=50.0, amplitude=1.0)
        l1 = _sine_ts(freq=80.0, amplitude=2.0)
        stack = _make_stack({"H1": h1, "L1": l1})
        psd = _flat_psd()
        psds = {"H1": psd, "L1": psd}

        expected = np.sqrt(_single_det_snr_numpy(h1, psd) ** 2 + _single_det_snr_numpy(l1, psd) ** 2)
        result = network_optimal_snr(stack, psds, low_frequency_cutoff=_FMIN)

        assert result > 0.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_identical_signals_scales_with_sqrt2(self) -> None:
        """Two identical detectors give SNR = sqrt(2) * single-detector SNR."""
        h1 = _sine_ts(freq=50.0, amplitude=1.0)
        l1 = _sine_ts(freq=50.0, amplitude=1.0)
        stack = _make_stack({"H1": h1, "L1": l1})
        psd = _flat_psd()
        psds = {"H1": psd, "L1": psd}

        snr_single = _single_det_snr_numpy(h1, psd)
        result = network_optimal_snr(stack, psds, low_frequency_cutoff=_FMIN)

        assert result == pytest.approx(np.sqrt(2.0) * snr_single, rel=1e-6)


class TestSensitivityToCorrelation:
    """Non-zero cross_psds changes the network SNR."""

    def test_positive_correlation_changes_result(self) -> None:
        """Introducing a real positive cross-PSD changes the SNR from the uncorrelated value."""
        h1 = _sine_ts(freq=50.0)
        l1 = _sine_ts(freq=50.0)
        stack = _make_stack({"H1": h1, "L1": l1})
        psd = _flat_psd()
        psds = {"H1": psd, "L1": psd}

        uncorr = network_optimal_snr(stack, psds, low_frequency_cutoff=_FMIN)

        cross_psd = 0.3 * psd.astype(complex)
        corr = network_optimal_snr(stack, psds, cross_psds={("H1", "L1"): cross_psd}, low_frequency_cutoff=_FMIN)

        assert corr != pytest.approx(uncorr, rel=1e-3)

    def test_cross_psd_conjugate_symmetry(self) -> None:
        """(H1,L1) and (L1,H1) keys produce the same result (Hermitian enforcement)."""
        h1 = _sine_ts(freq=50.0, amplitude=1.0)
        l1 = _sine_ts(freq=80.0, amplitude=0.5)
        stack = _make_stack({"H1": h1, "L1": l1})
        psd = _flat_psd()
        psds = {"H1": psd, "L1": psd}
        cross_psd = (0.2 + 0.1j) * psd

        result_hl = network_optimal_snr(stack, psds, cross_psds={("H1", "L1"): cross_psd}, low_frequency_cutoff=_FMIN)
        result_lh = network_optimal_snr(
            stack, psds, cross_psds={("L1", "H1"): np.conj(cross_psd)}, low_frequency_cutoff=_FMIN
        )

        assert result_hl == pytest.approx(result_lh, rel=1e-10)


class TestSingleDetector:
    """Single-detector stack matches the single-detector formula."""

    def test_single_detector_matches_formula(self) -> None:
        """Single-detector network SNR equals the single-detector formula."""
        h1 = _sine_ts(freq=50.0)
        stack = _make_stack({"H1": h1})
        psd = _flat_psd()
        psds = {"H1": psd}

        expected = _single_det_snr_numpy(h1, psd)
        result = network_optimal_snr(stack, psds, low_frequency_cutoff=_FMIN)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_top_level_import_resolves(self) -> None:
        """gwmock_signal.network_optimal_snr resolves via the lazy-load __getattr__."""
        import gwmock_signal

        assert callable(gwmock_signal.network_optimal_snr)


class TestFrequencyMask:
    """Bins outside [low_frequency_cutoff, high_frequency_cutoff] are excluded."""

    def test_low_bins_excluded(self) -> None:
        """Signal at 10 Hz is excluded by the default 20 Hz low cutoff."""
        low_freq_ts = _sine_ts(freq=10.0)  # 10 Hz * 4 s = 40 integer cycles → no leakage
        stack = _make_stack({"H1": low_freq_ts})
        psd = _flat_psd()
        psds = {"H1": psd}

        result_masked = network_optimal_snr(stack, psds, low_frequency_cutoff=20.0)
        result_unmasked = network_optimal_snr(stack, psds, low_frequency_cutoff=0.0)

        assert result_masked < result_unmasked * 0.01

    def test_high_frequency_cutoff_excludes_signal(self) -> None:
        """Signal at 50 Hz is excluded when high_frequency_cutoff=40 Hz."""
        ts = _sine_ts(freq=50.0)  # 50 Hz * 4 s = 200 integer cycles → no leakage
        stack = _make_stack({"H1": ts})
        psd = _flat_psd()
        psds = {"H1": psd}

        result_full = network_optimal_snr(stack, psds, low_frequency_cutoff=0.0)
        result_cut = network_optimal_snr(stack, psds, low_frequency_cutoff=0.0, high_frequency_cutoff=40.0)

        assert result_cut < result_full * 0.01
