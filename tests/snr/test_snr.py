"""Unit tests for gwmock_signal.snr (optimal_snr and matched_filter_snr)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.snr._pycbc import matched_filter_snr, optimal_snr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS = 512.0  # Hz — kept low so tests are fast
_DURATION = 2.0  # seconds
_N = int(_FS * _DURATION)
_FMIN = 20.0  # Hz


def _flat_psd(n_freq: int, delta_f: float) -> object:
    """Return a flat (white) pycbc FrequencySeries PSD."""
    pycbc_types = pytest.importorskip("pycbc.types")
    return pycbc_types.FrequencySeries(np.ones(n_freq), delta_f=delta_f)


def _sine_strain(freq: float = 50.0) -> TimeSeries:
    """Return a pure-tone gwpy TimeSeries."""
    t = np.arange(_N) / _FS
    return TimeSeries(np.sin(2 * np.pi * freq * t), sample_rate=_FS, t0=0.0)


# ---------------------------------------------------------------------------
# optimal_snr
# ---------------------------------------------------------------------------


class TestOptimalSNR:
    """Tests for optimal_snr."""

    def test_returns_positive_float(self) -> None:
        """optimal_snr returns a positive float for a non-zero signal."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        result = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
        assert isinstance(result, float)
        assert result > 0.0

    def test_zero_signal_returns_zero(self) -> None:
        """optimal_snr returns 0 for a zero-valued strain."""
        pytest.importorskip("pycbc")
        strain = TimeSeries(np.zeros(_N), sample_rate=_FS, t0=0.0)
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        result = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_larger_amplitude_gives_larger_snr(self) -> None:
        """Scaling strain by a factor k scales optimal_snr by k."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        snr1 = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
        strain2 = TimeSeries(strain.value * 3.0, sample_rate=_FS, t0=0.0)
        snr2 = optimal_snr(strain2, psd, low_frequency_cutoff=_FMIN)
        assert snr2 == pytest.approx(3.0 * snr1, rel=1e-6)

    def test_raises_importerror_without_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ImportError with 'pycbc' in the message is raised when pycbc is absent."""
        monkeypatch.setitem(sys.modules, "pycbc.filter", None)
        strain = TimeSeries(np.ones(_N), sample_rate=_FS, t0=0.0)
        with pytest.raises(ImportError, match="pycbc"):
            optimal_snr(strain, None)

    def test_top_level_import_resolves(self) -> None:
        """gwmock_signal.optimal_snr resolves via the lazy-load __getattr__."""
        import gwmock_signal

        assert callable(gwmock_signal.optimal_snr)

    def test_calls_sigma_via_mocked_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Success path runs and returns float even when pycbc is replaced by a mock."""
        mock_htilde = MagicMock()
        mock_strain = MagicMock()
        mock_strain.to_pycbc.return_value.to_frequencyseries.return_value = mock_htilde

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.sigma.return_value = 7.5
        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)

        result = optimal_snr(mock_strain, "mock_psd", low_frequency_cutoff=30.0)

        assert result == 7.5
        mock_pycbc_filter.sigma.assert_called_once_with(mock_htilde, psd="mock_psd", low_frequency_cutoff=30.0)


# ---------------------------------------------------------------------------
# matched_filter_snr
# ---------------------------------------------------------------------------


class TestMatchedFilterSNR:
    """Tests for matched_filter_snr."""

    def test_returns_gwpy_timeseries(self) -> None:
        """matched_filter_snr returns a gwpy TimeSeries."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        result = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN)
        assert isinstance(result, TimeSeries)
        assert len(result) == len(strain)

    def test_snr_is_non_negative(self) -> None:
        """All values of the returned SNR time series are non-negative."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        result = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN)
        assert np.all(result.value >= 0.0)

    def test_peak_snr_consistent_with_optimal(self) -> None:
        """Peak matched-filter SNR (template == data) is close to optimal SNR."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        snr_ts = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN)
        opt = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
        # The peak of the matched-filter SNR time series equals the optimal SNR
        # when template == data (zero noise).
        assert snr_ts.value.max() == pytest.approx(opt, rel=1e-4)

    def test_raises_importerror_without_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ImportError with 'pycbc' in the message is raised when pycbc is absent."""
        monkeypatch.setitem(sys.modules, "pycbc.filter", None)
        strain = TimeSeries(np.ones(_N), sample_rate=_FS, t0=0.0)
        with pytest.raises(ImportError, match="pycbc"):
            matched_filter_snr(strain, strain, None)

    def test_top_level_import_resolves(self) -> None:
        """gwmock_signal.matched_filter_snr resolves via the lazy-load __getattr__."""
        import gwmock_signal

        assert callable(gwmock_signal.matched_filter_snr)

    def test_calls_matched_filter_via_mocked_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Success path runs and returns a TimeSeries even when pycbc is replaced by a mock."""
        mock_data = MagicMock()
        mock_template = MagicMock()
        mock_snr_result = MagicMock()

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.matched_filter.return_value = mock_snr_result
        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)

        expected_ts = MagicMock(spec=TimeSeries)
        with patch("gwmock_signal.snr._pycbc.TimeSeries") as mock_ts_class:
            mock_ts_class.from_pycbc.return_value = expected_ts
            result = matched_filter_snr(mock_data, mock_template, "mock_psd", low_frequency_cutoff=30.0)

        assert result is expected_ts
        mock_pycbc_filter.matched_filter.assert_called_once_with(
            mock_template.to_pycbc(),
            mock_data.to_pycbc(),
            psd="mock_psd",
            low_frequency_cutoff=30.0,
        )
        mock_ts_class.from_pycbc.assert_called_once_with(abs(mock_snr_result))
