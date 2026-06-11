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
        mock_psd_interp = MagicMock()

        # duration=64.0 → natural_delta_f=1/64 ≤ 1/32 → no padding, htilde comes
        # directly from td.to_frequencyseries() so the mock chain stays simple.
        mock_td = MagicMock()
        mock_td.duration = 64.0
        mock_td.to_frequencyseries.return_value = mock_htilde

        mock_strain = MagicMock()
        mock_strain.to_pycbc.return_value = mock_td

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.sigma.return_value = 7.5
        mock_pycbc_psd = MagicMock()
        mock_pycbc_psd.interpolate.return_value = mock_psd_interp
        mock_pycbc_types = MagicMock()
        # The zero-replacement step calls pycbc_types.FrequencySeries to wrap
        # the processed PSD array; return mock_psd_interp so the sigma assertion holds.
        mock_pycbc_types.FrequencySeries.return_value = mock_psd_interp

        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)
        monkeypatch.setitem(sys.modules, "pycbc.psd", mock_pycbc_psd)
        monkeypatch.setitem(sys.modules, "pycbc.types", mock_pycbc_types)

        result = optimal_snr(mock_strain, "mock_psd", low_frequency_cutoff=30.0)

        assert result == 7.5
        mock_pycbc_filter.sigma.assert_called_once_with(mock_htilde, psd=mock_psd_interp, low_frequency_cutoff=30.0)

    def test_padding_branch_via_mocked_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Padding branch runs (natural_delta_f > target_delta_f) under mocked pycbc.

        Covers the zero-padding path (the strain is extended to reach target_delta_f)
        without requiring the optional pycbc backend, so coverage holds when pycbc
        is absent from the test environment.
        """
        mock_htilde = MagicMock()
        mock_psd_interp = MagicMock()

        # The padded TimeSeries that pycbc_types.TimeSeries(...) returns; its
        # to_frequencyseries() feeds sigma.
        mock_td_padded = MagicMock()
        mock_td_padded.to_frequencyseries.return_value = mock_htilde

        # duration=2.0 → natural_delta_f=0.5 > target=1/32 → padding fires.
        mock_td = MagicMock()
        mock_td.duration = 2.0
        mock_td.sample_rate = 512.0
        mock_td.__len__.return_value = 1024

        mock_strain = MagicMock()
        mock_strain.to_pycbc.return_value = mock_td

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.sigma.return_value = 9.0
        mock_pycbc_psd = MagicMock()
        mock_pycbc_psd.interpolate.return_value = mock_psd_interp
        mock_pycbc_types = MagicMock()
        mock_pycbc_types.TimeSeries.return_value = mock_td_padded
        mock_pycbc_types.FrequencySeries.return_value = mock_psd_interp

        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)
        monkeypatch.setitem(sys.modules, "pycbc.psd", mock_pycbc_psd)
        monkeypatch.setitem(sys.modules, "pycbc.types", mock_pycbc_types)

        result = optimal_snr(mock_strain, "mock_psd", low_frequency_cutoff=25.0)

        assert result == 9.0
        # The padded TimeSeries was constructed and its FFT was passed to sigma.
        mock_pycbc_types.TimeSeries.assert_called_once()
        mock_td_padded.to_frequencyseries.assert_called_once()
        mock_pycbc_filter.sigma.assert_called_once_with(mock_htilde, psd=mock_psd_interp, low_frequency_cutoff=25.0)


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
        """Peak matched-filter SNR (template == data) equals optimal SNR to <0.01%."""
        pytest.importorskip("pycbc")
        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        delta_f = pycbc_ts.to_frequencyseries().delta_f
        psd = _flat_psd(len(pycbc_ts.to_frequencyseries()), delta_f)
        snr_ts = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN)
        opt = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
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
        mock_htilde = MagicMock()
        mock_psd_interp = MagicMock()
        mock_snr_result = MagicMock()
        mock_snr_trimmed = MagicMock()

        # duration=64.0 → natural_delta_f=1/64 ≤ 1/32 → no padding path.
        mock_td_template = MagicMock()
        mock_td_template.duration = 64.0
        mock_td_template.to_frequencyseries.return_value = mock_htilde

        mock_td_data = MagicMock()

        mock_template = MagicMock()
        mock_template.to_pycbc.return_value = mock_td_template

        mock_data = MagicMock()
        mock_data.to_pycbc.return_value = mock_td_data

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.matched_filter.return_value = mock_snr_result

        mock_pycbc_psd = MagicMock()
        mock_pycbc_psd.interpolate.return_value = mock_psd_interp

        mock_pycbc_types = MagicMock()
        # FrequencySeries wraps the zero-replaced PSD; TimeSeries wraps the trimmed SNR.
        mock_pycbc_types.FrequencySeries.return_value = mock_psd_interp
        mock_pycbc_types.TimeSeries.return_value = mock_snr_trimmed

        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)
        monkeypatch.setitem(sys.modules, "pycbc.psd", mock_pycbc_psd)
        monkeypatch.setitem(sys.modules, "pycbc.types", mock_pycbc_types)

        expected_ts = MagicMock(spec=TimeSeries)
        with patch("gwmock_signal.snr._pycbc.TimeSeries") as mock_ts_class:
            mock_ts_class.from_pycbc.return_value = expected_ts
            result = matched_filter_snr(mock_data, mock_template, "mock_psd", low_frequency_cutoff=30.0)

        assert result is expected_ts
        mock_pycbc_filter.matched_filter.assert_called_once_with(
            mock_td_template,
            mock_td_data,
            psd=mock_psd_interp,
            low_frequency_cutoff=30.0,
        )
        mock_ts_class.from_pycbc.assert_called_once_with(mock_snr_trimmed)

    def test_padding_branch_via_mocked_pycbc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Padding branch runs (natural_delta_f > target_delta_f) under mocked pycbc.

        Covers the zero-padding path for both template and data — including the
        ``n_target > n_data`` data-extension branch — without the optional pycbc
        backend, so coverage holds when pycbc is absent from the environment.
        """
        mock_htilde = MagicMock()
        mock_psd_interp = MagicMock()

        # Single padded TimeSeries returned for the padded template, padded data,
        # and trimmed SNR; its to_frequencyseries() feeds the matched filter.
        mock_padded = MagicMock()
        mock_padded.to_frequencyseries.return_value = mock_htilde

        # template duration=2.0 → natural_delta_f=0.5 > target=1/32 → padding fires.
        # len(td_data)=0 → n_target = n_padded > n_data → data branch also runs.
        mock_td_template = MagicMock()
        mock_td_template.duration = 2.0
        mock_td_template.sample_rate = 512.0
        mock_td_data = MagicMock()

        mock_template = MagicMock()
        mock_template.to_pycbc.return_value = mock_td_template
        mock_data = MagicMock()
        mock_data.to_pycbc.return_value = mock_td_data

        mock_pycbc_filter = MagicMock()
        mock_pycbc_filter.matched_filter.return_value = MagicMock()
        mock_pycbc_psd = MagicMock()
        mock_pycbc_psd.interpolate.return_value = mock_psd_interp
        mock_pycbc_types = MagicMock()
        mock_pycbc_types.TimeSeries.return_value = mock_padded
        mock_pycbc_types.FrequencySeries.return_value = mock_psd_interp

        monkeypatch.setitem(sys.modules, "pycbc.filter", mock_pycbc_filter)
        monkeypatch.setitem(sys.modules, "pycbc.psd", mock_pycbc_psd)
        monkeypatch.setitem(sys.modules, "pycbc.types", mock_pycbc_types)

        expected_ts = MagicMock(spec=TimeSeries)
        with patch("gwmock_signal.snr._pycbc.TimeSeries") as mock_ts_class:
            mock_ts_class.from_pycbc.return_value = expected_ts
            result = matched_filter_snr(mock_data, mock_template, "mock_psd", low_frequency_cutoff=25.0)

        assert result is expected_ts
        # Both template and data were padded (plus the trimmed SNR) → ≥3 constructions.
        assert mock_pycbc_types.TimeSeries.call_count >= 3
        mock_padded.to_frequencyseries.assert_called_once()
        mock_pycbc_filter.matched_filter.assert_called_once()


# ---------------------------------------------------------------------------
# optimal_snr padding behaviour
# ---------------------------------------------------------------------------


class TestOptimalSNRPadding:
    """Tests for the zero-padding and PSD-interpolation logic in optimal_snr."""

    def test_optimal_snr_psd_delta_f_independence(self) -> None:
        """SNR is independent of the caller's PSD delta_f (<0.05%)."""
        pytest.importorskip("pycbc")
        from pycbc.types import FrequencySeries

        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        n_freq_coarse = int(len(pycbc_ts) / 2) + 1
        psd_coarse = FrequencySeries(np.ones(n_freq_coarse), delta_f=0.5)
        n_freq_fine = int(len(pycbc_ts) * 4 / 2) + 1
        psd_fine = FrequencySeries(np.ones(n_freq_fine), delta_f=0.125)
        snr_coarse = optimal_snr(strain, psd_coarse, low_frequency_cutoff=_FMIN)
        snr_fine = optimal_snr(strain, psd_fine, low_frequency_cutoff=_FMIN)
        assert abs(snr_coarse / snr_fine - 1.0) < 5e-4

    def test_optimal_snr_no_padding_for_already_fine_grid(self) -> None:
        """Long signals with natural delta_f <= 1/32 Hz are not padded."""
        pytest.importorskip("pycbc")
        from pycbc.types import FrequencySeries

        fs, duration = 512.0, 60.0
        n = int(fs * duration)
        strain = TimeSeries(
            np.sin(2 * np.pi * 50.0 * np.arange(n) / fs),
            sample_rate=fs,
            t0=0.0,
        )
        pycbc_ts = strain.to_pycbc()
        natural_df = 1.0 / float(pycbc_ts.duration)
        n_freq = int(len(pycbc_ts) / 2) + 1
        psd = FrequencySeries(np.ones(n_freq), delta_f=natural_df)
        snr_default = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN)
        snr_explicit = optimal_snr(strain, psd, low_frequency_cutoff=_FMIN, target_delta_f=natural_df)
        assert abs(snr_default / snr_explicit - 1.0) < 1e-4

    @pytest.mark.integration
    def test_optimal_snr_high_mass_bbh_bias_corrected(self) -> None:
        """Auto-padded SNR for 50+50 Msun BBH agrees with fine-grid reference to <0.02%."""
        pytest.importorskip("pycbc")
        pytest.importorskip("lalsimulation")
        from pycbc.psd.analytical import aLIGODesignSensitivityP1200087

        from gwmock_signal.waveform.backends import LALSimulationBackend

        hpc = LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=50.0,
            detector_frame_mass_2=50.0,
            luminosity_distance=400.0,
            inclination=0.0,
            coa_phase=0.0,
        )
        strain = hpc["plus"]
        delta_f_natural = 1.0 / float(strain.duration.value)
        psd = aLIGODesignSensitivityP1200087(
            length=int(len(strain) / 2) + 1,
            delta_f=delta_f_natural,
            low_freq_cutoff=20.0,
        )
        snr_auto = optimal_snr(strain, psd, low_frequency_cutoff=20.0)
        snr_fine = optimal_snr(strain, psd, low_frequency_cutoff=20.0, target_delta_f=1.0 / 64.0)
        assert abs(snr_auto / snr_fine - 1.0) < 2e-4


# ---------------------------------------------------------------------------
# matched_filter_snr padding behaviour
# ---------------------------------------------------------------------------


class TestMatchedFilterSNRPadding:
    """Tests for the zero-padding and PSD-interpolation logic in matched_filter_snr."""

    def test_matched_filter_snr_psd_delta_f_independence(self) -> None:
        """Peak SNR is independent of the caller's PSD delta_f (<0.05%)."""
        pytest.importorskip("pycbc")
        from pycbc.types import FrequencySeries

        strain = _sine_strain()
        pycbc_ts = strain.to_pycbc()
        n_freq_coarse = int(len(pycbc_ts) / 2) + 1
        psd_coarse = FrequencySeries(np.ones(n_freq_coarse), delta_f=0.5)
        n_freq_fine = int(len(pycbc_ts) * 4 / 2) + 1
        psd_fine = FrequencySeries(np.ones(n_freq_fine), delta_f=0.125)
        snr_coarse = matched_filter_snr(strain, strain, psd_coarse, low_frequency_cutoff=_FMIN)
        snr_fine = matched_filter_snr(strain, strain, psd_fine, low_frequency_cutoff=_FMIN)
        assert abs(snr_coarse.value.max() / snr_fine.value.max() - 1.0) < 5e-4

    def test_matched_filter_snr_long_data_no_data_padding(self) -> None:
        """When data is already at least as long as n_padded, only template is extended."""
        pytest.importorskip("pycbc")
        from pycbc.types import FrequencySeries

        fs = 512.0
        # Template: 1 s → natural_delta_f=1 Hz > target_delta_f=0.5 Hz → padding fires.
        # n_padded = round(512/0.5)=1024. Data: 4 s → n_data=2048.
        # n_target = max(1024, 2048) = 2048 = n_data → data branch NOT entered.
        template = TimeSeries(
            np.sin(2 * np.pi * 50.0 * np.arange(int(fs)) / fs),
            sample_rate=fs,
            t0=0.0,
        )
        data = TimeSeries(
            np.sin(2 * np.pi * 50.0 * np.arange(int(fs * 4)) / fs),
            sample_rate=fs,
            t0=0.0,
        )
        # Flat PSD at 0.5 Hz (513 bins); htilde is at 0.25 Hz (1025 bins) — same Nyquist.
        psd = FrequencySeries(np.ones(513), delta_f=0.5)
        result = matched_filter_snr(data, template, psd, low_frequency_cutoff=_FMIN, target_delta_f=0.5)
        assert isinstance(result, TimeSeries)
        assert len(result) == len(data)

    def test_matched_filter_snr_no_padding_for_already_fine_grid(self) -> None:
        """Long signals with natural delta_f <= 1/32 Hz are not padded."""
        pytest.importorskip("pycbc")
        from pycbc.types import FrequencySeries

        fs, duration = 512.0, 60.0
        n = int(fs * duration)
        strain = TimeSeries(
            np.sin(2 * np.pi * 50.0 * np.arange(n) / fs),
            sample_rate=fs,
            t0=0.0,
        )
        pycbc_ts = strain.to_pycbc()
        natural_df = 1.0 / float(pycbc_ts.duration)
        n_freq = int(len(pycbc_ts) / 2) + 1
        psd = FrequencySeries(np.ones(n_freq), delta_f=natural_df)
        snr_default = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN)
        snr_explicit = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=_FMIN, target_delta_f=natural_df)
        assert abs(snr_default.value.max() / snr_explicit.value.max() - 1.0) < 1e-4

    @pytest.mark.integration
    def test_matched_filter_snr_high_mass_bbh_bias_corrected(self) -> None:
        """Peak matched_filter_snr for 50+50 Msun BBH agrees with padded optimal_snr to <0.05%."""
        pytest.importorskip("pycbc")
        pytest.importorskip("lalsimulation")
        from pycbc.psd.analytical import aLIGODesignSensitivityP1200087

        from gwmock_signal.waveform.backends import LALSimulationBackend

        hpc = LALSimulationBackend().generate_td_waveform(
            "IMRPhenomD",
            tc=0.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            detector_frame_mass_1=50.0,
            detector_frame_mass_2=50.0,
            luminosity_distance=400.0,
            inclination=0.0,
            coa_phase=0.0,
        )
        strain = hpc["plus"]
        delta_f_natural = 1.0 / float(strain.duration.value)
        psd = aLIGODesignSensitivityP1200087(
            length=int(len(strain) / 2) + 1,
            delta_f=delta_f_natural,
            low_freq_cutoff=20.0,
        )
        snr_mf = matched_filter_snr(strain, strain, psd, low_frequency_cutoff=20.0)
        snr_opt = optimal_snr(strain, psd, low_frequency_cutoff=20.0)
        assert abs(snr_mf.value.max() / snr_opt - 1.0) < 5e-4
