"""Unit tests for gwmock_signal.snr.core."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

pytest.importorskip("pycbc", reason="pycbc not installed")

import pycbc.filter
import pycbc.psd
import pycbc.types

from gwmock_signal.snr.core import matched_filter_snr, noise_weighted_inner_product, optimal_snr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FS = 4096.0
DT = 1.0 / FS
F_LOW = 20.0


def _sine_wave_timeseries(
    f0_hz: float = 100.0,
    amplitude: float = 1.0,
    n_samples: int = 4096,
    dt: float = DT,
    t0: float = 0.0,
) -> TimeSeries:
    """Return a gwpy TimeSeries containing a pure sine wave."""
    t = np.arange(n_samples) * dt
    data = amplitude * np.sin(2.0 * np.pi * f0_hz * t)
    return TimeSeries(data, t0=t0, dt=dt)


def _flat_psd_array(n_rfft: int, s0: float = 1e-40) -> np.ndarray:
    """Return a flat PSD array with the DC bin set to inf."""
    arr = np.full(n_rfft, s0, dtype=float)
    arr[0] = np.inf
    return arr


# ---------------------------------------------------------------------------
# noise_weighted_inner_product
# ---------------------------------------------------------------------------


class TestNoiseWeightedInnerProduct:
    """Tests for noise_weighted_inner_product."""

    def test_analytic_flat_psd_sine_wave(self) -> None:
        """(h|h) for A*sin at an exact FFT bin with flat PSD equals A**2 * T / s0."""
        n, dt = 4096, DT
        t_duration = n * dt
        f0_hz = 100.0  # exact FFT bin: 100 Hz * 1 s = bin 100
        amplitude = 2.0
        s0 = 1e-40

        t = np.arange(n) * dt
        h = amplitude * np.sin(2.0 * np.pi * f0_hz * t)
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1] - freqs[0])
        h_fft = np.fft.rfft(h) * dt
        psd_arr = _flat_psd_array(len(h_fft), s0=s0)

        result = noise_weighted_inner_product(h_fft, h_fft, psd_arr, df)
        expected = amplitude**2 * t_duration / s0

        assert abs(result - expected) / expected < 1e-6

    def test_returns_real_float(self) -> None:
        """Return value is a real Python float."""
        n, dt = 256, DT
        h_fft = np.fft.rfft(np.random.default_rng(0).normal(size=n)) * dt
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1] - freqs[0])
        psd_arr = _flat_psd_array(len(h_fft))

        result = noise_weighted_inner_product(h_fft, h_fft, psd_arr, df)
        assert isinstance(result, float)

    def test_non_negative_for_self_inner_product(self) -> None:
        """(h|h) is always non-negative."""
        rng = np.random.default_rng(42)
        n, dt = 512, DT
        h_fft = np.fft.rfft(rng.normal(size=n)) * dt
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1] - freqs[0])
        psd_arr = _flat_psd_array(len(h_fft))

        result = noise_weighted_inner_product(h_fft, h_fft, psd_arr, df)
        assert result >= 0.0

    def test_dc_bin_excluded(self) -> None:
        """DC bin (index 0) does not contribute even with finite PSD there."""
        n, dt = 256, DT
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1] - freqs[0])

        h_fft = np.zeros(len(freqs), dtype=complex)
        h_fft[0] = 1e10

        psd_arr = np.ones(len(freqs)) * 1e-40
        result = noise_weighted_inner_product(h_fft, h_fft, psd_arr, df)
        assert result == pytest.approx(0.0, abs=1e-30)

    def test_inf_psd_bins_contribute_zero(self) -> None:
        """Bins with psd=inf contribute zero to the sum."""
        n, dt = 256, DT
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1] - freqs[0])
        h_fft = np.fft.rfft(np.ones(n)) * dt

        psd_all_inf = np.full(len(h_fft), np.inf)
        result = noise_weighted_inner_product(h_fft, h_fft, psd_all_inf, df)
        assert result == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# optimal_snr
# ---------------------------------------------------------------------------


class TestOptimalSnr:
    """Tests for optimal_snr."""

    def test_returns_positive_float(self) -> None:
        """optimal_snr returns a positive float for a non-zero signal."""
        ts = _sine_wave_timeseries(f0_hz=100.0, amplitude=1.0)
        n = len(ts)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd_arr = _flat_psd_array(len(freqs), s0=1e-40)
        result = optimal_snr(ts, psd_arr, f_low=F_LOW)
        assert isinstance(result, float)
        assert result > 0.0

    def test_analytic_flat_psd(self) -> None:
        """optimal_snr with flat PSD matches the analytic amplitude*sqrt(T/s0) result."""
        n, dt = 4096, DT
        t_duration = n * dt
        f0_hz = 100.0
        amplitude = 3.0
        s0 = 1e-40

        ts = _sine_wave_timeseries(f0_hz=f0_hz, amplitude=amplitude, n_samples=n, dt=dt)
        freqs = np.fft.rfftfreq(n, d=dt)
        psd_arr = _flat_psd_array(len(freqs), s0=s0)

        result = optimal_snr(ts, psd_arr, f_low=1.0)  # f_low below f0 so it is included
        expected = amplitude * np.sqrt(t_duration / s0)
        assert abs(result - expected) / expected < 1e-6

    def test_string_psd_aligo_design(self) -> None:
        """String 'aLIGO_design' is accepted and returns a positive SNR."""
        ts = _sine_wave_timeseries(f0_hz=100.0, amplitude=1e-21)
        result = optimal_snr(ts, "aLIGO_design", f_low=F_LOW)
        assert isinstance(result, float)
        assert result > 0.0

    def test_matches_pycbc_sigma_directly(self) -> None:
        """optimal_snr matches pycbc.filter.sigma called directly to < 1e-6."""
        n = 4096
        t = np.arange(n) * DT
        signal = 1e-21 * np.exp(-((t - 0.5) ** 2) / (2 * 0.01**2)) * np.sin(2.0 * np.pi * 100.0 * t)
        ts = TimeSeries(signal, t0=0.0, dt=DT)

        # Reference: call pycbc directly
        pycbc_ts = pycbc.types.TimeSeries(np.asarray(ts.value, dtype=float), delta_t=float(ts.dt.value))
        delta_f = 1.0 / (n * DT)
        length = n // 2 + 1
        psd_pc = pycbc.psd.aLIGODesignSensitivityP1200087(length, delta_f, F_LOW)
        ref = float(pycbc.filter.sigma(pycbc_ts, psd=psd_pc, low_frequency_cutoff=F_LOW))

        ours = optimal_snr(ts, "aLIGO_design", f_low=F_LOW)

        assert ref > 0.0
        assert abs(ours - ref) / ref < 1e-6

    def test_f_low_cutoff_reduces_snr(self) -> None:
        """Raising f_low removes frequency content and reduces SNR."""
        ts = _sine_wave_timeseries(f0_hz=30.0, amplitude=1e-21)
        snr_low = optimal_snr(ts, "aLIGO_design", f_low=20.0)
        snr_high = optimal_snr(ts, "aLIGO_design", f_low=50.0)
        assert snr_low > snr_high


# ---------------------------------------------------------------------------
# matched_filter_snr
# ---------------------------------------------------------------------------


class TestMatchedFilterSnr:
    """Tests for matched_filter_snr."""

    def test_returns_timeseries(self) -> None:
        """matched_filter_snr returns a gwpy TimeSeries."""
        ts = _sine_wave_timeseries()
        n = len(ts)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd_arr = _flat_psd_array(len(freqs))
        result = matched_filter_snr(ts, ts, psd_arr, f_low=1.0)
        assert isinstance(result, TimeSeries)

    def test_output_length_matches_data(self) -> None:
        """Output timeseries has the same number of samples as the input data."""
        ts = _sine_wave_timeseries()
        n = len(ts)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd_arr = _flat_psd_array(len(freqs))
        result = matched_filter_snr(ts, ts, psd_arr, f_low=1.0)
        assert len(result) == n

    def test_matched_peak_equals_optimal_snr(self) -> None:
        """For signal == data, peak |rho(t)| equals optimal_snr to < 1e-4."""
        n, dt = 4096, DT
        f0_hz = 100.0
        amplitude = 2.0
        s0 = 1e-40

        ts = _sine_wave_timeseries(f0_hz=f0_hz, amplitude=amplitude, n_samples=n, dt=dt)
        freqs = np.fft.rfftfreq(n, d=dt)
        psd_arr = _flat_psd_array(len(freqs), s0=s0)

        sigma = optimal_snr(ts, psd_arr, f_low=1.0)
        rho = matched_filter_snr(ts, ts, psd_arr, f_low=1.0)
        peak = float(np.max(np.abs(rho.value)))

        assert abs(peak - sigma) / sigma < 1e-4

    def test_t0_and_dt_preserved(self) -> None:
        """Output timeseries has the same t0 and dt as the data input."""
        ts = _sine_wave_timeseries(t0=1126259462.4)
        n = len(ts)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd_arr = _flat_psd_array(len(freqs))
        result = matched_filter_snr(ts, ts, psd_arr, f_low=1.0)
        assert float(result.t0.value) == pytest.approx(float(ts.t0.value))
        assert float(result.dt.value) == pytest.approx(float(ts.dt.value))

    def test_string_psd_accepted(self) -> None:
        """String PSD name is accepted without error."""
        ts = _sine_wave_timeseries(f0_hz=100.0, amplitude=1e-21)
        result = matched_filter_snr(ts, ts, "aLIGO_design", f_low=F_LOW)
        assert isinstance(result, TimeSeries)
        assert np.max(np.abs(result.value)) > 0.0

    def test_zero_sigma_returns_zeros(self) -> None:
        """Zero-signal input returns an all-zero SNR timeseries."""
        n = 256
        ts_zero = TimeSeries(np.zeros(n), t0=0.0, dt=DT)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd_arr = _flat_psd_array(len(freqs))
        result = matched_filter_snr(ts_zero, ts_zero, psd_arr, f_low=1.0)
        np.testing.assert_allclose(np.abs(result.value), 0.0, atol=1e-30)
