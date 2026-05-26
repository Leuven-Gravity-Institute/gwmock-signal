"""Unit tests for gwmock_signal.snr.psd."""

from __future__ import annotations

import numpy as np
import pycbc.types
import pytest

from gwmock_signal.snr.psd import evaluate_psd, from_numpy_psd, load_design_psd


class TestLoadDesignPsd:
    """Tests for load_design_psd."""

    def test_returns_frequency_series(self) -> None:
        """load_design_psd returns a pycbc FrequencySeries."""
        psd = load_design_psd("aLIGO_design", length=2049, delta_f=1.0)
        assert isinstance(psd, pycbc.types.FrequencySeries)

    def test_output_length(self) -> None:
        """Output FrequencySeries has the requested length."""
        length = 2049
        psd = load_design_psd("aLIGO_design", length=length, delta_f=1.0)
        assert len(psd) == length

    def test_aligo_value_at_100_hz(self) -> None:
        """ALIGO design PSD at 100 Hz is approximately 1e-46 1/Hz."""
        delta_f = 1.0
        psd = load_design_psd("aLIGO_design", length=2049, delta_f=delta_f)
        val = float(psd[100])  # bin 100 = 100 Hz at delta_f=1.0
        assert 1e-48 < val < 1e-44

    def test_dc_bin_is_zero(self) -> None:
        """The DC bin (f=0) is zero (pycbc convention for excluded bins)."""
        psd = load_design_psd("aLIGO_design", length=2049, delta_f=1.0)
        assert float(psd[0]) == 0.0

    def test_f_low_bins_are_zero(self) -> None:
        """Bins below f_low are set to zero by pycbc."""
        f_low = 30.0
        delta_f = 1.0
        psd = load_design_psd("aLIGO_design", length=2049, delta_f=delta_f, f_low=f_low)
        assert float(psd[int(f_low / delta_f) - 1]) == 0.0
        assert float(psd[50]) > 0.0

    def test_unknown_name_raises_value_error(self) -> None:
        """Unknown PSD name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown design PSD"):
            load_design_psd("not_a_psd", length=2049, delta_f=1.0)


class TestFromNumpyPsd:
    """Tests for from_numpy_psd."""

    def _power_law_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a simple f^(-2) PSD for testing."""
        freqs = np.linspace(10.0, 1000.0, 200)
        psd_values = freqs**-2
        return freqs, psd_values

    def test_returns_frequency_series(self) -> None:
        """from_numpy_psd returns a pycbc FrequencySeries."""
        freqs, vals = self._power_law_inputs()
        # f_low must be >= freqs[0] so pycbc does not extrapolate below the data
        psd = from_numpy_psd(freqs, vals, length=2049, delta_f=1.0, f_low=10.0)
        assert isinstance(psd, pycbc.types.FrequencySeries)

    def test_output_length(self) -> None:
        """Output FrequencySeries has the requested length."""
        freqs, vals = self._power_law_inputs()
        # (length-1)*delta_f must not exceed freqs[-1]=1000 Hz: 512*1.0=512 Hz < 1000 Hz
        length = 513
        psd = from_numpy_psd(freqs, vals, length=length, delta_f=1.0, f_low=10.0)
        assert len(psd) == length

    def test_value_between_knots(self) -> None:
        """Interpolated value between tabulated knots lies between adjacent values."""
        freqs = np.array([100.0, 200.0, 400.0])
        vals = np.array([1e-40, 5e-41, 2e-41])
        # f_low=100.0: within the data range (freqs starts at 100 Hz)
        psd = from_numpy_psd(freqs, vals, length=513, delta_f=1.0, f_low=100.0)
        val_150 = float(psd[150])
        assert 5e-41 < val_150 < 1e-40

    def test_f_low_bins_are_zero(self) -> None:
        """Bins below f_low are set to zero."""
        freqs, vals = self._power_law_inputs()
        f_low = 50.0
        psd = from_numpy_psd(freqs, vals, length=2049, delta_f=1.0, f_low=f_low)
        assert float(psd[20]) == 0.0
        assert float(psd[100]) > 0.0

    def test_positive_values_in_range(self) -> None:
        """Bins within the tabulated frequency range have positive PSD values."""
        freqs, vals = self._power_law_inputs()
        psd = from_numpy_psd(freqs, vals, length=513, delta_f=2.0, f_low=10.0)
        assert float(psd[50]) > 0.0  # 100 Hz at delta_f=2.0


class TestEvaluatePsd:
    """Tests for evaluate_psd."""

    def test_callable_psd(self) -> None:
        """Callable PSD is applied to the frequency array."""
        s0 = 1e-40
        freqs = np.array([0.0, 50.0, 100.0])
        result = evaluate_psd(lambda f: np.full_like(f, s0), freqs)
        np.testing.assert_allclose(result[1:], s0)
        assert np.isinf(result[0])

    def test_array_psd(self) -> None:
        """Array PSD is returned as-is (copy) with DC set to inf."""
        freqs = np.linspace(0.0, 100.0, 10)
        psd_arr = np.ones_like(freqs) * 1e-40
        result = evaluate_psd(psd_arr, freqs)
        np.testing.assert_allclose(result[1:], 1e-40)
        assert np.isinf(result[0])

    def test_array_psd_is_copy(self) -> None:
        """Returned array is a copy — modifying it does not change the input."""
        freqs = np.array([0.0, 100.0])
        psd_arr = np.array([1e-40, 1e-40])
        result = evaluate_psd(psd_arr, freqs)
        result[1] = 999.0
        assert psd_arr[1] == 1e-40

    def test_zero_bin_always_inf(self) -> None:
        """The f=0 bin is set to inf regardless of the PSD input."""
        freqs = np.array([0.0, 10.0, 100.0])
        psd_arr = np.array([1.0, 1.0, 1.0])
        result = evaluate_psd(psd_arr, freqs)
        assert np.isinf(result[0])

    def test_positive_frequencies_unchanged(self) -> None:
        """Positive-frequency bins are returned unchanged for array input."""
        freqs = np.array([10.0, 50.0, 100.0])
        psd_arr = np.array([3e-40, 2e-40, 1e-40])
        result = evaluate_psd(psd_arr, freqs)
        np.testing.assert_allclose(result, psd_arr)
