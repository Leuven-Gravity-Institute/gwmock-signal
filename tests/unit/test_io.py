"""Unit tests for DetectorStrainStack write / read I/O methods."""

from __future__ import annotations

try:
    import h5py
except ModuleNotFoundError as exc:  # Python 3.15: h5py does not install yet; its tests are skipped
    if exc.name != "h5py":
        raise
    h5py = None
import numpy as np
import pytest

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

from gwmock_signal.multichannel.stack import DetectorStrainStack

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 256.0
_N_SAMPLES = 256
_T0 = 1_000_000_000.0
_DT = 1.0 / _SAMPLE_RATE


def _make_stack() -> DetectorStrainStack:
    """Return a minimal 2-detector stack (256 samples at 256 Hz)."""
    rng = np.random.default_rng(0)
    h1 = TimeSeries(rng.standard_normal(_N_SAMPLES), t0=_T0, sample_rate=_SAMPLE_RATE, unit="strain", name="H1")
    l1 = TimeSeries(rng.standard_normal(_N_SAMPLES), t0=_T0, sample_rate=_SAMPLE_RATE, unit="strain", name="L1")
    return DetectorStrainStack(("H1", "L1"), (h1, l1))


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


class TestHDF5RoundTrip:
    """Test that the HDF5 format round-trips correctly."""

    @pytest.mark.requires_gwpy
    def test_arrays_preserved_to_machine_precision(self, tmp_path):
        """Test that the arrays are preserved to machine precision."""
        stack = _make_stack()
        path = tmp_path / "output.h5"

        stack.write(path, format="hdf5")
        restored = DetectorStrainStack.read(path, format="hdf5")

        for name in stack.detector_names:
            np.testing.assert_array_equal(
                np.asarray(stack[name].value),
                np.asarray(restored[name].value),
            )

    @pytest.mark.requires_gwpy
    def test_t0_preserved(self, tmp_path):
        """Test that the t0 is preserved."""
        stack = _make_stack()
        path = tmp_path / "output.h5"

        stack.write(path, format="hdf5")
        restored = DetectorStrainStack.read(path, format="hdf5")

        assert restored.t0 == pytest.approx(_T0, rel=0, abs=1e-6)

    @pytest.mark.requires_gwpy
    def test_dt_preserved(self, tmp_path):
        """Test that the dt is preserved."""
        stack = _make_stack()
        path = tmp_path / "output.h5"

        stack.write(path, format="hdf5")
        restored = DetectorStrainStack.read(path, format="hdf5")

        assert float(restored["H1"].dt) == pytest.approx(_DT)

    @pytest.mark.requires_gwpy
    def test_channel_names_preserved(self, tmp_path):
        """Test that the channel names are preserved."""
        stack = _make_stack()
        path = tmp_path / "output.h5"

        stack.write(path, format="hdf5")
        restored = DetectorStrainStack.read(path, format="hdf5")

        assert restored.detector_names == ("H1", "L1")

    @pytest.mark.requires_gwpy
    def test_file_is_created(self, tmp_path):
        """Test that the file is created."""
        stack = _make_stack()
        path = tmp_path / "output.h5"

        stack.write(path, format="hdf5")

        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.requires_h5py
    def test_hdf5_order_attribute_is_used(self, tmp_path):
        """If the top-level order attribute disagrees with file key order, reconstruction follows the attribute."""
        path = tmp_path / "order_test.h5"

        t0 = _T0
        dt = _DT
        a = np.array([1.0, 2.0, 3.0], dtype=float)
        b = np.array([4.0, 5.0, 6.0], dtype=float)

        with h5py.File(path, "w") as fh:
            # Create datasets in reverse order on purpose.
            ds_b = fh.create_dataset("B", data=b)
            ds_b.attrs["t0"] = float(t0)
            ds_b.attrs["dt"] = float(dt)
            ds_b.attrs["unit"] = "strain"

            ds_a = fh.create_dataset("A", data=a)
            ds_a.attrs["t0"] = float(t0)
            ds_a.attrs["dt"] = float(dt)
            ds_a.attrs["unit"] = "strain"

            # Force the desired reconstruction order.
            fh.attrs["gwmock_signal_detector_strain_stack_order"] = '["A", "B"]'

        restored = DetectorStrainStack.read(path, format="hdf5")
        assert restored.detector_names == ("A", "B")
        np.testing.assert_array_equal(restored["A"].value, a)
        np.testing.assert_array_equal(restored["B"].value, b)

    @pytest.mark.requires_h5py
    def test_hdf5_missing_order_attribute_falls_back_to_keys(self, tmp_path):
        """Files written by older versions without the order attribute should still be readable."""
        path = tmp_path / "order_missing_attr.h5"

        t0 = _T0
        dt = _DT
        a = np.array([1.0, 2.0, 3.0], dtype=float)
        b = np.array([4.0, 5.0, 6.0], dtype=float)

        with h5py.File(path, "w") as fh:
            ds_b = fh.create_dataset("B", data=b)
            ds_b.attrs["t0"] = float(t0)
            ds_b.attrs["dt"] = float(dt)
            ds_b.attrs["unit"] = "strain"

            ds_a = fh.create_dataset("A", data=a)
            ds_a.attrs["t0"] = float(t0)
            ds_a.attrs["dt"] = float(dt)
            ds_a.attrs["unit"] = "strain"
            expected = list(fh.keys())

        restored = DetectorStrainStack.read(path, format="hdf5")
        assert list(restored.detector_names) == expected


# ---------------------------------------------------------------------------
# npy round-trip
# ---------------------------------------------------------------------------


class TestNpyRoundTrip:
    """Test that the npy format round-trips correctly."""

    @pytest.mark.requires_gwpy
    def test_arrays_preserved_to_machine_precision(self, tmp_path):
        """Test that the arrays are preserved to machine precision."""
        stack = _make_stack()
        path = tmp_path / "output.npy"

        stack.write(path, format="npy")
        restored = DetectorStrainStack.read(path, format="npy")

        for name in stack.detector_names:
            np.testing.assert_array_equal(
                np.asarray(stack[name].value),
                np.asarray(restored[name].value),
            )

    @pytest.mark.requires_gwpy
    def test_t0_preserved(self, tmp_path):
        """Test that the t0 is preserved."""
        stack = _make_stack()
        path = tmp_path / "output.npy"

        stack.write(path, format="npy")
        restored = DetectorStrainStack.read(path, format="npy")

        assert restored.t0 == pytest.approx(_T0, rel=0, abs=1e-6)

    @pytest.mark.requires_gwpy
    def test_dt_preserved(self, tmp_path):
        """Test that the dt is preserved."""
        stack = _make_stack()
        path = tmp_path / "output.npy"

        stack.write(path, format="npy")
        restored = DetectorStrainStack.read(path, format="npy")

        assert float(restored["H1"].dt) == pytest.approx(_DT)

    @pytest.mark.requires_gwpy
    def test_channel_names_preserved(self, tmp_path):
        """Test that the channel names are preserved."""
        stack = _make_stack()
        path = tmp_path / "output.npy"

        stack.write(path, format="npy")
        restored = DetectorStrainStack.read(path, format="npy")

        assert list(restored.detector_names) == ["H1", "L1"]

    @pytest.mark.requires_gwpy
    def test_json_sidecar_is_created(self, tmp_path):
        """Test that a JSON sidecar is created."""
        stack = _make_stack()
        path = tmp_path / "output.npy"

        stack.write(path, format="npy")

        sidecar = tmp_path / "output.json"
        assert sidecar.exists()


# ---------------------------------------------------------------------------
# GWF write
# ---------------------------------------------------------------------------


class TestGWFWrite:
    """Test that writing a GWF file works."""

    @pytest.mark.requires_gwpy
    def test_file_exists_and_nonempty(self, tmp_path):
        """Test that the file exists and is non-empty."""
        pytest.importorskip("gwpy.io.gwf")
        stack = _make_stack()
        path = tmp_path / "output.gwf"

        stack.write(path, format="gwf")

        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.requires_gwpy
    def test_readable_by_gwpy(self, tmp_path):
        """Test that the file is readable by gwpy."""
        pytest.importorskip("gwpy.io.gwf")
        stack = _make_stack()
        path = tmp_path / "output.gwf"

        stack.write(path, format="gwf")

        ts = TimeSeries.read(str(path), channel="H1", start=_T0, end=_T0 + _N_SAMPLES / _SAMPLE_RATE)
        assert len(ts) > 0


# ---------------------------------------------------------------------------
# TXT write
# ---------------------------------------------------------------------------


class TestTxtWrite:
    """Test that writing a TXT file works."""

    @pytest.mark.requires_gwpy
    def test_file_exists_and_nonempty(self, tmp_path):
        """Test that the file exists and is non-empty."""
        stack = _make_stack()
        path = tmp_path / "output.txt"

        stack.write(path, format="txt")

        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.requires_gwpy
    def test_header_starts_with_t0(self, tmp_path):
        """Test that the header starts with t0."""
        stack = _make_stack()
        path = tmp_path / "output.txt"

        stack.write(path, format="txt")

        first_line = path.read_text().splitlines()[0]
        # numpy savetxt prefixes header lines with "# "
        assert first_line.startswith("# t0=")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestReadErrors:
    """Test that reading unsupported formats raises NotImplementedError."""

    def test_read_gwf_raises_not_implemented(self, tmp_path):
        """Reading a GWF file raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            DetectorStrainStack.read(tmp_path / "x.gwf", format="gwf")  # type: ignore[arg-type]

    def test_read_txt_raises_not_implemented(self, tmp_path):
        """Reading a TXT file raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            DetectorStrainStack.read(tmp_path / "x.txt", format="txt")  # type: ignore[arg-type]

    @pytest.mark.requires_gwpy
    def test_write_unknown_format_raises(self, tmp_path):
        """Writing an unknown format raises ValueError."""
        stack = _make_stack()
        with pytest.raises(ValueError, match="Unknown format"):
            stack.write(tmp_path / "x", format="csv")  # type: ignore[arg-type]
