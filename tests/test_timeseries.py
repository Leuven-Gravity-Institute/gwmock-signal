"""The package's own time series: NumPy or JAX samples, GPS timing, and conversions.

Every method has to keep the samples in their own array library, a JAX array on its device
included, and keep each sample at its GPS time. The timing checks are exact or stated in samples
of the float64 resolution at GPS magnitudes, because a sample moved by a fraction of a spacing is
the error the rest of the package is built to avoid.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from gwmock_signal.sampling_grid import SamplingGrid
from gwmock_signal.timeseries import TimeSeries, as_timeseries, register_pytree

_T0 = 1_400_000_000.25
_FS = 4096.0


def _series(n: int = 16, **kwargs: object) -> TimeSeries:
    return TimeSeries(np.arange(n, dtype=float), **{"t0": _T0, "sample_rate": _FS, **kwargs})


class TestConstruction:
    """The timing is validated; the samples are kept as given."""

    def test_a_numpy_array_is_kept_as_given(self):
        """No copy and no conversion, so a caller's array is the series' array."""
        data = np.arange(4.0)
        assert TimeSeries(data, t0=0.0, sample_rate=1.0).data is data

    def test_a_sequence_becomes_a_numpy_array(self):
        """Plain Python data is converted, so every series has an array namespace."""
        series = TimeSeries([1.0, 2.0], t0=0, sample_rate=2)

        assert isinstance(series.data, np.ndarray)
        assert series.t0 == 0.0
        assert isinstance(series.t0, float)
        assert isinstance(series.sample_rate, float)

    def test_a_scalar_has_no_time_axis(self):
        """Time is the last axis, so there must be one."""
        with pytest.raises(ValueError, match="at least one axis"):
            TimeSeries(np.float64(1.0), t0=0.0, sample_rate=1.0)

    @pytest.mark.parametrize("sample_rate", [0.0, -1.0, float("nan"), float("inf")])
    def test_the_sample_rate_must_be_positive_and_finite(self, sample_rate):
        """A lattice needs a positive, finite rate."""
        with pytest.raises(ValueError, match="sample_rate must be positive and finite"):
            TimeSeries(np.zeros(2), t0=0.0, sample_rate=sample_rate)

    @pytest.mark.parametrize("t0", [float("nan"), float("inf")])
    def test_t0_must_be_finite(self, t0):
        """A series must start somewhere."""
        with pytest.raises(ValueError, match="t0 must be finite"):
            TimeSeries(np.zeros(2), t0=t0, sample_rate=1.0)

    def test_t0_must_be_a_single_number(self):
        """An array of start times is rejected, not silently accepted unchecked."""
        with pytest.raises(TypeError, match="t0 must be a single number"):
            TimeSeries(np.zeros(2), t0=np.array([1.0, 2.0]), sample_rate=1.0)

    def test_equality_is_identity(self):
        """Comparing arrays elementwise has no single truth value, so equality is identity."""
        series = _series()
        same = series

        assert series == same
        assert series != series.with_data(series.data)


class TestDerivedQuantities:
    """Timing derived from ``t0``, the sample rate and the length of the last axis."""

    def test_spacing_duration_and_end(self):
        """Each follows from the rate and the number of samples."""
        series = _series(n=8192)

        assert series.dt == 1.0 / _FS
        assert series.n_samples == 8192
        assert series.duration == 2.0
        assert series.end == _T0 + 2.0

    def test_end_is_on_the_lattice(self):
        """``end`` is where sample ``n_samples`` would be, by the arithmetic the grid uses.

        At a rate whose spacing is not exact in binary and ``t0 = 0``, ``n / rate`` and
        ``n * (1 / rate)`` differ in the last bit, so this pins which one is used.
        """
        series = TimeSeries(np.zeros(5), t0=0.0, sample_rate=3000.0)

        assert series.end == float(series.grid.time_of(5))
        assert series.end == series.crop(start=series.end).t0
        assert series.duration == 5 * series.dt

    def test_value_is_the_data(self):
        """``value`` is GWpy's name for the samples, kept so code written against GWpy reads on."""
        series = _series()

        assert series.value is series.data

    @pytest.mark.parametrize("t0", [_T0, 0.0])
    def test_the_times_are_the_sampling_grid_times(self, t0):
        """Exactly the lattice ``SamplingGrid`` uses, so the two never disagree on where a sample is.

        At ``t0 = 0`` the last bit of each time is visible, so the two must also do the same
        arithmetic, not merely agree after rounding to a GPS magnitude.
        """
        series = _series(n=1000, sample_rate=1000.0, t0=t0)
        grid = SamplingGrid(epoch=t0, sampling_frequency=1000.0)

        assert np.array_equal(series.times, grid.time_of(np.arange(1000)))
        assert series.grid == grid

    def test_leading_axes_share_the_timing(self):
        """A ``(n_detectors, n_samples)`` series is timed by its last axis."""
        series = TimeSeries(np.zeros((3, 100)), t0=_T0, sample_rate=_FS)

        assert series.n_samples == 100
        assert series.times.shape == (100,)


class TestCrop:
    """``[start, end)`` in GPS seconds, judged against the float64 resolution of GPS times."""

    def test_a_bound_on_a_sample_keeps_that_sample(self):
        """At GPS magnitudes a sample's own timestamp must count as on it, or it would be dropped."""
        series = _series(n=4096)
        times = series.times

        cropped = series.crop(start=float(times[1000]), end=float(times[2000]))

        assert cropped.n_samples == 1000
        assert cropped.data[0] == 1000
        assert cropped.t0 == float(times[1000])

    def test_a_bound_computed_another_way_still_keeps_its_sample(self):
        """``t0 + k / rate`` lands a rounding error either side of sample ``k`` at GPS magnitudes.

        At 3 kHz near 1.4e9 s that error is ~7e-4 samples; for about half of all ``k`` it falls
        after the sample, and without the tolerance ``crop`` would drop sample ``k``.
        """
        series = _series(n=3000, sample_rate=3000.0)

        first = [int(series.crop(start=_T0 + k / 3000.0).data[0]) for k in range(3000)]

        assert first == list(range(3000))

    def test_a_bound_between_samples_starts_at_the_next_sample(self):
        """Half a sample past sample 10 keeps sample 11 onwards; the end is exclusive the same way."""
        series = _series(n=100)

        cropped = series.crop(start=_T0 + 10.5 * series.dt, end=_T0 + 20.5 * series.dt)

        assert list(cropped.data) == list(range(11, 21))

    def test_bounds_outside_the_series_are_clipped(self):
        """Cropping wider than the series keeps it whole; cropping past it leaves nothing."""
        series = _series(n=10)

        assert series.crop(start=_T0 - 5.0, end=_T0 + 5.0).n_samples == 10
        empty = series.crop(start=_T0 + 5.0)
        assert empty.n_samples == 0
        assert series.crop(end=_T0 - 1.0).n_samples == 0

    def test_open_bounds_keep_the_ends(self):
        """``None`` means from the first or to the last sample."""
        series = _series(n=10)

        assert series.crop().n_samples == 10
        assert list(series.crop(start=_T0 + 3 * series.dt).data) == list(range(3, 10))

    @pytest.mark.parametrize("bound", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize("which", ["start", "end"])
    def test_a_bound_that_is_not_finite_is_rejected(self, which, bound):
        """Said plainly, rather than as a failure to round NaN to an integer."""
        with pytest.raises(ValueError, match=f"{which} must be finite"):
            _series().crop(**{which: bound})

    def test_an_end_before_the_start_is_rejected(self):
        """An inverted interval is a caller error, not an empty result."""
        with pytest.raises(ValueError, match="before start"):
            _series().crop(start=_T0 + 1.0, end=_T0)

    def test_leading_axes_are_cropped_along_time_only(self):
        """Every row loses the same samples."""
        data = np.arange(30.0).reshape(3, 10)
        series = TimeSeries(data, t0=0.0, sample_rate=1.0)

        assert np.array_equal(series.crop(start=2.0, end=5.0).data, data[:, 2:5])


class TestPadCopyAndReplace:
    """New series from an old one keep its timing, name and unit."""

    def test_pad_adds_zeros_and_keeps_the_samples_at_their_times(self):
        """``t0`` moves back by the leading zeros; the old samples stay on their lattice points."""
        series = _series(n=4, name="H1", unit="strain")

        padded = series.pad(before=3, after=2)

        assert list(padded.data) == [0, 0, 0, 0, 1, 2, 3, 0, 0]
        assert padded.t0 == _T0 - 3 * series.dt
        index = float(padded.grid.index_of(series.t0))
        assert abs(index - 3) <= padded.grid.lattice_tolerance_samples(series.t0)
        assert (padded.name, padded.unit) == ("H1", "strain")

    def test_pad_pads_every_row(self):
        """Leading axes are padded along time only."""
        padded = TimeSeries(np.ones((2, 3)), t0=0.0, sample_rate=1.0).pad(after=2)

        assert np.array_equal(padded.data, [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])

    def test_a_negative_pad_is_rejected(self):
        """Removing samples is ``crop``'s job."""
        with pytest.raises(ValueError, match="must not be negative"):
            _series().pad(before=-1)

    def test_a_copy_does_not_share_samples(self):
        """Writing to the copy leaves the original alone."""
        series = _series(name="H1")
        copied = series.copy()
        copied.data[0] = -1.0

        assert series.data[0] == 0.0
        assert (copied.t0, copied.sample_rate, copied.name) == (series.t0, series.sample_rate, series.name)

    def test_with_data_keeps_everything_else(self):
        """Only the samples change."""
        series = _series(name="L1", unit="strain")
        replaced = series.with_data(np.ones(3))

        assert (replaced.t0, replaced.sample_rate, replaced.name, replaced.unit) == (_T0, _FS, "L1", "strain")

    def test_compatibility_is_the_rate_and_the_unit(self):
        """Series can be combined when they share a rate and a unit, wherever they start."""
        series = _series(unit="strain")

        assert series.is_compatible(_series(t0=0.0, unit="strain"))
        assert not series.is_compatible(_series(sample_rate=2048.0, unit="strain"))
        assert not series.is_compatible(_series(unit=None))


class TestGWpy:
    """Round trips with GWpy keep the samples, the timing, the name and the unit exactly."""

    @pytest.fixture
    def gwpy_timeseries(self):
        """GWpy's ``timeseries`` module, or a skip when GWpy is not installed."""
        return pytest.importorskip("gwpy.timeseries")

    @pytest.mark.parametrize("sample_rate", [4096.0, 1000.0, 3000.0, 44100.0])
    def test_to_gwpy_and_back(self, gwpy_timeseries, sample_rate):
        """Rates that are not powers of two included: GWpy keeps the rate it is given."""
        series = _series(sample_rate=sample_rate, name="H1:STRAIN", unit="strain")

        back = TimeSeries.from_gwpy(series.to_gwpy())

        assert np.array_equal(back.data, series.data)
        assert (back.t0, back.sample_rate, back.name, back.unit) == (_T0, sample_rate, "H1:STRAIN", "strain")

    def test_from_gwpy_and_back(self, gwpy_timeseries):
        """A GWpy series built from a spacing survives the trip with the same spacing and start."""
        original = gwpy_timeseries.TimeSeries(np.arange(5.0), t0=_T0, dt=1.0 / 1000.0, name="V1", unit="strain")

        back = TimeSeries.from_gwpy(original).to_gwpy()

        assert back.dx == original.dx
        assert back.x0 == original.x0
        assert (back.name, back.unit) == (original.name, original.unit)
        assert np.array_equal(back.value, original.value)

    def test_a_dimensionless_gwpy_series_has_no_unit(self, gwpy_timeseries):
        """GWpy's dimensionless unit is the absence of one here."""
        assert TimeSeries.from_gwpy(gwpy_timeseries.TimeSeries([1.0, 2.0])).unit is None

    def test_from_gwpy_shares_the_samples(self, gwpy_timeseries):
        """As documented: the samples are GWpy's array, not a copy."""
        original = gwpy_timeseries.TimeSeries(np.zeros(3), t0=0.0, sample_rate=1.0)

        assert np.shares_memory(TimeSeries.from_gwpy(original).data, original.value)

    def test_gwpy_takes_one_dimension_only(self):
        """A multi-row series has no GWpy ``TimeSeries`` counterpart."""
        with pytest.raises(ValueError, match="one-dimensional"):
            TimeSeries(np.zeros((2, 3)), t0=0.0, sample_rate=1.0).to_gwpy()


class TestPyCBC:
    """Round trips with PyCBC keep the samples and the timing."""

    def test_to_pycbc_and_back(self):
        """PyCBC has no name or unit, so those are given back explicitly."""
        pytest.importorskip("pycbc.types")
        series = _series(n=32)

        converted = series.to_pycbc()
        back = TimeSeries.from_pycbc(converted, name="H1", unit="strain")

        assert float(converted.delta_t) == series.dt
        assert np.array_equal(back.data, series.data)
        assert (back.t0, back.sample_rate, back.name, back.unit) == (_T0, _FS, "H1", "strain")

    def test_pycbc_takes_one_dimension_only(self):
        """A multi-row series has no PyCBC ``TimeSeries`` counterpart."""
        with pytest.raises(ValueError, match="one-dimensional"):
            TimeSeries(np.zeros((2, 3)), t0=0.0, sample_rate=1.0).to_pycbc()


class TestMissingExtras:
    """A conversion whose package is missing names the extra that installs it."""

    @pytest.mark.parametrize(
        ("call", "package", "extra"),
        [
            (lambda series: series.to_gwpy(), "gwpy", "gwpy"),
            (lambda series: series.to_pycbc(), "pycbc", "pycbc"),
            (lambda series: series.to_jax(), "jax", "jax"),
        ],
    )
    def test_the_extra_is_named(self, monkeypatch, call, package, extra):
        """The message gives the install command, and the original error is chained."""
        real_import = importlib.import_module

        def no_package(name, *args, **kwargs):
            if name == package or name.startswith(f"{package}."):
                raise ModuleNotFoundError(f"No module named {package!r}", name=package)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", no_package)

        with pytest.raises(ModuleNotFoundError, match=rf"pip install 'gwmock-signal\[{extra}\]'") as raised:
            call(_series())
        assert isinstance(raised.value.__cause__, ModuleNotFoundError)

    def test_a_missing_dependency_of_the_extra_is_not_relabelled(self, monkeypatch):
        """Only the extra's own package is reported as the missing extra; anything deeper is re-raised."""
        real_import = importlib.import_module

        def broken_gwpy(name, *args, **kwargs):
            if name == "gwpy.timeseries":
                raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", broken_gwpy)

        with pytest.raises(ModuleNotFoundError, match="No module named 'h5py'") as raised:
            _series().to_gwpy()
        assert "gwmock-signal[" not in str(raised.value)


def test_importing_the_module_does_not_import_jax():
    """NumPy-only users never pay for JAX."""
    code = "import sys, gwmock_signal.timeseries; print('jax' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip() == "False"


def test_the_package_exports_it():
    """``gwmock_signal.TimeSeries`` is this class."""
    import gwmock_signal

    assert gwmock_signal.TimeSeries is TimeSeries
    assert "TimeSeries" in gwmock_signal.__all__


class TestJAX:
    """JAX samples stay JAX, on their device, and a series passes through ``jit``, ``vmap`` and ``grad``."""

    @pytest.fixture(autouse=True)
    def _jax(self):
        """JAX in double precision for these tests only, or a skip when JAX is not installed."""
        self.jax = pytest.importorskip("jax")
        previous = self.jax.config.jax_enable_x64
        self.jax.config.update("jax_enable_x64", True)
        yield
        self.jax.config.update("jax_enable_x64", previous)

    def _jax_series(self, n: int = 16, **kwargs: object) -> TimeSeries:
        return TimeSeries(self.jax.numpy.arange(n, dtype=float), **{"t0": _T0, "sample_rate": _FS, **kwargs})

    def test_methods_keep_jax_arrays(self):
        """The namespace is JAX, and crop, pad, copy and times all stay in it."""
        series = self._jax_series()

        assert series.xp.__name__ == "jax.numpy"
        for result in (series.crop(start=_T0 + 2 * series.dt), series.pad(1, 1), series.copy()):
            assert isinstance(result.data, self.jax.Array)
        assert isinstance(series.times, self.jax.Array)
        assert np.array_equal(np.asarray(series.times), _series().times)

    def test_moving_between_numpy_and_jax(self):
        """``to_jax`` and ``to_numpy`` move the samples and nothing else."""
        series = _series(name="H1")

        on_device = series.to_jax(self.jax.devices()[0])
        back = on_device.to_numpy()

        assert isinstance(on_device.data, self.jax.Array)
        assert isinstance(back.data, np.ndarray)
        assert np.array_equal(back.data, series.data)
        assert (back.t0, back.name) == (_T0, "H1")

    def test_jit_passes_a_series_through(self):
        """Inside ``jit`` the start time is traced, and building a new series from it still works."""
        register_pytree()

        doubled = self.jax.jit(lambda series: series.with_data(series.data * 2))(self._jax_series(name="H1"))

        assert isinstance(doubled, TimeSeries)
        assert doubled.t0 == _T0
        assert doubled.name == "H1"
        assert np.array_equal(np.asarray(doubled.data), 2 * np.arange(16.0))

    def test_vmap_maps_over_start_times(self):
        """A batch of series with different starts is one pytree whose ``t0`` leaf is mapped."""
        register_pytree()
        starts = [_T0, _T0 + 1.0, _T0 + 7.5]
        batch = self.jax.tree.map(
            lambda *leaves: self.jax.numpy.stack(leaves), *(self._jax_series(t0=start) for start in starts)
        )

        times = self.jax.vmap(lambda series: series.times)(batch)

        for row, start in zip(np.asarray(times), starts, strict=True):
            assert np.array_equal(row, _series(t0=start).times)

    def test_grad_reaches_the_samples(self):
        """The samples are a differentiable leaf."""
        register_pytree()

        gradient = self.jax.grad(lambda series: self.jax.numpy.sum(series.data**2))(self._jax_series(n=4))

        assert np.array_equal(np.asarray(gradient.data), 2 * np.arange(4.0))

    def test_registering_twice_is_harmless(self):
        """Several modules register it; JAX would otherwise reject the second registration."""
        register_pytree()
        register_pytree()

    def test_concurrent_first_registrations_register_once(self):
        """Threads that call it together, before anything has registered, do not race into JAX twice.

        Run in a fresh process, since this one has already registered. A barrier releases every
        thread at once, and JAX's registration is slowed so they are all inside together; a cache
        alone would let each of them in, and JAX would reject every registration after the first.
        """
        code = (
            "import threading, time, jax\n"
            "original = jax.tree_util.register_pytree_node\n"
            "def slow(*args):\n    time.sleep(0.05)\n    original(*args)\n"
            "jax.tree_util.register_pytree_node = slow\n"
            "from gwmock_signal.timeseries import register_pytree\n"
            "barrier = threading.Barrier(16); errors = []\n"
            "def call():\n"
            "    barrier.wait()\n"
            "    try:\n        register_pytree()\n"
            "    except Exception as exc:\n        errors.append(repr(exc))\n"
            "threads = [threading.Thread(target=call) for _ in range(16)]\n"
            "[t.start() for t in threads]; [t.join() for t in threads]\n"
            "print(errors)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

        assert result.stdout.strip() == "[]"


class TestJAXIn32BitMode:
    """Without 64-bit mode JAX cannot hold GPS times, so the methods that need them refuse."""

    @pytest.fixture(autouse=True)
    def _x32(self):
        """JAX in its default 32-bit mode for these tests, or a skip when JAX is not installed."""
        self.jax = pytest.importorskip("jax")
        previous = self.jax.config.jax_enable_x64
        self.jax.config.update("jax_enable_x64", False)
        yield
        self.jax.config.update("jax_enable_x64", previous)

    def _jax_series(self) -> TimeSeries:
        return TimeSeries(self.jax.numpy.zeros(4), t0=_T0, sample_rate=_FS)

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
    def test_times_refuse(self):
        """In single precision every one of these samples would get the same time."""
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            _ = self._jax_series().times

    def test_moving_to_jax_refuses(self):
        """``to_jax`` would round double-precision samples to single."""
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            _series().to_jax()

    @pytest.mark.parametrize("transform", ["jit", "vmap", "grad"])
    def test_a_transformation_refuses(self, transform):
        """Each would trace the start time in single precision."""
        register_pytree()
        jax = self.jax
        series = self._jax_series()
        calls = {
            "jit": lambda: jax.jit(lambda s: s.data)(series),
            "vmap": lambda: jax.vmap(lambda s: s.data)(jax.tree.map(lambda leaf: jax.numpy.stack([leaf]), series)),
            "grad": lambda: jax.grad(lambda s: jax.numpy.sum(s.data))(series),
        }

        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            calls[transform]()

    def test_device_put_refuses(self):
        """Putting the series on a device would make its start time a single-precision array."""
        register_pytree()

        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            self.jax.device_put(_series())

    def test_structural_uses_still_work(self):
        """A tree map or a copy to the host leaves the start time a Python float, so needs no x64."""
        register_pytree()

        mapped = self.jax.tree.map(lambda leaf: leaf, _series(name="H1"))
        fetched = self.jax.device_get(self._jax_series())

        assert (mapped.t0, mapped.name) == (_T0, "H1")
        assert fetched.t0 == _T0
        assert isinstance(fetched.data, np.ndarray)

    def test_numpy_times_are_unaffected(self):
        """JAX's mode does not concern the times of NumPy samples."""
        assert _series().times.dtype == np.float64


class TestAsTimeSeries:
    """The coercion every function that takes a time series applies to its input."""

    def test_a_native_series_is_returned_as_is(self):
        """No copy, no conversion."""
        series = _series()

        assert as_timeseries(series) is series

    def test_a_gwpy_series_is_converted(self):
        """Samples, timing, name and unit carry over."""
        pytest.importorskip("gwpy.timeseries")
        converted = as_timeseries(_series(name="H1", unit="strain").to_gwpy())

        assert isinstance(converted, TimeSeries)
        assert (converted.t0, converted.sample_rate, converted.name, converted.unit) == (_T0, _FS, "H1", "strain")

    @pytest.mark.parametrize("other", [np.zeros(4), [1.0, 2.0], None])
    def test_anything_else_is_rejected(self, other):
        """A bare array has no timing, so it is not a time series."""
        with pytest.raises(TypeError, match=r"expected a gwmock_signal\.TimeSeries or a gwpy\.timeseries\.TimeSeries"):
            as_timeseries(other)

    def test_checking_does_not_import_gwpy(self):
        """Rejecting a non-series must not pull in GWpy just to compare against its class."""
        code = (
            "import sys; from gwmock_signal.timeseries import as_timeseries\n"
            "try:\n    as_timeseries(1.0)\nexcept TypeError:\n    pass\n"
            "print('gwpy' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

        assert result.stdout.strip() == "False"
