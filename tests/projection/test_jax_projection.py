"""Tests for the JAX detector-projection building blocks."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

jax = pytest.importorskip("jax", reason="jax not installed")
jax.config.update("jax_enable_x64", True)  # GPS times / Julian dates need float64

from gwmock_signal.projection.jax_projection import gmst_rad  # noqa: E402

# GPS times spanning the 36 s (pre-2017) and 37 s leap-second eras.
_GPS_TIMES = [1126259462.4, 1187008882.4, 1238166018.0, 1370000000.0]


def _astropy_reference(t_gps: float) -> tuple[float, float, float]:
    """Return (GMST, TAI-UTC, UT1-UTC) from Astropy for one GPS time."""
    t = Time(float(t_gps), format="gps", scale="utc", location=(0, 0))
    gmst = float(t.sidereal_time("mean").rad)
    jd_utc = t.utc.jd1 + t.utc.jd2
    tai_minus_utc = (t.tai.jd1 + t.tai.jd2 - jd_utc) * 86400.0
    dut1 = (t.ut1.jd1 + t.ut1.jd2 - jd_utc) * 86400.0
    return gmst, tai_minus_utc, dut1


def _wrapped_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a - b in radians."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize("t_gps", _GPS_TIMES)
def test_gmst_matches_astropy(t_gps: float) -> None:
    """Fed Astropy's leap seconds and DUT1, gmst_rad reproduces Astropy GMST to ~1e-6 rad.

    This anchors the JAX implementation against an external reference (Astropy's
    IAU sidereal time) rather than only internal consistency.
    """
    reference, tai_minus_utc, dut1 = _astropy_reference(t_gps)
    got = float(gmst_rad(t_gps, tai_minus_utc=tai_minus_utc, dut1=dut1))
    assert abs(_wrapped_diff(got, reference)) < 1e-6


def test_gmst_default_offsets_are_dut1_limited() -> None:
    """With default offsets (leap=37, dut1=0) a post-2017 time still matches to ~1e-4 rad.

    Documents the accuracy of the defaults: the only error is the neglected DUT1.
    """
    t_gps = 1370000000.0  # ~2023, leap-second era 37 s
    reference, _, _ = _astropy_reference(t_gps)
    got = float(gmst_rad(t_gps))  # defaults
    assert abs(_wrapped_diff(got, reference)) < 1e-4


def test_gmst_scalar_and_array_shapes() -> None:
    """gmst_rad returns a scalar for scalar input and preserves array shape."""
    assert gmst_rad(_GPS_TIMES[0]).shape == ()
    out = np.asarray(gmst_rad(np.array(_GPS_TIMES)))
    assert out.shape == (len(_GPS_TIMES),)
    assert ((out >= 0.0) & (out < 2.0 * np.pi)).all()


def test_gmst_is_jit_traceable() -> None:
    """gmst_rad is JAX-traceable (jit) and agrees with the eager result."""
    t_gps = _GPS_TIMES[0]
    eager = float(gmst_rad(t_gps))
    jitted = float(jax.jit(gmst_rad)(t_gps))
    # JIT may fuse float ops; agreement to 9 significant figures is far below the µs anchor.
    assert eager == pytest.approx(jitted, rel=1e-9)
