"""Tests for ``DetectorStrainStack``."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock_signal.multichannel import DetectorStrainStack


def _ch(n: int = 8, fs: float = 128.0, t0: float = 0.0, fill: float = 1.0) -> TimeSeries:
    return TimeSeries(np.full(n, fill), t0=t0, sample_rate=fs)


def test_from_mapping_order_and_shape():
    """Rows follow ``detector_names``; ``data`` has shape (n_det, n_samples)."""
    h1 = _ch(fill=1.0)
    l1 = _ch(fill=2.0)
    strains = {"H1": h1, "L1": l1}
    stack = DetectorStrainStack.from_mapping(["L1", "H1"], strains)
    assert stack.detector_names == ("L1", "H1")
    assert stack.data.shape == (2, 8)
    assert np.allclose(stack.data[0], 2.0)
    assert np.allclose(stack.data[1], 1.0)


def test_getitem_name_is_same_as_index():
    """String and integer access return the identical GWpy object."""
    strains = {"H1": _ch(fill=3.0), "L1": _ch(fill=4.0)}
    stack = DetectorStrainStack.from_mapping(["H1", "L1"], strains)
    assert stack["H1"] is stack[0]
    assert stack["L1"] is stack[1]


def test_to_dict_keys():
    """``to_dict`` contains all detectors."""
    names = ["H1", "L1", "V1"]
    strains = {n: _ch(fill=float(i)) for i, n in enumerate(names)}
    stack = DetectorStrainStack.from_mapping(names, strains)
    again = stack.to_dict()
    assert set(again) == set(names)


def test_missing_key_raises():
    """Missing detector in mapping raises ``KeyError``."""
    with pytest.raises(KeyError, match="Missing strain"):
        DetectorStrainStack.from_mapping(["H1", "X1"], {"H1": _ch()})


def test_mismatched_length_raises():
    """Different lengths raise ``ValueError``."""
    strains = {"H1": _ch(n=4), "L1": _ch(n=8)}
    with pytest.raises(ValueError, match="length"):
        DetectorStrainStack.from_mapping(["H1", "L1"], strains)


def test_mismatched_epoch_raises():
    """Same rate and length but a different ``t0`` is not an aligned stack.

    The check compares ``t0`` and ``dx`` instead of materialising both time indices, so this
    pins that the cheap comparison still rejects a genuine misalignment.
    """
    strains = {"H1": _ch(t0=0.0), "L1": _ch(t0=1.0)}
    with pytest.raises(ValueError, match="time grid does not match"):
        DetectorStrainStack.from_mapping(["H1", "L1"], strains)


def test_mismatched_sample_rate_names_the_channel():
    """A different sample rate is rejected, and the message says which channel.

    GWpy's own compatibility check raises instead of returning False, so its exception is
    re-raised with the channel index -- which is the part the caller cannot work out alone.
    """
    strains = {"H1": _ch(fs=128.0), "L1": _ch(fs=256.0)}
    with pytest.raises(ValueError, match="Channel 1 is not compatible") as raised:
        DetectorStrainStack.from_mapping(["H1", "L1"], strains)
    assert "sample sizes do not match" in str(raised.value)


def test_matching_channels_accepted_without_materialising_the_index():
    """Equal ``t0``, ``dx`` and length must be accepted, and the index left unbuilt.

    GWpy creates the time index lazily, so ``xindex`` being absent afterwards is what shows the
    validation no longer allocates one array per channel per segment.
    """
    strains = {"H1": _ch(t0=5.0), "L1": _ch(t0=5.0)}
    stack = DetectorStrainStack.from_mapping(["H1", "L1"], strains)
    assert stack.detector_names == ("H1", "L1")
    assert all(getattr(s, "_xindex", None) is None for s in strains.values())


def test_irregularly_sampled_channels_compared_directly():
    """Series built from an explicit ``times=`` array have no ``dx``, so fall back to the index.

    Nothing in gwmock_signal produces these, but the validation must not raise ``AttributeError``
    on a series a caller hands it.
    """
    times = np.array([0.0, 0.5, 1.7, 3.0])
    a = TimeSeries(np.zeros(4), times=times)
    # Matching indices must be accepted: `dx` raises for these, so this is the path that
    # exercises the fallback rather than the t0/dx comparison.
    stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": a, "L1": TimeSeries(np.ones(4), times=times)})
    assert stack.data.shape == (2, 4)
    # Differing indices are rejected by gwpy's own compatibility check, before the fallback.
    with pytest.raises(ValueError, match="Channel 1 is not compatible"):
        DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": a, "L1": TimeSeries(np.zeros(4), times=times + 1.0)})


def test_empty_detector_names_raises():
    """Empty ``detector_names`` is rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        DetectorStrainStack.from_mapping([], {"H1": _ch()})


def test_unknown_name_getitem():
    """Bad string key raises ``KeyError``."""
    stack = DetectorStrainStack.from_mapping(["H1"], {"H1": _ch()})
    with pytest.raises(KeyError):
        _ = stack["V1"]
