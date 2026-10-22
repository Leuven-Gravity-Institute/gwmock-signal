"""The LAL backend's frequency-to-time conditioning must not fold the waveform back on itself.

A periodic inverse transform of a spectrum cut hard at ``minimum_frequency`` rings with a 1/t tail,
and a buffer sized for the inspiral above the cutoff with a fixed tenth after coalescence leaves the
ringdown of a heavy remnant no room: both wrap round the buffer. For a 2302+248 solar-mass binary
at a 5 Hz cutoff that put 0.3 of the peak strain on the first sample of a 4 s buffer.

The references here are built from LALSimulation directly, not from the backend: the same model,
started where LALSimulation's own conditioning starts it and tapered with a raised cosine up to the
cutoff, on a buffer eight times longer -- and, for the band above the cutoff, the model's spectrum
cut at the cutoff, which is what the backend promises to leave untouched.
"""

from __future__ import annotations

import logging
import math

try:
    import lal
except ModuleNotFoundError as exc:  # Python 3.15: lalsuite does not install yet; its tests are skipped
    if exc.name != "lal":
        raise
    lal = None
try:
    import lalsimulation
except ModuleNotFoundError as exc:  # Python 3.15: lalsuite does not install yet; its tests are skipped
    if exc.name != "lalsimulation":
        raise
    lalsimulation = None
import numpy as np
import pytest

from gwmock_signal.waveform.backends.lal import LALSimulationBackend

_FS = 256.0
_F_MIN = 5.0

#: Heavy precessing binaries at a 5 Hz cutoff, where the previous conditioning was worst: an
#: intermediate-mass one whose cutoff lies above its innermost stable orbit, and a heavy stellar one.
_SOURCES = {
    "imbh": {
        "detector_frame_mass_1": 2302.4756575060846,
        "detector_frame_mass_2": 248.50156006705728,
        "spin_1x": 0.25546819814959526,
        "spin_1y": -0.04368782244375504,
        "spin_1z": 0.25309552265225915,
        "spin_2x": -0.43435885402297864,
        "spin_2y": 0.3721456501358579,
        "spin_2z": 0.54509962945348,
        "luminosity_distance": 17188.9,
        "inclination": 1.368138795003217,
        "coa_phase": 4.841402586182077,
    },
    "heavy_bbh": {
        "detector_frame_mass_1": 450.91599520000005,
        "detector_frame_mass_2": 438.16913359999995,
        "spin_1x": 0.09081025684208817,
        "spin_1y": 0.0717410592815908,
        "spin_1z": 0.26547104863888504,
        "spin_2x": 0.07932091562930543,
        "spin_2y": -0.03725986585443007,
        "spin_2z": 0.20102831825451478,
        "luminosity_distance": 109496.7,
        "inclination": 1.3750579856966285,
        "coa_phase": 6.247814799893307,
    },
}
_APPROXIMANT = "IMRPhenomXPHM"
_F_REF = 5.0


def _generated(name: str):
    """The backend's plus polarization, and the index coalescence sits on."""
    plus = LALSimulationBackend(f_ref=_F_REF).generate_td_waveform(_APPROXIMANT, 0.0, _FS, _F_MIN, **_SOURCES[name])[
        "plus"
    ]
    return np.asarray(plus.data), round(-plus.t0 * _FS)


def _start_frequency(source: dict[str, float]) -> float:
    """Where LALSimulation's frequency-domain conditioning starts this waveform."""
    m1, m2 = source["detector_frame_mass_1"], source["detector_frame_mass_2"]
    lowest_isco = 1.0 / (27.0 * math.pi * (m1 + m2) * lal.MTSUN_SI)
    frequency = min(_F_MIN, lowest_isco)
    chirp = lalsimulation.SimInspiralChirpTimeBound(
        frequency, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, source["spin_1z"], source["spin_2z"]
    )
    return lalsimulation.SimInspiralChirpStartFrequencyBound(1.1 * chirp, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI)


def _model_spectrum(source: dict[str, float], n_samples: int, start: float) -> tuple[np.ndarray, np.ndarray]:
    """LALSimulation's plus spectrum from *start*, on the grid of an *n_samples* buffer."""
    delta_f = _FS / n_samples
    hp, _ = lalsimulation.SimInspiralChooseFDWaveform(
        source["detector_frame_mass_1"] * lal.MSUN_SI,
        source["detector_frame_mass_2"] * lal.MSUN_SI,
        source["spin_1x"],
        source["spin_1y"],
        source["spin_1z"],
        source["spin_2x"],
        source["spin_2y"],
        source["spin_2z"],
        source["luminosity_distance"] * 1e6 * lal.PC_SI,
        source["inclination"],
        source["coa_phase"],
        0.0,
        0.0,
        0.0,
        delta_f,
        start,
        _FS / 2.0,
        _F_REF,
        lal.CreateDict(),
        lalsimulation.GetApproximantFromString(_APPROXIMANT),
    )
    spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
    data = np.asarray(hp.data.data)
    spectrum[: min(data.size, spectrum.size)] = data[: spectrum.size]
    return spectrum, np.arange(spectrum.size) * delta_f


def _placed(spectrum: np.ndarray, n_samples: int, index: int) -> np.ndarray:
    """The time series of a coalescence-at-zero *spectrum*, with coalescence moved to *index*."""
    return np.roll(np.fft.irfft(spectrum, n=n_samples) * _FS, index)


def _long_reference(name: str, n_samples: int, merger_index: int, *, tapered: bool):
    """The model on a buffer eight times longer, coalescence placed so the short buffer sits mid-way.

    Returns the long series and the index the short buffer's first sample corresponds to.
    """
    source = _SOURCES[name]
    start = _start_frequency(source)
    long_samples = 8 * n_samples
    spectrum, frequencies = _model_spectrum(source, long_samples, start)
    if tapered:
        ramp = np.clip((frequencies - start) / (_F_MIN - start), 0.0, 1.0)
        spectrum = spectrum * np.where(frequencies < start, 0.0, 0.5 - 0.5 * np.cos(np.pi * ramp))
    else:
        spectrum = np.where(frequencies >= _F_MIN, spectrum, 0.0)
    offset = long_samples // 2 - n_samples // 2
    return _placed(spectrum, long_samples, offset + merger_index), offset


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_the_buffer_matches_the_same_waveform_on_a_much_longer_one(name):
    """Nothing wraps: the buffer is the corresponding stretch of a buffer eight times longer.

    Measured here (256 Hz): 2.1e-06 (intermediate-mass) and 2.4e-05 (heavy stellar) of peak; at
    8192 Hz the previous conditioning left 0.34 and 1.5e-02 on these sources.
    """
    short, merger_index = _generated(name)
    long, offset = _long_reference(name, short.size, merger_index, tapered=True)
    window = long[offset : offset + short.size]
    peak = np.max(np.abs(window))
    assert np.max(np.abs(short - window)) < 1e-3 * peak


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_the_buffer_starts_and_ends_quietly(name):
    """The first and last samples are what is injected at the buffer's edges: no step into the data.

    The intermediate-mass source's first sample was 0.30 of peak under the previous conditioning.
    """
    short, _ = _generated(name)
    peak = np.max(np.abs(short))
    assert abs(short[0]) < 1e-3 * peak
    assert abs(short[-1]) < 1e-3 * peak


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_the_band_above_the_cutoff_is_the_models_own(name):
    """From the cutoff up, the buffer's spectrum is the model's, with nothing folded into it.

    Compared on the long grid, where the model's spectrum cut at the cutoff is exact. Measured here as
    a relative L2 norm over that band: 4.6e-06 (intermediate-mass) and 3.1e-05 (heavy stellar); the
    previous conditioning's wrapped-round ringing put it at the 1e-1 and 1e-2 level.
    """
    short, merger_index = _generated(name)
    long_cut, offset = _long_reference(name, short.size, merger_index, tapered=False)
    embedded = np.zeros(long_cut.size)
    embedded[offset : offset + short.size] = short
    frequencies = np.fft.rfftfreq(long_cut.size, d=1.0 / _FS)
    band = frequencies >= _F_MIN
    expected = np.fft.rfft(long_cut)[band]
    actual = np.fft.rfft(embedded)[band]
    assert np.linalg.norm(actual - expected) < 1e-3 * np.linalg.norm(expected)


@pytest.mark.requires_lalsuite
def test_a_quadrupole_only_mode_array_is_sized_for_the_quadrupole():
    """Higher harmonics enter the band before the quadrupole does, and only they need the extra room."""
    source = _SOURCES["heavy_bbh"]
    backend = LALSimulationBackend(f_ref=_F_REF)
    full = backend.pre_coalescence_duration(_APPROXIMANT, _FS, _F_MIN, **source)
    quadrupole = backend.pre_coalescence_duration(
        _APPROXIMANT, _FS, _F_MIN, waveform_arguments={"ModeArray": [(2, 2), (2, -2)]}, **source
    )
    assert full is not None
    assert quadrupole is not None
    assert quadrupole < full


@pytest.mark.requires_lalsuite
def test_an_approximant_that_cannot_start_below_the_cutoff_is_tapered_above_it(monkeypatch, caplog):
    """Generated from the cutoff instead, tapered above it, and said so -- never cut hard."""
    original = LALSimulationBackend._evaluate_fd

    def refuse_below_cutoff(self, approximant, p, grid):
        if grid.minimum_frequency < _F_MIN:
            raise RuntimeError("Internal function call failed: Input domain error")
        return original(self, approximant, p, grid)

    monkeypatch.setattr(LALSimulationBackend, "_evaluate_fd", refuse_below_cutoff)
    source = {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 25.0, "luminosity_distance": 400.0}
    with caplog.at_level(logging.WARNING, logger="gwmock_signal"):
        plus = LALSimulationBackend().generate_td_waveform("IMRPhenomD", 0.0, _FS, _F_MIN, **source)["plus"]

    assert "tapered over" in caplog.text
    data = np.asarray(plus.data)
    spectrum = np.abs(np.fft.rfft(data))
    frequencies = np.fft.rfftfreq(data.size, d=1.0 / _FS)
    peak = spectrum.max()
    # Nothing below the cutoff, and the band just above it rises from zero rather than stepping.
    assert np.max(spectrum[frequencies < _F_MIN - 0.5]) < 1e-6 * peak
    assert abs(data[0]) < 1e-3 * np.max(np.abs(data))


@pytest.mark.requires_lalsuite
def test_an_approximant_that_fails_from_the_cutoff_too_raises_the_original_error(monkeypatch):
    """A failure that is not about the start frequency is not hidden behind the fallback."""

    def always_refuse(self, approximant, p, grid):
        raise RuntimeError(f"refused at {grid.minimum_frequency:.3f} Hz")

    monkeypatch.setattr(LALSimulationBackend, "_evaluate_fd", always_refuse)
    source = {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 25.0, "luminosity_distance": 400.0}
    with pytest.raises(RuntimeError, match=r"refused at [0-4]\."):
        LALSimulationBackend().generate_td_waveform("IMRPhenomD", 0.0, _FS, _F_MIN, **source)


# ---------------------------------------------------------------------------
# Sizing for the largest harmonic the model emits
# ---------------------------------------------------------------------------


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize(
    ("approximant", "largest_m"),
    [
        ("IMRPhenomXPHM", 4),
        ("IMRPhenomXHM", 4),
        ("IMRPhenomHM", 4),
        ("IMRPhenomTPHM", 5),
        ("IMRPhenomD", 2),
        ("TaylorF2", 2),
        ("IMRPhenomPv2_NRTidalv2", 2),
        ("IMRPhenomNSBH", 2),
    ],
)
def test_the_buffer_is_sized_for_the_largest_harmonic_lal_reports(approximant, largest_m):
    """The sized harmonic is the largest ``|m|`` LALSimulation lists for the model, not a fixed one."""
    from gwmock_signal.waveform.backends import lal as lal_backend

    assert lal_backend._sizing_harmonic(approximant, {}) == largest_m
    # An explicit mode array is the caller's statement of what is generated.
    assert lal_backend._sizing_harmonic(approximant, {"ModeArray": [(2, 2), (2, -2), (3, 3)]}) == 3


@pytest.mark.requires_lalsuite
def test_a_higher_mode_model_whose_modes_lal_cannot_list_is_sized_for_the_largest_lal_harmonic(monkeypatch):
    """When the mode interface refuses, a higher-mode model is over-sized, never under-sized."""
    from gwmock_signal.waveform.backends import lal as lal_backend

    monkeypatch.setattr(lal_backend, "_reported_largest_m", lambda approximant, arguments: None)
    monkeypatch.setattr(lal_backend, "_DEFAULT_LARGEST_M", {})
    assert lal_backend._sizing_harmonic("SEOBNRv4HM_ROM", {}) == lal_backend._UNREPORTED_HIGHER_MODE_M == 5
    assert lal_backend._sizing_harmonic("IMRPhenomD", {}) == 2


@pytest.mark.requires_lalsuite
def test_an_unequal_mass_binary_with_higher_harmonics_keeps_its_early_m4_content():
    """A 250 + 5 solar-mass binary's (4, 4) harmonic enters the band long before the quadrupole.

    Sized for ``m = 3`` its buffer dropped 6.7e-3 of peak there, which then wrapped round; sized for
    the ``m = 4`` LALSimulation reports, what lies outside is 5e-5, the post-merger tail.
    """
    fs, approximant = 512.0, "IMRPhenomXHM"
    source = {
        "detector_frame_mass_1": 250.0,
        "detector_frame_mass_2": 5.0,
        "luminosity_distance": 1000.0,
        "inclination": 1.0,
        "coa_phase": 0.7,
    }
    plus = LALSimulationBackend(f_ref=_F_MIN).generate_td_waveform(approximant, 0.0, fs, _F_MIN, **source)["plus"]
    n_samples, merger_index = plus.n_samples, round(-plus.t0 * fs)

    m1, m2 = source["detector_frame_mass_1"], source["detector_frame_mass_2"]
    lowest_isco = 1.0 / (27.0 * math.pi * (m1 + m2) * lal.MTSUN_SI)
    chirp = lalsimulation.SimInspiralChirpTimeBound(min(_F_MIN, lowest_isco), m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0)
    start = lalsimulation.SimInspiralChirpStartFrequencyBound(1.1 * chirp, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI)
    long_samples = 4 * n_samples
    hp, _ = lalsimulation.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, source["luminosity_distance"] * 1e6 * lal.PC_SI,
        source["inclination"], source["coa_phase"], 0.0, 0.0, 0.0, fs / long_samples, start, fs / 2.0, _F_MIN,
        lal.CreateDict(), lalsimulation.GetApproximantFromString(approximant),
    )  # fmt: skip
    spectrum = np.zeros(long_samples // 2 + 1, dtype=complex)
    data = np.asarray(hp.data.data)[: spectrum.size]
    spectrum[: data.size] = data
    frequencies = np.arange(spectrum.size) * fs / long_samples
    ramp = np.clip((frequencies - start) / (_F_MIN - start), 0.0, 1.0)
    spectrum *= np.where(frequencies < start, 0.0, 0.5 - 0.5 * np.cos(np.pi * ramp))
    offset = long_samples // 2 - n_samples // 2
    long = np.roll(np.fft.irfft(spectrum, n=long_samples) * fs, offset + merger_index)
    outside = np.concatenate([long[:offset], long[offset + n_samples :]])
    assert np.max(np.abs(outside)) < 1e-3 * np.max(np.abs(long))
