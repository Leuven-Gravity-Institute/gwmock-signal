"""The finite-arm detector response.

Pinned against references that share no code with the implementation: the published single-arm
transfer function as jim writes it, PyCBC's implementation where PyCBC is installed, and -- for the
projection as a whole -- the definition itself: the strain of an analytic wave averaged by quadrature
along each arm's light path, with Earth's orientation at every sample.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gwmock_signal._detector_table import builtin_arm_lengths, builtin_detectors
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.projection import finite_arm, kernels
from gwmock_signal.projection.geometry import arm_lengths, reconstructed_geometry
from gwmock_signal.projection.network import _gmst_accurate_array, project_polarizations_to_network
from gwmock_signal.projection.resampling import DEFAULT_KAISER_BETA, DEFAULT_SINC_TAPS, SPEED_OF_LIGHT_M_S, edge_padding
from gwmock_signal.timeseries import TimeSeries

C = SPEED_OF_LIGHT_M_S
T0 = 1_000_000_000.0
SKY = {"right_ascension": 1.3, "declination": -0.4, "polarization_angle": 0.7}


def _jim_transfer(x, y):
    """Eq. (7) of Baral et al. 2023 as jim writes it: ``y`` projects the propagation direction on the arm."""
    return 0.5 * (
        np.exp(-1j * np.pi * x * (1 + y)) * np.sinc(x * (1 - y))
        + np.exp(1j * np.pi * x * (1 - y)) * np.sinc(x * (1 + y))
    )


def _triangle():
    """The ET triangle at the Meuse-Rhine site, from its preset."""
    return list(Network.from_preset("ET-Triangle-EMR").detector_names)


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["H1", "L1", "V1", "K1", "E1", "E2", "E3"])
def test_arm_directions_rebuild_the_response_tensor(code):
    """Half the difference of the arms' outer products is the response tensor the projection uses."""
    fields = builtin_detectors()[code]
    x_arm, y_arm = kernels.arm_directions(fields, xp=np)
    response, _ = reconstructed_geometry(code)
    np.testing.assert_allclose(0.5 * (np.outer(x_arm, x_arm) - np.outer(y_arm, y_arm)), response, rtol=0, atol=1e-15)
    np.testing.assert_allclose([np.linalg.norm(x_arm), np.linalg.norm(y_arm)], 1.0, rtol=0, atol=1e-15)


def test_each_triangle_arm_ends_at_a_neighbouring_vertex():
    """The tensor cannot tell an arm from its reverse; the sense, which the response needs, can be."""
    detectors = _triangle()
    vertices = [reconstructed_geometry(d.geometry_fields())[1] for d in detectors]
    for index, detector in enumerate(detectors):
        others = [vertex for other, vertex in enumerate(vertices) if other != index]
        for arm in kernels.arm_directions(detector.geometry_fields(), xp=np):
            end = vertices[index] + detector.arm_length_m * arm
            assert min(np.linalg.norm(end - vertex) for vertex in others) < 0.01


def test_built_in_arm_lengths_are_twice_lals_midpoints():
    """LAL records an arm by its midpoint, and nothing for a bar."""
    assert builtin_arm_lengths()["E1"] == (10_000.0, 10_000.0)
    assert arm_lengths("H1") == pytest.approx((3995.08, 3995.04), abs=0.01)
    assert arm_lengths("A1") is None  # a resonant bar has no arms


def test_presets_carry_their_design_arm_lengths():
    """10 km for the triangle, 15 km for the 2L."""
    assert {d.arm_length_m for d in Network.from_preset("ET-Triangle-EMR").detector_names} == {10_000.0}
    assert {d.arm_length_m for d in Network.from_preset("ET-2L-Misaligned").detector_names} == {15_000.0}


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_a_custom_detector_refuses_an_unphysical_arm_length(value):
    """A zero, negative or non-finite arm length is refused at construction."""
    with pytest.raises(ValueError, match="arm_length_m"):
        CustomDetector("X", 0.5, 0.1, 0.0, 0.0, math.pi / 2, arm_length_m=value)


def test_a_network_entry_gives_an_arm_length_only_to_a_custom_detector(tmp_path):
    """An arm length beside a built-in code would be ignored, so it is refused."""
    path = tmp_path / "net.yaml"
    path.write_text("name: n\ndetectors:\n  - name: H1\n    arm_length_m: 4000\n")
    with pytest.raises(ValueError, match="arm_length_m"):
        Network.from_file(path)


# --------------------------------------------------------------------------------------------------
# The transfer function
# --------------------------------------------------------------------------------------------------


def test_the_residual_is_the_published_transfer_function_less_the_midpoint_shift():
    """With the midpoint shift restored, the residual is Eq. (7) of Baral et al. as jim writes it."""
    rng = np.random.default_rng(1)
    x, mu = rng.uniform(0.0, 0.5, 2000), rng.uniform(-1.0, 1.0, 2000)
    published = _jim_transfer(x, -mu)  # jim's y is the propagation direction, -mu
    np.testing.assert_allclose(
        np.exp(1j * np.pi * x * mu) * kernels.finite_arm_residual(x, mu, xp=np), published, rtol=0, atol=1e-14
    )


def test_the_residual_agrees_with_pycbc():
    """PyCBC's single-arm response is the residual with the midpoint shift and one crossing's delay.

    Runs only where PyCBC is installed (``--extra pycbc``), which the continuous-integration jobs do not
    install, so there it is a check to run by hand. Run with PyCBC 2.11.0 when the response was added.
    """
    detector = pytest.importorskip("pycbc.detector")
    rng = np.random.default_rng(2)
    length = 10_000.0
    f, mu = rng.uniform(1.0, 4096.0, 2000), rng.uniform(-0.99, 0.99, 2000)
    x = f * length / C
    # PyCBC's n is the cosine to the source, as mu is, and it times the response a crossing later.
    ours = np.exp(1j * np.pi * x * mu) * kernels.finite_arm_residual(x, mu, xp=np) * np.exp(-2j * np.pi * x)
    np.testing.assert_allclose(detector.single_arm_frequency_response(f, mu, length), ours, rtol=0, atol=1e-12)


def test_the_residual_is_one_to_first_order():
    """The long-wavelength response is recovered at low frequency, to second order."""
    mu = np.linspace(-1.0, 1.0, 11)
    np.testing.assert_array_equal(kernels.finite_arm_residual(0.0, mu, xp=np), 1.0)
    small = kernels.finite_arm_residual(1e-4, mu, xp=np)
    assert np.max(np.abs(small - 1.0)) < 1e-7  # second order: (pi x)^2 / 6 is 1.6e-8


def test_alignment_nodes_interpolate_within_tolerance():
    """Interpolation between the nodes reproduces the residual at every alignment and frequency."""
    x_max = 0.5 * 8192 * 15_000.0 / C
    nodes = finite_arm.alignment_nodes(x_max)
    weights = finite_arm._chebyshev_points(len(nodes))[1]
    mu = np.linspace(-1.0, 1.0, 1001)
    for x in np.linspace(0.0, x_max, 7):
        basis = np.stack(list(finite_arm._interpolation_weights(mu, nodes, weights)))
        exact = kernels.finite_arm_residual(x, mu, xp=np)
        assert np.max(np.abs(kernels.finite_arm_residual(x, nodes, xp=np) @ basis - exact)) < 1e-13


def test_alignment_nodes_refuse_what_they_cannot_interpolate():
    """Arms far longer than any on the ground are refused rather than interpolated badly."""
    with pytest.raises(ValueError, match="cannot be interpolated"):
        finite_arm.alignment_nodes(50.0)


def test_a_sample_at_a_node_takes_that_nodes_value():
    """The barycentric form's division by zero at a node is avoided, not hit."""
    nodes, weights = finite_arm._chebyshev_points(5)
    rows = np.stack(list(finite_arm._interpolation_weights(nodes, nodes, weights)))
    np.testing.assert_array_equal(rows, np.eye(5))


def test_the_finite_residual_filter_matches_the_exact_one_below_its_band_edge():
    """The filter for a piece of a signal is as accurate as the resampling kernel, to 0.8 of Nyquist."""
    fs, length = 4096.0, 10_000.0
    t = np.arange(16384) / fs
    tone = np.sin(2 * np.pi * 1500.0 * t) * np.exp(-(((t - 2.0) / 0.4) ** 2))
    alignment = np.full(t.shape, 0.3)
    common = {"arm_length_m": length, "sample_rate": fs, "taps": DEFAULT_SINC_TAPS, "beta": DEFAULT_KAISER_BETA}
    exact = finite_arm.apply_residual(tone, alignment, tapered=True, **common)
    local = finite_arm.apply_residual(tone, alignment, tapered=False, **common)
    assert np.max(np.abs(exact - tone)) > 1e-2  # the residual is not negligible here
    assert np.max(np.abs(local - exact)) < 1e-7


# --------------------------------------------------------------------------------------------------
# The projection against the definition
# --------------------------------------------------------------------------------------------------

FS = 4096.0
N = int(4 * FS)
TC, WIDTH = 2.0, 0.05


def _wave(frequency):
    """An analytic, circularly polarised wave packet at ``frequency``, as functions of time after T0."""

    def plus(u):
        return np.exp(-(((u - TC) / WIDTH) ** 2)) * np.cos(2 * np.pi * frequency * (u - TC))

    def cross(u):
        return np.exp(-(((u - TC) / WIDTH) ** 2)) * np.sin(2 * np.pi * frequency * (u - TC))

    return plus, cross


def _polarizations(frequency):
    """Plus and cross of the analytic wave on the test grid."""
    plus, cross = _wave(frequency)
    t = np.arange(N) / FS
    return {"plus": TimeSeries(plus(t), t0=T0, sample_rate=FS), "cross": TimeSeries(cross(t), t0=T0, sample_rate=FS)}


def _by_definition(detector, frequency, times, *, frozen_at=None):
    """Return the detector's strain at ``times`` (seconds after T0), from the definition.

    The x arm's minus the y arm's, each the strain along its round trip averaged by quadrature, the light
    reflecting at each time. Earth's orientation is taken at every point of the trip, or held at
    ``frozen_at`` throughout when given.
    """
    plus, cross = _wave(frequency)
    fields = detector.geometry_fields()
    _, vertex = reconstructed_geometry(fields)
    crossing = detector.arm_length_m / C
    nodes, weights = np.polynomial.legendre.leggauss(40)
    elapsed = 0.5 * (nodes + 1.0) * crossing
    ra, dec, psi = SKY["right_ascension"], SKY["declination"], SKY["polarization_angle"]
    strain = 0.0
    for sign, arm in zip((1.0, -1.0), kernels.arm_directions(fields, xp=np), strict=True):
        for outward in (True, False):
            moment = times[:, None] + (elapsed - crossing if outward else elapsed)[None, :]
            sidereal = _gmst_accurate_array(T0 + (moment if frozen_at is None else np.full(moment.shape, frozen_at)))
            x, y = kernels._polarization_basis(np, sidereal, ra, dec, psi)
            pattern_plus = 0.5 * ((x @ arm) ** 2 - (y @ arm) ** 2)
            pattern_cross = (x @ arm) * (y @ arm)
            hour = sidereal - ra
            source = np.stack(
                [np.cos(dec) * np.cos(hour), -np.cos(dec) * np.sin(hour), np.full_like(hour, np.sin(dec))], -1
            )
            distance = (elapsed if outward else crossing - elapsed) * C
            position = vertex + distance[None, :, None] * arm
            geocentric = moment + np.einsum("kmi,kmi->km", source, position) / C
            leg = pattern_plus * plus(geocentric) + pattern_cross * cross(geocentric)
            strain = strain + sign * 0.5 * (leg * weights / 2.0).sum(axis=1)
    return strain


@pytest.mark.parametrize(
    ("frequency", "tolerance", "tapered"),
    # 1800 Hz is 0.88 of Nyquist, beyond the 0.8 an untapered piece is accurate to.
    [(100.0, 1e-10, True), (100.0, 1e-10, False), (700.0, 2e-9, True), (700.0, 2e-9, False), (1800.0, 1e-8, True)],
)
def test_the_rotating_projection_is_the_arm_average_along_the_light_path(frequency, tolerance, tapered):
    """The rotating projection is the strain averaged along each arm's light path, sample by sample.

    The difference left is bounded by the Doppler shift of Earth's rotation, 1.5e-6 at most, times the
    residual, as the residual is applied in detector time: 3e-11, 1.4e-9 and 9e-9 at these frequencies.
    The long-wavelength response is 1e-2 off.
    """
    detectors = _triangle()
    projected = project_polarizations_to_network(
        _polarizations(frequency), detectors, **SKY, earth_rotation=True, tapered=tapered, finite_arm_length=True
    )
    pad = edge_padding(FS)
    samples = np.arange(pad + int((TC - 0.15) * FS), pad + int((TC + 0.15) * FS), 5)
    times = (samples - pad) / FS
    for detector in detectors:
        expected = _by_definition(detector, frequency, times)
        error = np.max(np.abs(projected[detector.name].data[samples] - expected)) / np.max(np.abs(expected))
        assert error < tolerance, detector.name


def test_the_constant_projection_is_the_arm_average_with_earth_held_at_the_midpoint():
    """The constant branch is the same average with Earth's orientation fixed at the span's midpoint."""
    detectors = _triangle()
    projected = project_polarizations_to_network(
        _polarizations(700.0), detectors, **SKY, earth_rotation=False, finite_arm_length=True
    )
    pad = edge_padding(FS)
    samples = np.arange(pad + int((TC - 0.15) * FS), pad + int((TC + 0.15) * FS), 5)
    times = (samples - pad) / FS
    midpoint = 0.5 * ((N + 2 * pad - 1) / FS) - pad / FS
    for detector in detectors:
        expected = _by_definition(detector, 700.0, times, frozen_at=midpoint)
        error = np.max(np.abs(projected[detector.name].data[samples] - expected)) / np.max(np.abs(expected))
        assert error < 1e-11, detector.name


@pytest.mark.parametrize("earth_rotation", [True, False])
def test_the_triangle_null_stream_cancels_only_with_finite_arms(earth_rotation):
    """Each arm line is read by two interferometers; only finite arms read it at the same place."""
    detectors = _triangle()
    common = {**SKY, "earth_rotation": earth_rotation, "tapered": True}
    finite = project_polarizations_to_network(_polarizations(100.0), detectors, **common, finite_arm_length=True)
    vertex = project_polarizations_to_network(_polarizations(100.0), detectors, **common)
    peak = np.max(np.abs(finite[detectors[0].name].data))
    assert np.max(np.abs(sum(finite[d.name].data for d in detectors))) / peak < 2e-6
    assert np.max(np.abs(sum(vertex[d.name].data for d in detectors))) / peak > 1e-2


@pytest.mark.parametrize("earth_rotation", [True, False])
def test_a_short_arm_recovers_the_long_wavelength_response(earth_rotation):
    """With the arm a micrometre long the finite response is the long-wavelength one, by a separate path.

    A millimetre is not short enough: each arm's midpoint then sits half a millimetre from the vertex, a
    real difference of 2 pi f L / (2c), 3e-9 at 300 Hz.
    """
    detector = CustomDetector("X1", 0.7, 0.16, 100.0, 1.23, 2.80, arm_length_m=1e-6)
    common = {**SKY, "earth_rotation": earth_rotation, "tapered": True}
    finite = project_polarizations_to_network(_polarizations(300.0), [detector], **common, finite_arm_length=True)
    vertex = project_polarizations_to_network(_polarizations(300.0), [detector], **common)
    peak = np.max(np.abs(vertex["X1"].data))
    assert np.max(np.abs(finite["X1"].data - vertex["X1"].data)) / peak < 1e-10


def test_the_option_is_off_by_default():
    """The default is the long-wavelength response, unchanged."""
    detectors = _triangle()
    default = project_polarizations_to_network(_polarizations(300.0), detectors, **SKY, tapered=True)
    explicit = project_polarizations_to_network(
        _polarizations(300.0), detectors, **SKY, tapered=True, finite_arm_length=False
    )
    for detector in detectors:
        np.testing.assert_array_equal(default[detector.name].data, explicit[detector.name].data)


# --------------------------------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------------------------------


def test_the_device_backend_refuses_finite_arms():
    """The device projection has no finite-arm response, so it refuses the option."""
    with pytest.raises(ValueError, match="backend='numpy'"):
        project_polarizations_to_network(_polarizations(100.0), ["E1"], **SKY, backend="jax", finite_arm_length=True)


@pytest.mark.parametrize("detector", ["A1", CustomDetector("X1", 0.7, 0.16, 100.0, 1.23, 2.80)])
def test_a_detector_without_an_arm_length_is_refused(detector):
    """A bar, or a custom detector given no arm length, cannot have a finite-arm response."""
    with pytest.raises(ValueError, match="arm length"):
        project_polarizations_to_network(_polarizations(100.0), [detector], **SKY, finite_arm_length=True)


def test_a_midpoint_beyond_the_padding_is_refused():
    """A midpoint delay past the padding is refused on both branches, and one at its edge is read exactly.

    The padding covers the largest vertex delay and one sample more. At that limit the delay is a whole
    number of samples, so the resampled arm is the series shifted by it, not a partly zeroed copy.
    """
    from gwmock_signal.projection.network import _finite_arm_constant, _finite_arm_rotating, _kernel_input
    from gwmock_signal.projection.resampling import EARTH_RADIUS_M

    pad = edge_padding(FS)
    series = np.random.default_rng(3).standard_normal(512)
    kernel_input = _kernel_input(series, np.zeros(512), pad, tapered=False)
    padded = kernel_input[2]
    shift = math.ceil(EARTH_RADIUS_M / C * FS) + 1
    kernel = {"sinc_taps": DEFAULT_SINC_TAPS, "kaiser_beta": DEFAULT_KAISER_BETA}
    ones, rows = np.ones(512), np.arange(512)
    for sign in (1.0, -1.0):
        arm = _finite_arm_rotating(kernel_input, ones, ones, np.full(512, sign * shift / FS), FS, **kernel)
        np.testing.assert_array_equal(arm, padded[pad + rows - int(sign) * shift])
    with pytest.raises(ValueError, match="edge padding"):
        _finite_arm_rotating(kernel_input, ones, ones, np.full(512, -1.001 * shift / FS), FS, **kernel)
    spectra = (np.fft.rfft(series), np.fft.rfft(series), 512)
    with pytest.raises(ValueError, match="edge padding"):
        _finite_arm_constant(
            spectra,
            f_plus=1.0,
            f_cross=0.0,
            midpoint_delay=1.001 * shift / FS,
            alignment=0.0,
            length=10_000.0,
            sample_rate=FS,
        )
