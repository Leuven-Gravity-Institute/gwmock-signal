"""Unit tests for the Network named-catalog class."""

from __future__ import annotations

try:
    import lal
except ModuleNotFoundError as exc:  # Python 3.15: lalsuite does not install yet; its tests are skipped
    if exc.name != "lal":
        raise
    lal = None
import pytest

from gwmock_signal.network import _NETWORK_PRESETS, Network

# Keep the old alias so existing parametrize fixtures need no change.
_CATALOG = _NETWORK_PRESETS


class TestNetworkFromName:
    """Tests for Network.from_name."""

    @pytest.mark.parametrize("alias", list(_CATALOG.keys()))
    def test_from_name_round_trip(self, alias: str) -> None:
        """Test the round-trip from name to network and back."""
        net = Network.from_name(alias)
        assert net.name == alias
        assert net.detector_names == _CATALOG[alias]

    def test_h1l1v1_detector_names(self) -> None:
        """Test the detector names for the H1L1V1 network."""
        net = Network.from_name("H1L1V1")
        assert net.detector_names == ("H1", "L1", "V1")

    def test_unknown_alias_raises_value_error(self) -> None:
        """Test that an unknown alias raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown network 'unknown'"):
            Network.from_name("unknown")

    def test_unknown_alias_lists_known_names(self) -> None:
        """Test that an unknown alias lists the known networks."""
        with pytest.raises(ValueError, match="Known networks:"):
            Network.from_name("not-a-network")

    def test_et_triangle_has_three_detectors(self) -> None:
        """Test that the ET-triangle network has three detectors."""
        net = Network.from_name("ET-triangle")
        assert len(net.detector_names) == 3

    def test_et_triangle_detector_names(self) -> None:
        """Test the detector names for the ET-triangle network."""
        net = Network.from_name("ET-triangle")
        assert net.detector_names == ("E1", "E2", "E3")

    def test_hlvk_includes_kagra(self) -> None:
        """Test that the HLVK network includes the K1 detector."""
        net = Network.from_name("HLVK")
        assert "K1" in net.detector_names


class TestNetworkListNames:
    """Tests for Network.list_names."""

    def test_returns_sorted_list(self) -> None:
        """Test that the list of names is sorted."""
        names = Network.list_names()
        assert names == sorted(names)

    def test_contains_all_expected_aliases(self) -> None:
        """Test that the list of names contains all expected aliases."""
        names = Network.list_names()
        expected = {"H1L1", "H1L1V1", "HLVK", "ET-triangle", "ET-L", "ET-Triangle-Sardinia", "ET-Sardinia"}
        assert expected.issubset(set(names))

    def test_returns_at_least_five_aliases(self) -> None:
        """Test that the list of names contains at least five aliases."""
        assert len(Network.list_names()) >= 5


class TestNetworkProjection:
    """Integration-style: ET-triangle codes are valid LAL detector codes."""

    @pytest.mark.requires_lalsuite
    def test_et_triangle_resolves_in_lal_cache(self) -> None:
        """Test that the ET-triangle network resolves in the LAL detector cache."""
        net = Network.from_name("ET-triangle")
        for code in net.detector_names:
            assert code in lal.cached_detector_by_prefix

    @pytest.mark.requires_lalsuite
    def test_et_l_resolves_in_lal_cache(self) -> None:
        """Test that the ET-L network resolves in the LAL detector cache."""
        net = Network.from_name("ET-L")
        for code in net.detector_names:
            assert code in lal.cached_detector_by_prefix

    @pytest.mark.requires_gwpy
    def test_et_triangle_passes_through_project_polarizations_to_network(self) -> None:
        """ET-triangle codes produce valid strain output from the projection function."""
        import numpy as np
        from gwpy.timeseries import TimeSeries

        from gwmock_signal.projection.network import project_polarizations_to_network
        from gwmock_signal.projection.resampling import edge_padding

        n, fs, t0 = 128, 4096.0, 1000000000.0
        t = np.arange(n) / fs
        hp = TimeSeries(np.sin(2 * np.pi * 10.0 * t), t0=t0, sample_rate=fs)
        hc = TimeSeries(np.cos(2 * np.pi * 10.0 * t), t0=t0, sample_rate=fs)

        net = Network.from_name("ET-triangle")
        result = project_polarizations_to_network(
            {"plus": hp, "cross": hc},
            list(net.detector_names),
            right_ascension=0.5,
            declination=0.3,
            polarization_angle=0.1,
            earth_rotation=False,
        )
        assert set(result.keys()) == set(net.detector_names)
        # Padded at each end against the geocentre delay, by the same amount for every detector.
        pad = edge_padding(fs)
        for code in net.detector_names:
            assert result[code].n_samples == n + 2 * pad
            assert result[code].t0 == t0 - pad / fs


class TestNetworkFromDetectors:
    """Tests for Network.from_detectors."""

    def test_valid_lal_codes_accepted(self) -> None:
        """from_detectors accepts any LAL-available code without hard-coding."""
        net = Network.from_detectors(["H1", "L1"])
        assert "H1" in net.detector_names
        assert "L1" in net.detector_names

    def test_auto_name_is_comma_joined(self) -> None:
        """When name is omitted the network name is the joined detector codes."""
        net = Network.from_detectors(["H1", "L1", "V1"])
        assert net.name == "H1,L1,V1"

    def test_explicit_name_is_preserved(self) -> None:
        """An explicitly supplied name overrides the auto-generated one."""
        net = Network.from_detectors(["H1", "L1"], name="MyNetwork")
        assert net.name == "MyNetwork"

    def test_unknown_code_raises_value_error(self) -> None:
        """A code not in LAL's runtime list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LAL detector code"):
            Network.from_detectors(["NOT_A_REAL_CODE"])

    def test_empty_sequence_raises_value_error(self) -> None:
        """An empty detectors sequence raises ValueError."""
        with pytest.raises(ValueError, match="detectors must be a non-empty sequence"):
            Network.from_detectors([])

    def test_all_lal_codes_accepted(self) -> None:
        """Every code in list_lal_detectors() can be used in from_detectors."""
        for code in Network.list_lal_detectors():
            net = Network.from_detectors([code])
            assert code in net.detector_names

    def test_single_detector_network(self) -> None:
        """A single-detector network is valid."""
        net = Network.from_detectors(["V1"])
        assert net.detector_names == ("V1",)


class TestNetworkListLalDetectors:
    """Tests for Network.list_lal_detectors."""

    def test_returns_sorted_list(self) -> None:
        """list_lal_detectors() returns a sorted list."""
        codes = Network.list_lal_detectors()
        assert codes == sorted(codes)

    def test_contains_standard_detectors(self) -> None:
        """Standard operational detectors are always present."""
        codes = Network.list_lal_detectors()
        for expected in ("H1", "L1", "V1", "K1"):
            assert expected in codes, f"{expected} missing from list_lal_detectors()"

    def test_contains_et_codes(self) -> None:
        """ET codes used in named presets are present."""
        codes = Network.list_lal_detectors()
        for et in ("E0", "E1", "E2", "E3"):
            assert et in codes, f"{et} missing from list_lal_detectors()"

    def test_returns_list_type(self) -> None:
        """Return type is a plain list."""
        assert isinstance(Network.list_lal_detectors(), list)


class TestNetworkImport:
    """Network is importable from the top-level package."""

    def test_import_from_package(self) -> None:
        """Test the import from package."""
        from gwmock_signal import Network as NetworkFromPkg

        assert NetworkFromPkg is Network
