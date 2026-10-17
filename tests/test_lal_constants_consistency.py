# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""The scheduled check of the built-in constants against the oldest and newest ``lalsuite``.

The policy these tests pin, with every ``lalsuite`` replaced by a table of values so nothing is
installed or fetched:

    the built-in values must equal the newest release's, and the floor must give the newest release's
    values; when it does not, the proposed floor is the lowest release from which *every* release up to
    the newest agrees.

The script proposes and never edits, so the only outputs are the exit status and the report. A wrong
answer is silent in both directions: a missed mismatch leaves a one-ulp error in every waveform, and a
wrong proposed floor admits a release that disagrees.
"""

from __future__ import annotations

import math

import pytest
from lal_constants_consistency import (
    Mismatch,
    builtin_constants,
    check,
    compare,
    installable_versions,
    lalsuite_floor,
    lowest_consistent,
)
from packaging.tags import Tag
from packaging.version import Version

from gwmock_signal import constants

GOOD = {"MSUN_SI": 1.9884098706980507e30, "PC_SI": 3.085677581491367e16}
#: ``GOOD`` with the solar mass one ulp higher.
OFF = {**GOOD, "MSUN_SI": math.nextafter(GOOD["MSUN_SI"], math.inf)}


def _reader(table: dict[str, dict[str, float] | None]):
    """Return a ``read`` that serves ``table`` and records which versions were asked for."""
    asked = []

    def read(version):
        asked.append(str(version))
        return table[str(version)]

    read.asked = asked
    return read


def _versions(*raw: str) -> list[Version]:
    return [Version(version) for version in raw]


class TestTheFloorIsReadFromPyproject:
    """The floor comes from the core dependency's ``>=`` clause."""

    @pytest.mark.parametrize(
        ("dependency", "floor"),
        [("lalsuite>=7.24", "7.24"), ("lalsuite>=7.24,<8", "7.24"), ("lalsuite >= 7.26.1", "7.26.1")],
    )
    def test_the_floor_is_the_lower_bound(self, dependency, floor):
        """An upper bound or spacing does not change the floor."""
        pyproject = f'[project]\ndependencies = ["numpy", "{dependency}"]\n'

        assert lalsuite_floor(pyproject) == Version(floor)

    def test_no_lalsuite_dependency_is_an_error(self):
        """Without the dependency there is no floor to check."""
        with pytest.raises(ValueError, match="not a core dependency"):
            lalsuite_floor('[project]\ndependencies = ["numpy"]\n')

    def test_a_lalsuite_without_a_floor_is_an_error(self):
        """An upper bound alone is not a floor."""
        with pytest.raises(ValueError, match="no '>=' floor"):
            lalsuite_floor('[project]\ndependencies = ["lalsuite<8"]\n')

    def test_the_repository_floor_is_readable(self):
        """The real ``pyproject.toml`` parses, so the workflow does not fail before checking anything."""
        from lal_constants_consistency import REPO_ROOT

        assert lalsuite_floor((REPO_ROOT / "pyproject.toml").read_text()) >= Version("7.24")


class TestOnlyInstallableReleasesAreCandidates:
    """A release counts only if it ships an unyanked wheel the running interpreter can install."""

    TAGS = (Tag("cp312", "cp312", "manylinux_2_28_x86_64"),)

    @staticmethod
    def _wheel(version, python, *, yanked=False):
        return {"filename": f"lalsuite-{version}-{python}-{python}-manylinux_2_28_x86_64.whl", "yanked": yanked}

    def test_the_filters(self):
        """Wrong interpreter, yanked, sdist-only and pre-releases are all excluded."""
        releases = {
            "7.24": [self._wheel("7.24", "cp312")],
            "7.25": [self._wheel("7.25", "cp314")],  # no wheel for this interpreter
            "7.26": [self._wheel("7.26", "cp312", yanked=True)],
            "7.26.1": [{"filename": "lalsuite-7.26.1.tar.gz", "yanked": False}],  # sdist only
            "7.27rc1": [self._wheel("7.27rc1", "cp312")],  # pre-release
            "7.27": [self._wheel("7.27", "cp312"), self._wheel("7.27", "cp314")],
        }

        assert installable_versions(releases, self.TAGS) == _versions("7.24", "7.27")

    def test_versions_are_ordered_numerically(self):
        """7.26.10 sorts above 7.26.9, which string order would get wrong."""
        releases = {raw: [self._wheel(raw, "cp312")] for raw in ("7.26.10", "7.26.9", "7.3")}

        assert installable_versions(releases, self.TAGS) == _versions("7.3", "7.26.9", "7.26.10")


class TestComparison:
    """Exact equality, with the difference reported in ulps."""

    def test_equal_values_give_no_mismatch(self):
        """Identical values compare clean."""
        assert compare(GOOD, dict(GOOD)) == []

    def test_one_ulp_is_a_mismatch_of_one_ulp(self):
        """The smallest possible difference is caught and reported as one ulp."""
        (mismatch,) = compare(GOOD, OFF)

        assert mismatch.name == "MSUN_SI"
        assert mismatch.ulps == 1

    def test_a_name_on_one_side_only_is_a_mismatch(self):
        """A constant one provider lacks is a mismatch, with no ulp figure."""
        (mismatch,) = compare(GOOD, {"MSUN_SI": GOOD["MSUN_SI"]})

        assert mismatch == Mismatch("PC_SI", GOOD["PC_SI"], None)
        assert mismatch.ulps is None


class TestTheProposedFloor:
    """The lowest release from which every release up to the newest agrees."""

    def test_the_walk_stops_at_the_first_disagreement_from_the_top(self):
        """A release that agrees below one that does not is not a valid floor."""
        read = _reader({"1": GOOD, "2": OFF, "3": GOOD, "4": GOOD})

        lowest, unreadable = lowest_consistent(Version("5"), _versions("1", "2", "3", "4"), GOOD, read)

        # 1 agrees, but a floor of 1 would admit 2, which does not.
        assert lowest == Version("3")
        assert unreadable == []
        assert read.asked == ["4", "3", "2"]

    def test_when_nothing_below_agrees_the_newest_is_the_floor(self):
        """With no agreeing release below it, the newest is the only valid floor."""
        lowest, _ = lowest_consistent(Version("3"), _versions("1", "2"), GOOD, _reader({"1": OFF, "2": OFF}))

        assert lowest == Version("3")

    def test_an_unreadable_release_is_passed_over_and_reported(self):
        """A release that cannot be installed neither stops the walk nor goes unmentioned."""
        read = _reader({"1": OFF, "2": GOOD, "3": None, "4": GOOD})

        lowest, unreadable = lowest_consistent(Version("5"), _versions("1", "2", "3", "4"), GOOD, read)

        # The walk continues past 3 to the agreeing 2 and stops only at the disagreeing 1.
        assert lowest == Version("2")
        assert unreadable == [Version("3")]
        assert read.asked == ["4", "3", "2", "1"]


class TestThePolicy:
    """Exit status and report for each outcome."""

    def test_everything_agrees(self):
        """Built-in, floor and newest agree: success."""
        status, report = check(
            floor=Version("1"), versions=_versions("1", "2"), builtin=GOOD, read=_reader({"1": GOOD, "2": GOOD})
        )

        assert status == 0
        assert "The built-in values equal lalsuite 2's." in report
        assert "The floor lalsuite 1 gives the same values as lalsuite 2." in report

    def test_built_in_values_that_differ_from_the_newest_fail(self):
        """A built-in value one ulp off the newest release fails, with the table."""
        status, report = check(
            floor=Version("1"), versions=_versions("1", "2"), builtin=OFF, read=_reader({"1": GOOD, "2": GOOD})
        )

        assert status == 1
        assert "## Built-in values differ from lalsuite 2" in report
        assert "| `MSUN_SI` |" in report
        assert "+1 ulp" in report

    def test_a_floor_that_differs_from_the_newest_fails_and_proposes_one(self):
        """A stale floor fails and the report names the floor to move to."""
        read = _reader({"1": OFF, "2": OFF, "3": GOOD, "4": GOOD})

        status, report = check(floor=Version("1"), versions=_versions("1", "2", "3", "4"), builtin=GOOD, read=read)

        assert status == 1
        assert "## The floor lalsuite 1 differs from lalsuite 4" in report
        assert "Raise the floor to `lalsuite>=3`" in report
        assert "The built-in values equal lalsuite 4's." in report

    def test_a_floor_that_is_the_newest_is_read_once(self):
        """When the floor is the newest release, it is installed once, not twice."""
        read = _reader({"2": GOOD})

        status, _ = check(floor=Version("2"), versions=_versions("2"), builtin=GOOD, read=read)

        assert status == 0
        assert read.asked == ["2"]

    def test_no_installable_release_cannot_be_checked(self):
        """Nothing to install is a failure to check, not a pass."""
        status, report = check(floor=Version("2"), versions=[], builtin=GOOD, read=_reader({}))

        assert status == 2
        assert "No installable lalsuite release" in report

    def test_an_unreadable_newest_release_cannot_be_checked(self):
        """A newest release that cannot be read is a failure to check, not a pass."""
        status, report = check(
            floor=Version("1"), versions=_versions("1", "2"), builtin=GOOD, read=_reader({"1": GOOD, "2": None})
        )

        assert status == 2
        assert "Could not read the constants of lalsuite 2" in report


def test_the_built_in_provider_is_read_without_importing_the_package():
    """The script reads ``_builtin.py`` directly; it must see exactly the package's constants."""
    assert builtin_constants() == {name: getattr(constants, name) for name in constants.CONSTANT_NAMES}
