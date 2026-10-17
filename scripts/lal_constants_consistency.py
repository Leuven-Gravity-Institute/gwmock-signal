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

"""Check the built-in constants against the oldest and newest supported ``lalsuite``.

``gwmock_signal.constants`` converts units with constants written out in the source, which must equal
LAL's bit-for-bit: LALSimulation converts with its own constants internally, so a built-in value that
differed in the last bit would put a one-ulp error into every waveform generated through it. The test
suite checks this against whichever ``lalsuite`` the environment resolves. It cannot notice a *new*
``lalsuite`` release that changes a constant, because nothing in this repository changes when one is
published. This script checks that case, on a schedule.

It reads the ``lalsuite`` floor from ``pyproject.toml`` and the newest release from PyPI. It then
installs each into a throwaway environment and reads the constants through
``gwmock_signal/constants/_lal.py``, so the two providers are compared name by name. The policy it
enforces:

- the built-in values must equal those of the newest release;
- the floor must give the same values as the newest release. If it does not, the releases between
  them are read from the newest downwards, and the walk stops at the first one that disagrees. The
  lowest release above that point is the floor to move to, because every release it admits agrees with
  the newest. Bisection would not do: a constant could change and then change back, and a floor below
  the change-back would still admit the releases in between.

It proposes; it does not edit. Changing a constant changes every waveform, so a maintainer applies the
values and the floor in a reviewed pull request. The report is markdown, written to ``--report`` and to
stdout. The exit status is 0 when everything agrees, 1 on any mismatch, and 2 when the check could not
be made.

Run it with ``packaging`` supplied on the fly, so it need not become a dependency::

    uv run --no-project --with packaging python scripts/lal_constants_consistency.py

It lives in ``scripts/`` rather than in ``src/``: it is maintenance tooling and has no business in the
wheel.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import subprocess
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import Tag, sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_DIR = REPO_ROOT / "src" / "gwmock_signal" / "constants"
PYPI_URL = "https://pypi.org/pypi/lalsuite/json"
#: The throwaway environments use the interpreter version running this script, so the wheels counted
#: as installable (from :func:`packaging.tags.sys_tags`) are the ones those environments install.
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"

#: Prints the LAL provider's constants as JSON hex strings; run inside an environment with ``lalsuite``.
_READ_LAL_PROVIDER = (
    "import json, runpy, sys; "
    "ns = runpy.run_path(sys.argv[1]); "
    "print(json.dumps({k: v.hex() for k, v in ns.items() if k.isupper()}))"
)

Constants = Mapping[str, float]


@dataclass(frozen=True)
class Mismatch:
    """One constant whose value differs between two sources."""

    name: str
    expected: float | None
    actual: float | None

    @property
    def ulps(self) -> float | None:
        """The difference in units of the expected value's last place, or ``None`` if either is missing."""
        if self.expected is None or self.actual is None:
            return None
        return (self.actual - self.expected) / math.ulp(self.expected)


def lalsuite_floor(pyproject: str) -> Version:
    """Return the lowest ``lalsuite`` version the core dependencies allow.

    Raises:
        ValueError: If ``lalsuite`` is not a core dependency, or its specifier sets no ``>=`` floor.
    """
    dependencies = tomllib.loads(pyproject)["project"]["dependencies"]
    for dependency in dependencies:
        requirement = Requirement(dependency)
        if requirement.name != "lalsuite":
            continue
        floors = [Version(spec.version) for spec in requirement.specifier if spec.operator == ">="]
        if not floors:
            raise ValueError(f"lalsuite has no '>=' floor in {dependency!r}")
        return max(floors)
    raise ValueError("lalsuite is not a core dependency in pyproject.toml")


def installable_versions(release_files: Mapping[str, list[dict]], tags: Iterable[Tag]) -> list[Version]:
    """Return the final, unyanked releases that ship a wheel this interpreter can install, ascending.

    Args:
        release_files: PyPI's ``releases`` mapping, version string to the list of its files.
        tags: The wheel tags the target interpreter supports, as :func:`packaging.tags.sys_tags` gives them.
    """
    supported = set(tags)
    versions = []
    for raw, files in release_files.items():
        version = Version(raw)
        if version.is_prerelease or version.is_devrelease:
            continue
        for file in files:
            if file.get("yanked") or not file["filename"].endswith(".whl"):
                continue
            if supported & set(parse_wheel_filename(file["filename"])[3]):
                versions.append(version)
                break
    return sorted(versions)


def compare(expected: Constants, actual: Constants) -> list[Mismatch]:
    """Return every name whose value differs, including names only one side defines."""
    return [
        Mismatch(name, expected.get(name), actual.get(name))
        for name in sorted(set(expected) | set(actual))
        if expected.get(name) != actual.get(name)
    ]


def lowest_consistent(
    newest: Version,
    candidates: Iterable[Version],
    reference: Constants,
    read: Callable[[Version], Constants | None],
) -> tuple[Version, list[Version]]:
    """Return the lowest version from which every release up to ``newest`` gives ``reference``.

    The candidates (releases below ``newest``) are read from the highest down, stopping at the first that
    disagrees, so every release at or above the answer agrees. ``read`` returns ``None`` for a version it
    could not install; such a version is passed over and returned in the second element, so the caller
    can say which releases the answer could not account for.
    """
    lowest = newest
    unreadable = []
    for version in sorted(candidates, reverse=True):
        values = read(version)
        if values is None:
            unreadable.append(version)
            continue
        if compare(reference, values):
            break
        lowest = version
    return lowest, unreadable


def builtin_constants() -> dict[str, float]:
    """Return the built-in provider's constants, without importing the package."""
    namespace = runpy.run_path(str(CONSTANTS_DIR / "_builtin.py"))
    return {name: value for name, value in namespace.items() if name.isupper()}


def lal_constants(version: Version) -> dict[str, float] | None:
    """Return the LAL provider's constants with ``lalsuite==version`` installed, or ``None`` if it fails."""
    command = [
        "uv", "run", "--no-project", "--isolated", "--quiet", "--python", PYTHON,
        "--with", f"lalsuite=={version}",
        "python", "-c", _READ_LAL_PROVIDER, str(CONSTANTS_DIR / "_lal.py"),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        print(f"lalsuite {version}: could not read constants\n{result.stderr}", file=sys.stderr)
        return None
    return {name: float.fromhex(value) for name, value in json.loads(result.stdout).items()}


def _table(mismatches: list[Mismatch], expected_label: str, actual_label: str) -> list[str]:
    def cell(value: float | None) -> str:
        return "missing" if value is None else f"`{value!r}` (`{value.hex()}`)"

    rows = [f"| Constant | {expected_label} | {actual_label} | Difference |", "|---|---|---|---|"]
    for mismatch in mismatches:
        ulps = "n/a" if mismatch.ulps is None else f"{mismatch.ulps:+.0f} ulp"
        rows.append(f"| `{mismatch.name}` | {cell(mismatch.expected)} | {cell(mismatch.actual)} | {ulps} |")
    return rows


def check(
    *,
    floor: Version,
    versions: list[Version],
    builtin: Constants,
    read: Callable[[Version], Constants | None],
) -> tuple[int, str]:
    """Run the policy and return ``(exit status, markdown report)``."""
    lines = ["# Built-in constants against lalsuite", ""]
    newest = versions[-1] if versions else None
    if newest is None or newest < floor:
        lines.append(f"No installable lalsuite release at or above the floor {floor}.")
        return 2, "\n".join(lines)

    newest_values = read(newest)
    floor_values = newest_values if floor == newest else read(floor)
    if newest_values is None or floor_values is None:
        unreadable = newest if newest_values is None else floor
        lines.append(f"Could not read the constants of lalsuite {unreadable}; see the log.")
        return 2, "\n".join(lines)

    lines.append(f"Floor: lalsuite {floor}. Newest installable on Python {PYTHON}: lalsuite {newest}.")
    status = 0

    builtin_vs_newest = compare(newest_values, builtin)
    lines.append("")
    if builtin_vs_newest:
        status = 1
        lines += [
            f"## Built-in values differ from lalsuite {newest}",
            "",
            "Update `src/gwmock_signal/constants/_builtin.py` to the newest release's values:",
            "",
            *_table(builtin_vs_newest, f"lalsuite {newest}", "built-in"),
        ]
    else:
        lines.append(f"The built-in values equal lalsuite {newest}'s.")

    floor_vs_newest = compare(newest_values, floor_values)
    lines.append("")
    if floor_vs_newest:
        status = 1
        candidates = [version for version in versions if floor < version < newest]
        proposed, unreadable = lowest_consistent(newest, candidates, newest_values, read)
        lines += [
            f"## The floor lalsuite {floor} differs from lalsuite {newest}",
            "",
            *_table(floor_vs_newest, f"lalsuite {newest}", f"lalsuite {floor}"),
            "",
            (
                f"Every release from **{proposed}** to {newest} gives lalsuite {newest}'s values. Raise the "
                f"floor to `lalsuite>={proposed}`, in the same pull request as any change to the built-in values."
            ),
        ]
        if unreadable:
            skipped = ", ".join(str(version) for version in sorted(unreadable))
            lines += ["", f"Could not read, so not accounted for: {skipped}."]
    else:
        lines.append(f"The floor lalsuite {floor} gives the same values as lalsuite {newest}.")

    return status, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the check against PyPI and write the report."""
    parser = argparse.ArgumentParser(description="Check the built-in constants against lalsuite.")
    parser.add_argument("--report", type=Path, help="Also write the markdown report to this file.")
    args = parser.parse_args(argv)

    floor = lalsuite_floor((REPO_ROOT / "pyproject.toml").read_text())
    with urllib.request.urlopen(PYPI_URL, timeout=60) as response:
        releases = json.load(response)["releases"]
    versions = [version for version in installable_versions(releases, sys_tags()) if version >= floor]

    status, report = check(
        floor=floor,
        versions=versions,
        builtin=builtin_constants(),
        read=lal_constants,
    )
    print(report)
    if args.report:
        args.report.write_text(report + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
