#
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
#
"""The delay between the geocentre and the solar-system barycentre, for every source.

A wave reaches the geocentre at a different time than it reaches the solar-system barycentre (SSB), by
an amount that changes as the Earth moves. Written as the SSB time of a wavefront that passes the
geocentre at ``t``, ``t_SSB = t + D(t)``, the delay ``D`` has three parts, each switchable through
:class:`~gwmock_signal.projection.effects.ProjectionEffects`:

* **Roemer**: ``n . r(t) / c``, the geocentre's position ``r`` relative to the SSB along the direction
  ``n`` to the source (J2000). Up to 500 s, changing by up to 1e-4 s per second.
* **Einstein**: TDB minus TT, the relativistic time dilation of a clock on Earth, 1.7 ms over a year.
* **Shapiro**: the Sun's gravitational delay, ``-2 G M / c^3 ln(1 + cos)`` in effect, entering ``D``
  with a minus sign; up to ~1e-4 s for a source near the Sun.

This is LALPulsar's ``XLALBarycenterEarth`` and ``XLALBarycenter`` in their ``TIMECORRECTION_ORIGINAL``
time system, without the site term, which projection adds from the detector geometry, and agrees with
``lalpulsar.Barycenter`` at the geocentre to 1e-13 s: the Earth's position from the JPL DE405 tables
LALPulsar ships (second-order Taylor expansion about the nearest entry, 7200 s apart, which misses the
Earth's jerk by up to 5e-8 s between entries, as LALPulsar does), the
Einstein delay from the 40 largest terms of TEMPO's series, and the Shapiro delay including a ray that
passes through the Sun. The arithmetic follows the JAX port of those functions in ripplegw
(``ripplegw.waveforms.cw.earth`` and ``barycenter``, MIT licence), which the continuous-wave
simulator used before this module existed.

The tables are read from ``earth00-40-DE405.dat.gz`` and ``sun00-40-DE405.dat.gz`` by default, which
cover GPS 630720013 to 1893024018 (2000 to 2040). A file not found locally is downloaded once from the
LALSuite repository and cached; on a machine without network access, place the files in the cache
(:func:`cache_directory`) or pass their paths.
"""

from __future__ import annotations

import gzip
import math
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

#: Where the standard tables are downloaded from when not found locally.
LALSUITE_EPHEMERIS_URL = "https://git.ligo.org/lscsoft/lalsuite/-/raw/master/lalpulsar/lib"

#: The tables used unless others are given: JPL DE405, 2000 to 2040.
DEFAULT_EARTH_EPHEMERIS = "earth00-40-DE405.dat.gz"
DEFAULT_SUN_EPHEMERIS = "sun00-40-DE405.dat.gz"

#: LALPulsar's naming of its standard tables; only such a bare name is ever downloaded.
_STANDARD_NAME = re.compile(r"\A(earth|sun)\d{2}-\d{2}-DE\d{3}\.dat(\.gz)?\Z")

#: Values in a table's header: first time, spacing, number of rows.
_HEADER_LENGTH = 3

#: Astronomical unit over the speed of light, seconds, and the Sun's radius in light-seconds, as
#: LALPulsar's Shapiro delay uses them.
_AU_OVER_C = 149597870700.0 / 299792458.0
_SUN_RADIUS_LIGHT_SECONDS = 2.322

#: The Einstein delay's series: (amplitude in microseconds, rate in radians per Julian millennium,
#: phase in radians), the 20 largest terms of TEMPO's expansion and the 20 next, as in
#: ``XLALBarycenterEarth``.
_EINSTEIN_TERMS = np.array(
    [
        (1656.674564, 6283.075849991, 6.240054195),
        (22.417471, 5753.384884897, 4.296977442),
        (13.839792, 12566.151699983, 6.196904410),
        (4.770086, 529.690965095, 0.444401603),
        (4.676740, 6069.776754553, 4.021195093),
        (2.256707, 213.299095438, 5.543113262),
        (1.694205, -3.523118349, 5.025132748),
        (1.554905, 77713.771467920, 5.198467090),
        (1.276839, 7860.419392439, 5.988822341),
        (1.193379, 5223.693919802, 3.649823730),
        (1.115322, 3930.209696220, 1.422745069),
        (0.794185, 11506.769769794, 2.322313077),
        (0.447061, 26.298319800, 3.615796498),
        (0.435206, -398.149003408, 4.349338347),
        (0.600309, 1577.343542448, 2.678271909),
        (0.496817, 6208.294251424, 5.696701824),
        (0.486306, 5884.926846583, 0.520007179),
        (0.432392, 74.781598567, 2.435898309),
        (0.468597, 6244.942814354, 5.866398759),
        (0.375510, 5507.553238667, 4.103476804),
        (0.243085, -775.522611324, 3.651837925),
        (0.173435, 18849.227549974, 6.153743485),
        (0.230685, 5856.477659115, 4.773852582),
        (0.203747, 12036.460734888, 4.333987818),
        (0.143935, -796.298006816, 5.957517795),
        (0.159080, 10977.078804699, 1.890075226),
        (0.119979, 38.133035638, 4.551585768),
        (0.118971, 5486.777843175, 1.914547226),
        (0.116120, 1059.381930189, 0.873504123),
        (0.137927, 11790.629088659, 1.135934669),
        (0.098358, 2544.314419883, 0.092793886),
        (0.101868, -5573.142801634, 5.984503847),
        (0.080164, 206.185548437, 2.095377709),
        (0.079645, 4694.002954708, 2.949233637),
        (0.062617, 20.775395492, 2.654394814),
        (0.075019, 2942.463423292, 4.980931759),
        (0.064397, 5746.271337896, 1.280308748),
        (0.063814, 5760.498431898, 4.167901731),
        (0.048042, 2146.165416475, 1.495846011),
        (0.048373, 155.420399434, 2.251573730),
    ]
)

#: Seconds per Julian millennium, the Einstein series' time unit.
_SECONDS_PER_MILLENNIUM = 3.6525e5 * 8.64e4

#: Spacing of the grid on which :class:`DelayModel` evaluates the Einstein delay, in seconds. Its fastest
#: term has a 30-day period, so cubic-Hermite interpolation between nodes this far apart is exact to
#: about 1e-21 s.
EINSTEIN_GRID_SECONDS = 64.0


@dataclass(frozen=True)
class EphemerisTable:
    """One body's tabulated position, velocity and acceleration relative to the SSB.

    Attributes:
        gps0: GPS time of the first entry, seconds.
        spacing: Seconds between entries.
        position: Light-seconds, shape ``(n, 3)``.
        velocity: In units of ``c``, shape ``(n, 3)``.
        acceleration: Per second, shape ``(n, 3)``.
        source: Where the table was read from.
    """

    gps0: float
    spacing: float
    position: np.ndarray = field(repr=False)
    velocity: np.ndarray = field(repr=False)
    acceleration: np.ndarray = field(repr=False)
    source: str = ""

    @property
    def gps_end(self) -> float:
        """GPS time of the last entry."""
        return self.gps0 + (len(self.position) - 1) * self.spacing

    def state(self, gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return position and velocity at ``gps``, Taylor-expanded about the nearest entry as LALPulsar does.

        Raises:
            ValueError: If a time lies outside the table, where LALPulsar refuses too.
        """
        gps = np.asarray(gps, dtype=float)
        offset = gps - self.gps0
        entry = np.floor(offset / self.spacing + 0.5).astype(np.int64)
        if np.any(entry < 0) or np.any(entry >= len(self.position)):
            raise ValueError(
                f"GPS times {float(np.min(gps)):.1f} to {float(np.max(gps)):.1f} reach outside the ephemeris "
                f"{self.source or 'table'}, which covers {self.gps0:.1f} to {self.gps_end:.1f}."
            )
        since = (offset - self.spacing * entry)[..., None]
        position = self.position[entry] + self.velocity[entry] * since + 0.5 * self.acceleration[entry] * since**2
        velocity = self.velocity[entry] + self.acceleration[entry] * since
        return position, velocity


def cache_directory() -> Path:
    """Return where downloaded tables are kept.

    ``$GWMOCK_SIGNAL_CACHE_DIR/ephemeris`` if that variable is set; otherwise
    ``$XDG_CACHE_HOME/gwmock-signal/ephemeris`` (``~/.cache`` when unset), or
    ``~/Library/Caches/gwmock-signal/ephemeris`` on macOS.
    """
    override = os.environ.get("GWMOCK_SIGNAL_CACHE_DIR")
    if override:
        return Path(override) / "ephemeris"
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "gwmock-signal" / "ephemeris"


def resolve_table(name_or_path: str | os.PathLike[str]) -> Path:
    """Return a local path for a table, downloading a standard one into the cache if it is not there.

    An existing path (with or without ``.gz``) is used as it is. A bare standard name, such as
    ``earth00-40-DE405.dat.gz``, is looked up in :func:`cache_directory` and downloaded there from
    :data:`LALSUITE_EPHEMERIS_URL` if absent. Nothing else is downloaded.

    Raises:
        FileNotFoundError: If the file is neither local nor a standard name, or the download failed.
    """
    path = Path(name_or_path)
    for candidate in (path, Path(f"{path}.gz")):
        if candidate.exists():
            return candidate
    if path.parent != Path() or not _STANDARD_NAME.match(path.name):
        raise FileNotFoundError(
            f"Ephemeris {str(name_or_path)!r} does not exist, and is not a standard LALPulsar name such as "
            f"{DEFAULT_EARTH_EPHEMERIS!r} that can be downloaded."
        )
    cached = cache_directory() / path.name
    if cached.exists():
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=cached.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with (
            urllib.request.urlopen(f"{LALSUITE_EPHEMERIS_URL}/{path.name}", timeout=60) as response,  # noqa: S310
            os.fdopen(handle, "wb") as target,
        ):
            shutil.copyfileobj(response, target)
        os.replace(partial, cached)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        Path(partial).unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Ephemeris {path.name!r} is not in {cached.parent} and could not be downloaded from "
            f"{LALSUITE_EPHEMERIS_URL}: {exc}. On a machine without network access, copy it into that "
            f"directory or pass its path."
        ) from exc
    return cached


@cache
def read_table(name_or_path: str) -> EphemerisTable:
    """Read a LALPulsar ephemeris table, once per process.

    The file holds a header ``gps0 spacing count``, then ``count`` rows of ``gps x y z vx vy vz ax ay az``.

    Raises:
        FileNotFoundError: See :func:`resolve_table`.
        ValueError: If the header disagrees with the rows, or the rows are not evenly spaced.
    """
    path = resolve_table(name_or_path)
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
    with gzip.open(path, "rt") if compressed else path.open() as text:
        values = [
            token for line in text if line.strip() and not line.lstrip().startswith("#") for token in line.split()
        ]
    if len(values) < _HEADER_LENGTH:
        raise ValueError(f"Ephemeris {path} has no header.")
    spacing, count = float(values[1]), int(float(values[2]))
    body = np.asarray(values[3:], dtype=float)
    if count <= 0 or body.size != 10 * count:
        raise ValueError(f"Ephemeris {path} declares {count} rows but holds {body.size} values.")
    rows = body.reshape(count, 10)
    if count > 1 and not np.allclose(np.diff(rows[:, 0]), spacing, rtol=0.0, atol=1e-6):
        raise ValueError(f"Ephemeris {path} is not evenly spaced at its declared {spacing} s.")
    position, velocity, acceleration = (np.ascontiguousarray(rows[:, k : k + 3]) for k in (1, 4, 7))
    for array in (position, velocity, acceleration):
        array.setflags(write=False)
    return EphemerisTable(float(rows[0, 0]), spacing, position, velocity, acceleration, str(path))


@dataclass(frozen=True)
class SolarSystemEphemeris:
    """The Earth and Sun tables the solar-system delay is computed from.

    Attributes:
        earth: Path or standard name of the Earth table.
        sun: Path or standard name of the Sun table.
    """

    earth: str = DEFAULT_EARTH_EPHEMERIS
    sun: str = DEFAULT_SUN_EPHEMERIS

    def tables(self) -> tuple[EphemerisTable, EphemerisTable]:
        """Return the two tables, reading (and if need be downloading) each once per process."""
        return read_table(str(self.earth)), read_table(str(self.sun))


def _einstein(gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Einstein delay TDB - TT in seconds and its rate, as ``XLALBarycenterEarth`` computes it."""
    millennia = (-7300.5 + (gps + 51.184) / 8.64e4) / 3.6525e5
    amplitude, rate, phase = (_EINSTEIN_TERMS[:, k, None] for k in range(3))
    argument = rate * np.ravel(millennia)[None, :] + phase
    value = 1e-6 * np.sum(amplitude * np.sin(argument), axis=0)
    derivative = 1e-6 * np.sum(amplitude * rate * np.cos(argument), axis=0) / _SECONDS_PER_MILLENNIUM
    return value.reshape(np.shape(gps)), derivative.reshape(np.shape(gps))


def _shapiro(sun_to_earth: np.ndarray, sun_to_earth_rate: np.ndarray, source: np.ndarray) -> tuple[Any, Any]:
    """Return the Sun's Shapiro delay in seconds and its rate, as ``XLALBarycenter`` computes them.

    Below the Sun's limb, when the ray passes through the Sun, the delay inside a uniform Sun is used.
    """
    distance = np.linalg.norm(sun_to_earth, axis=-1)
    along = sun_to_earth @ source
    distance_rate = np.sum(sun_to_earth * sun_to_earth_rate, axis=-1) / distance
    along_rate = sun_to_earth_rate @ source
    impact = np.sqrt(np.maximum(distance**2 - along**2, 0.0))
    through = (impact < _SUN_RADIUS_LIGHT_SECONDS) & (along < 0.0)
    usual = 9.852e-6 * np.log(_AU_OVER_C / (distance + along))
    usual_rate = -9.852e-6 * (distance_rate + along_rate) / (distance + along)
    root = np.sqrt(_SUN_RADIUS_LIGHT_SECONDS**2 + along**2)
    inside = 9.852e-6 * np.log(_AU_OVER_C / (along + root)) + 19.704e-6 * (1.0 - impact / _SUN_RADIUS_LIGHT_SECONDS)
    impact_rate = np.where(
        impact > 0.0, (distance * distance_rate - along * along_rate) / np.where(impact > 0.0, impact, 1.0), 0.0
    )
    inside_rate = -9.852e-6 * along_rate / root - 19.704e-6 * impact_rate / _SUN_RADIUS_LIGHT_SECONDS
    return np.where(through, inside, usual), np.where(through, inside_rate, usual_rate)


def source_direction(right_ascension: float, declination: float) -> np.ndarray:
    """Return the unit vector towards a source at J2000 right ascension and declination."""
    return np.array(
        [
            math.cos(declination) * math.cos(right_ascension),
            math.cos(declination) * math.sin(right_ascension),
            math.sin(declination),
        ]
    )


def delay(  # noqa: PLR0913 -- the times, the sky, the tables and the three switches
    gps: np.ndarray,
    right_ascension: float,
    declination: float,
    ephemeris: SolarSystemEphemeris,
    *,
    roemer: bool = True,
    einstein: bool = True,
    shapiro: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the delay ``D`` with ``t_SSB = t + D`` at geocentre times ``gps``, and its rate.

    Args:
        gps: GPS times at the geocentre.
        right_ascension: J2000 right ascension of the source, radians.
        declination: J2000 declination, radians.
        ephemeris: The Earth and Sun tables.
        roemer: Include the Roemer delay.
        einstein: Include the Einstein delay.
        shapiro: Include the Shapiro delay, which enters with a minus sign.

    Returns:
        ``(D, dD/dt)``, each of the shape of ``gps``.
    """
    gps = np.asarray(gps, dtype=float)
    value = np.zeros(gps.shape)
    rate = np.zeros(gps.shape)
    if not (roemer or einstein or shapiro):
        return value, rate
    earth_table, sun_table = ephemeris.tables()
    source = source_direction(right_ascension, declination)
    earth_position, earth_velocity = earth_table.state(gps)
    if roemer:
        value = value + earth_position @ source
        rate = rate + earth_velocity @ source
    if einstein:
        einstein_value, einstein_rate = _einstein(gps)
        value, rate = value + einstein_value, rate + einstein_rate
    if shapiro:
        sun_position, sun_velocity = sun_table.state(gps)
        shapiro_value, shapiro_rate = _shapiro(earth_position - sun_position, earth_velocity - sun_velocity, source)
        value, rate = value - shapiro_value, rate - shapiro_rate
    return value, rate


@dataclass(frozen=True)
class DelayModel:
    """The delay over a span of geocentre times, ready to evaluate at every sample on the host or device.

    The Roemer and Shapiro parts are evaluated exactly as LALPulsar evaluates them, by Taylor expansion
    about the nearest table entry, from the few entries the span needs; the Einstein part, a smooth
    series whose fastest term has a 30-day period, from cubic-Hermite interpolation of its value and rate
    on a grid :data:`EINSTEIN_GRID_SECONDS` apart. Interpolating the Roemer part instead would smooth over
    the steps LALPulsar's expansion takes between entries, up to 5e-8 s, and disagree with it by that.

    Times are passed relative to :attr:`epoch`, so a device in any precision never handles the GPS epoch.

    Attributes:
        epoch: GPS time the times passed to :meth:`evaluate` are measured from.
        source: Unit vector towards the source, J2000.
        components: ``(roemer, einstein, shapiro)``, which parts are included.
        earth: Earth-table entries covering the span: ``(offset of the first entry from the epoch,
            spacing, position, velocity, acceleration)``.
        sun: The same for the Sun.
        einstein_grid: ``(offset of the first node from the epoch, spacing, value, rate)``.
    """

    epoch: float
    source: np.ndarray
    components: tuple[bool, bool, bool]
    earth: tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
    sun: tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
    einstein_grid: tuple[float, float, np.ndarray, np.ndarray]

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        start: float,
        end: float,
        right_ascension: float,
        declination: float,
        ephemeris: SolarSystemEphemeris,
        *,
        roemer: bool = True,
        einstein: bool = True,
        shapiro: bool = True,
    ) -> DelayModel:
        """Return the model for geocentre GPS times in ``[start, end]``, measured from ``start``.

        Raises:
            ValueError: If the span reaches outside the tables.
        """
        earth_table, sun_table = ephemeris.tables()
        pieces = []
        for table in (earth_table, sun_table):
            table.state(np.array([start, end]))  # refuses a span outside the table
            # A row count fixed by the span's length alone, so a compiled device kernel serves every
            # span of that length: the entries nearest each time in the span, and one to spare.
            count = math.ceil((end - start) / table.spacing) + 2
            first = int(np.floor((start - table.gps0) / table.spacing + 0.5))
            first = max(0, min(first, len(table.position) - count))
            rows = slice(first, first + count)
            pieces.append(
                (
                    table.gps0 + first * table.spacing - start,
                    table.spacing,
                    table.position[rows],
                    table.velocity[rows],
                    table.acceleration[rows],
                )
            )
        nodes = start + EINSTEIN_GRID_SECONDS * np.arange(-1, math.ceil((end - start) / EINSTEIN_GRID_SECONDS) + 2)
        value, rate = _einstein(nodes)
        return cls(
            epoch=float(start),
            source=source_direction(right_ascension, declination),
            components=(roemer, einstein, shapiro),
            earth=pieces[0],
            sun=pieces[1],
            einstein_grid=(-EINSTEIN_GRID_SECONDS, EINSTEIN_GRID_SECONDS, value, rate),
        )

    def arrays(self) -> tuple[Any, ...]:
        """Return the model as a flat tuple of arrays, for :func:`evaluate_arrays` on a device."""
        earth_first, earth_spacing, *earth = self.earth
        sun_first, sun_spacing, *sun = self.sun
        grid_first, grid_spacing, value, rate = self.einstein_grid
        offsets = np.array([earth_first, earth_spacing, sun_first, sun_spacing, grid_first, grid_spacing])
        return (offsets, self.source, *earth, *sun, value, rate)

    def evaluate(self, since_epoch: Any, *, xp: Any = np) -> Any:
        """Return the delay ``D`` at geocentre times ``since_epoch`` seconds after :attr:`epoch`."""
        return evaluate_arrays(self.arrays(), self.components, since_epoch, xp=xp)


def evaluate_arrays(arrays: tuple[Any, ...], components: tuple[bool, bool, bool], since_epoch: Any, *, xp: Any) -> Any:
    """Return the delay of :meth:`DelayModel.arrays` at ``since_epoch``, in the namespace ``xp``.

    ``components`` is ``(roemer, einstein, shapiro)``; on a device it is static, the arrays traced.
    """
    offsets, source, *rest = arrays
    earth = (offsets[0], offsets[1], *rest[0:3])
    sun = (offsets[2], offsets[3], *rest[3:6])
    grid = (offsets[4], offsets[5], *rest[6:8])
    roemer, einstein, shapiro = components
    source = xp.asarray(source)
    total = xp.zeros_like(since_epoch)
    if roemer or shapiro:
        earth_position = _table_position(earth, since_epoch, xp)
    if roemer:
        total = total + earth_position @ source
    if einstein:
        total = total + _hermite(grid, since_epoch, xp)
    if shapiro:
        sun_to_earth = earth_position - _table_position(sun, since_epoch, xp)
        distance = xp.sqrt(xp.sum(sun_to_earth * sun_to_earth, axis=-1))
        along = sun_to_earth @ source
        impact = xp.sqrt(xp.maximum(distance * distance - along * along, 0.0))
        usual = 9.852e-6 * xp.log(_AU_OVER_C / (distance + along))
        inside = 9.852e-6 * xp.log(
            _AU_OVER_C / (along + xp.sqrt(_SUN_RADIUS_LIGHT_SECONDS**2 + along * along))
        ) + 19.704e-6 * (1.0 - impact / _SUN_RADIUS_LIGHT_SECONDS)
        total = total - xp.where((impact < _SUN_RADIUS_LIGHT_SECONDS) & (along < 0.0), inside, usual)
    return total


def _table_position(piece: tuple, since_epoch: Any, xp: Any) -> Any:
    """Return positions from a table slice, Taylor-expanded about the nearest entry as LALPulsar does."""
    first, spacing, position, velocity, acceleration = piece
    offset = since_epoch - first
    entry = xp.clip(xp.floor(offset / spacing + 0.5), 0, len(position) - 1).astype(xp.int32)
    since = (offset - spacing * entry)[..., None]
    return (
        xp.asarray(position)[entry]
        + xp.asarray(velocity)[entry] * since
        + 0.5 * xp.asarray(acceleration)[entry] * since * since
    )


def _hermite(grid: tuple, since_epoch: Any, xp: Any) -> Any:
    """Return a cubic-Hermite interpolation of ``(first node, spacing, value, rate)`` at ``since_epoch``."""
    first, spacing, value, rate = grid
    position = (since_epoch - first) / spacing
    index = xp.clip(xp.floor(position), 0, len(value) - 2).astype(xp.int32)
    u = position - index
    value, slope = xp.asarray(value), xp.asarray(rate) * spacing
    return (
        (1 + 2 * u) * (1 - u) ** 2 * value[index]
        + u * (1 - u) ** 2 * slope[index]
        + u**2 * (3 - 2 * u) * value[index + 1]
        + u**2 * (u - 1) * slope[index + 1]
    )
