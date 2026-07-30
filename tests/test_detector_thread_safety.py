"""Registering a CustomDetector must be safe from several threads at once.

``lal.cached_detector_by_prefix`` is process-global and mutable, and registration is a
check-then-act: read the registry for a free prefix, then write to it. Two threads doing that
concurrently can settle on the same prefix, both register, and the last write wins — after which
``resolve_detectors`` hands two channels the same lookup key, and with it the same geometry, with no
error anywhere. A network that silently contains one detector twice still looks physically plausible,
which is what makes this worth a lock rather than a note.

The race window is microseconds wide, so a bare threaded test passes almost always whether the lock
is there or not. These tests widen it deliberately, by making the LAL call that sits between the
check and the write take measurable time. That is the difference between a test that reproduces the
defect and one that merely runs threads.
"""

from __future__ import annotations

import threading
import time

import lal
import pytest

from gwmock_signal.detector import CustomDetector, _generate_detector_prefix
from gwmock_signal.projection.geometry import reconstructed_geometry, resolve_detectors

_THREADS = 8
#: Long enough that every thread is inside the check-and-register window together, short enough not
#: to matter to the suite.
_WINDOW_SECONDS = 0.02


def _detector(name: str, longitude: float) -> CustomDetector:
    """A detector at a chosen longitude, with an auto-allocated prefix."""
    return CustomDetector(
        name=name,
        latitude_rad=0.7615,
        longitude_rad=longitude,
        elevation_m=51.884,
        xarm_azimuth_rad=0.3387,
        yarm_azimuth_rad=1.3861,
    )


@pytest.fixture
def crowded_prefix_space(monkeypatch: pytest.MonkeyPatch):
    """Shrink the prefix alphabet so a collision is likely rather than a one-in-a-thousand event.

    ``_generate_detector_prefix`` seeds its search from ``uuid4`` over 62*62 = 3844 slots, so eight
    threads racing collide only about 0.7% of the time. A test that reproduces the defect less than
    once per hundred runs is not a regression test. With a three-character alphabet there are nine
    slots for eight detectors, so an unsynchronised search collides almost every run while a
    synchronised one still finds a free slot for each.
    """
    monkeypatch.setattr("gwmock_signal.detector._PREFIX_ALPHABET", "ABC")


@pytest.fixture
def slow_registration(monkeypatch: pytest.MonkeyPatch):
    """Widen the window between the prefix check and the registry write.

    Patches ``lal.CreateDetector``, which the production code calls after choosing a prefix and
    before publishing it. Under the lock this serialises; without it, every thread sits in the window
    at once and they collide reliably.
    """
    real_create = lal.CreateDetector

    def _slow_create(*args: object, **kwargs: object) -> object:
        time.sleep(_WINDOW_SECONDS)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(lal, "CreateDetector", _slow_create)


def _register_concurrently(detectors: list[CustomDetector]) -> list[Exception | str]:
    """Call ``to_lal`` on every detector at once; return each prefix or the exception raised."""
    outcomes: list[Exception | str | None] = [None] * len(detectors)
    barrier = threading.Barrier(len(detectors))

    def _worker(index: int) -> None:
        barrier.wait()
        try:
            outcomes[index] = detectors[index].to_lal().frDetector.prefix
        except Exception as exc:  # noqa: BLE001 - recording the failure *is* the measurement
            outcomes[index] = exc

    threads = [threading.Thread(target=_worker, args=(index,)) for index in range(len(detectors))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [outcome for outcome in outcomes if outcome is not None]


def test_concurrent_registration_gives_every_detector_its_own_prefix(
    slow_registration: None, crowded_prefix_space: None
) -> None:
    """Distinct detectors registering at once must not end up sharing a lookup key.

    This is the aliasing hazard: sharing a prefix means sharing a geometry, silently.
    """
    detectors = [_detector(f"THREADED{index}", -1.5 + 0.2 * index) for index in range(_THREADS)]
    outcomes = _register_concurrently(detectors)

    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert not failures, f"registration raised under contention: {failures}"
    prefixes = sorted(outcomes)
    assert len(set(prefixes)) == _THREADS, f"prefixes collided under contention: {prefixes}"


def test_concurrently_registered_detectors_keep_distinct_geometry(
    slow_registration: None, crowded_prefix_space: None
) -> None:
    """The consequence, checked through the public resolver rather than on the prefixes.

    Two channels resolving to one key would be caught by ``resolve_detectors``, but only if the
    prefixes actually differ; this checks the geometry behind them differs too, since that is what a
    caller ultimately gets.
    """
    detectors = [_detector(f"GEOMETRY{index}", -1.5 + 0.3 * index) for index in range(4)]
    _register_concurrently(detectors)

    keys = [key for _, key in resolve_detectors(detectors)]
    assert len(set(keys)) == len(detectors)
    locations = [tuple(reconstructed_geometry(key)[1]) for key in keys]
    assert len(set(locations)) == len(detectors), "two concurrently registered detectors share a location"


def test_an_explicit_prefix_clash_still_raises_under_contention(slow_registration: None) -> None:
    """With one prefix named explicitly by several detectors, exactly one may win.

    An auto-allocated prefix is reallocated on a clash; an explicit one is a caller error and must
    stay an error even when the clash is produced by a race rather than by sequential calls.
    """
    detectors = [
        CustomDetector(
            name=f"EXPLICIT{index}",
            latitude_rad=0.7615,
            longitude_rad=-1.5 + 0.2 * index,
            elevation_m=51.884,
            xarm_azimuth_rad=0.3387,
            yarm_azimuth_rad=1.3861,
            prefix="Z9",
        )
        for index in range(_THREADS)
    ]
    outcomes = _register_concurrently(detectors)

    succeeded = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    failed = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(succeeded) == 1, f"{len(succeeded)} detectors claimed the same explicit prefix"
    assert all(isinstance(exc, ValueError) for exc in failed), f"unexpected exception types: {failed}"


def test_every_prefix_allocation_is_inside_the_lock() -> None:
    """Every call to ``_generate_detector_prefix`` must be lexically inside a ``with`` on the lock.

    It searches the registry another thread may be writing, so an unlocked call reintroduces the race
    one level down. Checked structurally rather than by timing, because a missing lock here is a code
    property, not a probabilistic one.

    Via the AST, not a substring search. An earlier version asked whether ``_REGISTRY_LOCK`` appeared
    anywhere earlier in the enclosing function, which a mere docstring mention would satisfy -- a
    check that cannot fail for the reason it claims to.
    """
    import ast
    import inspect

    from gwmock_signal import detector as detector_module

    tree = ast.parse(inspect.getsource(detector_module))

    def _calls_to_allocator(node: ast.AST) -> list[ast.Call]:
        return [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_generate_detector_prefix"
        ]

    def _guards_the_lock(with_node: ast.With) -> bool:
        return any(
            isinstance(item.context_expr, ast.Name) and item.context_expr.id == "_REGISTRY_LOCK"
            for item in with_node.items
        )

    every_call = _calls_to_allocator(tree)
    assert every_call, "no call to _generate_detector_prefix found; this test is watching nothing"

    guarded = {
        id(call)
        for node in ast.walk(tree)
        if isinstance(node, ast.With) and _guards_the_lock(node)
        for call in _calls_to_allocator(node)
    }
    unguarded = [call for call in every_call if id(call) not in guarded]
    assert not unguarded, (
        f"unlocked call(s) to _generate_detector_prefix at line(s) {sorted(call.lineno for call in unguarded)}"
    )
    assert callable(_generate_detector_prefix)
