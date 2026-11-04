"""The projection effects: their defaults, the long-wavelength preset, and the deprecated spellings."""

from __future__ import annotations

import dataclasses

import pytest

from gwmock_signal.projection.effects import ProjectionEffects
from gwmock_signal.projection.network import _resolve_effects


def test_every_effect_is_on_by_default():
    """The default includes every effect; the long-wavelength preset none, unless Earth rotation is asked for."""
    names = [field.name for field in dataclasses.fields(ProjectionEffects)]
    assert all(getattr(ProjectionEffects(), name) for name in names)
    assert not any(getattr(ProjectionEffects.long_wavelength(), name) for name in names)
    rotating = ProjectionEffects.long_wavelength(earth_rotation=True)
    assert rotating.earth_rotation
    assert not any(getattr(rotating, name) for name in names if name != "earth_rotation")


def test_solar_system_is_any_of_the_three_delays():
    """``solar_system`` is true when any one of the Roemer, Einstein and Shapiro delays is on."""
    none = ProjectionEffects(roemer_delay=False, einstein_delay=False, shapiro_delay=False)
    assert not none.solar_system
    for name in ("roemer_delay", "einstein_delay", "shapiro_delay"):
        assert dataclasses.replace(none, **{name: True}).solar_system


@pytest.mark.parametrize(
    ("keyword", "field"), [("earth_rotation", "earth_rotation"), ("precess_source_direction", "source_precession")]
)
def test_a_deprecated_keyword_overrides_its_field_and_warns(keyword, field):
    """The separate keywords from before ``ProjectionEffects`` still work, and say what replaces them."""
    with pytest.warns(DeprecationWarning, match=f"{keyword}=.*ProjectionEffects\\({field}="):
        resolved = _resolve_effects(None, **{keyword: False})
    assert resolved == dataclasses.replace(ProjectionEffects(), **{field: False})


def test_no_deprecated_keyword_leaves_the_effects_alone(recwarn):
    """Without a deprecated keyword the effects pass through unchanged, and nothing warns."""
    effects = ProjectionEffects(shapiro_delay=False)
    assert _resolve_effects(effects) is effects
    assert _resolve_effects(None) == ProjectionEffects()
    assert not [warning for warning in recwarn if issubclass(warning.category, DeprecationWarning)]
