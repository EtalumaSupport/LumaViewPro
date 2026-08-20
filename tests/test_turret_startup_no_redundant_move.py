# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: homing seeds the turret position cache so a redundant
select-position-1 at startup does not bounce Z.

Bug
---
The firmware homes the turret to position 1 (home() homes Z -> T -> X/Y, and
home(axis='T') homes T), but Lumascope.motion never seeded _last_turret_position --
it stayed None. The existing idempotent skip in tmove() ("if
_last_turret_position == position: return") therefore could not fire, so the
startup select-position-1 ran a full Z-retract -> rotate -> Z-restore even
though the turret was already at 1.

Fix
---
home() (when T is present) and home(axis='T') seed _last_turret_position = 1 on
success, so a following tmove(1) is a no-op.
"""

from modules.lumascope_api import Lumascope


def test_home_seeds_turret_position_one():
    scope = Lumascope(simulate=True)
    assert scope.capabilities.has_turret, 'simulated scope is expected to have a turret'
    scope.motion.home()
    assert scope.motion._last_turret_position == 1


def test_thome_seeds_turret_position_one():
    scope = Lumascope(simulate=True)
    assert scope.capabilities.has_turret
    scope.motion.home(axis='T')
    assert scope.motion._last_turret_position == 1


def test_tmove_to_one_after_home_is_noop():
    """tmove(1) right after homing must skip the move (and its Z bounce)."""
    scope = Lumascope(simulate=True)
    assert scope.capabilities.has_turret
    scope.motion.home()
    moves = []
    original = scope.motion.move_absolute
    scope.motion.move_absolute = lambda *a, **k: moves.append((a, k))
    try:
        scope.motion.tmove(1)
    finally:
        scope.motion.move_absolute = original
    assert moves == [], f'tmove(1) after home should be a no-op; issued moves: {moves}'
