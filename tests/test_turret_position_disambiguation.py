# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#488 regression: when two turret positions hold the same objective,
the lookup must use the user's intended position, not silently pick
the lowest-numbered match.

Three lookup ranks (per ``Lumascope.get_turret_position_for_objective_id``):
    1. Current physical T position, if it matches.
    2. Persisted position from settings, if it matches.
    3. First-match dict iteration (today's fallback).

The bug: rank 1 fails post-home (T is at 1 by convention), and there
was no rank 2 -- so the lookup always landed on rank 3 = position 1.
This file exercises rank 2 directly so the disambiguation survives
restarts and post-home situations.
"""

from pathlib import Path
import re

from modules.lumascope_api import Lumascope
from modules.lumascope_api.motion import MotionAPI
from modules.lumascope_api.runtime_state import RuntimeState


def _make_scope_with_turret(turret_config, current_pos=None):
    """Construct a minimal Lumascope with just enough state for the
    lookup. Avoids the full hardware init path.
    """
    scope = Lumascope.__new__(Lumascope)
    scope.runtime_state = RuntimeState(scope)
    scope.runtime_state._turret_config = turret_config
    # MotionAPI hosts the relocated body. __new__ skips __init__,
    # which is what sets scope.motion in production, so the test
    # installs the sub-API first so monkeypatches land on the canonical
    # surface. driver=None is OK -- this lookup reads only scope-side
    # state, no driver calls.
    scope.motion = MotionAPI(scope, None)
    if current_pos is not None:
        # Bypass the full position-cache plumbing for the test.
        scope.motion.get_current_position = lambda axis=None: current_pos
    else:

        def _raise(*_a, **_kw):
            raise RuntimeError('current pos unavailable in test')

        scope.motion.get_current_position = _raise
    return scope


def test_persisted_position_wins_over_first_match():
    """Two slots hold '4x Oly' (positions 1 and 4). Current physical
    is 1 (post-home). Persisted is 4. Lookup must return 4.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
        current_pos=1,
    )
    result = scope.motion.get_turret_position_for_objective_id(
        objective_id='4x Oly',
        persisted_position=4,
    )
    assert result == 4, (
        f'Expected position 4 (persisted), got {result}. The persisted '
        f'tier of the lookup must rank above first-match. (#488)'
    )


def test_current_position_still_wins_when_it_matches():
    """If physical T is already at 4 and 4 holds the objective, return
    4 even without a persisted hint -- preserves existing prefer_current
    semantics from f99add7.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
        current_pos=4,
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='4x Oly')
    assert result == 4


def test_persisted_position_ignored_when_objective_changed():
    """If the user moved a different objective into the persisted slot
    between sessions, persisted_position no longer matches -> fall
    through to first-match. No silent wrong-position move.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x Oly'},
        current_pos=1,
    )
    result = scope.motion.get_turret_position_for_objective_id(
        objective_id='4x Oly',
        persisted_position=4,
    )
    assert result == 1, (
        'When persisted slot no longer holds the objective, lookup '
        'must fall through to first-match.'
    )


def test_no_persisted_falls_back_to_first_match():
    """Migration case: existing settings.json has no turret_position
    key. settings.get(...) returns None. Lookup must keep working,
    returning today's first-match value.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
        current_pos=1,
    )
    # current_pos=1 holds '4x Oly' so prefer_current returns 1 first;
    # disable that to exercise the pure first-match fallback.
    result = scope.motion.get_turret_position_for_objective_id(
        objective_id='4x Oly',
        prefer_current=False,
        persisted_position=None,
    )
    assert result == 1


def test_objective_not_in_turret_returns_none():
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly'},
        current_pos=1,
    )
    result = scope.motion.get_turret_position_for_objective_id(
        objective_id='100x Oly',
        persisted_position=2,  # not matching
    )
    assert result is None


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text()


def test_callers_pass_persisted_position():
    """Static-source guard: every call site of
    ``get_turret_position_for_objective_id`` must pass
    ``persisted_position=...``. Without this, the persisted tier is
    dead -- same bug shape as the original #488 if a future caller
    forgets it.

    History: the startup + reconnect call sites were lifted into
    ``ScopeSession.start_application_session`` (2026-05-04), then
    retired entirely when startup stopped looking the position up --
    every session now starts at turret position 1 and adopts that
    slot's objective, so the lookup's remaining production caller is
    step navigation.
    """
    callers = [
        'ui/step_navigation.py',
    ]
    pattern = re.compile(
        r'get_turret_position_for_objective_id\([^)]*\)',
        re.DOTALL,
    )
    for rel in callers:
        src = _read(rel)
        calls = pattern.findall(src)
        assert calls, (
            f'No call to get_turret_position_for_objective_id found in '
            f'{rel} -- has the lookup site moved? Update the test.'
        )
        for call in calls:
            assert 'persisted_position' in call, (
                f'{rel} calls get_turret_position_for_objective_id '
                f'without passing persisted_position. The persisted '
                f'tier of the lookup is dead for this caller -- falls '
                f'through to first-match (#488 regression).\n\nCall:\n{call}'
            )
