# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#488 regression: when two turret positions hold the same objective,
the lookup must use the user's intended position, not silently pick
the lowest-numbered match.

Three lookup ranks (per ``MotionAPI.get_turret_position_for_objective_id``):
    1. The preferred slot -- the last slot a turret move landed on,
       seeded at bring-up from the saved turret position -- if it matches.
    2. The turret's current slot, if it matches.
    3. The lowest-numbered slot carrying it.

The bug: the current slot is wrong post-home (T is at 1 by convention),
and nothing remembered the person's choice -- so the lookup always landed
on position 1. The preference lives in the API, so a run and a person's
step navigation read it through the one lookup and choose the same slot.
"""

from pathlib import Path
import re

from modules.lumascope_api import Lumascope
from modules.lumascope_api.motion import MotionAPI
from modules.lumascope_api.runtime_state import RuntimeState


def _make_scope_with_turret(turret_config, current_pos=None, preferred=None):
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
    # The turret's current position is the slot the last turret command
    # left it in (None: not known), never the step counter -- which is made
    # to raise, so a lookup that read it would fail here.
    scope.motion._last_turret_position = current_pos
    scope.motion.seed_preferred_turret_slot(preferred)

    def _raise(*_a, **_kw):
        raise RuntimeError('the step counter is not a slot')

    scope.motion.get_current_position = _raise
    return scope


def test_the_preferred_slot_wins_over_the_post_home_slot():
    """Two slots hold '4x Oly' (positions 1 and 4). Current is 1
    (post-home). Preferred is 4. Lookup must return 4.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
        current_pos=1,
        preferred=4,
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='4x Oly')
    assert result == 4, (
        f'Expected position 4 (preferred), got {result}. The preferred '
        f'slot must rank above the post-home slot and first-match. (#488)'
    )


def test_current_position_still_wins_when_it_matches():
    """If T is already at 4 and 4 holds the objective, return 4 even
    without a preference.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
        current_pos=4,
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='4x Oly')
    assert result == 4


def test_the_preference_is_ignored_when_its_slot_no_longer_holds_the_objective():
    """If a different objective now sits in the preferred slot, the
    preference no longer matches -> fall through. No silent wrong-slot move.
    """
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x Oly'},
        current_pos=None,
        preferred=4,
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='4x Oly')
    assert result == 1, (
        'When the preferred slot no longer holds the objective, lookup '
        'must fall through to first-match.'
    )


def test_no_preference_and_no_known_slot_falls_back_to_first_match():
    """Nothing saved, turret not in a known slot: the lowest-numbered."""
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '4x Oly'},
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='4x Oly')
    assert result == 1


def test_objective_not_in_turret_returns_none():
    scope = _make_scope_with_turret(
        turret_config={1: '4x Oly', 2: '10x Oly'},
        current_pos=1,
        preferred=2,  # not matching
    )
    result = scope.motion.get_turret_position_for_objective_id(objective_id='100x Oly')
    assert result is None


def test_a_turret_move_sets_the_preference_and_a_home_does_not():
    """On the simulated scope: the preference follows the moves a person
    or a run makes, and survives the home that returns the turret to 1."""
    from tests.scope_fakes import homed_sim_scope

    scope = homed_sim_scope()
    try:
        assert scope.motion.get_preferred_turret_slot() is None
        scope.motion.move_turret(3)
        assert scope.motion.get_preferred_turret_slot() == 3
        # Asking for the slot the turret is already in is still a choice.
        scope.motion.move_turret(3)
        assert scope.motion.get_preferred_turret_slot() == 3
        assert scope.motion._home_impl()
        assert scope.motion.get_turret_slot() == 1
        assert scope.motion.get_preferred_turret_slot() == 3
    finally:
        scope.disconnect()


def test_a_seed_that_names_no_slot_is_refused():
    import pytest

    from modules.exceptions import PositionOutOfRangeError

    scope = _make_scope_with_turret(turret_config={1: '4x Oly'})
    with pytest.raises(PositionOutOfRangeError):
        scope.motion.seed_preferred_turret_slot(9)


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text()


def test_step_navigation_asks_the_same_lookup_the_run_asks():
    """Static-source guard: step navigation passes nothing but the
    objective, as the run does, so the two cannot choose different slots
    for a step. A caller-supplied hint was how they came to disagree.
    """
    pattern = re.compile(
        r'get_turret_position_for_objective_id\(([^)]*)\)',
        re.DOTALL,
    )
    for rel in ('ui/step_navigation.py', 'modules/protocol_step_runner.py'):
        calls = pattern.findall(_read(rel))
        assert calls, f'No slot lookup found in {rel} -- has it moved? Update the test.'
        for args in calls:
            assert args.split('=')[0].strip() == 'objective_id' and args.count('=') == 1, (
                f'{rel} passes more than the objective to the slot lookup:\n{args}'
            )
