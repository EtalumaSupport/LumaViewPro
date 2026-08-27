# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Assigning an objective to a turret position, and staying quiet about it.

Two defects lived in this flow.

The slot key: `turret_objectives` is loaded from JSON, so its keys are
strings, while the Set/Reset handlers pick the slot with `range(1, 5)` and
so held an int. Writing the int added a SECOND entry beside the string one
-- the string key kept its old value, every reader keyed by string saw the
stale answer, and the saved file carried a duplicate key that resolved
last-wins on the next load.

The dialog: selecting an objective the turret does not hold raised a popup
saying the selection could not be made, immediately before making it. That
is the first half of assigning the objective, so the interruption landed in
the middle of the workflow that resolves it.
"""

import ast
import inspect
import json


class TestTurretSlotKeyMatchesTheFile:
    def test_string_and_int_keys_are_not_interchangeable(self):
        """The premise, run rather than asserted: this is why the int write
        was invisible to string-keyed readers."""
        loaded = json.loads('{"1": "4x Oly", "2": null, "3": null, "4": null}')
        loaded[2] = '20x w/collar'
        assert loaded.get('2') is None
        assert loaded.get(2) == '20x w/collar'
        assert json.dumps(loaded).count('"2"') == 2, 'the int key round-trips as a duplicate'

    def test_set_and_reset_write_the_string_slot_key(self):
        import ui.vertical_control as vc

        for name in ('set_turret_objective', 'reset_turret_objective'):
            src = inspect.getsource(getattr(vc.VerticalControl, name))
            assert "settings['turret_objectives'][str(selected_turret)]" in src, (
                f'{name} must key the slot by str(): the dict comes from JSON and is '
                f'string-keyed, so an int key writes a second entry instead of the one'
            )

    def test_a_populated_slot_map_survives_a_json_round_trip(self):
        """A dict whose keys all agree carries every slot across a save/load
        with no duplicates and no lost assignment."""
        slots = {'1': '4x Oly', '2': None, '3': None, '4': None}
        slots[str(2)] = '20x w/collar'
        reloaded = json.loads(json.dumps(slots))
        assert reloaded == {'1': '4x Oly', '2': '20x w/collar', '3': None, '4': None}
        assert len(reloaded) == 4


class TestSelectingAnObjectiveIsQuiet:
    def test_objective_selection_raises_no_notification(self):
        # The guard stays -- it is how the condition reaches the log -- but it
        # must not interrupt. The refusals that matter live at protocol
        # create / modify / add / run, and those really do refuse.
        import textwrap

        import ui.vertical_control as vc

        src = ast.unparse(
            ast.parse(textwrap.dedent(inspect.getsource(vc.VerticalControl.select_objective)))
        )
        assert 'notifications.' not in src, (
            'selecting an objective must not raise a notification: it is the first '
            'step of assigning one, and the write below it always succeeds'
        )
        assert 'turret_objectives' in src, (
            'the unassigned-objective condition should still reach the log'
        )
