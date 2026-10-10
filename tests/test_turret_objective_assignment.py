# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Assigning an objective to a turret position, and staying quiet about it.

Two defects lived in this flow.

The slot key: `turret_objectives` is loaded from JSON, so its keys are
strings, while the Set/Reset handlers picked the slot with `range(1, 5)` and
so held an int (the writers now live on the Session, keyed the same way). Writing the int added a SECOND entry beside the string one
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
from tests.installation_fixtures import copy_installation_files


class TestTurretSlotKeyMatchesTheFile:
    def test_string_and_int_keys_are_not_interchangeable(self):
        """The premise, run rather than asserted: this is why the int write
        was invisible to string-keyed readers."""
        loaded = json.loads('{"1": "4x Oly", "2": null, "3": null, "4": null}')
        loaded[2] = '20x w/collar'
        assert loaded.get('2') is None
        assert loaded.get(2) == '20x w/collar'
        assert json.dumps(loaded).count('"2"') == 2, 'the int key round-trips as a duplicate'

    def test_assign_and_clear_write_the_runtime_key_type(self):
        from modules.scope_session import ScopeSession

        for name in ('assign_turret_objective', 'clear_turret_objective'):
            src = inspect.getsource(getattr(ScopeSession, name))
            assert "self.settings['turret_objectives'][position]" in src, (
                f'{name} must key the slot by the int the pipeline guarantees; '
                f'a str() key writes a second entry beside the real one'
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
        # On a turreted scope a selection assigns the slot in the light path,
        # so there is no unassigned selection to interrupt. The refusals that
        # matter live at protocol create / modify / add / run.
        import textwrap

        from modules.scope_session import ScopeSession

        src = ast.unparse(
            ast.parse(textwrap.dedent(inspect.getsource(ScopeSession.select_objective)))
        )
        assert 'notifications.' not in src, (
            'selecting an objective must not raise a notification: on a turreted '
            'scope it is the assignment itself'
        )
        assert 'self.assign_turret_objective(slot, objective_id)' in src, (
            'a selection on a turreted scope must assign the slot in the light path'
        )


_REQUIRED_KEYS = {
    'frame': {'width': 1920, 'height': 1200},
    'live_folder': '.',
    'microscope': 'LS850T',
}


class TestBothHostsAgreeOnTheSlotKeyType:
    """The saved file used to carry eight entries for a four-slot turret --
    the stale int-keyed set the GUI produced, then the string keys the
    assignment path wrote, both emitted by a plain json.dump. It read back
    correctly only because json.loads is last-wins.
    """

    def _prepared(self, tmp_path, on_disk):
        import json as _json
        import logging

        import modules.settings_init as si

        payload = {**_REQUIRED_KEYS, **on_disk}
        data = tmp_path / 'data'
        data.mkdir()
        (data / 'current.json').write_text(_json.dumps(payload))
        (data / 'settings.json').write_text(_json.dumps(payload))
        prepared, _ = si.prepare_settings(
            logging.getLogger('t'), str(tmp_path), fall_back_to_template=True
        )
        return prepared

    def test_the_pipeline_hands_every_host_int_keys(self, tmp_path):
        prepared = self._prepared(
            tmp_path, {'turret_objectives': {'1': '4x Oly', '2': None, '3': None, '4': None}}
        )
        assert list(prepared['turret_objectives']) == [1, 2, 3, 4]

    def test_a_file_already_holding_duplicates_saves_back_clean(self, tmp_path, monkeypatch):
        """Field files carry the duplicates already. Assigning and saving
        through the session must leave four entries in the written file.

        The assertion parses with object_pairs_hook rather than json.load:
        json.load is last-wins, so it resolves the very state under test and
        would pass while the duplicate was still on disk.
        """
        import json as _json
        import os
        import shutil

        import modules.settings_init as si
        from modules.scope_session import ScopeSession

        shipped = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        data = tmp_path / 'data'
        data.mkdir()
        shutil.copy(os.path.join(shipped, 'settings.json'), data / 'settings.json')
        copy_installation_files(data)

        # A real field file: duplicate keys, exactly as the bench bundle held them.
        required = _json.dumps(_REQUIRED_KEYS)[1:-1]
        (data / 'current.json').write_text(
            '{' + required + ', "turret_objectives": {"1": "40x w/collar", '
            '"2": "10x Oly", "3": "20x Phase", "4": "10x Phase", '
            '"4": "60x w/collar", "3": "4x Oly", "2": "20x Oly", '
            '"1": "1.25x Oly"}}'
        )
        # load_user_settings reuses these module globals when they are populated,
        # and would then never read the file written above.
        monkeypatch.setattr(si, 'settings', None)
        monkeypatch.setattr(si, 'rejected_current_json', None)

        session = ScopeSession.create(
            ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
        )
        # Last-wins already resolves slot 2 to '20x Oly', so assigning that
        # value would pass even with the save path gutted. This one changes it.
        session.assign_turret_objective(2, '4x Oly')
        session.save_settings('./data/current.json')

        saved = _json.loads(
            (data / 'current.json').read_text(), object_pairs_hook=lambda pairs: pairs
        )
        slots = next(value for key, value in saved if key == 'turret_objectives')
        assert len(slots) == 4, f'saved file still carries duplicates: {slots}'
        assert dict(slots)['2'] == '4x Oly', f'assignment did not reach the file: {slots}'
