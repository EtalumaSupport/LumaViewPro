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


class TestTurretSlotKeyMatchesTheFile:
    def test_string_and_int_keys_are_not_interchangeable(self):
        """The premise, run rather than asserted: this is why the int write
        was invisible to string-keyed readers."""
        loaded = json.loads('{"1": "4x Oly", "2": null, "3": null, "4": null}')
        loaded[2] = '20x w/collar'
        assert loaded.get('2') is None
        assert loaded.get(2) == '20x w/collar'
        assert json.dumps(loaded).count('"2"') == 2, 'the int key round-trips as a duplicate'

    def test_the_conversion_has_exactly_one_home(self):
        """A turret position is a number, so the slot map is int-keyed at
        runtime and string-keyed only on disk, where JSON forces it.

        The conversion belongs to the shared settings pipeline, which every
        host runs. It used to live in the GUI's settings load instead, so a
        headless caller kept string keys while the GUI had ints -- the two
        hosts disagreed about the type of the same dict, which duplicated
        keys in the saved file and raised KeyError off the GUI. Pinning it
        here keeps a second converter from reappearing somewhere a headless
        caller never reaches.
        """
        import modules.settings_init as si
        import ui.microscope_settings as ms
        import ui.vertical_control as vc

        assert hasattr(si, '_normalize_turret_slot_keys')

        for module in (ms, vc):
            src = inspect.getsource(module)
            assert 'int(k): v for k, v in' not in src, (
                f'{module.__name__} rebuilds the slot map keys; the shared '
                f'settings pipeline is the only place that may convert them'
            )

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
        # The guard stays -- it is how the condition reaches the log -- but it
        # must not interrupt. The refusals that matter live at protocol
        # create / modify / add / run, and those really do refuse.
        import textwrap

        from modules.scope_session import ScopeSession

        src = ast.unparse(
            ast.parse(textwrap.dedent(inspect.getsource(ScopeSession.select_objective)))
        )
        assert 'notifications.' not in src, (
            'selecting an objective must not raise a notification: it is the first '
            'step of assigning one, and the write below it always succeeds'
        )
        assert 'turret_objectives' in src, (
            'the unassigned-objective condition should still reach the log'
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

    def test_a_file_already_holding_duplicates_saves_back_clean(self, tmp_path):
        """Field files carry the duplicates already. Last-wins resolves them
        to the current values on load, and the next save must emit four."""
        import json as _json

        # A real field file: duplicate keys, exactly as the bench bundle held them.
        required = _json.dumps(_REQUIRED_KEYS)[1:-1]
        raw = (
            '{' + required + ', "turret_objectives": {"1": "40x w/collar", '
            '"2": "10x Oly", "3": "20x Phase", "4": "10x Phase", '
            '"4": "60x w/collar", "3": "4x Oly", "2": "20x Oly", '
            '"1": "1.25x Oly"}}'
        )
        data = tmp_path / 'data'
        data.mkdir()
        (data / 'current.json').write_text(raw)
        (data / 'settings.json').write_text(raw)

        import logging

        import modules.settings_init as si

        prepared, _ = si.prepare_settings(
            logging.getLogger('t'), str(tmp_path), fall_back_to_template=True
        )
        prepared['turret_objectives'][2] = '20x Oly'

        dumped = _json.dumps(prepared['turret_objectives'])
        assert dumped.count('":') == 4, f'saved file still carries duplicates: {dumped}'
        assert _json.loads(dumped)['2'] == '20x Oly'

    def test_an_assignment_does_not_add_a_parallel_entry(self, tmp_path):
        import json as _json

        prepared = self._prepared(
            tmp_path, {'turret_objectives': {'1': None, '2': None, '3': None, '4': None}}
        )
        prepared['turret_objectives'][2] = '20x Oly'
        assert len(prepared['turret_objectives']) == 4
        assert _json.dumps(prepared['turret_objectives']).count('":') == 4
