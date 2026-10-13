# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Assigning an objective to a turret position, and picking one.

The slot key: `turret_objectives` is loaded from JSON, so its keys are
strings, while the Set/Reset handlers picked the slot with `range(1, 5)` and
so held an int (the writers now live on the Session, keyed the same way). Writing the int added a SECOND entry beside the string one
-- the string key kept its old value, every reader keyed by string saw the
stale answer, and the saved file carried a duplicate key that resolved
last-wins on the next load.

Picking an objective on a turret scope turns the turret to the one slot
holding it, and writes no slot: the pick used to assign the slot in the
light path, so one objective could end up on two slots. A scope with no
turret refuses every slot write.
"""

import json

import pytest

from modules.exceptions import ArgumentRefusedError, HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.installation_fixtures import copy_installation_files
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.settings_fixtures import complete_settings


class TestTurretSlotKeyMatchesTheFile:
    def test_string_and_int_keys_are_not_interchangeable(self):
        """The premise, run rather than asserted: this is why the int write
        was invisible to string-keyed readers."""
        loaded = json.loads('{"1": "4x Oly", "2": null, "3": null, "4": null}')
        loaded[2] = '20x w/collar'
        assert loaded.get('2') is None
        assert loaded.get(2) == '20x w/collar'
        assert json.dumps(loaded).count('"2"') == 2, 'the int key round-trips as a duplicate'

    def test_assign_and_clear_write_the_runtime_key_type(self, sim_turreted_session):
        """A slot written and cleared stays keyed by its int, with no str twin."""
        session = sim_turreted_session
        before = dict(session.settings['turret_objectives'])
        try:
            session.assign_turret_objective(4, '4x Oly')
            assert session.settings['turret_objectives'][4] == '4x Oly'
            session.clear_turret_objective(4)
            slots = session.settings['turret_objectives']
            assert slots[4] is None
            assert {type(k) for k in slots} == {int}, slots
        finally:
            if before[4] is not None:
                session.assign_turret_objective(4, before[4])

    def test_a_populated_slot_map_survives_a_json_round_trip(self):
        """A dict whose keys all agree carries every slot across a save/load
        with no duplicates and no lost assignment."""
        slots = {'1': '4x Oly', '2': None, '3': None, '4': None}
        slots[str(2)] = '20x w/collar'
        reloaded = json.loads(json.dumps(slots))
        assert reloaded == {'1': '4x Oly', '2': '20x w/collar', '3': None, '4': None}
        assert len(reloaded) == 4


@pytest.fixture
def turret_session(tmp_path):
    """A homed LS850T session whose slots are the test objectives, the turret on slot 1."""
    session = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            live_folder=str(tmp_path),
        ),
        simulate=True,
    )
    home_sim_scope(session.scope)
    session.scope.motion.move_turret(1)
    yield session
    session.shutdown()


@pytest.fixture
def turretless_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(microscope='LS850', live_folder=str(tmp_path)), simulate=True
    )
    yield session
    session.shutdown()


class TestPickingAnObjectiveTurnsTheTurret:
    """``select_objective(X)`` on a turret scope moves to the one slot holding X.

    It records nothing: which glass is in which slot is the assignment's
    (``assign_turret_objective``), so a pick can never put one objective
    on two slots.
    """

    def test_the_pick_moves_to_the_slot_holding_it_and_writes_no_slot(self, turret_session):
        before = dict(turret_session.settings['turret_objectives'])

        assert turret_session.select_objective('20x Oly') is True

        assert turret_session.scope.motion.get_turret_slot() == 2
        assert turret_session.settings['turret_objectives'] == before
        assert turret_session.scope.runtime_state.get_current_objective_id() == '20x Oly'

    def test_an_objective_two_slots_hold_is_refused_naming_them(self, tmp_path):
        session = ScopeSession.create(
            complete_settings(
                microscope='LS850T',
                turret_objectives={1: '4x Oly', 2: '10x Oly', 3: '10x Oly', 4: None},
                live_folder=str(tmp_path),
            ),
            simulate=True,
        )
        try:
            home_sim_scope(session.scope)
            session.scope.motion.move_turret(1)

            with pytest.raises(ArgumentRefusedError) as excinfo:
                session.select_objective('10x Oly')

            assert excinfo.value.reason == 'objective_in_several_slots'
            assert excinfo.value.offered == (2, 3)
            assert session.scope.motion.get_turret_slot() == 1
        finally:
            session.shutdown()

    def test_an_objective_no_slot_holds_is_refused_naming_what_the_turret_carries(
        self, turret_session
    ):
        with pytest.raises(ArgumentRefusedError) as excinfo:
            turret_session.select_objective('40x Phase')

        assert excinfo.value.reason == 'objective_not_on_turret'
        assert excinfo.value.offered == ('10x Oly', '20x Oly', '4x Oly')
        assert turret_session.scope.motion.get_turret_slot() == 1

    def test_a_confirmation_on_a_turret_scope_names_its_slot(self, turret_session):
        """The answer records glass in a slot; with no slot named it has none to record."""
        with pytest.raises(ArgumentRefusedError) as excinfo:
            turret_session.confirm_objective('20x Oly')

        assert excinfo.value.reason == 'turret_position_required'
        assert turret_session.settings['turret_objectives'][1] == '10x Oly'


class TestAScopeWithNoTurretRefusesEverySlotWrite:
    """Row 34: a slot write on a turretless scope is refused before its catalogue check.

    Accepted, it lay in the settings until a model change to a turret scope
    made it that turret's assignment.
    """

    def test_each_slot_writer_is_refused_for_the_absent_turret(self, turretless_session):
        # Not a catalogue key: the absent turret is the first thing said.
        writes = {
            'assign_turret_objective': lambda s: s.assign_turret_objective(1, 'no such glass'),
            'clear_turret_objective': lambda s: s.clear_turret_objective(1),
            'clear_current_turret_objective': lambda s: s.clear_current_turret_objective(),
            'confirm_objective': lambda s: s.confirm_objective('no such glass', turret_position=1),
        }
        before = dict(turretless_session.settings['turret_objectives'])
        wrong = {}
        for member, write in writes.items():
            try:
                write(turretless_session)
            except HardwareCommandRefusedError as e:
                if e.reason != 'axis_absent':
                    wrong[member] = e.reason
            except Exception as e:  # each member's answer is collected, so all four are named
                wrong[member] = repr(e)
            else:
                wrong[member] = 'accepted'

        assert wrong == {}, f'refused for something other than the absent turret: {wrong}'
        assert turretless_session.settings['turret_objectives'] == before


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
            ScopeSession.load_user_settings(str(tmp_path)),
            source_path=str(tmp_path),
            simulate=True,
            settings_file=str(data / 'current.json'),
        )
        # Last-wins already resolves slot 2 to '20x Oly', so assigning that
        # value would pass even with the save path gutted. This one changes it.
        session.assign_turret_objective(2, '4x Oly')
        session.shutdown()

        saved = _json.loads(
            (data / 'current.json').read_text(), object_pairs_hook=lambda pairs: pairs
        )
        slots = next(value for key, value in saved if key == 'turret_objectives')
        assert len(slots) == 4, f'saved file still carries duplicates: {slots}'
        assert dict(slots)['2'] == '4x Oly', f'assignment did not reach the file: {slots}'
