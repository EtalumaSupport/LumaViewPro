# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol step records the turret slot it was made at, and the run turns there (#488).

A step stored only its objective's name. With one objective in two turret
slots, the slot came from one preference for the whole scope -- the last
slot any step's move landed on -- so changing one step to slot 1 sent every
step naming that objective to slot 1; and the startup move to slot 1
overwrote the saved preference on every launch. Each step now carries its
own ``Turret Slot``. A step that cannot say which slot it means -- one from
an older protocol whose objective is in two slots, or one whose slot now
holds another objective -- is asked about when the protocol is opened, and
refused by a run and by going to it until answered.
"""

from __future__ import annotations

import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ArgumentRefusedError, ProtocolRunRefusedError
from modules.image_mode import ImageCaptureConfig
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

DOUBLED = '10x Oly'
# One objective in slots 1 and 2, as in the reporter's turret.
TURRET = {1: DOUBLED, 2: DOUBLED, 3: '20x Oly', 4: None}


@pytest.fixture
def session(tmp_path):
    settings = complete_settings(microscope='LS850T', live_folder=str(tmp_path))
    settings['turret_objectives'] = dict(TURRET)
    built = ScopeSession.create(settings, simulate=True)
    home_sim_scope(built.scope)
    assert built.scope.capabilities.has_turret
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _acquire_bf_only(session):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'


def _a_step_at_each_slot(session, slots=(1, 2)):
    _acquire_bf_only(session)
    protocol = session.create_empty_protocol()
    for slot in slots:
        session.scope.motion.move_turret(slot)
        session.add_step(protocol, after_step=protocol.num_steps() - 1)
    return protocol


def _slots(protocol):
    return list(protocol.steps()['Turret Slot'])


def _targets(session, protocol):
    return [
        session.scope.protocols.step_targets(protocol, idx).turret_slot
        for idx in range(protocol.num_steps())
    ]


def _prepare(session, protocol, tmp_path):
    return session.sequenced_capture_runner.prepare(
        protocol=protocol,
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='t',
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        autogain_settings={},
        parent_dir=tmp_path / 'out',
    )


def _as_an_old_file(session, protocol, tmp_path):
    """``protocol`` saved, then rewritten as a v8 file: no slot column."""
    path = tmp_path / 'old.tsv'
    protocol.to_file(path)
    lines = path.read_text().splitlines()
    header = lines.index('Steps') + 1
    drop = lines[header].split('\t').index('Turret Slot')
    rewritten = []
    for i, line in enumerate(lines):
        if line.startswith('Version'):
            line = 'Version\t8'
        elif i >= header and line:
            cells = line.split('\t')
            del cells[drop]
            line = '\t'.join(cells)
        rewritten.append(line)
    path.write_text('\n'.join(rewritten) + '\n')
    return path


class TestEachStepKeepsItsOwnSlot:
    def test_an_add_records_the_slot_it_was_made_at(self, session):
        protocol = _a_step_at_each_slot(session)

        assert _slots(protocol) == [1, 2]

    def test_a_run_and_the_navigation_turn_each_step_to_its_own_slot_in_either_order(self, session):
        protocol = _a_step_at_each_slot(session, slots=(2, 1, 2))

        assert _targets(session, protocol) == [2, 1, 2]
        for idx, slot in enumerate((2, 1, 2)):
            session.go_to_step(protocol, idx)
            session.scope.motion.wait_until_finished_moving(timeout_s=30)
            assert session.scope.motion.get_turret_slot() == slot

    def test_changing_one_step_to_a_slot_leaves_the_others_where_they_are(self, session):
        # The reporter's retest: change one step to T1 and every step went to T1.
        protocol = _a_step_at_each_slot(session, slots=(2, 2))
        session.scope.motion.move_turret(1)

        session.update_step(protocol, 0, layer='BF')

        assert _slots(protocol) == [1, 2]
        assert _targets(session, protocol) == [1, 2]

    def test_a_new_protocol_records_the_slot_in_the_light_path(self, session):
        _acquire_bf_only(session)
        session.scope.motion.move_turret(2)

        protocol = session.new_protocol()

        assert set(_slots(protocol)) == {2}

    def test_a_saved_protocol_reads_back_with_its_slots(self, session, tmp_path):
        protocol = _a_step_at_each_slot(session)
        path = tmp_path / 'p.tsv'
        protocol.to_file(path)

        loaded = session.scope.protocols.load_protocol(path)

        assert _slots(loaded) == [1, 2]
        assert session.scope.protocols.turret_slot_questions(loaded) == []


class TestOldProtocols:
    def test_a_step_whose_objective_is_in_one_slot_gets_it_and_nothing_is_asked(
        self, session, tmp_path
    ):
        _acquire_bf_only(session)
        session.scope.motion.move_turret(3)
        protocol = session.create_empty_protocol()
        session.add_step(protocol, after_step=-1)

        loaded = session.scope.protocols.load_protocol(_as_an_old_file(session, protocol, tmp_path))

        assert _slots(loaded) == [3]
        assert session.scope.protocols.turret_slot_questions(loaded) == []

    def test_steps_whose_objective_is_in_two_slots_are_asked_about_and_refused_until_answered(
        self, session, tmp_path
    ):
        old = _as_an_old_file(session, _a_step_at_each_slot(session), tmp_path)

        loaded = session.scope.protocols.load_protocol(old)

        assert _slots(loaded) == [-1, -1]
        (question,) = session.scope.protocols.turret_slot_questions(loaded)
        assert (question.objective_id, question.steps, question.reassigned, question.choices) == (
            DOUBLED,
            (0, 1),
            (),
            (1, 2),
        )
        with pytest.raises(ProtocolRunRefusedError) as at_prepare:
            _prepare(session, loaded, tmp_path)
        assert at_prepare.value.reason == 'turret_slot_ambiguous'
        with pytest.raises(ProtocolRunRefusedError) as at_step:
            session.scope.protocols.step_targets(loaded, 0)
        assert at_step.value.reason == 'turret_slot_ambiguous'

        assert session.scope.protocols.point_steps_at_turret_slot(loaded, DOUBLED, 2) == (0, 1)

        assert _targets(session, loaded) == [2, 2]
        assert session.scope.protocols.turret_slot_questions(loaded) == []

    def test_an_answer_naming_a_slot_without_the_objective_is_refused_and_changes_nothing(
        self, session, tmp_path
    ):
        loaded = session.scope.protocols.load_protocol(
            _as_an_old_file(session, _a_step_at_each_slot(session), tmp_path)
        )

        with pytest.raises(ArgumentRefusedError) as refused:
            session.scope.protocols.point_steps_at_turret_slot(loaded, DOUBLED, 3)

        assert (refused.value.reason, refused.value.offered) == (
            'slot_not_holding_objective',
            (1, 2),
        )
        assert _slots(loaded) == [-1, -1]


class TestAReassignedSlot:
    def test_a_step_whose_slot_now_holds_another_objective_loads_and_is_refused_until_answered(
        self, session, tmp_path
    ):
        protocol = _a_step_at_each_slot(session)
        path = tmp_path / 'p.tsv'
        protocol.to_file(path)
        session.assign_turret_objective(2, '20x Oly')

        loaded = session.scope.protocols.load_protocol(path)

        (question,) = session.scope.protocols.turret_slot_questions(loaded)
        assert (question.steps, question.reassigned, question.choices) == ((1,), (1,), (1,))
        with pytest.raises(ProtocolRunRefusedError) as at_prepare:
            _prepare(session, loaded, tmp_path)
        assert at_prepare.value.reason == 'turret_slot_reassigned'
        with pytest.raises(ProtocolRunRefusedError) as at_step:
            session.scope.protocols.step_targets(loaded, 1)
        assert at_step.value.reason == 'turret_slot_reassigned'

        session.scope.protocols.point_steps_at_turret_slot(loaded, DOUBLED, 1)

        assert _targets(session, loaded) == [1, 1]
