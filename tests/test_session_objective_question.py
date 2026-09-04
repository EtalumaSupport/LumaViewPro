# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session owns the objective question, its answer and the plain writers.

The objective sets the pixel size stamped into every capture. Whether it is
unknowable -- never confirmed on this install, or the turret sitting on an
unassigned slot -- is a fact the Session decides and exposes as state, so a
REST caller learns it the same way the GUI does; the answer goes back through
``confirm_objective``. The GUI prompt only renders the question. The four
plain writers (objective, slot assign / clear, turret position) live here so
that the settings store and the scope's runtime state can never be written
apart, and so the resolved-optics record fires for every host.
"""

import ast
import json
import logging
import os
import shutil

import pytest

from modules import settings_init
from modules.exceptions import ConfigError
import modules.scope_session as scope_session_module
from modules.scope_session import ScopeSession
from tests.ast_seams import iter_package_modules, parse_module
from tests.settings_fixtures import complete_settings


TURRET_MODEL = 'LS850T'  # data/scopes.json Models: the only Turret:true family
NON_TURRET_MODEL = 'LS850'  # complete_settings()'s own default, Turret:false


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _session(**overrides):
    """A real sim Session on the settings the case needs."""
    return ScopeSession.create_headless(settings=complete_settings(**overrides))


@pytest.fixture
def sessions():
    """Build sessions and shut every one of them down."""
    built = []

    def make(**overrides):
        session = _session(**overrides)
        built.append(session)
        return session

    yield make
    for session in reversed(built):
        try:
            session.shutdown()
        except Exception:
            logging.getLogger(__name__).debug(
                'teardown noise is not the measurement', exc_info=True
            )


@pytest.fixture(autouse=True)
def not_provisional(monkeypatch):
    """Default every case to keepable settings (the single-flight file's shape)."""
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)


def _turret_settings(**overrides):
    base = {
        'microscope': TURRET_MODEL,
        'objective_confirmed': False,
        'turret_position': 1,
        'turret_objectives': {'1': '4x Oly', '2': None, '3': None, '4': None},
        'objective_id': '20x Oly',
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def fresh_log():
    """The suite installs a MagicMock as ``lvp_logger``, so the Session's
    lines never reach pytest's log capture; they are read off the mock,
    reset per case because the mock is process-wide."""
    scope_session_module.logger.reset_mock()


def _clear_log():
    """Drop the bring-up's own lines so a case measures only the act."""
    scope_session_module.logger.reset_mock()


def _lines():
    """Every message the Session and its helpers put on the log, any level."""
    return [str(call.args[0]) for call in scope_session_module.logger.method_calls if call.args]


# ---------------------------------------------------------------------------
# T8 -- objective_question()
# ---------------------------------------------------------------------------


class TestT8ObjectiveQuestion:
    def test_fresh_install_on_a_turret_model_asks_for_position_1(self, sessions):
        session = sessions(**_turret_settings())
        question = session.objective_question()
        assert question is not None
        assert question.turret_position == 1
        assert question.proposed == '4x Oly'
        assert question.choices == tuple(session.objective_helper.get_objectives_list())

    def test_a_non_turret_model_asks_without_a_position(self, sessions):
        session = sessions(**_turret_settings(microscope=NON_TURRET_MODEL, objective_id='10x Oly'))
        question = session.objective_question()
        assert question is not None
        assert question.turret_position is None
        assert question.proposed == '10x Oly'

    def test_confirmed_and_assigned_asks_nothing(self, sessions):
        session = sessions(**_turret_settings(objective_confirmed=True))
        assert session.objective_question() is None

    def test_confirmed_but_the_current_slot_is_empty_asks_for_that_slot(self, sessions):
        session = sessions(
            **_turret_settings(
                objective_confirmed=True,
                turret_position=2,
                turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
            )
        )
        question = session.objective_question()
        assert question is not None
        assert question.turret_position == 2

    def test_the_position_comes_from_settings(self, sessions):
        session = sessions(**_turret_settings(objective_confirmed=True, turret_position=3))
        question = session.objective_question()
        assert question is not None
        assert question.turret_position == 3

    def test_no_hardware_withholds_the_owed_question_and_says_so(self, sessions):
        session = sessions(**_turret_settings())
        session.scope._no_hardware = True
        assert session.scope.no_hardware is True  # the lever works
        assert session.objective_question() is None
        assert any('no hardware' in line for line in _lines()), _lines()

    def test_no_hardware_with_nothing_owed_says_nothing(self, sessions):
        session = sessions(**_turret_settings(objective_confirmed=True))
        session.scope._no_hardware = True
        assert session.scope.no_hardware is True  # the lever works
        assert session.objective_question() is None
        assert not any('no hardware' in line for line in _lines()), _lines()

    def test_provisional_settings_defer_the_question_and_say_so(self, sessions, monkeypatch):
        session = sessions(**_turret_settings())
        monkeypatch.setattr(
            settings_init, 'rejected_current_json', ('/data/current.json', 'invalid JSON')
        )
        assert session.objective_question() is None
        assert any('provisional' in line for line in _lines()), _lines()

    def test_a_proposal_outside_the_catalogue_falls_back_to_the_first_choice(self, sessions):
        # '4' rather than 'banana': the loader PREFIX-matches, so '4' is the
        # only kind of non-catalogue id that survives bring-up and reaches
        # the question as a stored proposal (an id with no close match makes
        # `initialize` raise, so that state cannot be constructed at all).
        session = sessions(
            **_turret_settings(
                turret_objectives={'1': None, '2': None, '3': None, '4': None},
                objective_id='4',
            )
        )
        assert session.settings['objective_id'] == '4'
        assert '4' not in session.objective_helper.get_objectives_list()
        question = session.objective_question()
        assert question is not None
        assert question.proposed == question.choices[0]

    def test_an_empty_catalogue_raises_at_the_api(self, sessions, monkeypatch):
        session = sessions(**_turret_settings())
        monkeypatch.setattr(
            session.objective_helper, 'get_objectives_list', lambda: [], raising=False
        )
        with pytest.raises(ConfigError):
            session.objective_question()

    def test_a_digit_string_position_is_coerced(self, sessions):
        # Nothing types turret_position where the file is read, so a
        # hand-edited "3" reaches the session as a string.
        session = sessions(**_turret_settings(objective_confirmed=True, turret_position='3'))
        question = session.objective_question()
        assert question is not None
        assert question.turret_position == 3

    def test_a_non_numeric_position_raises(self, sessions):
        session = sessions(**_turret_settings(turret_position='three'))
        with pytest.raises(ConfigError, match='turret_position'):
            session.objective_question()

    def test_a_session_without_a_catalogue_raises(self, sessions):
        # The helper is None when objectives.json did not load under the
        # data root; the question names that rather than failing on it.
        session = sessions(**_turret_settings())
        session.objective_helper = None
        with pytest.raises(ConfigError, match='catalogue'):
            session.objective_question()

    def test_a_returned_question_logs_nothing(self, sessions):
        # The renderer logs its own show; a polled read must not log per poll.
        session = sessions(**_turret_settings())
        assert session.objective_question() is not None
        assert not any('objective question' in line for line in _lines()), _lines()


# ---------------------------------------------------------------------------
# T9 -- confirm_objective()
# ---------------------------------------------------------------------------


class TestT9ConfirmObjective:
    def test_a_confirm_writes_both_stores_the_slot_and_the_flag(self, sessions):
        session = sessions(**_turret_settings(turret_position=2))
        assert session.confirm_objective('10x Oly', turret_position=2) is True
        assert session.settings['objective_id'] == '10x Oly'
        assert session.scope.runtime_state.get_current_objective_id() == '10x Oly'
        assert session.settings['turret_objectives'][2] == '10x Oly'
        assert session.settings['objective_confirmed'] is True

    def test_confirming_the_held_objective_reports_no_change(self, sessions):
        session = sessions(**_turret_settings(turret_position=2))
        assert session.confirm_objective('10x Oly', turret_position=2) is True
        assert session.confirm_objective('10x Oly', turret_position=2) is False
        assert session.settings['objective_confirmed'] is True

    def test_a_confirm_binds_the_slot_it_names(self, sessions):
        session = sessions(**_turret_settings(turret_position=2))
        session.confirm_objective('4x Oly', turret_position=2)
        assert session.settings['objective_id'] == '4x Oly'
        assert session.scope.runtime_state.get_current_objective_id() == '4x Oly'
        assert session.settings['turret_objectives'][2] == '4x Oly'
        assert session.settings['objective_confirmed'] is True

    @pytest.mark.parametrize('bad', ['banana', '4', ''])
    def test_a_non_catalogue_id_raises_before_any_write(self, sessions, bad):
        session = sessions(**_turret_settings(turret_position=2))
        before = (
            session.settings['objective_id'],
            session.scope.runtime_state.get_current_objective_id(),
            dict(session.settings['turret_objectives']),
            session.settings.get('objective_confirmed'),
        )
        with pytest.raises(ConfigError):
            session.confirm_objective(bad, turret_position=2)
        after = (
            session.settings['objective_id'],
            session.scope.runtime_state.get_current_objective_id(),
            dict(session.settings['turret_objectives']),
            session.settings.get('objective_confirmed'),
        )
        assert before == after

    def test_a_confirm_without_a_position_writes_no_slot(self, sessions):
        session = sessions(**_turret_settings(microscope=NON_TURRET_MODEL))
        before = dict(session.settings['turret_objectives'])
        session.confirm_objective('4x Oly')
        assert dict(session.settings['turret_objectives']) == before
        assert session.settings['objective_confirmed'] is True

    def test_after_a_confirm_the_question_is_answered(self, sessions):
        session = sessions(**_turret_settings(turret_position=2))
        session.confirm_objective('4x Oly', turret_position=2)
        assert session.objective_question() is None


# ---------------------------------------------------------------------------
# T10 -- the plain writers
# ---------------------------------------------------------------------------


class TestT10SelectObjective:
    def test_selecting_the_held_id_is_a_no_op(self, sessions):
        session = sessions(**_turret_settings())
        held = session.settings['objective_id']
        assert held == '4x Oly'  # slot 1's, adopted at bring-up
        _clear_log()
        assert session.select_objective(held) is False
        assert not any('[Optics' in line for line in _lines()), _lines()

    def test_a_new_id_writes_both_stores_and_records_the_optics(self, sessions):
        session = sessions(**_turret_settings())
        _clear_log()
        assert session.select_objective('10x Oly') is True
        assert session.settings['objective_id'] == '10x Oly'
        assert session.scope.runtime_state.get_current_objective_id() == '10x Oly'
        optics = [line for line in _lines() if line.startswith('[Optics')]
        assert len(optics) == 1, _lines()
        assert 'objective=10x Oly' in optics[0]
        assert 'um/px' in optics[0] and 'no image scale' not in optics[0], optics[0]

    @pytest.mark.parametrize('bad', ['4', '', 'banana'])
    def test_a_non_catalogue_id_is_refused_before_any_write(self, sessions, bad):
        # '4' is a PREFIX of a catalogue key: the loader would bind it to the
        # first match, and '' to the first entry, so the member checks for
        # an exact key itself.
        session = sessions(**_turret_settings())
        before = (
            session.settings['objective_id'],
            session.scope.runtime_state.get_current_objective_id(),
        )
        with pytest.raises(ConfigError):
            session.select_objective(bad)
        assert (
            session.settings['objective_id'],
            session.scope.runtime_state.get_current_objective_id(),
        ) == before

    def test_the_held_id_is_refused_when_it_is_not_a_key(self, sessions):
        # A prefix id survives bring-up (the loader matches it), so a stored
        # non-key id is constructible; it must not read as "no change".
        session = sessions(
            **_turret_settings(
                turret_objectives={'1': None, '2': None, '3': None, '4': None},
                objective_id='4',
            )
        )
        assert session.settings['objective_id'] == '4'
        with pytest.raises(ConfigError):
            session.select_objective('4')


class TestT10TurretWriters:
    def test_assign_writes_the_int_key_and_the_runtime_config(self, sessions):
        session = sessions(**_turret_settings())
        session.assign_turret_objective(2, '10x Oly')
        assert session.settings['turret_objectives'][2] == '10x Oly'
        assert session.scope.runtime_state.get_turret_config()[2] == '10x Oly'

    @pytest.mark.parametrize('position', ['2', 5])
    def test_a_bad_position_raises_value_error(self, sessions, position):
        session = sessions(**_turret_settings())
        with pytest.raises(ValueError):
            session.assign_turret_objective(position, '10x Oly')

    def test_a_bad_id_raises_and_leaves_the_slot(self, sessions):
        session = sessions(**_turret_settings())
        before = dict(session.settings['turret_objectives'])
        with pytest.raises(ConfigError):
            session.assign_turret_objective(2, 'banana')
        assert dict(session.settings['turret_objectives']) == before

    def test_clear_empties_the_slot(self, sessions):
        session = sessions(**_turret_settings())
        session.clear_turret_objective(1)
        assert session.settings['turret_objectives'][1] is None
        assert session.scope.runtime_state.get_turret_config()[1] is None


class TestT10TurretPosition:
    def test_setting_a_position_onto_an_empty_slot_warns(self, sessions):
        session = sessions(**_turret_settings())
        _clear_log()
        session.set_turret_position(3)
        assert session.settings['turret_position'] == 3
        assert any('no objective assigned' in line for line in _lines()), _lines()

    def test_recording_the_current_position_again_is_silent(self, sessions):
        session = sessions(**_turret_settings())
        session.set_turret_position(3)
        _clear_log()
        session.set_turret_position(3)
        assert session.settings['turret_position'] == 3
        assert not any('no objective assigned' in line for line in _lines()), _lines()

    def test_a_string_position_raises_type_error(self, sessions):
        session = sessions(**_turret_settings())
        with pytest.raises(TypeError):
            session.set_turret_position('3')

    def test_a_slot_key_outside_the_turret_is_recorded_and_silent(self, sessions):
        # No layer refuses a slot key outside 1-4 from a hand-edited file,
        # and a position that reaches this member through the objective
        # lookup names a slot that holds an objective; the member records
        # what the motion landed on rather than turning it into an error.
        session = sessions(
            **_turret_settings(
                turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None, '9': '10x Oly'}
            )
        )
        _clear_log()
        session.set_turret_position(9)
        assert session.settings['turret_position'] == 9
        assert not any('no objective assigned' in line for line in _lines()), _lines()

    def test_start_application_session_writes_the_position_through_the_member(
        self, sessions, monkeypatch
    ):
        session = sessions(**_turret_settings())
        seen = []
        monkeypatch.setattr(
            type(session), 'set_turret_position', lambda self, position: seen.append(position)
        )
        session.start_application_session(
            disable_homing=False,
            home_fn=lambda axis: True,
            turret_fn=lambda position: None,
        )
        assert seen == [1]


# ---------------------------------------------------------------------------
# T14 -- the resolved-optics record is the Session's, single-homed
# ---------------------------------------------------------------------------


class TestT14OpticsRecord:
    def test_the_record_fires_once_at_bring_up_with_the_scope_s_binning(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            ScopeSession,
            '_log_resolved_optics',
            lambda self, objective_id, focal_length: seen.append((objective_id, focal_length)),
        )
        session = ScopeSession.create_headless(settings=complete_settings(**_turret_settings()))
        try:
            assert seen == [('4x Oly', session.get_objective_info('4x Oly')['focal_length'])]
        finally:
            session.shutdown()

    def test_the_bring_up_record_carries_a_real_scale(self):
        session = ScopeSession.create_headless(settings=complete_settings(**_turret_settings()))
        try:
            optics = [line for line in _lines() if line.startswith('[Optics')]
            assert len(optics) == 1, _lines()
            assert 'um/px' in optics[0] and 'no image scale' not in optics[0], optics[0]
            assert f'binning={session.scope.imaging.get_binning_size()}' in optics[0]
        finally:
            session.shutdown()

    def test_the_gui_no_longer_records_the_optics(self):
        # The record fires from the Session's writers, so a GUI call site
        # would be a second home for the same fact.
        modules = list(iter_package_modules(('ui',)))
        modules.append(('lumaviewpro.py', parse_module('lumaviewpro.py')))
        offenders = [
            f'{rel_path}:{node.lineno}'
            for rel_path, tree in modules
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
            and (getattr(node, 'id', None) or getattr(node, 'attr', None)) == 'log_resolved_optics'
        ]
        assert offenders == [], offenders

    def test_the_gui_getters_no_longer_define_the_record(self):
        import modules.config_ui_getters as config_ui_getters

        assert not hasattr(config_ui_getters, 'log_resolved_optics')


class TestPopulatedSlotsSurviveTheLoad:
    """A behaviour-preservation guard for the settings pipeline's ordering.

    ``prepare_settings`` merges the shipped template into the user's file
    BEFORE it normalizes the slot keys to ints. Normalizing first would
    let the merge add the template's string-keyed empty slots beside the
    user's int keys, and the later normalization would let the empties
    win -- every assignment wiped on load. The template's slots must
    DIFFER from the user's for the case to be able to fail, so the
    shipped template is the second input, never a copy of the first.
    """

    def test_a_populated_current_json_keeps_every_assignment(self, tmp_path):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = tmp_path / 'data'
        data.mkdir()
        shutil.copy(os.path.join(repo, 'data', 'settings.json'), data / 'settings.json')
        template = settings_init.read_settings_json(str(data / 'settings.json'))
        assert all(value is None for value in template['turret_objectives'].values())
        user = dict(template)
        user['turret_objectives'] = {'1': '4x Oly', '2': '10x Oly', '3': None, '4': '20x Oly'}
        (data / 'current.json').write_text(json.dumps(user))

        settings, rejected = settings_init.prepare_settings(
            logging.getLogger('test'), str(tmp_path), fall_back_to_template=False
        )

        assert rejected is None
        assert settings['turret_objectives'] == {1: '4x Oly', 2: '10x Oly', 3: None, 4: '20x Oly'}
