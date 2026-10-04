# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Going to a protocol step is one Session member, driven on the protocol's plate.

The GUI's step navigation composed the move itself: it looked the turret
slot up, then moved X and Y through the motion API's plate frame, which
converts against the plate the SESSION has selected. A protocol written for
one plate, navigated after the operator picked another, was driven to the
wrong place; a run of the same protocol went to the right one, because the
run engine converted against the protocol's plate with its own copy of the
conversion. A script had neither.

``ScopeSession.go_to_step`` goes to a step as a click does, through
``ProtocolsAPI.step_targets``, the one conversion the run engine now calls
too, as one task on the IO lane that asks about every axis once before
anything moves.
"""

from __future__ import annotations

import ast
import copy
import datetime
import logging

import pandas as pd
import pytest

from modules.exceptions import AxisStateUnknownError, ConfigError, ProtocolRunRefusedError
from modules.protocol import Protocol, StepNotFoundError
from modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT, parse_module
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

_PLATE = '96 well microplate'
_OTHER_PLATE = 'Four-Slide Holder'
_MOVE_WAIT_S = 30.0


def _step(x: float, y: float, z: float, objective: str) -> dict:
    return {
        'Name': 'step',
        'X': x,
        'Y': y,
        'Z': z,
        'Auto_Focus': False,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': objective,
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
        'Auto_Named': False,
        'Label': '',
    }


def _protocol(labware: str, *steps: dict) -> Protocol:
    return Protocol(
        tiling_configs_file_loc=REPO_ROOT / 'data' / 'tiling.json',
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(list(steps)),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': labware,
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _objective_of(built: ScopeSession) -> str:
    return built.scope.runtime_state.get_current_objective_id()


def _settled_position(built: ScopeSession) -> dict:
    assert built.scope.motion.wait_until_finished_moving(timeout_s=_MOVE_WAIT_S)
    return {axis: built.scope.motion.get_current_position(axis) for axis in ('X', 'Y', 'Z')}


def _spy_moves(monkeypatch, built: ScopeSession) -> list:
    """Every move the Session commands, recorded before the motion API sees it."""
    moves = []
    motion = built.scope.motion
    real_absolute, real_turret = motion.move_absolute, motion.move_turret

    def absolute(axis, position, *args, **kwargs):
        moves.append((axis, position))
        return real_absolute(axis, position, *args, **kwargs)

    def turret(position, *args, **kwargs):
        moves.append(('T', position))
        return real_turret(position, *args, **kwargs)

    monkeypatch.setattr(motion, 'move_absolute', absolute)
    monkeypatch.setattr(motion, 'move_turret', turret)
    return moves


def _session_for(microscope: str, *, homed: bool) -> ScopeSession:
    built = ScopeSession.create(complete_settings(microscope=microscope), simulate=True)
    if homed:
        home_sim_scope(built.scope)
    return built


def _close(built: ScopeSession) -> None:
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def unhomed_session():
    built = _session_for('LS850', homed=False)
    yield built
    _close(built)


@pytest.fixture
def no_motor_session():
    built = _session_for('LS560', homed=False)
    assert not built.scope.motor_connected
    yield built
    _close(built)


@pytest.fixture
def turret_session():
    settings = complete_settings(microscope='LS850T')
    settings['turret_objectives'] = dict(TEST_TURRET_OBJECTIVES)
    built = ScopeSession.create(settings, simulate=True)
    home_sim_scope(built.scope)
    assert built.scope.capabilities.has_turret
    yield built
    _close(built)


class TestTheMove:
    def test_the_step_is_driven_on_the_protocols_plate_not_the_selected_one(self, session):
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))
        assert session.select_labware(_OTHER_PLATE)

        session.go_to_step(protocol, 0)

        arrived = _settled_position(session)
        on_protocol_plate = session.scope.protocols.plate_to_stage(protocol, 60.0, 40.0)
        assert (arrived['X'], arrived['Y']) == pytest.approx(on_protocol_plate, abs=1.0)
        assert arrived['Z'] == pytest.approx(5000.0, abs=1.0)
        # The two plates differ, so a move on the selected one would have
        # landed elsewhere: the assertion above is not vacuous.
        on_selected_plate = session.scope.protocols.plate_to_stage(
            _protocol(_OTHER_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session))), 60.0, 40.0
        )
        assert abs(on_selected_plate[0] - on_protocol_plate[0]) > 100.0

    def test_the_turret_turns_to_the_slot_carrying_the_steps_objective(self, turret_session):
        slots = {obj: slot for slot, obj in TEST_TURRET_OBJECTIVES.items() if obj}
        objective = '20x Oly'
        protocol = _protocol('6 well microplate', _step(30.0, 20.0, 4000.0, objective))

        turret_session.go_to_step(protocol, 0)

        _settled_position(turret_session)
        assert turret_session.scope.motion.get_turret_slot() == slots[objective]
        targets = turret_session.scope.protocols.step_targets(protocol, 0)
        assert targets.turret_slot == slots[objective]

    def test_a_scope_with_no_motor_goes_to_the_step_without_moving(
        self, no_motor_session, monkeypatch
    ):
        moves = _spy_moves(monkeypatch, no_motor_session)
        protocol = _protocol(
            'Center Plate', _step(10.0, 10.0, 0.0, _objective_of(no_motor_session))
        )

        no_motor_session.go_to_step(protocol, 0)

        assert moves == []


class TestTheRefusals:
    def test_a_step_the_protocol_lacks_is_refused_before_anything_moves(self, session, monkeypatch):
        moves = _spy_moves(monkeypatch, session)
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))

        with pytest.raises(StepNotFoundError):
            session.go_to_step(protocol, 1)
        with pytest.raises(StepNotFoundError):
            session.go_to_step(protocol, -1)

        assert moves == []

    def test_an_axis_that_does_not_know_its_position_refuses_the_whole_step(
        self, unhomed_session, monkeypatch
    ):
        moves = _spy_moves(monkeypatch, unhomed_session)
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(unhomed_session)))

        with pytest.raises(AxisStateUnknownError):
            unhomed_session.go_to_step(protocol, 0)

        assert moves == []

    def test_glass_the_turret_does_not_carry_is_refused_before_anything_moves(
        self, turret_session, monkeypatch
    ):
        moves = _spy_moves(monkeypatch, turret_session)
        assert '40x Oly' not in TEST_TURRET_OBJECTIVES.values()
        protocol = _protocol('6 well microplate', _step(30.0, 20.0, 4000.0, '40x Oly'))

        with pytest.raises(ProtocolRunRefusedError):
            turret_session.go_to_step(protocol, 0)

        assert moves == []


class TestOneTargetComputation:
    """The run engine and the Session both take a step's targets from ``step_targets``.

    A conversion kept in either would be the second implementation this
    file retired, free to drift from the other again.
    """

    _CALLERS = ('modules/protocol_step_runner.py', 'modules/scope_session.py')
    _OWN = frozenset({'plate_to_stage', 'get_turret_position_for_objective_id'})

    @staticmethod
    def _called_attrs(tree) -> set[str]:
        return {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

    @pytest.mark.parametrize('caller', _CALLERS)
    def test_a_step_mover_takes_its_targets_from_the_protocols_api(self, caller):
        called = self._called_attrs(parse_module(caller))
        assert 'step_targets' in called, f'{caller} does not call step_targets'
        assert 'get_turret_position_for_objective_id' not in called, (
            f'{caller} looks a turret slot up itself'
        )

    def test_the_session_converts_no_plate_coordinate_itself(self):
        called = self._called_attrs(parse_module('modules/scope_session.py'))
        assert 'plate_to_stage' not in called

    def test_the_gui_navigation_moves_no_axis_itself(self):
        called = self._called_attrs(parse_module('ui/step_navigation.py'))
        assert not called & {'move_absolute', 'move_turret', *self._OWN}, (
            'ui/step_navigation.py composes a step move itself: ' + ', '.join(sorted(called))
        )


class TestTheStepIntoItsLayer:
    def test_the_layer_takes_the_steps_values_and_acquires_as_the_step_does(self, session):
        step = _step(60.0, 40.0, 5000.0, _objective_of(session))
        step.update({'Gain': 7.0, 'Exposure': 33.0, 'Illumination': 120.0, 'Acquire': 'video'})
        protocol = _protocol(_PLATE, step)
        session.settings['BF']['acquire'] = None
        session.settings['BF']['stim_config'] = {'enabled': True}

        session.go_to_step(protocol, 0)

        layer = session.settings['BF']
        assert layer['gain_db'] == 7.0
        assert layer['exposure_ms'] == 33.0
        assert layer['illumination_ma'] == 120.0
        assert layer['focus'] == 5000.0
        assert layer['acquire'] == 'video'
        assert layer['video_config'] == step['Video Config']
        assert layer['video_config'] is not step['Video Config'], 'the protocol owns its dicts'
        # A layer set to acquire stops stimulating: set_layer_acquire's rule.
        assert layer['stim_config']['enabled'] is False

    def test_a_scope_with_no_motor_still_loads_the_step_into_its_layer(self, no_motor_session):
        step = _step(10.0, 10.0, 0.0, _objective_of(no_motor_session))
        step['Gain'] = 9.0
        protocol = _protocol('Center Plate', step)

        no_motor_session.go_to_step(protocol, 0)

        assert no_motor_session.settings['BF']['gain_db'] == 9.0

    def test_a_refused_navigation_writes_nothing(self, unhomed_session):
        step = _step(60.0, 40.0, 5000.0, _objective_of(unhomed_session))
        step['Gain'] = 9.0
        protocol = _protocol(_PLATE, step)
        before = copy.deepcopy(unhomed_session.settings['BF'])

        with pytest.raises(AxisStateUnknownError):
            unhomed_session.go_to_step(protocol, 0)

        assert unhomed_session.settings['BF'] == before

    def test_a_stimulation_naming_no_layer_is_refused_before_anything_changes(
        self, session, monkeypatch
    ):
        moves = _spy_moves(monkeypatch, session)
        step = _step(60.0, 40.0, 5000.0, _objective_of(session))
        step['Stim_Config'] = {'Purple': {'enabled': True}}
        protocol = _protocol(_PLATE, step)
        before = copy.deepcopy(session.settings['BF'])

        with pytest.raises(ConfigError):
            session.go_to_step(protocol, 0)

        assert moves == []
        assert session.settings['BF'] == before
        assert 'Purple' not in session.settings


class TestTheLedPreview:
    @staticmethod
    def _lit(built: ScopeSession) -> set[str]:
        illumination = built.scope.illumination
        return {
            layer
            for layer in ('BF', 'Blue', 'Green', 'Red')
            if illumination.get_led_state(channel=layer)['enabled']
        }

    def test_going_to_a_step_lights_its_channel_when_the_preview_is_on(self, session):
        session.settings['protocol_led_on'] = True
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))

        session.go_to_step(protocol, 0)

        assert self._lit(session) == {'BF'}

    def test_going_to_a_step_darkens_everything_when_the_preview_is_off(self, session):
        session.settings['protocol_led_on'] = False
        session.scope.illumination.led_on('Blue', 50.0)
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))

        session.go_to_step(protocol, 0)

        assert self._lit(session) == set()

    def test_re_selecting_the_same_step_leaves_a_user_lit_channel_lit(self, session):
        session.settings['protocol_led_on'] = True
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))
        session.go_to_step(protocol, 0)
        session.scope.illumination.led_on('Blue', 50.0)
        session.scope.illumination.led_off('BF')

        session.go_to_step(protocol, 0)

        assert self._lit(session) == {'Blue'}, 'a re-click re-drove the preview'

    def test_a_different_step_previews_again(self, session):
        session.settings['protocol_led_on'] = True
        objective = _objective_of(session)
        protocol = _protocol(
            _PLATE, _step(60.0, 40.0, 5000.0, objective), _step(61.0, 40.0, 5000.0, objective)
        )
        session.go_to_step(protocol, 0)
        session.scope.illumination.led_off('BF')

        session.go_to_step(protocol, 1)

        assert self._lit(session) == {'BF'}

    def test_a_run_transition_forgets_the_last_step(self, session):
        session.settings['protocol_led_on'] = True
        protocol = _protocol(_PLATE, _step(60.0, 40.0, 5000.0, _objective_of(session)))
        session.go_to_step(protocol, 0)
        session.scope.illumination.led_off('BF')
        session.notify_run_state()

        session.go_to_step(protocol, 0)

        assert self._lit(session) == {'BF'}


# The settings key each step column lands in.
STEP_TO_SETTINGS = {
    'Auto_Focus': 'autofocus',
    'False_Color': 'false_color',
    'Illumination': 'illumination_ma',
    'Gain': 'gain_db',
    'Auto_Gain': 'auto_gain',
    'Exposure': 'exposure_ms',
    'Sum': 'sum',
    'Acquire': 'acquire',
    'Z': 'focus',
    'Video Config': 'video_config',
}


class TestEveryStepValueReachesTheLayer:
    def test_every_column_lands_and_the_dicts_are_copies(self, no_motor_session):
        step = _step(10.0, 10.0, 0.0, _objective_of(no_motor_session))
        step.update(
            {
                'Auto_Focus': True,
                'False_Color': True,
                'Illumination': 123.0,
                'Gain': 6.0,
                'Auto_Gain': True,
                'Exposure': 40.0,
                'Sum': 4,
                'Acquire': 'video',
                'Z': 1234.0,
                'Video Config': {'duration': 30},
                'Stim_Config': {'Green': {'enabled': True, 'illumination_ma': 200}},
            }
        )
        protocol = _protocol('Center Plate', step)

        no_motor_session.go_to_step(protocol, 0)

        layer = no_motor_session.settings['BF']
        for column, key in STEP_TO_SETTINGS.items():
            assert layer[key] == step[column], f"{key} did not take the step's {column}"
        assert no_motor_session.settings['Green']['stim_config'] == step['Stim_Config']['Green']
        # The step's dicts are the protocol's: a later edit to them must not
        # reach into the live settings.
        step['Video Config']['duration'] = 999
        step['Stim_Config']['Green']['illumination_ma'] = 999
        assert layer['video_config']['duration'] == 30
        assert no_motor_session.settings['Green']['stim_config']['illumination_ma'] == 200

    def test_a_scope_with_no_motor_previews_the_step(self, no_motor_session):
        no_motor_session.settings['protocol_led_on'] = True
        protocol = _protocol(
            'Center Plate', _step(10.0, 10.0, 0.0, _objective_of(no_motor_session))
        )

        no_motor_session.go_to_step(protocol, 0)

        assert no_motor_session.scope.illumination.get_led_state(channel='BF')['enabled']


class TestTheSlotLookupCannotDisagreeWithTheRule:
    def test_a_missing_slot_after_the_rule_passed_is_a_defect_not_a_move(
        self, turret_session, monkeypatch
    ):
        """The rule and the lookup read the same store, so this cannot happen.

        When it does, the two have disagreed and the honest answer is to
        stop. What this replaced logged, raised a dialog, and then moved X,
        Y and Z without the objective -- a capture through whatever glass
        happened to be in the path, named for the glass the step asked for.
        """
        moves = _spy_moves(monkeypatch, turret_session)
        monkeypatch.setattr(
            turret_session.scope.motion,
            'get_turret_position_for_objective_id',
            lambda objective_id, persisted_position=None: None,
        )
        protocol = _protocol('6 well microplate', _step(30.0, 20.0, 4000.0, '20x Oly'))

        with pytest.raises(RuntimeError) as defect:
            turret_session.go_to_step(protocol, 0)

        assert 'slot' in str(defect.value).lower()
        assert moves == []
