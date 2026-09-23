# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A jog's step comes from the active objective, and an unknown one is refused.

The jog buttons used to read the objective themselves and, when it could not
be read, log a warning and do nothing -- so on an un-homed turret or an
unassigned slot the stage and focus silently would not move. The step is
now the API's answer, and an unknown objective is refused there with the
reason, which the buttons show.
"""

import ast

import pytest

from modules.exceptions import ObjectiveUnknownError
from modules.scope_session import ScopeSession
from tests.ast_seams import find_def
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def turret_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
        ),
        simulate=True,
    )
    yield session
    session.shutdown()


def test_the_step_is_the_active_objective_s(turret_session):
    motion = turret_session.scope.motion
    home_sim_scope(turret_session.scope)
    motion.move_turret(1)
    info = turret_session.get_objective_info('4x Oly')

    assert motion.jog_step('Z', coarse=True) == info['z_coarse']
    assert motion.jog_step('Z', coarse=False) == info['z_fine']
    assert motion.jog_step('X', coarse=True) == info['xy_coarse']
    assert motion.jog_step('Y', coarse=False) == info['xy_fine']


@pytest.mark.parametrize('slot', [None, 2], ids=['slot_unknown', 'slot_unassigned'])
def test_an_unknown_objective_is_refused_with_its_reason(turret_session, slot):
    motion = turret_session.scope.motion
    if slot is not None:
        home_sim_scope(turret_session.scope)
        motion.move_turret(slot)

    with pytest.raises(ObjectiveUnknownError) as excinfo:
        motion.jog_step('Z', coarse=True)

    assert excinfo.value.reason == ('slot_unknown' if slot is None else 'slot_unassigned')


def test_an_axis_without_a_jog_is_refused(turret_session):
    with pytest.raises(ValueError, match="'T'"):
        turret_session.scope.motion.jog_step('T', coarse=True)


@pytest.mark.parametrize(
    'rel_path, class_name, method',
    [
        ('ui/vertical_control.py', 'VerticalControl', '_z_jog'),
        ('ui/motion_settings.py', 'XYStageControl', '_xy_jog'),
    ],
)
def test_the_buttons_ask_the_api_and_show_its_refusal(rel_path, class_name, method):
    fn = find_def(rel_path, method, class_name=class_name)
    assert fn is not None
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    }
    assert 'jog_step' in calls
    assert 'show_jog_refusal' in calls
    assert 'get_current_objective_info' not in calls
