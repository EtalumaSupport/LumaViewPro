# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A jog is the API's: its step comes from the active objective, asked only once the scope may move.

The jog buttons used to read the objective themselves and, when it could not
be read, log a warning and do nothing -- so on an un-homed turret or an
unassigned slot the stage and focus silently would not move. Later they
asked the API for the step before the move, so after a failed home the
objective's refusal ("home the turret or move it to a slot") answered
first, naming two remedies the motion gate refuses. ``start_jog`` takes the
gesture whole and refuses in the order every move does: the lane's, the
motion gate's, then the objective's.
"""

import ast

import pytest

from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    AxisStateUnknownError,
    ControlSessionRefusedError,
    HardwareCommandRefusedError,
    HomingFailedError,
    ObjectiveUnknownError,
)
from _lumascope.modules.scope_session import ScopeSession
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


@pytest.fixture
def failed_home_session(tmp_path):
    """An LS850T on the real board firmware whose home failed: X stalled, every axis UNKNOWN."""
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            simulator_tier='firmware',
            microscope='LS850T',
            turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
        ),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        session.scope.simulation.fail_home('X', 'stall')
        with pytest.raises(HomingFailedError):
            session.scope.motion._home_impl()
        yield session
    finally:
        session.shutdown()


def _homed_at(session, slot):
    home_sim_scope(session.scope)
    session.scope.motion.move_turret(slot)
    return session.scope.motion


def test_the_step_is_the_active_objective_s(turret_session):
    motion = _homed_at(turret_session, 1)
    objective = turret_session.scope.objectives['4x Oly']

    assert motion.jog_step_um('Z', coarse=True) == objective.jog_z_coarse_um
    assert motion.jog_step_um('Z', coarse=False) == objective.jog_z_fine_um
    assert motion.jog_step_um('X', coarse=True) == objective.jog_xy_coarse_um
    assert motion.jog_step_um('Y', coarse=False) == objective.jog_xy_fine_um


def test_a_jog_moves_its_steps_of_the_objective_s_step(turret_session):
    motion = _homed_at(turret_session, 1)
    objective = turret_session.scope.objectives['4x Oly']
    z = motion.get_target_positions()['Z']

    motion.start_jog('Z', 1, coarse=False).result()
    motion.start_jog('Z', 1.5, coarse=False).result()

    # The board holds Z in microsteps, so the target is the distance to within one.
    assert motion.get_target_positions()['Z'] == pytest.approx(
        z + 2.5 * objective.jog_z_fine_um, abs=0.01
    )


def test_a_jog_after_a_failed_home_is_refused_by_the_motion_gate_not_the_objective(
    failed_home_session,
):
    """The turret has no slot, so the objective is unknown too; the gate answers first."""
    motion = failed_home_session.scope.motion
    before = motion.get_target_positions()

    for axis, steps, coarse in (('Z', 1, False), ('X', -1, True)):
        with pytest.raises(AxisStateUnknownError) as refused:
            motion.start_jog(axis, steps, coarse)
        assert refused.value.reason == 'position_unknown'
        assert set(refused.value.axes_without_position) == {'X', 'Y', 'Z', 'T'}

    assert motion.get_target_positions() == before


def test_an_unknown_objective_on_a_scope_that_may_move_is_refused_with_its_reason(
    turret_session,
):
    motion = _homed_at(turret_session, 2)

    with pytest.raises(ObjectiveUnknownError) as refused:
        motion.start_jog('Z', 1, coarse=True)

    assert refused.value.reason == 'slot_unassigned'


def test_a_jog_while_another_session_holds_control_is_refused_for_the_session(turret_session):
    """Before the objective: slot 2 has no objective, and the session's refusal answers."""
    _homed_at(turret_session, 2)
    session_id = turret_session.take_control('plate reader')
    try:
        with pytest.raises(ControlSessionRefusedError) as refused:
            turret_session.scope.motion.start_jog('Z', 1, coarse=True)
        assert refused.value.reason == 'control_session_held'
    finally:
        turret_session.end_control(session_id)


def test_a_jog_while_an_activity_holds_the_scope_is_refused_for_the_activity(turret_session):
    _homed_at(turret_session, 2)
    diagnostic = turret_session._activity_claim.try_claim('diagnostic')
    assert diagnostic
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            turret_session.scope.motion.start_jog('Z', 1, coarse=True)
        assert refused.value.reason == 'exclusive_activity_running'
    finally:
        diagnostic.release()


def test_an_axis_without_a_jog_is_refused(turret_session):
    with pytest.raises(ArgumentRefusedError) as refused:
        turret_session.scope.motion.start_jog('T', 1, coarse=True)
    assert (refused.value.reason, refused.value.offered) == ('not_offered', ('X', 'Y', 'Z'))


def test_steps_that_are_not_a_finite_number_are_refused_and_nothing_moves(turret_session):
    motion = _homed_at(turret_session, 1)
    before = motion.get_target_positions()

    for steps in (float('nan'), float('inf')):
        with pytest.raises(ArgumentRefusedError) as refused:
            motion.start_jog('Z', steps, coarse=False)
        assert refused.value.reason == 'not_a_number'

    assert motion.get_target_positions() == before


def test_every_gui_jog_hands_the_gesture_to_the_api_and_sizes_nothing():
    """The buttons and scroll-to-focus call ``jog``; none reads the step, so none orders the refusals."""
    callers = (
        ('ui/vertical_control.py', 'VerticalControl', '_z_jog'),
        ('ui/motion_settings.py', 'XYStageControl', '_xy_jog'),
        ('ui/shader.py', 'ShaderViewer', '_flush_scroll_z'),
    )
    wrong = []
    for rel_path, class_name, method in callers:
        fn = find_def(rel_path, method, class_name=class_name)
        assert fn is not None, (rel_path, method)
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
        }
        if 'jog' not in calls or 'jog_step_um' in calls:
            wrong.append(f'{rel_path}::{method}')
    assert wrong == []
