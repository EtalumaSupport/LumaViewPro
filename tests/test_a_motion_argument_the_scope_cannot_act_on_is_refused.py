# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motion or diagnostics argument no scope can act on is refused as the request's.

A name that was no axis met seven answers: five members raised a bare
``ValueError``, a home raised its own, ``get_axis_state`` answered
``'unknown'``, the position reads answered None, ``position_is_known``
answered False, ``get_axis_limits`` raised whatever the driver raised, and
the drive-status read passed it to the driver. A home took ``'all'``, and a
fan duty of 101 reached the board. Each is now one check, asked first,
refused with ``ArgumentRefusedError`` whose cause is the request's, so a
client is told to change it rather than to retry. An axis the model lacks is
still a name, and keeps the answer a missing axis gives. A value that is not
a name or a number at all -- None, ``True``, NaN for a duty -- is the
``@api`` door's (``tests/test_an_argument_of_another_type_is_refused_at_the_door.py``).
"""

import math

import pytest

from modules.exceptions import ArgumentRefusedError, RefusalCause
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session

EVERY_AXIS = ('X', 'Y', 'Z', 'T')

AXIS_MEMBERS = {
    'position_is_known': lambda s, a: s.motion.position_is_known(a),
    'get_axis_state': lambda s, a: s.motion.get_axis_state(a),
    'get_actual_position': lambda s, a: s.motion.get_actual_position(a),
    'get_target_position': lambda s, a: s.motion.get_target_position(a),
    'get_current_position': lambda s, a: s.motion.get_current_position(a),
    'get_axis_limits': lambda s, a: s.motion.get_axis_limits(a),
    'set_precision_mode': lambda s, a: s.motion.set_precision_mode(a, True),
    'get_target_status': lambda s, a: s.motion.get_target_status(a),
    'get_limit_switch_status': lambda s, a: s.motion.get_limit_switch_status(a),
    'move_absolute': lambda s, a: s.motion.move_absolute(a, 0),
    'move_relative': lambda s, a: s.motion.move_relative(a, 0),
    'start_move_absolute': lambda s, a: s.motion.start_move_absolute(a, 0),
    'start_move_relative': lambda s, a: s.motion.start_move_relative(a, 0),
    'refuse_unknown_positions': lambda s, a: s.motion.refuse_unknown_positions(
        [a], recording=True, then='move it'
    ),
    'read_motor_drv_status': lambda s, a: s.diagnostics.read_motor_drv_status(a),
}


@pytest.fixture(scope='module')
def scope(tmp_path_factory):
    session = _session(tmp_path_factory.mktemp('c4'), 'LS850T')
    yield session.scope
    session.shutdown()


def _refused(call, reason):
    with pytest.raises(ArgumentRefusedError) as refused:
        call()
    assert refused.value.reason == reason
    assert refused.value.cause == RefusalCause.REQUEST
    return refused.value


def _motion_state(scope):
    motion = scope.motion
    states = {axis: motion.get_axis_state(axis) for axis in EVERY_AXIS}
    return states, motion.get_target_position(), motion.get_turret_slot()


NO_AXIS = [(member, name) for member in AXIS_MEMBERS for name in ('Q', 'x')]


@pytest.mark.parametrize('member, name', NO_AXIS)
def test_a_name_that_is_no_axis_is_refused_and_nothing_moves(scope, member, name):
    before = _motion_state(scope)

    refusal = _refused(lambda: AXIS_MEMBERS[member](scope, name), 'axis_unknown')

    assert (refusal.argument, refusal.value, refusal.offered) == ('axis', name, EVERY_AXIS)
    assert _motion_state(scope) == before


@pytest.mark.parametrize('name', ['X', 'all', 'z'])
@pytest.mark.parametrize('member', ['home', 'start_home'])
def test_a_home_takes_z_the_turret_or_all_in_capitals(scope, member, name):
    refusal = _refused(lambda: getattr(scope.motion, member)(name), 'axis_unknown')
    assert refusal.offered == ('Z', 'T', 'ALL')


@pytest.mark.parametrize(
    'member', ['move_absolute', 'start_move_relative', 'set_precision_mode', 'home']
)
def test_a_name_that_is_no_axis_is_refused_whatever_the_scope_is_doing(tmp_path, member):
    """Refused at the door, so a shut scope does not answer for the name."""
    session = _session(tmp_path, 'LS850')
    session.shutdown()
    calls = {
        'move_absolute': lambda m: m.move_absolute('Q', 0),
        'start_move_relative': lambda m: m.start_move_relative('Q', 0),
        'set_precision_mode': lambda m: m.set_precision_mode('Q', True),
        'home': lambda m: m.home('Q'),
    }
    _refused(lambda: calls[member](session.scope.motion), 'axis_unknown')


def test_an_axis_the_model_lacks_is_a_name_and_keeps_its_answers(tmp_path):
    session = _session(tmp_path, 'LS850')
    try:
        assert session.scope.motion.get_axis_state('T') == 'unknown'
        assert session.scope.motion.position_is_known('T') is False
        assert session.scope.diagnostics.read_motor_drv_status('T') == 0
    finally:
        session.shutdown()


def test_a_frame_is_refused_before_the_hardware_is_asked(tmp_path):
    session = _session(tmp_path, 'LS820')
    try:
        refusal = _refused(
            lambda: session.scope.motion.move_absolute('X', 0, frame='bogus'), 'frame_unknown'
        )
        assert refusal.offered == ('stage', 'plate')
    finally:
        session.shutdown()


def test_a_plate_target_for_z_is_refused_by_the_plate_owner(scope):
    refusal = _refused(
        lambda: scope.motion.move_absolute('Z', 0, frame='plate'), 'plate_frame_axis'
    )
    assert (refusal.value, refusal.offered) == ('Z', ('X', 'Y'))


@pytest.mark.parametrize(
    'duty, reason',
    [
        (101, 'fan_duty_out_of_range'),
        (-1, 'fan_duty_out_of_range'),
    ],
)
def test_a_fan_duty_off_the_percent_scale_is_refused_and_nothing_is_written(
    scope, monkeypatch, duty, reason
):
    written = []
    monkeypatch.setattr(scope._motion_driver, 'set_fan_duty', written.append)

    refusal = _refused(lambda: scope.diagnostics.set_motor_fan_duty(duty), reason)

    assert refusal.argument == 'duty_pct'
    assert written == []


def test_a_fan_duty_on_the_scale_is_written(scope, monkeypatch):
    written = []
    monkeypatch.setattr(scope._motion_driver, 'set_fan_duty', written.append)
    scope.diagnostics.set_motor_fan_duty(0)
    scope.diagnostics.set_motor_fan_duty(100)
    assert written == [0, 100]


def test_a_reason_takes_offered_values_exactly_when_its_words_name_them():
    with pytest.raises(TypeError):
        ArgumentRefusedError('axis_unknown', argument='axis', value='Q')
    with pytest.raises(TypeError):
        ArgumentRefusedError('not_a_number', argument='gain_db', value=math.nan, offered=('X',))
