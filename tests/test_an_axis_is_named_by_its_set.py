# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An axis is named by its set, in what the scope answers as in what it takes.

The axis names are string enums (``modules.axes``): ``Axis`` (X, Y, Z, T)
and ``StageAxis`` (X, Y, Z), the axes with a position. What the scope
answers names its axes by those members -- the positions, the targets, the
unknown positions, the capabilities, the status, a position event -- so a
caller compares or keys by the member or its plain string alike. A member
that takes a stage axis refuses the turret as it refuses any other name:
the turret has a slot, not a position, and there is no bookmark of it.

An axis's state is an ``AxisState`` the same way: answered as the member,
and judged where a record holding one is built, so an ``AxisPosition`` or a
``TurretStatus`` a caller builds with a plain string holds the member and
one built with any other word is refused.
"""

from __future__ import annotations

import pytest

from modules.axes import Axis, StageAxis
from modules.exceptions import ArgumentRefusedError
from modules.lumascope_api import AxisPosition, AxisState, TurretStatus
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture
def session(tmp_path):
    s = _session(tmp_path, 'LS850T')
    yield s
    s.shutdown()


def test_the_answers_name_their_axes_by_the_members(session):
    motion = session.scope.motion
    seen = []
    motion.add_position_listener(lambda axis, at: seen.append(axis))
    motion.move_z(motion.get_target_positions()['Z'] + 1.0)
    motion.wait_until_finished_moving()

    answered = {
        'axis_positions': list(motion.axis_positions()),
        'get_target_positions': list(motion.get_target_positions()),
        'Status.axes': list(session.status.axes),
        'position event': seen[:1],
    }
    assert {name: [(type(a), a) for a in axes] for name, axes in answered.items()} == {
        'axis_positions': [(StageAxis, a) for a in 'XYZ'],
        'get_target_positions': [(StageAxis, a) for a in 'XYZ'],
        'Status.axes': [(StageAxis, a) for a in 'XYZ'],
        'position event': [(StageAxis, 'Z')],
    }
    assert [(type(a), a) for a in session.scope.capabilities.axes] == [(Axis, a) for a in 'XYZT']
    assert motion.axis_positions()['Z'] == motion.axis_positions()[StageAxis.Z]


def test_an_axis_without_a_position_is_answered_by_its_member(tmp_path):
    session = _session(tmp_path, 'LS850T', homed=False)
    try:
        unknown = session.scope.motion.axes_without_position()
        assert [(type(a), a) for a in unknown] == [(Axis, a) for a in 'XYZT']
    finally:
        session.shutdown()


@pytest.mark.parametrize('member', ['save_bookmark', 'start_go_to_bookmark'])
def test_a_bookmark_of_the_turret_is_refused_and_nothing_is_written(session, member):
    before = dict(session.settings['bookmark'])

    with pytest.raises(ArgumentRefusedError) as refused:
        getattr(session, member)(['T'])

    assert (refused.value.reason, refused.value.argument, refused.value.offered) == (
        'not_offered',
        'axes[0]',
        ('X', 'Y', 'Z'),
    )
    assert session.settings['bookmark'] == before


def test_an_axis_state_is_answered_as_the_member(session):
    motion = session.scope.motion

    answered = [
        motion.get_axis_state('Z'),
        motion.axis_positions()['Z'].state,
        motion.turret_status().state,
        session.status.turret.state,
    ]

    assert [(type(state), state) for state in answered] == [(AxisState, 'idle')] * 4


def test_a_position_record_built_with_a_plain_state_holds_the_member():
    built = [AxisPosition('idle', 1.0).state, TurretStatus('homing', None).state]
    assert [(type(state), state) for state in built] == [
        (AxisState, 'idle'),
        (AxisState, 'homing'),
    ]


def test_a_position_record_built_with_a_word_that_is_no_state_is_refused():
    states = ('unknown', 'idle', 'moving', 'homing')
    for build, argument in (
        (lambda: AxisPosition('parked', 1.0), 'AxisPosition.state'),
        (lambda: TurretStatus('parked', 2), 'TurretStatus.state'),
    ):
        with pytest.raises(ArgumentRefusedError) as refused:
            build()
        assert (refused.value.reason, refused.value.argument, refused.value.offered) == (
            'not_offered',
            argument,
            states,
        )
