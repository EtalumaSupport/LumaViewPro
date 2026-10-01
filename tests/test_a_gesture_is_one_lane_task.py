# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A person's several-axis gesture is one task on the IO lane.

Going to a step, clicking the stage map and clicking to centre each asked
the motion API on the GUI thread whether the scope knew where its axes
were, then submitted each axis as its own lane task. A home or a stop
landing in between was then refused axis by axis, one popup per axis,
after the gesture had moved on. ``submit_gesture`` asks and moves in one
task, on a real lane here, and runs the gesture's GUI work only when the
moves went through.
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
from modules.exceptions import AxisStateUnknownError, PositionOutOfRangeError
from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from tests.shown_outcomes import capture_shown
from ui import ui_helpers

_WAIT_S = 2.0


@pytest.fixture
def lane():
    ex = SequentialIOExecutor(name='TEST_IO')
    ex.start()
    yield ex
    ex.shutdown(wait=False)


@pytest.fixture
def env(monkeypatch, lane):
    from modules import notification_center, sequential_io_executor

    shown = capture_shown(monkeypatch)
    monkeypatch.setattr(sequential_io_executor, 'notifications', notification_center.notifications)
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    monkeypatch.setattr(ui_helpers, '_user_motion_locked', lambda label: False)

    record: list = []
    motion = types.SimpleNamespace(
        refuse_unknown_positions=MagicMock(
            side_effect=lambda axes, **kw: record.append(('ask', tuple(axes), kw))
        ),
    )
    vertical_control = types.SimpleNamespace(
        update_gui=MagicMock(side_effect=lambda: record.append(('draw', 'Z'))),
        show_turret_state=MagicMock(side_effect=lambda: record.append(('draw', 'T'))),
    )
    motion_settings = types.SimpleNamespace(
        ids={'verticalcontrol_id': vertical_control},
        update_xy_stage_control_gui=MagicMock(side_effect=lambda: record.append(('draw', 'XY'))),
    )
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        types.SimpleNamespace(
            scope=types.SimpleNamespace(motion=motion),
            io_executor=lane,
            motion_settings=motion_settings,
        ),
    )
    return types.SimpleNamespace(
        lane=lane,
        motion=motion,
        record=record,
        shown=shown,
        vertical_control=vertical_control,
    )


def _settle(lane):
    # Everything the lane owes a submit has been answered once a task queued
    # after it has run.
    assert lane.call(IOTask(action=lambda: None), 'settle', _WAIT_S) is None


def _gesture(env, moves, *, axes=('X', 'Y', 'Z', 'T')):
    ui_helpers.submit_gesture(
        'GESTURE',
        axes=axes,
        then='go to the step',
        moves=moves,
        on_moved=lambda: env.record.append(('moved',)),
    )
    _settle(env.lane)


def test_asks_once_then_moves_on_the_lane_then_redraws_then_its_gui_work(env):
    where = []

    def moves():
        where.append(threading.current_thread().name)
        env.record.append(('move',))

    _gesture(env, moves)

    assert where == ['TEST_IO_WORKER']
    assert env.record == [
        ('ask', ('X', 'Y', 'Z', 'T'), {'recording': False, 'then': 'go to the step'}),
        ('move',),
        ('draw', 'XY'),
        ('draw', 'Z'),
        ('draw', 'T'),
        ('moved',),
    ]
    assert env.shown == []


def test_a_refusal_moves_nothing_is_shown_once_and_still_redraws(env):
    def refuse(axes, **kw):
        raise AxisStateUnknownError({'X': 'unknown', 'Y': 'unknown'}, then='go to the step')

    env.motion.refuse_unknown_positions.side_effect = refuse

    _gesture(env, lambda: env.record.append(('move',)))

    assert ('move',) not in env.record
    assert ('moved',) not in env.record, "a refused gesture ran the gesture's GUI work"
    assert ('draw', 'XY') in env.record
    assert len(env.shown) == 1, env.shown


def test_a_move_refused_part_way_leaves_what_moved_and_skips_the_gui_work(env):
    def moves():
        env.record.append(('move', 'X'))
        raise PositionOutOfRangeError('Y', 999999.0, 0.0, 100.0)

    _gesture(env, moves)

    assert ('move', 'X') in env.record
    assert ('moved',) not in env.record
    assert ('draw', 'XY') in env.record
    assert len(env.shown) == 1, env.shown


def test_a_gesture_with_no_axes_asks_nothing_and_still_runs(env):
    _gesture(env, lambda: env.record.append(('move',)), axes=())

    env.motion.refuse_unknown_positions.assert_not_called()
    assert env.record == [('move',), ('moved',)]


def test_a_locked_control_surface_submits_nothing(env, monkeypatch):
    monkeypatch.setattr(ui_helpers, '_user_motion_locked', lambda label: True)

    _gesture(env, lambda: env.record.append(('move',)))

    assert env.record == []


def test_a_lane_that_takes_no_work_skips_the_gui_work(env):
    env.lane.shutdown(wait=False)

    ui_helpers.submit_gesture(
        'GESTURE',
        axes=('X',),
        then='move the stage',
        moves=lambda: env.record.append(('move',)),
        on_moved=lambda: env.record.append(('moved',)),
    )

    assert ('move',) not in env.record
    assert ('moved',) not in env.record
