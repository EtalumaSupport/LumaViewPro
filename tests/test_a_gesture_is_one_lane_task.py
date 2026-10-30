# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A person's several-axis gesture is one task on the IO lane.

Going to a step, clicking the stage map and clicking to centre each asked
the motion API on the GUI thread whether the scope knew where its axes
were, then submitted each axis as its own lane task. A home or a stop
landing in between was then refused axis by axis, one popup per axis,
after the gesture had moved on. ``submit_gesture`` asks and moves in one
task, on a simulated session's IO lane here, and runs the gesture's GUI
work only when the moves went through.
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import pytest

from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from ui import ui_helpers

_WAIT_S = 2.0

# A turret scope: a gesture names every axis it moves, the turret's among them.
pytestmark = pytest.mark.parametrize(
    'sim_session',
    [{'microscope': 'LS850T', 'turret_objectives': dict(TEST_TURRET_OBJECTIVES)}],
    indirect=True,
)


@pytest.fixture
def unhomed(gui_ctx, monkeypatch, centre_posts):
    """The GUI over a simulated turret session as it comes up, no axis knowing its position.

    The gesture runs on the session's own IO lane. Each ask of the motion
    API is recorded, with the reason it refused, then made; the redraws
    are the test's own widgets, recorded.
    """
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn: fn(0))
    monkeypatch.setattr(ui_helpers, '_user_motion_locked', lambda label: False)

    record: list = []
    motion = gui_ctx.scope.motion
    real_ask = motion.refuse_unknown_positions

    def ask(axes, **kw):
        record.append(('ask', tuple(axes), kw))
        try:
            real_ask(axes, **kw)
        except Exception as e:
            record.append(('refused', e.reason))
            raise

    monkeypatch.setattr(motion, 'refuse_unknown_positions', ask)
    vertical_control = types.SimpleNamespace(
        update_gui=MagicMock(side_effect=lambda: record.append(('draw', 'Z'))),
        show_turret_state=MagicMock(side_effect=lambda: record.append(('draw', 'T'))),
    )
    gui_ctx.motion_settings = types.SimpleNamespace(
        ids={'verticalcontrol_id': vertical_control},
        update_xy_stage_control_gui=MagicMock(side_effect=lambda: record.append(('draw', 'XY'))),
    )
    return types.SimpleNamespace(
        ctx=gui_ctx,
        lane=gui_ctx.io_executor,
        record=record,
        posts=centre_posts,
        vertical_control=vertical_control,
    )


@pytest.fixture
def env(unhomed):
    """``unhomed``, homed: every axis knows its position."""
    home_sim_scope(unhomed.ctx.scope)
    return unhomed


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

    lane_worker = env.lane.call(
        IOTask(action=lambda: threading.current_thread().name), 'name', _WAIT_S
    )
    assert where == [lane_worker]
    assert env.record == [
        ('ask', ('X', 'Y', 'Z', 'T'), {'recording': False, 'then': 'go to the step'}),
        ('move',),
        ('draw', 'XY'),
        ('draw', 'Z'),
        ('draw', 'T'),
        ('moved',),
    ]
    assert env.posts == []


def test_a_refusal_moves_nothing_is_shown_once_and_still_redraws(unhomed):
    _gesture(unhomed, lambda: unhomed.record.append(('move',)))

    assert ('refused', 'position_unknown') in unhomed.record
    assert ('move',) not in unhomed.record
    assert ('moved',) not in unhomed.record, "a refused gesture ran the gesture's GUI work"
    assert ('draw', 'XY') in unhomed.record
    assert len(unhomed.posts) == 1, unhomed.posts


def test_a_move_refused_part_way_leaves_what_moved_and_skips_the_gui_work(env):
    refused = []

    def moves():
        env.ctx.scope.motion.start_move_z(5000.0)
        try:
            env.ctx.scope.motion.start_move_xy(y_um=999_999.0)
        except Exception as e:
            refused.append(e.reason)
            raise

    _gesture(env, moves)

    assert refused == ['position_out_of_range']
    assert env.ctx.scope.motion.get_target_positions()['Z'] == 5000.0, 'the Z that moved was undone'
    assert ('moved',) not in env.record
    assert ('draw', 'XY') in env.record
    assert len(env.posts) == 1, env.posts


def test_a_gesture_with_no_axes_asks_nothing_and_still_runs(env):
    _gesture(env, lambda: env.record.append(('move',)), axes=())

    assert env.record == [('move',), ('moved',)]


def test_a_locked_control_surface_submits_nothing(env, monkeypatch):
    monkeypatch.setattr(ui_helpers, '_user_motion_locked', lambda label: True)

    _gesture(env, lambda: env.record.append(('move',)))

    assert env.record == []


def test_a_lane_that_takes_no_work_skips_the_gui_work(env):
    closed = SequentialIOExecutor(name='CLOSED_IO')
    closed.start()
    closed.shutdown(wait=False)
    env.ctx.io_executor = closed

    ui_helpers.submit_gesture(
        'GESTURE',
        axes=('X',),
        then='move the stage',
        moves=lambda: env.record.append(('move',)),
        on_moved=lambda: env.record.append(('moved',)),
    )

    assert ('move',) not in env.record
    assert ('moved',) not in env.record
