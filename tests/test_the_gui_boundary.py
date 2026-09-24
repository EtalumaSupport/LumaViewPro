# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI hands an API call's outcome to the one reporter, then redraws from the API.

run_reported runs a call that does not wait on a lane where it is made;
submit_reported runs one that may wait on the GUI's worker pool. Either way
the outcome is shown as its type says, the redraw runs whatever happened, and
a redraw that raises is one reported fault, never an exit from a clock
callback. A blocking call handed to the inline form fails at once, by name,
before anything waits: on the GUI's thread a wait freezes the window.
"""

from __future__ import annotations

import threading
import types

import pytest

import modules.app_context as _app_ctx
from modules.exceptions import HardwareCommandRefusedError
from modules.notification_center import Severity
from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from tests.shown_outcomes import capture_shown
from ui import ui_helpers
from ui.ui_helpers import run_reported, submit_reported

_WAIT_S = 2.0


@pytest.fixture
def shown(monkeypatch):
    return capture_shown(monkeypatch)


@pytest.fixture
def lane():
    ex = SequentialIOExecutor(name='TEST_IO')
    ex.start()
    yield ex
    ex.shutdown(wait=False)


@pytest.fixture
def pool(monkeypatch):
    ex = SequentialIOExecutor(name='WORKER_POOL', priority_aware=True, lane=False)
    ex.start()
    monkeypatch.setattr(_app_ctx, 'ctx', types.SimpleNamespace(worker_pool=ex))
    # Headless: a scheduled redraw runs as soon as it is scheduled.
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    yield ex
    ex.shutdown(wait=False)


def _refuse():
    raise HardwareCommandRefusedError('exclusive_activity_running', 'select_objective', 'protocol')


def _fail():
    raise RuntimeError('board gone')


def _submit_and_wait(call, redraw, label):
    done = threading.Event()

    def _redraw():
        try:
            if redraw is not None:
                redraw()
        finally:
            done.set()

    submit_reported(call, _redraw, label)
    assert done.wait(_WAIT_S)


class TestTheInlineForm:
    def test_runs_the_call_on_the_calling_thread(self, shown):
        where = []
        run_reported(lambda: where.append(threading.current_thread()), None, 'TEST')
        assert where == [threading.current_thread()]

    @pytest.mark.parametrize(
        ('call', 'expected'),
        [
            (lambda: None, []),
            (_refuse, [(Severity.WARNING, 'Microscope Busy')]),
            (_fail, [(Severity.ERROR, 'Operation failed')]),
        ],
        ids=['success', 'refusal', 'fault'],
    )
    def test_shows_the_outcome_as_typed_and_redraws_after_it(self, shown, call, expected):
        order = []
        run_reported(call, lambda: order.append([(n.severity, n.title) for n in shown]), 'TEST')
        assert order == [expected], 'the redraw ran once, after the outcome was shown'

    def test_a_redraw_that_raises_is_one_reported_fault(self, shown):
        run_reported(lambda: None, _fail, 'TEST')
        assert [(n.severity, n.category) for n in shown] == [(Severity.ERROR, 'UI:TEST')]

    def test_a_call_that_would_wait_on_a_lane_fails_by_name_and_never_waits(self, shown, lane):
        ran = []
        run_reported(
            lambda: lane.call(IOTask(action=ran.append, args=(1,)), 'move', 5.0), None, 'T'
        )
        assert ran == []
        assert [(n.severity, n.category) for n in shown] == [(Severity.ERROR, 'UI:T')]

    def test_a_waited_motion_call_fails_by_name_and_is_never_submitted(self, shown):
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(), simulate=True)
        try:
            ran = []
            run_reported(
                lambda: session.scope.motion._submit_motion(
                    lambda: ran.append(1), 'move_absolute', wait_timeout=5.0
                ),
                None,
                'T',
            )
            assert ran == []
            assert [(n.severity, n.category) for n in shown] == [(Severity.ERROR, 'UI:T')]
        finally:
            session.shutdown()

    def test_the_mark_does_not_outlive_the_call(self, shown, lane):
        run_reported(_fail, None, 'T')
        run_reported(lambda: run_reported(lambda: None, None, 'INNER'), None, 'OUTER')
        # Outside the form a blocking call waits as it always did.
        assert lane.call(IOTask(action=lambda: 'answered'), 'read', 5.0) == 'answered'


class TestThePoolForm:
    def test_runs_the_call_on_the_worker_pool(self, shown, pool):
        where = []
        _submit_and_wait(lambda: where.append(threading.current_thread().name), None, 'TEST')
        assert where and where[0] != threading.current_thread().name

    @pytest.mark.parametrize(
        ('call', 'expected'),
        [
            (lambda: None, []),
            (_refuse, [(Severity.WARNING, 'Microscope Busy')]),
            (_fail, [(Severity.ERROR, 'Operation failed')]),
        ],
        ids=['success', 'refusal', 'fault'],
    )
    def test_shows_the_outcome_and_redraws_after_it(self, shown, pool, call, expected):
        seen_at_redraw = []
        _submit_and_wait(
            call, lambda: seen_at_redraw.append([(n.severity, n.title) for n in shown]), 'T'
        )
        assert seen_at_redraw == [expected]

    def test_a_blocking_member_waits_on_its_lane_from_the_pool(self, shown, pool, lane):
        answers = []
        _submit_and_wait(
            lambda: answers.append(lane.call(IOTask(action=lambda: 'moved'), 'move', 5.0)),
            None,
            'T',
        )
        assert answers == ['moved']
        assert shown == []

    def test_a_pool_that_takes_no_work_still_redraws(self, shown, pool):
        pool.disable()
        redrawn = []
        submit_reported(lambda: None, lambda: redrawn.append(True), 'T')
        assert redrawn == [True]


def test_a_burst_of_scroll_ticks_is_one_move_of_the_last_ticks_step(monkeypatch):
    from ui import shader

    steps = []
    moves = []
    monkeypatch.setattr(
        shader._app_ctx,
        'ctx',
        types.SimpleNamespace(
            scope=types.SimpleNamespace(
                motion=types.SimpleNamespace(
                    jog_step=lambda axis, coarse: (
                        steps.append(coarse) or (100.0 if coarse else 10.0)
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(ui_helpers, 'move_relative', lambda axis, um, **k: moves.append((axis, um)))
    viewer = types.SimpleNamespace(_scroll_z_pending=None)
    for pending in [(1.0, False), (2.0, False), (-1.5, True)]:
        viewer._scroll_z_pending = pending
    shader.ShaderViewer._flush_scroll_z(viewer, 0)

    assert moves == [('Z', -150.0)]
    assert steps == [True], 'the step is asked for once, at the move, with the last tick coarseness'
