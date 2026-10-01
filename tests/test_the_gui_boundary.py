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
import time
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
        pool.shutdown(wait=False)
        redrawn = []
        submit_reported(lambda: None, lambda: redrawn.append(True), 'T')
        assert redrawn == [True]

    def test_a_stop_goes_ahead_of_queued_requests(self, shown, pool):
        started = threading.Event()
        release = threading.Event()
        order = []

        def _hold():
            started.set()
            release.wait(_WAIT_S)

        submit_reported(_hold, None, 'HOLD')
        assert started.wait(_WAIT_S), 'the pool never ran the first request'
        submit_reported(lambda: order.append('queued'), None, 'QUEUED')
        done = threading.Event()
        submit_reported(lambda: order.append('stop'), done.set, 'STOP', stop=True)
        release.set()
        assert done.wait(_WAIT_S)
        assert order[0] == 'stop', f'a Stop waited behind earlier requests: {order}'


class TestTheLaneForm:
    """A call submitted to its member's lane redraws exactly once, whatever the lane does.

    A widget that shows a request as pending is cleared only by its redraw:
    a lost redraw leaves it dead, so every outcome is counted, including the
    lane refusing the task after it was queued.
    """

    @pytest.fixture
    def headless(self, monkeypatch, shown):
        from modules import notification_center, sequential_io_executor

        monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
        # The lane reports its refusals through the name it imported.
        monkeypatch.setattr(
            sequential_io_executor, 'notifications', notification_center.notifications
        )

    @pytest.fixture
    def claim(self, lane):
        from modules.activity_claim import ActivityClaim

        c = ActivityClaim()
        lane.ask_claim(c)
        return c

    @staticmethod
    def _settle(lane):
        # Everything the lane owes this submit has been answered once a task
        # queued after it has run.
        assert lane.call(IOTask(action=lambda: None), 'settle', _WAIT_S) is None

    def _submit(self, lane, call, label='T'):
        redrawn = []
        submit_reported(call, lambda: redrawn.append(label), label, lane=lane)
        self._settle(lane)
        return redrawn

    def test_runs_the_call_on_the_lane_and_redraws_once(self, shown, headless, lane):
        where = []
        redrawn = self._submit(lane, lambda: where.append(threading.current_thread().name))
        assert where == ['TEST_IO_WORKER']
        assert redrawn == ['T']
        assert shown == []

    def test_a_call_that_raises_is_one_fault_and_one_redraw(self, shown, headless, lane):
        redrawn = self._submit(lane, _fail)
        assert [(n.severity, n.category) for n in shown] == [(Severity.ERROR, 'UI:T')]
        assert redrawn == ['T']

    def test_a_refusal_at_submit_redraws_once(self, shown, headless, lane, claim):
        held = claim.try_claim('protocol', run_trigger_source='test')
        ran = []
        redrawn = []
        try:
            submit_reported(lambda: ran.append(1), lambda: redrawn.append('T'), 'T', lane=lane)
        finally:
            held.release()
        self._settle(lane)
        assert ran == []
        assert [n.title for n in shown] == ['Microscope Busy']
        assert redrawn == ['T']

    def test_a_refusal_while_queued_redraws_once(self, shown, headless, lane, claim):
        gate = threading.Event()
        started = threading.Event()

        def _busy():
            started.set()
            gate.wait(_WAIT_S)

        lane.put(IOTask(action=_busy))
        assert started.wait(_WAIT_S), 'the worker is running the task ahead of ours'
        ran = []
        redrawn = []
        submit_reported(lambda: ran.append(1), lambda: redrawn.append('T'), 'T', lane=lane)
        held = claim.try_claim('protocol', run_trigger_source='test')
        try:
            gate.set()
            deadline = time.monotonic() + _WAIT_S
            while not redrawn and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            held.release()
        self._settle(lane)
        assert ran == []
        assert [n.title for n in shown] == ['Microscope Busy']
        assert redrawn == ['T']

    def test_a_lane_that_takes_no_work_redraws_once(self, shown, headless, lane):
        lane.shutdown(wait=False)
        redrawn = []
        submit_reported(lambda: None, lambda: redrawn.append('T'), 'T', lane=lane)
        assert redrawn == ['T']

    def test_a_refusal_names_the_gesture_not_the_wrapper(self, shown, headless, lane, claim):
        held = claim.try_claim('protocol', run_trigger_source='test')
        try:
            submit_reported(lambda: None, None, 'LED_Blue', lane=lane)
        finally:
            held.release()
        self._settle(lane)
        assert [n.category for n in shown] == ['Task:UI:LED_Blue']


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
